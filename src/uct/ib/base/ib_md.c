/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2020. ALL RIGHTS RESERVED.
 * Copyright (C) The University of Tennessee and The University
 *               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ib_md.h"
#include "ib_device.h"
#include "ib_log.h"

#include <ucs/arch/atomic.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/module.h>
#include <ucs/sys/ptr_arith.h>
#include <ucs/sys/string.h>
#include <ucs/time/time.h>
#include <ucm/api/ucm.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <uct/api/v2/uct_v2.h>
#include <pthread.h>
#ifdef HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif
#include <sys/resource.h>


#define UCT_IB_MD_RCACHE_DEFAULT_ALIGN 16

static UCS_CONFIG_DEFINE_ARRAY(pci_bw,
                               sizeof(ucs_config_bw_spec_t),
                               UCS_CONFIG_TYPE_BW_SPEC);

static const char *uct_ib_devx_objs[] = {
    [UCT_IB_DEVX_OBJ_RCQP]  = "rcqp",
    [UCT_IB_DEVX_OBJ_RCSRQ] = "rcsrq",
    [UCT_IB_DEVX_OBJ_DCT]   = "dct",
    [UCT_IB_DEVX_OBJ_DCSRQ] = "dcsrq",
    [UCT_IB_DEVX_OBJ_DCI]   = "dci",
    [UCT_IB_DEVX_OBJ_CQ]    = "cq",
    NULL
};

ucs_config_field_t uct_ib_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_ib_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* TODO take default from device */
     ucs_offsetof(uct_ib_md_config_t, reg_cost.c), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
     ucs_offsetof(uct_ib_md_config_t, reg_cost.m), UCS_CONFIG_TYPE_TIME},

    {"FORK_INIT", "try",
     "Initialize a fork-safe IB library with ibv_fork_init().",
     ucs_offsetof(uct_ib_md_config_t, fork_init), UCS_CONFIG_TYPE_TERNARY},

    {"ASYNC_EVENTS", "y",
     "Enable listening for async events on the device",
     ucs_offsetof(uct_ib_md_config_t, async_events), UCS_CONFIG_TYPE_BOOL},

    {"ETH_PAUSE_ON", "y",
     "Whether or not 'Pause Frame' is enabled on an Ethernet network.\n"
     "Pause frame is a mechanism for temporarily stopping the transmission of data to\n"
     "ensure zero loss under congestion on Ethernet family computer networks.\n"
     "This parameter, if set to 'no', will disqualify IB transports that may not perform\n"
     "well on a lossy fabric when working with RoCE.",
     ucs_offsetof(uct_ib_md_config_t, ext.eth_pause), UCS_CONFIG_TYPE_BOOL},

    {"ODP_PREFETCH", "n",
     "Force prefetch of memory regions created with ODP.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.odp.prefetch), UCS_CONFIG_TYPE_BOOL},

    {"DEVICE_SPECS", "",
     "Array of custom device specification. Each element is a string of the following format:\n"
     "  <vendor-id>:<device-id>[:name[:<flags>[:<priority>]]]\n"
     "where:\n"
     "  <vendor-id> - (mandatory) pci vendor id, integer or hexadecimal.\n"
     "  <device-id> - (mandatory) pci device id, integer or hexadecimal.\n"
     "  <name>      - (optional) device name.\n"
     "  <flags>     - (optional) empty, or a combination of:\n"
     "                             '4' - mlx4 device\n"
     "                             '5' - mlx5 device\n"
     "                             'd' - DC version 1 (Connect-IB, ConnectX-4)\n"
     "                             'D' - DC version 2 (ConnectX-5 and above)\n"
     "                             'a' - Compact address vector support\n"
     "  <priority>  - (optional) device priority, integer.\n"
     "\n"
     "Example: The value '0x02c9:4115:ConnectX4:5d' would specify a device named ConnectX-4\n"
     "to match vendor id 0x2c9, device id 4115, with DC version 1 support.",
     ucs_offsetof(uct_ib_md_config_t, custom_devices), UCS_CONFIG_TYPE_STRING_ARRAY},

    {"PREFER_NEAREST_DEVICE", "y",
     "Prefer nearest device to cpu when selecting a device from NET_DEVICES list.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.prefer_nearest_device), UCS_CONFIG_TYPE_BOOL},

    {"INDIRECT_ATOMIC", "y",
     "Use indirect atomic\n",
     ucs_offsetof(uct_ib_md_config_t, ext.enable_indirect_atomic), UCS_CONFIG_TYPE_BOOL},

    {"GID_INDEX", "auto",
     "Port GID index to use.",
     ucs_offsetof(uct_ib_md_config_t, ext.gid_index), UCS_CONFIG_TYPE_ULUNITS},

    {"SUBNET_PREFIX", "",
     "Infiniband subnet prefix to filter ports by, empty means no filter. "
     "Relevant for IB link layer only\n"
     "For example a filter for the default subnet prefix can be specified as: fe80:0:0:0",
     ucs_offsetof(uct_ib_md_config_t, subnet_prefix), UCS_CONFIG_TYPE_STRING},

    {"GPU_DIRECT_RDMA", "try",
     "Use GPU Direct RDMA for HCA to access GPU pages directly\n",
     ucs_offsetof(uct_ib_md_config_t, enable_gpudirect_rdma), UCS_CONFIG_TYPE_TERNARY},

    {"PCI_BW", "",
     "Maximum effective data transfer rate of PCI bus connected to HCA\n",
     ucs_offsetof(uct_ib_md_config_t, pci_bw), UCS_CONFIG_TYPE_ARRAY(pci_bw)},

    {"MLX5_DV", "try",
     "MLX5 support\n",
     ucs_offsetof(uct_ib_md_config_t, mlx5dv), UCS_CONFIG_TYPE_TERNARY},

    {"MLX5_DEVX", "try",
     "DEVX support\n",
     ucs_offsetof(uct_ib_md_config_t, devx), UCS_CONFIG_TYPE_TERNARY},

    {"MLX5_DEVX_OBJECTS", "rcqp,rcsrq,dct,dcsrq,dci,cq",
     "Objects to be created by DEVX\n",
     ucs_offsetof(uct_ib_md_config_t, devx_objs),
     UCS_CONFIG_TYPE_BITMAP(uct_ib_devx_objs)},

    {"REG_MT_THRESH", "4G",
     "Minimal MR size to be register using multiple parallel threads.\n"
     "Number of threads used will be determined by number of CPUs which "
     "registering thread is bound to by hard affinity.",
     ucs_offsetof(uct_ib_md_config_t, ext.min_mt_reg), UCS_CONFIG_TYPE_MEMUNITS},

    {"REG_MT_CHUNK", "2G",
     "Size of single chunk used in multithreaded registration.\n"
     "Must be power of 2.",
     ucs_offsetof(uct_ib_md_config_t, ext.mt_reg_chunk), UCS_CONFIG_TYPE_MEMUNITS},

    {"REG_MT_BIND", "n",
     "Enable setting CPU affinity of memory registration threads.",
     ucs_offsetof(uct_ib_md_config_t, ext.mt_reg_bind), UCS_CONFIG_TYPE_BOOL},

    {"PCI_RELAXED_ORDERING", "auto",
     "Enable relaxed ordering for PCIe transactions to improve performance on some systems.",
     ucs_offsetof(uct_ib_md_config_t, mr_relaxed_order), UCS_CONFIG_TYPE_TERNARY_AUTO},

    {"MAX_IDLE_RKEY_COUNT", "16",
     "Maximal number of invalidated memory keys that are kept idle before reuse.",
     ucs_offsetof(uct_ib_md_config_t, ext.max_idle_rkey_count), UCS_CONFIG_TYPE_UINT},

    {"REG_RETRY_CNT", "inf",
     "Number of memory registration attempts.",
     ucs_offsetof(uct_ib_md_config_t, ext.reg_retry_cnt), UCS_CONFIG_TYPE_ULUNITS},

    {"SMKEY_BLOCK_SIZE", "8",
     "Number of indexes in a symmetric block. More can lead to less contention.",
     ucs_offsetof(uct_ib_md_config_t, ext.smkey_block_size), UCS_CONFIG_TYPE_UINT},

    {"XGVMI_UMR_ENABLE", "y",
     "Enable UMR optimization for XGVMI mkeys export/import.",
     ucs_offsetof(uct_ib_md_config_t, xgvmi_umr_enable), UCS_CONFIG_TYPE_BOOL},

    {"ODP_MEM_TYPES", "host",
     "Advertise non-blocking registration for these memory types, when ODP is "
     "enabled.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.odp.mem_types),
     UCS_CONFIG_TYPE_BITMAP(ucs_memory_type_names)},

    {NULL}
};

#ifdef ENABLE_STATS
static ucs_stats_class_t uct_ib_md_stats_class = {
    .name          = "",
    .num_counters  = UCT_IB_MD_STAT_LAST,
    .class_id      = UCS_STATS_CLASS_ID_INVALID,
    .counter_names = {
        [UCT_IB_MD_STAT_MEM_ALLOC]   = "mem_alloc",
        [UCT_IB_MD_STAT_MEM_REG]     = "mem_reg"
    }
};
#endif


extern uct_tl_t UCT_TL_NAME(rc_verbs);
extern uct_tl_t UCT_TL_NAME(ud_verbs);

static uct_tl_t *uct_ib_tls[] = {
#ifdef HAVE_TL_RC
    &UCT_TL_NAME(rc_verbs),
#endif
#ifdef HAVE_TL_UD
    &UCT_TL_NAME(ud_verbs),
#endif
};

static uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(verbs);

UCS_LIST_HEAD(uct_ib_ops);

typedef struct {
    uct_ib_mem_t        super;
    uct_ib_mr_t         mrs[];
} uct_ib_verbs_mem_t;

typedef struct {
    pthread_t                     thread;
    uct_ib_md_t                   *md;
    void                          *address;
    size_t                        length;
    size_t                        first_mr_size;
    const uct_md_mem_reg_params_t *params;
    uint64_t                      access_flags;
    struct ibv_mr                 **mrs;
} uct_ib_md_mem_reg_thread_t;

ucs_status_t uct_ib_md_query(uct_md_h uct_md, uct_md_attr_v2_t *md_attr)
{
    uct_ib_md_t *md              = ucs_derived_of(uct_md, uct_ib_md_t);
    size_t component_name_length = strlen(md->super.component->name);
    uint64_t guid                = IBV_DEV_ATTR(&md->dev, sys_image_guid);

    uct_md_base_md_query(md_attr);
    md_attr->max_alloc                 = ULONG_MAX; /* TODO query device */
    md_attr->max_reg                   = ULONG_MAX; /* TODO query device */
    md_attr->flags                     = md->cap_flags;
    md_attr->access_mem_types          = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->reg_mem_types             = md->reg_mem_types;
    md_attr->gva_mem_types             = md->gva_mem_types;
    md_attr->reg_nonblock_mem_types    = md->reg_nonblock_mem_types;
    md_attr->cache_mem_types           = UCS_MASK(UCS_MEMORY_TYPE_LAST);
    md_attr->rkey_packed_size          = UCT_IB_MD_PACKED_RKEY_SIZE;
    md_attr->reg_cost                  = md->reg_cost;
    md_attr->exported_mkey_packed_size = sizeof(uct_ib_md_packed_mkey_t);

    ucs_sys_cpuset_copy(&md_attr->local_cpus, &md->dev.local_cpus);
    UCS_STATIC_ASSERT(sizeof(guid) <=
                      (UCT_MD_GLOBAL_ID_MAX - UCT_COMPONENT_NAME_MAX));
    memcpy(md_attr->global_id, md->super.component->name, component_name_length);
    memcpy(UCS_PTR_BYTE_OFFSET(md_attr->global_id, component_name_length),
           &guid, sizeof(guid));

    return UCS_OK;
}

static void
uct_ib_md_print_mem_reg_err_msg(const char *title, void *address, size_t length,
                                uint64_t access_flags, int err, int silent)
{
    ucs_log_level_t level = silent ? UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR;
    UCS_STRING_BUFFER_ONSTACK(msg, 256);
    size_t page_size;
    size_t unused;

    ucs_string_buffer_appendf(
            &msg, "%s(address=%p, length=%zu, access=0x%lx) failed: %s", title,
            address, length, access_flags, strerror(err));

    if (err == EINVAL) {
        /* Check if huge page is used */
        ucs_get_mem_page_size(address, length, &unused, &page_size);
        if (page_size != ucs_get_page_size()) {
            ucs_string_buffer_appendf(&msg,
                                      ". Application is using HUGE pages. "
                                      "Please set environment variable "
                                      "RDMAV_HUGEPAGES_SAFE=1");
        }
    }

    uct_ib_memlock_limit_msg(&msg, err);

    ucs_log(level, "%s", ucs_string_buffer_cstr(&msg));
}

void *uct_ib_md_mem_handle_thread_func(void *arg)
{
    uct_ib_md_mem_reg_thread_t *ctx = arg;
    size_t chunk_size               = ctx->md->config.mt_reg_chunk;
    ucs_time_t UCS_V_UNUSED t0      = ucs_get_time();
    void UCS_V_UNUSED *start        = ctx->address;
    int mr_idx                      = 0;
    size_t length                   = ctx->first_mr_size;
    ucs_status_t status;

    while (ctx->length > 0) {
        if (ctx->params != NULL) {
            status = uct_ib_reg_mr(ctx->md, ctx->address, length, ctx->params,
                                   ctx->access_flags, NULL, &ctx->mrs[mr_idx]);
            if (status != UCS_OK) {
                goto err_dereg;
            }
        } else {
            status = uct_ib_dereg_mr(ctx->mrs[mr_idx]);
            if (status != UCS_OK) {
                goto err;
            }
        }
        ctx->address = UCS_PTR_BYTE_OFFSET(ctx->address, length);
        ctx->length -= length;
        length       = ucs_min(ctx->length, chunk_size);
        mr_idx++;
    }

    ucs_trace("%s %p..%p (first_mr_size %zu) took %f usec\n",
              (ctx->params != NULL) ? "reg_mr" : "dereg_mr",
              start, ctx->address, ctx->first_mr_size,
              ucs_time_to_usec(ucs_get_time() - t0));
    return UCS_STATUS_PTR(UCS_OK);

err_dereg:
    for (; mr_idx >= 0; --mr_idx) {
        uct_ib_dereg_mr(ctx->mrs[mr_idx]);
    }
err:
    return UCS_STATUS_PTR(status);
}

ucs_status_t
uct_ib_md_handle_mr_list_mt(uct_ib_md_t *md, void *address, size_t length,
                            const uct_md_mem_reg_params_t *params,
                            uint64_t access_flags, size_t mr_num,
                            struct ibv_mr **mrs)
{
    size_t chunk_size = md->config.mt_reg_chunk;
    int thread_num_mrs, thread_num, thread_idx, mr_idx, cpu_id;
    ucs_sys_cpuset_t parent_set, thread_set;
    uct_ib_md_mem_reg_thread_t *ctxs, *ctx;
    char UCS_V_UNUSED affinity_str[64];
    pthread_attr_t attr;
    ucs_status_t status;
    void *thread_status;
    uint64_t offset;
    size_t padding;
    int ret;

    status = ucs_sys_pthread_getaffinity(&parent_set);
    if (status != UCS_OK) {
        return status;
    }

    thread_num = ucs_min(CPU_COUNT(&parent_set), mr_num);
    if (thread_num == 1) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_trace("multithreaded %s %p..%p threads %d affinity %s\n",
              (params != NULL) ? "reg" : "dereg", address,
              UCS_PTR_BYTE_OFFSET(address, length), thread_num,
              ucs_make_affinity_str(&parent_set, affinity_str,
                                    sizeof(affinity_str)));

    ctxs = ucs_calloc(thread_num, sizeof(*ctxs), "ib mr ctxs");
    if (ctxs == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    pthread_attr_init(&attr);

    status = UCS_OK;
    mr_idx = 0;
    cpu_id = 0;
    offset = 0;
    for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        /* calculate number of mrs for each thread so each one will
         * get proportional amount */
        thread_num_mrs    = ucs_div_round_up(mr_num - mr_idx,
                                             thread_num - thread_idx);
        ctx               = &ctxs[thread_idx];
        ctx->md           = md;
        ctx->address      = UCS_PTR_BYTE_OFFSET(address, offset);
        ctx->params       = params;
        ctx->access_flags = access_flags;
        ctx->mrs          = &mrs[mr_idx];

        /* First MR size can be different to align further MRs */
        padding            = ucs_padding((uintptr_t)ctx->address, chunk_size);
        ctx->first_mr_size = (padding > 0) ? padding : chunk_size;
        ctx->first_mr_size = ucs_min(ctx->first_mr_size, length - offset);
        ucs_assertv((ctx->address == address) || (padding == 0),
                    "thread_idx=%d address=%p padding=%zu",
                    thread_idx, address, padding);

        ctx->length        = (thread_num_mrs - 1) * chunk_size +
                             ctx->first_mr_size;
        ctx->length        = ucs_min(ctx->length, length - offset);
        offset            += ctx->length;

        if (md->config.mt_reg_bind) {
            while (!CPU_ISSET(cpu_id, &parent_set)) {
                cpu_id++;
            }

            CPU_ZERO(&thread_set);
            CPU_SET(cpu_id, &thread_set);
            cpu_id++;
            pthread_attr_setaffinity_np(&attr, sizeof(ucs_sys_cpuset_t),
                                        &thread_set);
        }

        ret = pthread_create(&ctx->thread, &attr,
                             uct_ib_md_mem_handle_thread_func, ctx);
        if (ret != 0) {
            ucs_error("pthread_create() failed: %m");
            status     = UCS_ERR_IO_ERROR;
            thread_num = thread_idx;
            break;
        }

        mr_idx += thread_num_mrs;
    }

    for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        ctx = &ctxs[thread_idx];
        pthread_join(ctx->thread, &thread_status);
        if (UCS_PTR_IS_ERR(thread_status)) {
            status = UCS_PTR_STATUS(thread_status);
        }
    }

    ucs_free(ctxs);
    pthread_attr_destroy(&attr);

    if (status != UCS_OK) {
        for (mr_idx = 0; mr_idx < mr_num; mr_idx++) {
            /* coverity[check_return] */
            uct_ib_dereg_mr(mrs[mr_idx]);
        }
    }

    return status;
}

ucs_status_t uct_ib_reg_mr(uct_ib_md_t *md, void *address, size_t length,
                           const uct_md_mem_reg_params_t *params,
                           uint64_t access_flags, struct ibv_dm *dm,
                           struct ibv_mr **mr_p)
{
    ucs_time_t UCS_V_UNUSED start_time = ucs_get_time();
    unsigned long retry                = 0;
    size_t UCS_V_UNUSED dmabuf_offset;
    const char *title;
    struct ibv_mr *mr;
    uint64_t flags;
    int dmabuf_fd;

    flags         = UCT_MD_MEM_REG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0);
    dmabuf_fd     = UCS_PARAM_VALUE(UCT_MD_MEM_REG_FIELD, params, dmabuf_fd,
                                    DMABUF_FD, UCT_DMABUF_FD_INVALID);
    dmabuf_offset = UCS_PARAM_VALUE(UCT_MD_MEM_REG_FIELD, params, dmabuf_offset,
                                    DMABUF_OFFSET, 0);

    if (dm != NULL) {
#if HAVE_IBV_DM
        title = "ibv_reg_dm_mr";
        mr    = UCS_PROFILE_CALL_ALWAYS(ibv_reg_dm_mr, md->pd, dm, 0, length,
                                        access_flags | IBV_ACCESS_ZERO_BASED);
#else
        return UCS_ERR_UNSUPPORTED;
#endif
    } else if (dmabuf_fd == UCT_DMABUF_FD_INVALID) {
        title = "ibv_reg_mr";
        do {
            /* when access_flags contains IBV_ACCESS_ON_DEMAND ibv_reg_mr() may
             * fail with EAGAIN. It means prefetch failed due to collision
             * with invalidation */
            mr = UCS_PROFILE_CALL_ALWAYS(ibv_reg_mr, md->pd, address, length,
                                         access_flags);
        } while ((mr == NULL) && (errno == EAGAIN) &&
                 (retry++ < md->config.reg_retry_cnt));
    } else {
#if HAVE_DECL_IBV_REG_DMABUF_MR
        title = "ibv_reg_dmabuf_mr";
        mr = UCS_PROFILE_CALL_ALWAYS(ibv_reg_dmabuf_mr, md->pd, dmabuf_offset,
                                     length, (uintptr_t)address, dmabuf_fd,
                                     access_flags);
#else
        return UCS_ERR_UNSUPPORTED;
#endif
    }
    if (mr == NULL) {
        uct_ib_md_print_mem_reg_err_msg(title, address, length, access_flags,
                                        errno,
                                        flags & UCT_MD_MEM_FLAG_HIDE_ERRORS);
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("%s(pd=%p addr=%p len=%zu fd=%d offset=%zu access=0x%" PRIx64 "):"
              " mr=%p lkey=0x%x retry=%lu took %.3f ms",
              title, md->pd, address, length, dmabuf_fd, dmabuf_offset,
              access_flags, mr, mr->lkey, retry,
              ucs_time_to_msec(ucs_get_time() - start_time));
    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_REG, +1);

    *mr_p = mr;
    return UCS_OK;
}

ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr)
{
    int ret;

    ucs_trace("ibv_dereg_mr(mr=%p addr=%p length=%zu)", mr, mr->addr,
              mr->length);

    ret = UCS_PROFILE_CALL(ibv_dereg_mr, mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_ib_mem_prefetch(uct_ib_md_t *md, uct_ib_mem_t *ib_memh,
                                 void *addr, size_t length)
{
#if HAVE_DECL_IBV_ADVISE_MR
    unsigned long retry = 0;
    struct ibv_sge sg_list;
    int ret;

    if (!(ib_memh->flags & UCT_IB_MEM_FLAG_ODP)) {
        return UCS_OK;
    }

    ucs_debug("memh %p prefetch %p length %zu", ib_memh, addr, length);

    sg_list.lkey   = ib_memh->lkey;
    sg_list.addr   = (uintptr_t)addr;
    sg_list.length = length;

    do {
        ret = UCS_PROFILE_CALL(ibv_advise_mr, md->pd,
                               IBV_ADVISE_MR_ADVICE_PREFETCH_WRITE,
                               IBV_ADVISE_MR_FLAG_FLUSH, &sg_list, 1);
    } while ((ret == EAGAIN) && (retry++ < md->config.reg_retry_cnt));

    if (ret) {
        ucs_diag("ibv_advise_mr(addr=%p length=%zu key=%x) returned %d: %m",
                 addr, length, ib_memh->lkey, ret);
        return UCS_ERR_IO_ERROR;
    }
#endif
    return UCS_OK;
}

ucs_status_t uct_ib_mem_advise(uct_md_h uct_md, uct_mem_h memh, void *addr,
                               size_t length, unsigned advice)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    ucs_debug("memh %p advice %d", memh, advice);
    if ((advice == UCT_MADV_WILLNEED) && !md->config.odp.prefetch) {
        return uct_ib_mem_prefetch(md, memh, addr, length);
    }

    return UCS_OK;
}

ucs_status_t uct_ib_memh_alloc(uct_ib_md_t *md, size_t length,
                               unsigned mem_flags, size_t memh_base_size,
                               size_t mr_size, uct_ib_mem_t **memh_p)
{
    int num_mrs = md->relaxed_order ?
                          2 /* UCT_IB_MR_DEFAULT and UCT_IB_MR_STRICT_ORDER */ :
                          1 /* UCT_IB_MR_DEFAULT */;
    uct_ib_mem_t *memh;

    memh = ucs_calloc(1, memh_base_size + (mr_size * num_mrs), "ib_memh");
    if (memh == NULL) {
        ucs_error("%s: failed to allocated memh struct",
                  uct_ib_device_name(&md->dev));
        return UCS_ERR_NO_MEMORY;
    }

    memh->lkey = UCT_IB_INVALID_MKEY;
    memh->rkey = UCT_IB_INVALID_MKEY;

    if ((mem_flags & UCT_MD_MEM_FLAG_NONBLOCK) && (length > 0) &&
        (md->reg_nonblock_mem_types != 0)) {
        /* Registration will fail if memory does not actually support it */
        memh->flags |= UCT_IB_MEM_FLAG_ODP;
    }

    if (mem_flags & UCT_MD_MEM_ACCESS_REMOTE_ATOMIC) {
        memh->flags |= UCT_IB_MEM_ACCESS_REMOTE_ATOMIC;
    }

    if (mem_flags &
        (UCT_MD_MEM_ACCESS_REMOTE_GET | UCT_MD_MEM_ACCESS_REMOTE_PUT)) {
        memh->flags |= UCT_IB_MEM_ACCESS_REMOTE_RMA;
    }

    if (mem_flags & UCT_MD_MEM_GVA) {
        memh->flags |= UCT_IB_MEM_FLAG_GVA;
    }

    *memh_p = memh;
    return UCS_OK;
}

uint64_t uct_ib_memh_access_flags(uct_ib_mem_t *memh, int relaxed_order)
{
    uint64_t access_flags = UCT_IB_MEM_ACCESS_FLAGS;

    if (memh->flags & UCT_IB_MEM_FLAG_ODP) {
        access_flags |= IBV_ACCESS_ON_DEMAND;
    }

    if (relaxed_order) {
        access_flags |= IBV_ACCESS_RELAXED_ORDERING;
    }

    return access_flags;
}

ucs_status_t uct_ib_verbs_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                  const uct_md_mem_reg_params_t *params,
                                  uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    struct ibv_mr *mr_default;
    uct_ib_verbs_mem_t *memh;
    uct_ib_mem_t *ib_memh;
    uint64_t access_flags;
    ucs_status_t status;

    status = uct_ib_memh_alloc(md, length,
                               UCT_MD_MEM_REG_FIELD_VALUE(params, flags,
                                                          FIELD_FLAGS, 0),
                               sizeof(*memh), sizeof(memh->mrs[0]), &ib_memh);
    if (status != UCS_OK) {
        goto err;
    }

    memh         = ucs_derived_of(ib_memh, uct_ib_verbs_mem_t);
    access_flags = uct_ib_memh_access_flags(&memh->super, md->relaxed_order);

    status = uct_ib_reg_mr(md, address, length, params, access_flags, NULL,
                           &mr_default);
    if (status != UCS_OK) {
        goto err_free;
    }

    memh->super.lkey                = mr_default->lkey;
    memh->super.rkey                = mr_default->rkey;
    memh->mrs[UCT_IB_MR_DEFAULT].ib = mr_default;

    if (md->relaxed_order) {
        status = uct_ib_reg_mr(md, address, length, params,
                               access_flags & ~IBV_ACCESS_RELAXED_ORDERING,
                               NULL, &memh->mrs[UCT_IB_MR_STRICT_ORDER].ib);
        if (status != UCS_OK) {
            goto err_dereg_default;
        }
    }

    if (md->config.odp.prefetch) {
        uct_ib_mem_prefetch(md, &memh->super, address, length);
    }

    *memh_p = memh;
    return UCS_OK;

err_dereg_default:
    uct_ib_dereg_mr(mr_default);
err_free:
    ucs_free(memh);
err:
    return status;
}

ucs_status_t
uct_ib_verbs_mem_dereg(uct_md_h uct_md, const uct_md_mem_dereg_params_t *params)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_verbs_mem_t *memh;
    ucs_status_t status;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    memh = params->memh;

    if (md->relaxed_order) {
        status = uct_ib_dereg_mr(memh->mrs[UCT_IB_MR_STRICT_ORDER].ib);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = uct_ib_dereg_mr(memh->mrs[UCT_IB_MR_DEFAULT].ib);
    if (status != UCS_OK) {
        return status;
    }

    ucs_free(memh);
    return UCS_OK;
}

ucs_status_t uct_ib_verbs_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                                    void *address, size_t length,
                                    const uct_md_mkey_pack_params_t *params,
                                    void *mkey_buffer)
{
    uct_ib_md_t *md                 = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_verbs_mem_t *memh        = uct_memh;
    uct_ib_mr_type_t atomic_mr_type = uct_ib_md_get_atomic_mr_type(md);
    unsigned flags;

    flags = UCS_PARAM_VALUE(UCT_MD_MKEY_PACK_FIELD, params, flags, FLAGS, 0);
    if (flags &
        (UCT_MD_MKEY_PACK_FLAG_INVALIDATE_RMA |
         UCT_MD_MKEY_PACK_FLAG_INVALIDATE_AMO | UCT_MD_MKEY_PACK_FLAG_EXPORT)) {
        return UCS_ERR_UNSUPPORTED;
    }

    uct_ib_md_pack_rkey(memh->super.rkey, memh->mrs[atomic_mr_type].ib->rkey,
                        mkey_buffer);
    return UCS_OK;
}

ucs_status_t uct_ib_rkey_unpack(uct_component_t *component,
                                const void *rkey_buffer, uct_rkey_t *rkey_p,
                                void **handle_p)
{
    uint64_t packed_rkey = *(const uint64_t*)rkey_buffer;

    *rkey_p   = packed_rkey;
    *handle_p = NULL;
    ucs_trace("unpacked rkey 0x%llx: direct 0x%x atomic 0x%x",
              (unsigned long long)packed_rkey, uct_ib_md_direct_rkey(*rkey_p),
              uct_ib_md_atomic_rkey(*rkey_p));
    return UCS_OK;
}

static const char *uct_ib_device_transport_type_name(struct ibv_device *device)
{
    switch (device->transport_type) {
    case IBV_TRANSPORT_IB:
        return "InfiniBand";
    case IBV_TRANSPORT_IWARP:
        return "iWARP";
#if HAVE_DECL_IBV_TRANSPORT_USNIC
    case IBV_TRANSPORT_USNIC:
        return "usNIC";
#endif
#if HAVE_DECL_IBV_TRANSPORT_USNIC_UDP
    case IBV_TRANSPORT_USNIC_UDP:
        return "usNIC UDP";
#endif
#if HAVE_DECL_IBV_TRANSPORT_UNSPECIFIED
    case IBV_TRANSPORT_UNSPECIFIED:
        return "Unspecified";
#endif
    default:
        return "Unknown";
    }
}

static int uct_ib_device_is_supported(struct ibv_device *device)
{
    /* TODO: enable additional transport types when ready */
    int ret = device->transport_type == IBV_TRANSPORT_IB;
    if (!ret) {
        ucs_debug("device %s of type %s is not supported",
                  device->dev_name, uct_ib_device_transport_type_name(device));
    }

    return ret;
}

int uct_ib_device_is_accessible(struct ibv_device *device)
{
    /* Enough place to hold the full path */
    char device_path[IBV_SYSFS_PATH_MAX];
    struct stat st;

    ucs_snprintf_safe(device_path, sizeof(device_path), "%s%s",
                      "/dev/infiniband/", device->dev_name);

    /* Could not stat the path or
       the path is not a char device file or
       the device cannot be accessed for read & write
    */
    if ((stat(device_path, &st) != 0) || !S_ISCHR(st.st_mode) ||
        (access(device_path, R_OK | W_OK) != 0)) {
        return 0;
    }

    return uct_ib_device_is_supported(device);
}

ucs_status_t uct_ib_query_md_resources(uct_component_t *component,
                                       uct_md_resource_desc_t **resources_p,
                                       unsigned *num_resources_p)
{
    int num_resources = 0;
    uct_md_resource_desc_t *resources;
    struct ibv_device **device_list;
    ucs_status_t status;
    int i, num_devices;

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if ((device_list == NULL) || (num_devices == 0)) {
        *resources_p     = NULL;
        *num_resources_p = 0;

        if (device_list != NULL) {
            ucs_debug("no devices are found");
            status = UCS_OK;
            goto out_free_device_list;
        } else if (errno == ENOSYS) {
            ucs_debug("failed to get ib device list: no kernel support for "
                      "rdma");
        } else {
            ucs_debug("failed to get ib device list: %m");
        }

        return UCS_OK;
    }

    resources = ucs_calloc(num_devices, sizeof(*resources), "ib_resources");
    if (resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_device_list;
    }

    for (i = 0; i < num_devices; ++i) {
        /* Skip non-existent and non-accessible devices */
        if (!uct_ib_device_is_accessible(device_list[i])) {
            continue;
        }

        ucs_snprintf_zero(resources[num_resources].md_name,
                          sizeof(resources[num_resources].md_name),
                          "%s", ibv_get_device_name(device_list[i]));
        num_resources++;
    }

    *resources_p     = resources;
    *num_resources_p = num_resources;
    status = UCS_OK;

out_free_device_list:
    ibv_free_device_list(device_list);
    return status;
}

static void uct_ib_fork_warn()
{
    ucs_warn("IB: ibv_fork_init() was disabled or failed, yet a fork() has been issued.");
    ucs_warn("IB: data corruption might occur when using registered memory.");
}

static void uct_ib_fork_warn_enable()
{
    static volatile uint32_t enabled = 0;
    int ret;

    if (ucs_atomic_cswap32(&enabled, 0, 1) != 0) {
        return;
    }

    ret = pthread_atfork(uct_ib_fork_warn, NULL, NULL);
    if (ret) {
        ucs_warn("registering fork() warning failed: %m");
    }
}

static void uct_ib_md_release_device_config(uct_ib_md_t *md)
{
    unsigned i;

    for (i = 0; i < md->custom_devices.count; ++i) {
        free((char*)md->custom_devices.specs[i].name);
    }
    ucs_free(md->custom_devices.specs);
}

static ucs_status_t
uct_ib_md_parse_device_config(uct_ib_md_t *md, const uct_ib_md_config_t *md_config)
{
    char *flags_str = NULL;
    uct_ib_device_spec_t *spec;
    ucs_status_t status;
    char *p;
    unsigned i, count;
    int nfields;

    count = md->custom_devices.count = md_config->custom_devices.count;
    if (count == 0) {
        md->custom_devices.specs = NULL;
        md->custom_devices.count = 0;
        return UCS_OK;
    }

    md->custom_devices.specs = ucs_calloc(count, sizeof(*md->custom_devices.specs),
                                          "ib_custom_devices");
    if (md->custom_devices.specs == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    for (i = 0; i < count; ++i) {
        spec = &md->custom_devices.specs[i];
        nfields = sscanf(md_config->custom_devices.spec[i],
                         "%hi:%hi:%m[^:]:%m[^:]:%hhu",
                         &spec->pci_id.vendor, &spec->pci_id.device, &spec->name,
                         &flags_str, &spec->priority);
        if (nfields < 2) {
            ucs_error("failed to parse device config '%s' (parsed: %d/%d)",
                      md_config->custom_devices.spec[i], nfields, 5);
            status = UCS_ERR_INVALID_PARAM;
            goto err_free;
        }

        if (nfields >= 4) {
            /* Check that 'flags_str' is not NULL to suppress the Coverity warning */
            ucs_assert(flags_str != NULL);

            for (p = flags_str; *p != 0; ++p) {
                if (*p == '4') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_MLX4_PRM;
                } else if (*p == '5') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
                } else if (*p == 'd') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_DC_V1;
                } else if (*p == 'D') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_DC_V2;
                } else if (*p == 'a') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_AV;
                } else {
                    ucs_error("invalid device flag: '%c'", *p);
                    free(flags_str);
                    flags_str = NULL;
                    status = UCS_ERR_INVALID_PARAM;
                    goto err_free;
                }
            }
            free(flags_str);
            flags_str = NULL;
        }

        ucs_trace("added device '%s' vendor_id 0x%x device_id %d flags %c%c prio %d",
                  spec->name, spec->pci_id.vendor, spec->pci_id.device,
                  (spec->flags & UCT_IB_DEVICE_FLAG_MLX4_PRM) ? '4' : '-',
                  (spec->flags & UCT_IB_DEVICE_FLAG_MLX5_PRM) ? '5' : '-',
                  spec->priority);
    }

    return UCS_OK;

err_free:
    uct_ib_md_release_device_config(md);
err:
    return status;
}

static ucs_status_t
uct_ib_md_parse_subnet_prefix(const char *subnet_prefix_str,
                              uint64_t *subnet_prefix)
{
    uint16_t pfx[4] = {0};
    uint64_t pfx64 = 0;
    int res, i;

    res = sscanf(subnet_prefix_str, "%hx:%hx:%hx:%hx",
                 &pfx[0], &pfx[1], &pfx[2], &pfx[3]);
    if (res != 4) {
        ucs_error("subnet filter '%s' is invalid", subnet_prefix_str);
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < 4; i++) {
        pfx64 = pfx[i] + (pfx64 << 16);
    }

    *subnet_prefix = htobe64(pfx64);
    return UCS_OK;
}

static void
uct_ib_md_set_pci_bw(uct_ib_md_t *md, const uct_ib_md_config_t *md_config)
{
    const char *device_name = uct_ib_device_name(&md->dev);
    unsigned i;

    for (i = 0; i < md_config->pci_bw.count; i++) {
        if (!strcmp(device_name, md_config->pci_bw.device[i].name)) {
            if (UCS_CONFIG_DBL_IS_AUTO(md_config->pci_bw.device[i].bw)) {
                break; /* read data from system */
            }

            md->pci_bw = md_config->pci_bw.device[i].bw;
            return;
        }
    }

    /* Did not find a matching configuration - take from underlying device */
    md->pci_bw = md->dev.pci_bw;
}

static ucs_status_t uct_ib_component_md_open(struct ibv_device *ib_device,
                                             const uct_ib_md_config_t *md_config,
                                             const char *md_name,
                                             struct uct_ib_md **md_p)
{
    struct uct_ib_md *md;
    ucs_status_t status;
    uct_ib_md_ops_entry_t *entry;

    ucs_list_for_each(entry, &uct_ib_ops, list) {
        status = entry->ops->open(ib_device, md_config, &md);
        if (status == UCS_ERR_UNSUPPORTED) {
            ucs_debug("%s: md open by '%s' failed, trying next", md_name,
                      entry->name);
            continue;
        } else if (status == UCS_OK) {
            ucs_debug("%s: md open by '%s' is successful", md_name,
                      entry->name);
            *md_p = md;
            return UCS_OK;
        } else {
            return status;
        }
    }

    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t
uct_ib_get_device_by_name(struct ibv_device **ib_device_list, int num_devices,
                          const char *md_name, struct ibv_device** ibv_device_p)
{
    int i;

    for (i = 0; i < num_devices; ++i) {
        if (!strcmp(ibv_get_device_name(ib_device_list[i]), md_name)) {
            *ibv_device_p = ib_device_list[i];
            return UCS_OK;
        }
    }

    ucs_debug("IB device %s not found", md_name);
    return UCS_ERR_NO_DEVICE;
}

ucs_status_t
uct_ib_fork_init(const uct_ib_md_config_t *md_config, int *fork_init_p)
{
    int ret;

    *fork_init_p = 0;

    if (md_config->fork_init == UCS_NO) {
        uct_ib_fork_warn_enable();
        return UCS_OK;
    }

    ret = ibv_fork_init();
    if (ret) {
        if (md_config->fork_init == UCS_YES) {
            ucs_error("ibv_fork_init() failed: %m");
            return UCS_ERR_IO_ERROR;
        }

        ucs_debug("ibv_fork_init() failed: %m, continuing, but fork may be unsafe.");
        uct_ib_fork_warn_enable();
        return UCS_OK;
    }

    *fork_init_p = 1;
    return UCS_OK;
}

static ucs_status_t
uct_ib_md_open(uct_component_t *component, const char *md_name,
               const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    const uct_ib_md_config_t *md_config = ucs_derived_of(uct_md_config,
                                                         uct_ib_md_config_t);
    ucs_status_t status = UCS_ERR_UNSUPPORTED;
    uct_ib_md_t *md = NULL;
    struct ibv_device **ib_device_list, *ib_device;
    int num_devices, fork_init = 0;

    ucs_trace("opening IB device %s", md_name);

    /* Get device list from driver */
    ib_device_list = ibv_get_device_list(&num_devices);
    if (ib_device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    status = uct_ib_get_device_by_name(ib_device_list, num_devices, md_name,
                                       &ib_device);
    if (status != UCS_OK) {
        goto out_free_dev_list;
    }

    status = uct_ib_fork_init(md_config, &fork_init);
    if (status != UCS_OK) {
        goto out_free_dev_list;
    }

    status = uct_ib_component_md_open(ib_device, md_config, md_name, &md);
    if (status != UCS_OK) {
        if (status == UCS_ERR_UNSUPPORTED) {
            ucs_debug("Unsupported IB device %s", md_name);
        }

        goto out_free_dev_list;
    }

    /* cppcheck-suppress autoVariables */
    *md_p         = &md->super;
    md->fork_init = fork_init;

out_free_dev_list:
    ibv_free_device_list(ib_device_list);
out:
    return status;
}

void uct_ib_md_parse_relaxed_order(uct_ib_md_t *md,
                                   const uct_ib_md_config_t *md_config,
                                   int is_supported)
{
    int have_relaxed_order = (IBV_ACCESS_RELAXED_ORDERING != 0) && is_supported;

    if (md_config->mr_relaxed_order == UCS_YES) {
        if (have_relaxed_order) {
            md->relaxed_order = 1;
        } else {
            ucs_warn("%s: relaxed order memory access requested, but "
                     "unsupported",
                     uct_ib_device_name(&md->dev));
            return;
        }
    } else if (md_config->mr_relaxed_order == UCS_TRY) {
        md->relaxed_order = have_relaxed_order;
    } else if (md_config->mr_relaxed_order == UCS_AUTO) {
        md->relaxed_order = have_relaxed_order &&
                            ucs_cpu_prefer_relaxed_order();
    }

    ucs_debug("%s: relaxed order memory access is %sabled",
              uct_ib_device_name(&md->dev), md->relaxed_order ? "en" : "dis");
}

static void uct_ib_check_gpudirect_driver(uct_ib_md_t *md, const char *file,
                                          ucs_memory_type_t mem_type)
{
    if (md->reg_mem_types & UCS_BIT(mem_type)) {
        return;
    }

    if (!access(file, F_OK)) {
        md->reg_mem_types |= UCS_BIT(mem_type);
    }

    ucs_debug("%s: %s GPUDirect RDMA is %sdetected by checking %s",
              uct_ib_device_name(&md->dev), ucs_memory_type_names[mem_type],
              md->reg_mem_types & UCS_BIT(mem_type) ? "" : "not ", file);
}

static void uct_ib_md_check_dmabuf(uct_ib_md_t *md)
{
#if HAVE_DECL_IBV_REG_DMABUF_MR
    static const int bad_fd = -1;
    struct ibv_mr *mr;

    mr = ibv_reg_dmabuf_mr(md->pd, 0, ucs_get_page_size(), 0, bad_fd,
                           UCT_IB_MEM_ACCESS_FLAGS);
    if (mr != NULL) {
        ibv_dereg_mr(mr);
        /* dmabuf is supported */
    } else if (errno == EBADF) {
        /* dmabuf is supported */
    } else {
        /* Error code is not bad-fd, which means dmabuf registration is not
           supported by the driver */
        ucs_debug("%s: ibv_reg_dmabuf_mr(fd=%d) returned %m, dmabuf is not "
                  "supported",
                  uct_ib_device_name(&md->dev), bad_fd);
        return;
    }

    ucs_debug("%s: dmabuf is supported", uct_ib_device_name(&md->dev));
    md->cap_flags |= UCT_MD_FLAG_REG_DMABUF;
#endif
}

int uct_ib_md_check_odp_common(uct_ib_md_t *md, const char **reason_ptr)
{
    if (IBV_ACCESS_ON_DEMAND == 0) {
        *reason_ptr = "IBV_ACCESS_ON_DEMAND is not supported";
        return 0;
    }

    if (!IBV_DEVICE_HAS_ODP(&md->dev)) {
        *reason_ptr = "device does not support IBV_ACCESS_ON_DEMAND";
        return 0;
    }

    return 1;
}

void uct_ib_md_check_odp(uct_ib_md_t *md)
{
    const char *device_name = uct_ib_device_name(&md->dev);
    const char *reason;

    if (!uct_ib_md_check_odp_common(md, &reason)) {
        ucs_debug("%s: ODP is disabled because %s", device_name, reason);
        return;
    }

    md->reg_nonblock_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    ucs_debug("%s: ODP is supported, version 1", device_name);
}

ucs_status_t uct_ib_md_open_common(uct_ib_md_t *md,
                                   struct ibv_device *ib_device,
                                   const uct_ib_md_config_t *md_config)
{
    ucs_status_t status;

    md->super.component = &uct_ib_component;
    md->config          = md_config->ext;
    md->cap_flags      |= UCT_MD_FLAG_REG |
                          UCT_MD_FLAG_NEED_MEMH |
                          UCT_MD_FLAG_NEED_RKEY |
                          UCT_MD_FLAG_ADVISE;
    md->reg_cost        = md_config->reg_cost;
    md->relaxed_order   = 0;

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&md->stats, &uct_ib_md_stats_class,
                                  ucs_stats_get_root(), "%s-%p",
                                  ibv_get_device_name(ib_device), md);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_device_init(&md->dev, ib_device, md_config->async_events
                                UCS_STATS_ARG(md->stats));
    if (status != UCS_OK) {
        goto err_release_stats;
    }

    if (strlen(md_config->subnet_prefix) > 0) {
        status = uct_ib_md_parse_subnet_prefix(md_config->subnet_prefix,
                                               &md->subnet_filter);

        if (status != UCS_OK) {
            goto err_cleanup_device;
        }

        md->check_subnet_filter = 1;
    }

    md->reg_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST) |
                        md->reg_nonblock_mem_types;

    /* Check for GPU-direct support */
    if (md_config->enable_gpudirect_rdma != UCS_NO) {
        /* Check peer memory driver is loaded, different driver versions use 
         * different paths */
        uct_ib_check_gpudirect_driver(
                md, "/sys/kernel/mm/memory_peers/nv_mem/version",
                UCS_MEMORY_TYPE_CUDA);
        uct_ib_check_gpudirect_driver(
                md, "/sys/module/nvidia_peermem/version",
                UCS_MEMORY_TYPE_CUDA);
        uct_ib_check_gpudirect_driver(
                md, "/sys/module/nv_peer_mem/version",
                UCS_MEMORY_TYPE_CUDA);
                

        /* check if ROCM KFD driver is loaded */
        uct_ib_check_gpudirect_driver(md, "/dev/kfd", UCS_MEMORY_TYPE_ROCM);

        /* Check for dma-buf support */
        uct_ib_md_check_dmabuf(md);
    }

    if (!(md->reg_mem_types & ~UCS_MEMORY_TYPES_CPU_ACCESSIBLE) &&
        !(md->cap_flags & UCT_MD_FLAG_REG_DMABUF) &&
        (md_config->enable_gpudirect_rdma == UCS_YES)) {
        ucs_error("%s: Couldn't enable GPUDirect RDMA. Please make sure "
                  "nv_peer_mem or amdgpu plugin installed correctly, or dmabuf "
                  "is supported.",
                  uct_ib_device_name(&md->dev));
        status = UCS_ERR_UNSUPPORTED;
        goto err_cleanup_device;
    }

    md->dev.max_zcopy_log_sge = INT_MAX;
    if (md->reg_mem_types & ~UCS_BIT(UCS_MEMORY_TYPE_HOST)) {
        md->dev.max_zcopy_log_sge = 1;
    }

    uct_ib_md_set_pci_bw(md, md_config);

    return UCS_OK;

err_cleanup_device:
    uct_ib_device_cleanup(&md->dev);
err_release_stats:
    UCS_STATS_NODE_FREE(md->stats);
err:
    return status;
}

void uct_ib_md_close_common(uct_ib_md_t *md)
{
    uct_ib_device_cleanup(&md->dev);
    UCS_STATS_NODE_FREE(md->stats);
}

void uct_ib_md_device_context_close(struct ibv_context *ctx)
{
    int ret = ibv_close_device(ctx);
    if (ret != 0) {
        ucs_warn("ibv_close_device(%s) of failed: %m",
                 ibv_get_device_name(ctx->device));
    }
}

uct_ib_md_t* uct_ib_md_alloc(size_t size, const char *name,
                             struct ibv_context *ctx)
{
    uct_ib_md_t *md;

    md = ucs_calloc(1, size, name);
    if (md == NULL) {
        ucs_error("failed to allocate memory for md");
        goto err;
    }

    md->dev.ibv_context = ctx;
    md->pd              = ibv_alloc_pd(md->dev.ibv_context);
    if (md->pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        goto err_md_free;
    }

    return md;

err_md_free:
    ucs_free(md);
err:
    return NULL;
}

void uct_ib_md_free(uct_ib_md_t *md)
{
    int ret;

    ret = ibv_dealloc_pd(md->pd);
    /* Do not print a warning if PD deallocation failed with EINVAL, because
     * it fails from time to time on BF/ARM (TODO: investigate) */
    if ((ret != 0) && (errno != EINVAL)) {
        ucs_warn("ibv_dealloc_pd() failed: %m");
    }

    ucs_free(md);
}

void uct_ib_md_ece_check(uct_ib_md_t *md)
{
#if HAVE_DECL_IBV_SET_ECE
    struct ibv_context *ibv_context = md->dev.ibv_context;
    struct ibv_pd *pd               = md->pd;
    struct ibv_ece ece              = {};
    struct ibv_qp *dummy_qp;
    struct ibv_cq *cq;
    struct ibv_qp_init_attr qp_init_attr;

    cq = ibv_create_cq(ibv_context, 1, NULL, NULL, 0);
    if (cq == NULL) {
        uct_ib_check_memlock_limit_msg(ibv_context, UCS_LOG_LEVEL_DEBUG,
                                       "ibv_create_cq()");
        return;
    }

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq          = cq;
    qp_init_attr.recv_cq          = cq;
    qp_init_attr.qp_type          = IBV_QPT_RC;
    qp_init_attr.cap.max_send_wr  = 1;
    qp_init_attr.cap.max_recv_wr  = 1;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;

    dummy_qp = ibv_create_qp(pd, &qp_init_attr);
    if (dummy_qp == NULL) {
        uct_ib_check_memlock_limit_msg(ibv_context, UCS_LOG_LEVEL_DEBUG,
                                       "ibv_create_qp(RC)");
        goto free_cq;
    }

    /* ibv_set_ece check whether ECE is supported */
    if ((ibv_query_ece(dummy_qp, &ece) == 0) &&
        (ibv_set_ece(dummy_qp, &ece) == 0)) {
        md->ece_enable = 1;
    }

    ibv_destroy_qp(dummy_qp);
free_cq:
    ibv_destroy_cq(cq);
#endif
}

static uct_ib_md_ops_t uct_ib_verbs_md_ops;

static ucs_status_t uct_ib_verbs_md_open(struct ibv_device *ibv_device,
                                         const uct_ib_md_config_t *md_config,
                                         uct_ib_md_t **p_md)
{
    uct_ib_device_t *dev;
    ucs_status_t status;
    uct_ib_md_t *md;
    struct ibv_context *ctx;

    if (md_config->devx == UCS_YES) {
        ucs_error("DEVX requested but not supported");
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    /* Open verbs context */
    ctx = ibv_open_device(ibv_device);
    if (ctx == NULL) {
        ucs_diag("ibv_open_device(%s) failed: %m",
                 ibv_get_device_name(ibv_device));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    md = uct_ib_md_alloc(sizeof(*md), "ib_verbs_md", ctx);
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free_context;
    }

    dev = &md->dev;

    status = uct_ib_device_query(dev, ibv_device);
    if (status != UCS_OK) {
        goto err_md_free;
    }

    if (IBV_DEVICE_ATOMIC_HCA(dev) ||
        IBV_DEVICE_ATOMIC_GLOB(dev)) {
        dev->atomic_arg_sizes = sizeof(uint64_t);
    }

    if (IBV_DEVICE_ATOMIC_GLOB(dev)) {
        dev->pci_fadd_arg_sizes = sizeof(uint64_t);
        dev->pci_cswap_arg_sizes = sizeof(uint64_t);
    }

    status  = uct_ib_md_parse_device_config(md, md_config);
    if (status != UCS_OK) {
        goto err_device_config_release;
    }

    md->super.ops = &uct_ib_verbs_md_ops.super;

    status = uct_ib_md_open_common(md, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_md_free;
    }

    md->dev.flags  = uct_ib_device_spec(&md->dev)->flags;
    md->name       = UCT_IB_MD_NAME(verbs);
    md->flush_rkey = UCT_IB_MD_INVALID_FLUSH_RKEY;

    uct_ib_md_ece_check(md);
    uct_ib_md_parse_relaxed_order(md, md_config, 0);
    uct_ib_md_check_odp(md);

    *p_md = md;
    return UCS_OK;

err_device_config_release:
    uct_ib_md_release_device_config(md);
err_md_free:
    uct_ib_md_free(md);
err_free_context:
    uct_ib_md_device_context_close(ctx);
err:
    return status;
}

void uct_ib_md_close(uct_md_h tl_md)
{
    uct_ib_md_t *md         = ucs_derived_of(tl_md, uct_ib_md_t);
    struct ibv_context *ctx = md->dev.ibv_context;

    uct_ib_md_close_common(md);
    uct_ib_md_free(md);
    uct_ib_md_device_context_close(ctx);
}

static uct_ib_md_ops_t uct_ib_verbs_md_ops = {
    .super = {
        .close              = uct_ib_md_close,
        .query              = uct_ib_md_query,
        .mem_reg            = uct_ib_verbs_mem_reg,
        .mem_dereg          = uct_ib_verbs_mem_dereg,
        .mem_attach         = ucs_empty_function_return_unsupported,
        .mem_advise         = uct_ib_mem_advise,
        .mkey_pack          = uct_ib_verbs_mkey_pack,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    },
    .open = uct_ib_verbs_md_open,
};

static UCT_IB_MD_DEFINE_ENTRY(verbs, uct_ib_verbs_md_ops);

uct_component_t uct_ib_component = {
    .query_md_resources = uct_ib_query_md_resources,
    .md_open            = uct_ib_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ib_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "ib",
    .md_config          = {
        .name           = "IB memory domain",
        .prefix         = UCT_IB_CONFIG_PREFIX,
        .table          = uct_ib_md_config_table,
        .size           = sizeof(uct_ib_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ib_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};

void UCS_F_CTOR uct_ib_init()
{
    UCS_MODULE_FRAMEWORK_DECLARE(uct_ib);
    ssize_t i;

    ucs_list_add_head(&uct_ib_ops, &UCT_IB_MD_OPS_NAME(verbs).list);
    uct_component_register(&uct_ib_component);

    for (i = 0; i < ucs_static_array_size(uct_ib_tls); i++) {
        uct_tl_register(&uct_ib_component, uct_ib_tls[i]);
    }

    UCS_MODULE_FRAMEWORK_LOAD(uct_ib, 0);
}

void UCS_F_DTOR uct_ib_cleanup()
{
    ssize_t i;

    for (i = ucs_static_array_size(uct_ib_tls) - 1; i >= 0; i--) {
        uct_tl_unregister(uct_ib_tls[i]);
    }

    uct_component_unregister(&uct_ib_component);
    ucs_list_del(&UCT_IB_MD_OPS_NAME(verbs).list);
}
