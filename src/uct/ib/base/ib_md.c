/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
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
#include <ucs/sys/math.h>
#include <ucs/sys/module.h>
#include <ucs/sys/string.h>
#include <ucs/time/time.h>
#include <ucm/api/ucm.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/vfs/base/vfs_obj.h>
#include <pthread.h>
#ifdef HAVE_PTHREAD_NP_H
#include <pthread_np.h>
#endif
#include <sys/resource.h>


#define UCT_IB_MD_RCACHE_DEFAULT_ALIGN 16

#define UCT_IB_MD_MEM_DEREG_CHECK_PARAMS(_ib_md, _params) \
    UCT_MD_MEM_DEREG_CHECK_PARAMS(_params, \
                                  (_ib_md)->cap_flags & UCT_MD_FLAG_INVALIDATE)


static UCS_CONFIG_DEFINE_ARRAY(pci_bw,
                               sizeof(ucs_config_bw_spec_t),
                               UCS_CONFIG_TYPE_BW_SPEC);

static const char *uct_ib_devx_objs[] = {
    [UCT_IB_DEVX_OBJ_RCQP]  = "rcqp",
    [UCT_IB_DEVX_OBJ_RCSRQ] = "rcsrq",
    [UCT_IB_DEVX_OBJ_DCT]   = "dct",
    [UCT_IB_DEVX_OBJ_DCSRQ] = "dcsrq",
    [UCT_IB_DEVX_OBJ_DCI]   = "dci",
    NULL
};

static ucs_config_field_t uct_ib_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_ib_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"REG_METHODS", "rcache,odp,direct",
     "List of registration methods in order of preference. Supported methods are:\n"
     "  odp         - implicit on-demand paging\n"
     "  rcache      - userspace registration cache\n"
     "  direct      - direct registration\n",
     ucs_offsetof(uct_ib_md_config_t, reg_methods), UCS_CONFIG_TYPE_STRING_ARRAY},

    {"", "RCACHE_ADDR_ALIGN=" UCS_PP_MAKE_STRING(UCT_IB_MD_RCACHE_DEFAULT_ALIGN), NULL,
     ucs_offsetof(uct_ib_md_config_t, rcache),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_rcache_table)},

    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* TODO take default from device */
     ucs_offsetof(uct_ib_md_config_t, uc_reg_cost.c), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
     ucs_offsetof(uct_ib_md_config_t, uc_reg_cost.m), UCS_CONFIG_TYPE_TIME},

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

    {"ODP_NUMA_POLICY", "preferred",
     "Override NUMA policy for ODP regions, to avoid extra page migrations.\n"
     " - default: Do no change existing policy.\n"
     " - preferred/bind:\n"
     "     Unless the memory policy of the current thread is MPOL_BIND, set the\n"
     "     policy of ODP regions to MPOL_PREFERRED/MPOL_BIND, respectively.\n"
     "     If the numa node mask of the current thread is not defined, use the numa\n"
     "     nodes which correspond to its cpu affinity mask.",
     ucs_offsetof(uct_ib_md_config_t, ext.odp.numa_policy),
     UCS_CONFIG_TYPE_ENUM(ucs_numa_policy_names)},

    {"ODP_PREFETCH", "n",
     "Force prefetch of memory regions created with ODP.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.odp.prefetch), UCS_CONFIG_TYPE_BOOL},

    {"ODP_MAX_SIZE", "auto",
     "Maximal memory region size to enable ODP for. 0 - disable.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.odp.max_size), UCS_CONFIG_TYPE_MEMUNITS},

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

#ifdef HAVE_EXP_UMR
    {"MAX_INLINE_KLM_LIST", "inf",
     "When posting a UMR, KLM lists shorter or equal to this value will be posted as inline.\n"
     "The actual maximal length is also limited by device capabilities.",
     ucs_offsetof(uct_ib_md_config_t, ext.max_inline_klm_list), UCS_CONFIG_TYPE_UINT},
#endif

    {"PCI_BW", "",
     "Maximum effective data transfer rate of PCI bus connected to HCA\n",
     ucs_offsetof(uct_ib_md_config_t, pci_bw), UCS_CONFIG_TYPE_ARRAY(pci_bw)},

    {"MLX5_DEVX", "try",
     "DEVX support\n",
     ucs_offsetof(uct_ib_md_config_t, devx), UCS_CONFIG_TYPE_TERNARY},

    {"MLX5_DEVX_OBJECTS", "rcqp,rcsrq,dct,dcsrq,dci",
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
     ucs_offsetof(uct_ib_md_config_t, mr_relaxed_order), UCS_CONFIG_TYPE_ON_OFF_AUTO},

    {"MAX_IDLE_RKEY_COUNT", "16",
     "Maximal number of invalidated memory keys that are kept idle before reuse.",
     ucs_offsetof(uct_ib_md_config_t, ext.max_idle_rkey_count), UCS_CONFIG_TYPE_UINT},

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


extern uct_tl_t UCT_TL_NAME(dc_mlx5);
extern uct_tl_t UCT_TL_NAME(rc_verbs);
extern uct_tl_t UCT_TL_NAME(rc_mlx5);
extern uct_tl_t UCT_TL_NAME(ud_verbs);
extern uct_tl_t UCT_TL_NAME(ud_mlx5);

static uct_tl_t *uct_ib_tls[] = {
#ifdef HAVE_TL_DC
    &UCT_TL_NAME(dc_mlx5),
#endif
#ifdef HAVE_TL_RC
    &UCT_TL_NAME(rc_verbs),
#endif
#if defined (HAVE_TL_RC) && defined (HAVE_MLX5_HW)
    &UCT_TL_NAME(rc_mlx5),
#endif
#ifdef HAVE_TL_UD
    &UCT_TL_NAME(ud_verbs),
#endif
#if defined (HAVE_TL_UD) && defined (HAVE_MLX5_HW_UD)
    &UCT_TL_NAME(ud_mlx5)
#endif
};

extern uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(devx);
extern uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(dv);
extern uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(exp);
static uct_ib_md_ops_entry_t UCT_IB_MD_OPS_NAME(verbs);

static uct_ib_md_ops_entry_t *uct_ib_ops[] = {
#if defined (HAVE_MLX5_HW)
#  if defined (HAVE_MLX5_DV) && defined (HAVE_DEVX)
    &UCT_IB_MD_OPS_NAME(devx),
#  endif
#  if defined (HAVE_MLX5_DV)
    &UCT_IB_MD_OPS_NAME(dv),
#  endif
#  if defined (HAVE_VERBS_EXP_H)
    &UCT_IB_MD_OPS_NAME(exp),
#  endif
#endif /* HAVE_MLX5_HW */
    &UCT_IB_MD_OPS_NAME(verbs)
};

typedef struct uct_ib_verbs_mem {
    uct_ib_mem_t        super;
    uct_ib_mr_t         mrs[];
} uct_ib_verbs_mem_t;

typedef struct {
    pthread_t     thread;
    void          *addr;
    size_t        len;
    size_t        chunk;
    uint64_t      access;
    struct ibv_pd *pd;
    struct ibv_mr **mr;
    int           silent;
} uct_ib_md_mem_reg_thread_t;


static ucs_status_t uct_ib_md_query(uct_md_h uct_md, uct_md_attr_t *md_attr)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    md_attr->cap.max_alloc        = ULONG_MAX; /* TODO query device */
    md_attr->cap.max_reg          = ULONG_MAX; /* TODO query device */
    md_attr->cap.flags            = md->cap_flags;
    md_attr->cap.alloc_mem_types  = 0;
    md_attr->cap.access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.reg_mem_types    = md->reg_mem_types;
    md_attr->rkey_packed_size     = UCT_IB_MD_PACKED_RKEY_SIZE;
    md_attr->reg_cost             = md->reg_cost;
    ucs_sys_cpuset_copy(&md_attr->local_cpus, &md->dev.local_cpus);

    return UCS_OK;
}

static void uct_ib_md_print_mem_reg_err_msg(void *address, size_t length,
                                            uint64_t access_flags, int err,
                                            int silent)
{
    ucs_log_level_t level = silent ? UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR;
    UCS_STRING_BUFFER_ONSTACK(msg, 256);
    size_t page_size;
    size_t unused;

    ucs_string_buffer_appendf(&msg, "%s(address=%p, length=%zu, access=0x%lx)",
                              ibv_reg_mr_func_name, address, length,
                              access_flags);

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

    uct_ib_mem_lock_limit_msg(ucs_string_buffer_cstr(&msg), err, level);
}

void *uct_ib_md_mem_handle_thread_func(void *arg)
{
    uct_ib_md_mem_reg_thread_t *ctx = arg;
    ucs_status_t status;
    int mr_idx = 0;
    size_t size = 0;
    ucs_time_t UCS_V_UNUSED t0 = ucs_get_time();

    while (ctx->len) {
        size = ucs_min(ctx->len, ctx->chunk);
        if (ctx->access != UCT_IB_MEM_DEREG) {
            ctx->mr[mr_idx] = UCS_PROFILE_NAMED_CALL(ibv_reg_mr_func_name,
                                                     ibv_reg_mr, ctx->pd,
                                                     ctx->addr, size,
                                                     ctx->access);
            if (ctx->mr[mr_idx] == NULL) {
                uct_ib_md_print_mem_reg_err_msg(ctx->addr, size, ctx->access,
                                                errno, ctx->silent);
                return UCS_STATUS_PTR(UCS_ERR_IO_ERROR);
            }
        } else {
            status = uct_ib_dereg_mr(ctx->mr[mr_idx]);
            if (status != UCS_OK) {
                return UCS_STATUS_PTR(status);
            }
        }
        ctx->addr  = UCS_PTR_BYTE_OFFSET(ctx->addr, size);
        ctx->len  -= size;
        mr_idx++;
    }

    ucs_trace("%s %p..%p took %f usec\n",
              (ctx->access == UCT_IB_MEM_DEREG) ? "dereg_mr" : "reg_mr",
              ctx->mr[0]->addr,
              UCS_PTR_BYTE_OFFSET(ctx->mr[mr_idx-1]->addr, size),
              ucs_time_to_usec(ucs_get_time() - t0));

    return UCS_STATUS_PTR(UCS_OK);
}

ucs_status_t
uct_ib_md_handle_mr_list_multithreaded(uct_ib_md_t *md, void *address,
                                       size_t length, uint64_t access_flags,
                                       size_t chunk, struct ibv_mr **mrs,
                                       int silent)
{
    int thread_num_mrs, thread_num, thread_idx, mr_idx = 0, cpu_id = 0;
    int mr_num = ucs_div_round_up(length, chunk);
    ucs_status_t status;
    void *thread_status;
    ucs_sys_cpuset_t parent_set, thread_set;
    uct_ib_md_mem_reg_thread_t *ctxs, *cur_ctx;
    pthread_attr_t attr;
    char UCS_V_UNUSED affinity_str[64];
    int ret;

    ret = pthread_getaffinity_np(pthread_self(), sizeof(ucs_sys_cpuset_t),
                                 &parent_set);
    if (ret != 0) {
        ucs_error("pthread_getaffinity_np() failed: %m");
        return UCS_ERR_INVALID_PARAM;
    }

    thread_num = ucs_min(CPU_COUNT(&parent_set), mr_num);

    ucs_trace("multithreaded handle %p..%p access %lx threads %d affinity %s\n",
              address, UCS_PTR_BYTE_OFFSET(address, length), access_flags, thread_num,
              ucs_make_affinity_str(&parent_set, affinity_str, sizeof(affinity_str)));

    if (thread_num == 1) {
        return UCS_ERR_UNSUPPORTED;
    }

    ctxs = ucs_calloc(thread_num, sizeof(*ctxs), "ib mr ctxs");
    if (ctxs == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    pthread_attr_init(&attr);

    status = UCS_OK;
    for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        /* calculate number of mrs for each thread so each one will
         * get proportional amount */
        thread_num_mrs  = ucs_div_round_up(mr_num - mr_idx, thread_num - thread_idx);

        cur_ctx         = &ctxs[thread_idx];
        cur_ctx->pd     = md->pd;
        cur_ctx->addr   = UCS_PTR_BYTE_OFFSET(address, mr_idx * chunk);
        cur_ctx->len    = ucs_min(thread_num_mrs * chunk, length - (mr_idx * chunk));
        cur_ctx->access = access_flags;
        cur_ctx->mr     = &mrs[mr_idx];
        cur_ctx->chunk  = chunk;
        cur_ctx->silent = silent;

        if (md->config.mt_reg_bind) {
            while (!CPU_ISSET(cpu_id, &parent_set)) {
                cpu_id++;
            }

            CPU_ZERO(&thread_set);
            CPU_SET(cpu_id, &thread_set);
            cpu_id++;
            pthread_attr_setaffinity_np(&attr, sizeof(ucs_sys_cpuset_t), &thread_set);
        }

        ret = pthread_create(&cur_ctx->thread, &attr,
                             uct_ib_md_mem_handle_thread_func, cur_ctx);
        if (ret) {
            ucs_error("pthread_create() failed: %m");
            status     = UCS_ERR_IO_ERROR;
            thread_num = thread_idx;
            break;
        }

        mr_idx += thread_num_mrs;
    }

    for (thread_idx = 0; thread_idx < thread_num; thread_idx++) {
        cur_ctx = &ctxs[thread_idx];
        pthread_join(cur_ctx->thread, &thread_status);
        if (UCS_PTR_IS_ERR(UCS_OK)) {
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

static ucs_status_t uct_ib_md_reg_mr(uct_ib_md_t *md, void *address,
                                     size_t length, uint64_t access_flags,
                                     int silent, uct_ib_mem_t *memh,
                                     uct_ib_mr_type_t mr_type)
{
    ucs_status_t status;

    if (length >= md->config.min_mt_reg) {
        UCS_PROFILE_CODE("reg ksm") {
            status = md->ops->reg_multithreaded(md, address, length,
                                                access_flags, memh, mr_type,
                                                silent);
        }

        if (status != UCS_ERR_UNSUPPORTED) {
            if (status == UCS_OK) {
                memh->flags |= UCT_IB_MEM_MULTITHREADED;
            }

            return status;
        } /* if unsupported - fallback to regular registration */
    }

    return md->ops->reg_key(md, address, length, access_flags, memh, mr_type,
                            silent);
}

ucs_status_t uct_ib_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
                           uint64_t access_flags, struct ibv_mr **mr_p,
                           int silent)
{
    ucs_time_t UCS_V_UNUSED start_time = ucs_get_time();
    struct ibv_mr *mr;
#if HAVE_DECL_IBV_EXP_REG_MR
    struct ibv_exp_reg_mr_in in = {};

    in.pd         = pd;
    in.addr       = addr;
    in.length     = length;
    in.exp_access = access_flags;
    mr = UCS_PROFILE_CALL(ibv_exp_reg_mr, &in);
#else
    mr = UCS_PROFILE_CALL(ibv_reg_mr, pd, addr, length, access_flags);
#endif
    if (mr == NULL) {
        uct_ib_md_print_mem_reg_err_msg(addr, length, access_flags,
                                        errno, silent);
        return UCS_ERR_IO_ERROR;
    }

    *mr_p = mr;

    /* to prevent clang dead code */
    ucs_trace("ibv_reg_mr(pd=%p addr=%p length=%zu): mr=%p took %.3f msec", pd,
              addr, length, mr, ucs_time_to_msec(ucs_get_time() - start_time));
    return UCS_OK;
}

ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr)
{
    int ret;

    if (mr == NULL) {
        return UCS_OK;
    }

    ucs_trace("ibv_dereg_mr(mr=%p addr=%p length=%zu)", mr, mr->addr,
              mr->length);

    ret = UCS_PROFILE_CALL(ibv_dereg_mr, mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t uct_ib_dereg_mrs(struct ibv_mr **mrs, size_t mr_num)
{
    ucs_status_t s, status = UCS_OK;
    int i;

    for (i = 0; i < mr_num; i++) {
        s = uct_ib_dereg_mr(mrs[i]);
        if (s != UCS_OK) {
            status = s;
        }
    }

    return status;
}

static ucs_status_t uct_ib_memh_dereg_key(uct_ib_md_t *md, uct_ib_mem_t *memh,
                                          uct_ib_mr_type_t mr_type)
{
    if (memh->flags & UCT_IB_MEM_MULTITHREADED) {
        return md->ops->dereg_multithreaded(md, memh, mr_type);
    } else {
        return md->ops->dereg_key(md, memh, mr_type);
    }
}


static ucs_status_t uct_ib_memh_dereg(uct_ib_md_t *md, uct_ib_mem_t *memh)
{
    ucs_status_t s, status = UCS_OK;

    if (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) {
        memh->flags &= ~UCT_IB_MEM_FLAG_ATOMIC_MR;
        s = md->ops->dereg_atomic_key(md, memh);
        if (s != UCS_OK) {
            status = s;
        }
    }

    if (memh->flags & UCT_IB_MEM_FLAG_RELAXED_ORDERING) {
        s = uct_ib_memh_dereg_key(md, memh, UCT_IB_MR_STRICT_ORDER);
        memh->flags &= ~UCT_IB_MEM_FLAG_RELAXED_ORDERING;
        if (s != UCS_OK) {
            status = s;
        }
    }

    s = uct_ib_memh_dereg_key(md, memh, UCT_IB_MR_DEFAULT);
    if (s != UCS_OK) {
        status = s;
    }

    return status;
}

static void uct_ib_memh_free(uct_ib_mem_t *memh)
{
    ucs_free(memh);
}

static uct_ib_mem_t *uct_ib_memh_alloc(uct_ib_md_t *md)
{
    return ucs_calloc(1, md->memh_struct_size, "ib_memh");
}

static uint64_t uct_ib_md_access_flags(uct_ib_md_t *md, unsigned flags,
                                       size_t length)
{
    uint64_t access_flags = UCT_IB_MEM_ACCESS_FLAGS;

    if ((flags & UCT_MD_MEM_FLAG_NONBLOCK) && (length > 0) &&
        (length <= md->config.odp.max_size)) {
        access_flags |= IBV_ACCESS_ON_DEMAND;
    }

    if (md->relaxed_order) {
        access_flags |= IBV_ACCESS_RELAXED_ORDERING;
    }

    return access_flags;
}

#if HAVE_NUMA
static ucs_status_t uct_ib_mem_set_numa_policy(uct_ib_md_t *md, void *address,
                                               size_t length, uct_ib_mem_t *memh)
{
    int ret, old_policy, new_policy;
    struct bitmask *nodemask;
    uintptr_t start, end;
    ucs_status_t status;

    if (!(memh->flags & UCT_IB_MEM_FLAG_ODP) ||
        (md->config.odp.numa_policy == UCS_NUMA_POLICY_DEFAULT) ||
        (numa_available() < 0))
    {
        status = UCS_OK;
        goto out;
    }

    nodemask = numa_allocate_nodemask();
    if (nodemask == NULL) {
        ucs_warn("Failed to allocate numa node mask");
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    ret = get_mempolicy(&old_policy, numa_nodemask_p(nodemask),
                        numa_nodemask_size(nodemask), NULL, 0);
    if (ret < 0) {
        ucs_warn("get_mempolicy(maxnode=%zu) failed: %m",
                 numa_nodemask_size(nodemask));
        status = UCS_ERR_INVALID_PARAM;
        goto out_free;
    }

    switch (old_policy) {
    case MPOL_DEFAULT:
        /* if no policy is defined, use the numa node of the current cpu */
        numa_get_thread_node_mask(&nodemask);
        break;
    case MPOL_BIND:
        /* if the current policy is BIND, keep it as-is */
        status = UCS_OK;
        goto out_free;
    default:
        break;
    }

    switch (md->config.odp.numa_policy) {
    case UCS_NUMA_POLICY_BIND:
        new_policy = MPOL_BIND;
        break;
    case UCS_NUMA_POLICY_PREFERRED:
        new_policy = MPOL_PREFERRED;
        break;
    default:
        ucs_error("unexpected numa policy %d", md->config.odp.numa_policy);
        status = UCS_ERR_INVALID_PARAM;
        goto out_free;
    }

    if (new_policy != old_policy) {
        start = ucs_align_down_pow2((uintptr_t)address, ucs_get_page_size());
        end   = ucs_align_up_pow2((uintptr_t)address + length,
                                  ucs_get_page_size());
        ucs_trace("0x%lx..0x%lx: changing numa policy from %d to %d, "
                  "nodemask[0]=0x%lx", start, end, old_policy, new_policy,
                  numa_nodemask_p(nodemask)[0]);

        ret = UCS_PROFILE_CALL(mbind, (void*)start, end - start, new_policy,
                               numa_nodemask_p(nodemask),
                               numa_nodemask_size(nodemask), 0);
        if (ret < 0) {
            ucs_warn("mbind(addr=0x%lx length=%ld policy=%d) failed: %m",
                     start, end - start, new_policy);
            status = UCS_ERR_IO_ERROR;
            goto out_free;
        }
    }

    status = UCS_OK;

out_free:
    numa_free_nodemask(nodemask);
out:
    return status;
}
#else
static ucs_status_t uct_ib_mem_set_numa_policy(uct_ib_md_t *md, void *address,
                                               size_t length, uct_ib_mem_t *memh)
{
    return UCS_OK;
}
#endif /* UCT_MD_DISABLE_NUMA */

static void uct_ib_mem_init(uct_ib_mem_t *memh, unsigned uct_flags,
                            uint64_t access_flags)
{
    memh->lkey          = UCT_IB_INVALID_MKEY;
    memh->rkey          = UCT_IB_INVALID_MKEY;
    memh->atomic_rkey   = UCT_IB_INVALID_MKEY;
    memh->indirect_rkey = UCT_IB_INVALID_MKEY;
    memh->flags         = 0;

    /* coverity[dead_error_condition] */
    if (access_flags & IBV_ACCESS_ON_DEMAND) {
        memh->flags |= UCT_IB_MEM_FLAG_ODP;
    }

    if (uct_flags & UCT_MD_MEM_ACCESS_REMOTE_ATOMIC) {
        memh->flags |= UCT_IB_MEM_ACCESS_REMOTE_ATOMIC;
    }
}

static ucs_status_t uct_ib_mem_reg_internal(uct_md_h uct_md, void *address,
                                            size_t length, unsigned flags,
                                            int silent, uct_ib_mem_t *memh)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_status_t status;
    uint64_t access_flags;

    access_flags = uct_ib_md_access_flags(md, flags, length);
    uct_ib_mem_init(memh, flags, access_flags);
    status = uct_ib_md_reg_mr(md, address, length, access_flags, silent, memh,
                              UCT_IB_MR_DEFAULT);
    if (status != UCS_OK) {
        return status;
    }

    if (md->relaxed_order) {
        status = uct_ib_md_reg_mr(md, address, length,
                                  access_flags & ~IBV_ACCESS_RELAXED_ORDERING,
                                  silent, memh, UCT_IB_MR_STRICT_ORDER);
        if (status != UCS_OK) {
            goto err;
        }

        memh->flags |= UCT_IB_MEM_FLAG_RELAXED_ORDERING;
    }

    ucs_trace("registered memory %p..%p on %s lkey 0x%x rkey 0x%x "
              "access 0x%lx flags 0x%x", address,
              UCS_PTR_BYTE_OFFSET(address, length),
              uct_ib_device_name(&md->dev), memh->lkey, memh->rkey,
              access_flags, flags);

    uct_ib_mem_set_numa_policy(md, address, length, memh);

    if (md->config.odp.prefetch) {
        md->ops->mem_prefetch(md, memh, address, length);
    }

    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_REG, +1);
    return UCS_OK;

err:
    uct_ib_memh_dereg(md, memh);
    return status;
}

static ucs_status_t uct_ib_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                   unsigned flags, uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_status_t status;
    uct_ib_mem_t *memh;

    memh = uct_ib_memh_alloc(md);
    if (memh == NULL) {
        uct_md_log_mem_reg_error(flags,
                                 "md %p: failed to allocate memory handle", md);
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_ib_mem_reg_internal(uct_md, address, length, flags, 0, memh);
    if (status != UCS_OK) {
        uct_ib_memh_free(memh);
        return status;
    }
    *memh_p = memh;

    return UCS_OK;
}

static ucs_status_t uct_ib_mem_dereg(uct_md_h uct_md,
                                     const uct_md_mem_dereg_params_t *params)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_mem_t *ib_memh;
    ucs_status_t status;

    UCT_IB_MD_MEM_DEREG_CHECK_PARAMS(md, params);

    ib_memh = params->memh;
    status  = uct_ib_memh_dereg(md, ib_memh);
    uct_ib_memh_free(ib_memh);
    if (UCT_MD_MEM_DEREG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0) &
        UCT_MD_MEM_DEREG_FLAG_INVALIDATE) {
        ucs_assert(params->comp != NULL); /* suppress coverity false-positive */
        uct_invoke_completion(params->comp, UCS_OK);
    }

    return status;
}

static ucs_status_t uct_ib_verbs_reg_key(uct_ib_md_t *md, void *address,
                                         size_t length, uint64_t access_flags,
                                         uct_ib_mem_t *ib_memh,
                                         uct_ib_mr_type_t mr_type, int silent)
{
    uct_ib_verbs_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_verbs_mem_t);

    return uct_ib_reg_key_impl(md, address, length, access_flags,
                               ib_memh, &memh->mrs[mr_type], mr_type, silent);
}

ucs_status_t uct_ib_reg_key_impl(uct_ib_md_t *md, void *address,
                                 size_t length, uint64_t access_flags,
                                 uct_ib_mem_t *memh, uct_ib_mr_t *mr,
                                 uct_ib_mr_type_t mr_type, int silent)
{
    ucs_status_t status;

    status = uct_ib_reg_mr(md->pd, address, length, access_flags, &mr->ib,
                           silent);
    if (status != UCS_OK) {
        return status;
    }

    if (mr_type == UCT_IB_MR_DEFAULT) {
        uct_ib_memh_init_keys(memh, mr->ib->lkey, mr->ib->rkey);
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_verbs_dereg_key(uct_ib_md_t *md,
                                           uct_ib_mem_t *ib_memh,
                                           uct_ib_mr_type_t mr_type)
{
    uct_ib_verbs_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_verbs_mem_t);

    return uct_ib_dereg_mr(memh->mrs[mr_type].ib);
}

static ucs_status_t uct_ib_verbs_reg_atomic_key(uct_ib_md_t *ibmd,
                                                uct_ib_mem_t *ib_memh)
{
    uct_ib_mr_type_t mr_type = uct_ib_memh_get_atomic_base_mr_type(ib_memh);
    uct_ib_verbs_mem_t *memh = ucs_derived_of(ib_memh, uct_ib_verbs_mem_t);

    if (mr_type != UCT_IB_MR_STRICT_ORDER) {
        return UCS_ERR_UNSUPPORTED;
    }

    memh->super.atomic_rkey = memh->mrs[mr_type].ib->rkey;
    return UCS_OK;
}

static ucs_status_t
uct_ib_mem_advise(uct_md_h uct_md, uct_mem_h memh, void *addr,
                        size_t length, unsigned advice)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    ucs_debug("memh %p advice %d", memh, advice);
    if ((advice == UCT_MADV_WILLNEED) && !md->config.odp.prefetch) {
        return md->ops->mem_prefetch(md, memh, addr, length);
    }

    return UCS_OK;
}

static ucs_status_t
uct_ib_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                 const uct_md_mkey_pack_params_t *params,
                 void *rkey_buffer)
{
    uct_ib_md_t *md     = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_mem_t *memh  = uct_memh;
    unsigned flags      = UCS_PARAM_VALUE(UCT_MD_MKEY_PACK_FIELD, params, flags,
                                          FLAGS, 0);
    uint32_t atomic_rkey;
    uint32_t rkey;
    ucs_status_t status;

    /* create umr only if a user requested atomic access to the
     * memory region and the hardware supports it.
     */
    if ((memh->flags & (UCT_IB_MEM_ACCESS_REMOTE_ATOMIC |
                        UCT_IB_MEM_FLAG_RELAXED_ORDERING)) &&
        !(memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) &&
        (memh != md->global_odp))
    {
        /* create UMR on-demand */
        UCS_PROFILE_CODE("reg atomic key") {
            status = md->ops->reg_atomic_key(md, memh);
        }

        if (status == UCS_OK) {
            memh->flags |= UCT_IB_MEM_FLAG_ATOMIC_MR;
            ucs_trace("created atomic key 0x%x for 0x%x", memh->atomic_rkey,
                      memh->lkey);
        } else if (status == UCS_ERR_UNSUPPORTED) {
            /* ignore for atomic MR */
        } else {
            return status;
        }
    }

    if (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) {
        atomic_rkey = memh->atomic_rkey;
    } else {
        atomic_rkey = UCT_IB_INVALID_MKEY;
    }

    /* Register indirect key, that does not support atomic operations, only if
     * we have a dedicated atomic key or atomic support wasn't requested */
    if ((flags & UCT_MD_MKEY_PACK_FLAG_INVALIDATE) &&
        ((atomic_rkey != UCT_IB_INVALID_MKEY) ||
         !(memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC))) {
        if (memh->indirect_rkey == UCT_IB_INVALID_MKEY) {
            status = md->ops->reg_indirect_key(md, memh);
            if (status != UCS_OK) {
                return status;
            }
        }

        rkey = memh->indirect_rkey;
    } else {
        rkey = memh->rkey;
    }

    uct_ib_md_pack_rkey(rkey, atomic_rkey, rkey_buffer);
    return UCS_OK;
}

static ucs_status_t uct_ib_rkey_unpack(uct_component_t *component,
                                       const void *rkey_buffer, uct_rkey_t *rkey_p,
                                       void **handle_p)
{
    uint64_t packed_rkey = *(const uint64_t*)rkey_buffer;

    *rkey_p   = packed_rkey;
    *handle_p = NULL;
    ucs_trace("unpacked rkey 0x%llx: direct 0x%x indirect 0x%x",
              (unsigned long long)packed_rkey,
              uct_ib_md_direct_rkey(*rkey_p), uct_ib_md_indirect_rkey(*rkey_p));
    return UCS_OK;
}

static uct_md_ops_t uct_ib_md_ops = {
    .close              = uct_ib_md_close,
    .query              = uct_ib_md_query,
    .mem_reg            = uct_ib_mem_reg,
    .mem_dereg          = uct_ib_mem_dereg,
    .mem_advise         = uct_ib_mem_advise,
    .mkey_pack          = uct_ib_mkey_pack,
    .detect_memory_type = ucs_empty_function_return_unsupported,
};

static inline uct_ib_rcache_region_t* uct_ib_rcache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_ib_rcache_region_t, memh);
}

static ucs_status_t uct_ib_mem_rcache_reg(uct_md_h uct_md, void *address,
                                          size_t length, unsigned flags,
                                          uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;
    uct_ib_mem_t *memh;

    status = ucs_rcache_get(md->rcache, address, length, PROT_READ|PROT_WRITE,
                            &flags, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    memh = &ucs_derived_of(rregion, uct_ib_rcache_region_t)->memh;
    /* The original region was registered without atomic access
     * so update the access flags. Actual umr creation will happen
     * when uct_ib_mkey_pack() is called.
     */
    if (flags & UCT_MD_MEM_ACCESS_REMOTE_ATOMIC) {
        memh->flags |= UCT_IB_MEM_ACCESS_REMOTE_ATOMIC;
    }
    *memh_p = memh;
    return UCS_OK;
}

static void uct_ib_mem_region_invalidate_cb(void *arg)
{
    uct_completion_t *comp = arg;

    uct_invoke_completion(comp, UCS_OK);
}

static ucs_status_t
uct_ib_mem_rcache_dereg(uct_md_h uct_md,
                        const uct_md_mem_dereg_params_t *params)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_rcache_region_t *region;

    UCT_IB_MD_MEM_DEREG_CHECK_PARAMS(md, params);

    region = uct_ib_rcache_region_from_memh(params->memh);
    if (UCT_MD_MEM_DEREG_FIELD_VALUE(params, flags, FIELD_FLAGS, 0) &
        UCT_MD_MEM_DEREG_FLAG_INVALIDATE) {
        ucs_rcache_region_invalidate(md->rcache, &region->super,
                                     uct_ib_mem_region_invalidate_cb,
                                     params->comp);
    }

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t uct_ib_md_rcache_ops = {
    .close                  = uct_ib_md_close,
    .query                  = uct_ib_md_query,
    .mem_reg                = uct_ib_mem_rcache_reg,
    .mem_dereg              = uct_ib_mem_rcache_dereg,
    .mem_advise             = uct_ib_mem_advise,
    .mkey_pack              = uct_ib_mkey_pack,
    .is_sockaddr_accessible = ucs_empty_function_return_zero_int,
    .detect_memory_type     = ucs_empty_function_return_unsupported,
};

static ucs_status_t uct_ib_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                             void *arg, ucs_rcache_region_t *rregion,
                                             uint16_t rcache_mem_reg_flags)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_md_t *md = context;
    int *flags      = arg;
    int silent      = (rcache_mem_reg_flags & UCS_RCACHE_MEM_REG_HIDE_ERRORS) ||
                      (*flags & UCT_MD_MEM_FLAG_HIDE_ERRORS);

    return uct_ib_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                   region->super.super.end - region->super.super.start,
                                   *flags, silent, &region->memh);
}

static void uct_ib_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                       ucs_rcache_region_t *rregion)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_md_t *md = (uct_ib_md_t *)context;

    (void)uct_ib_memh_dereg(md, &region->memh);
}

static void uct_ib_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion, char *buf,
                                         size_t max)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_mem_t *memh = &region->memh;

    snprintf(buf, max, "lkey 0x%x rkey 0x%x atomic_rkey 0x%x",
             memh->lkey, memh->rkey,
             (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) ? memh->atomic_rkey :
                             UCT_IB_INVALID_MKEY
             );
}

static ucs_rcache_ops_t uct_ib_rcache_ops = {
    .mem_reg     = uct_ib_rcache_mem_reg_cb,
    .mem_dereg   = uct_ib_rcache_mem_dereg_cb,
    .dump_region = uct_ib_rcache_dump_region_cb
};

static ucs_status_t uct_ib_md_odp_query(uct_md_h uct_md, uct_md_attr_t *md_attr)
{
    ucs_status_t status;

    status = uct_ib_md_query(uct_md, md_attr);
    if (status != UCS_OK) {
        return status;
    }

    /* ODP supports only host memory */
    md_attr->cap.reg_mem_types &= UCS_BIT(UCS_MEMORY_TYPE_HOST);
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_global_odp_reg(uct_md_h uct_md, void *address,
                                              size_t length, unsigned flags,
                                              uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_mem_t *memh = md->global_odp;

    ucs_assert(md->global_odp != NULL);
    if (flags & UCT_MD_MEM_FLAG_LOCK) {
        return uct_ib_mem_reg(uct_md, address, length, flags, memh_p);
    }

    if (md->config.odp.prefetch) {
        md->ops->mem_prefetch(md, memh, address, length);
    }

    /* cppcheck-suppress autoVariables */
    *memh_p = md->global_odp;
    return UCS_OK;
}

static ucs_status_t
uct_ib_mem_global_odp_dereg(uct_md_h uct_md,
                            const uct_md_mem_dereg_params_t *params)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_mem_t *ib_memh;
    ucs_status_t status;

    UCT_IB_MD_MEM_DEREG_CHECK_PARAMS(md, params);

    if (params->memh == md->global_odp) {
        return UCS_OK;
    }

    ib_memh = params->memh;
    status  = uct_ib_memh_dereg(md, ib_memh);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_memh_free(ib_memh);
    return status;
}

static uct_md_ops_t UCS_V_UNUSED uct_ib_md_global_odp_ops = {
    .close              = uct_ib_md_close,
    .query              = uct_ib_md_odp_query,
    .mem_reg            = uct_ib_mem_global_odp_reg,
    .mem_dereg          = uct_ib_mem_global_odp_dereg,
    .mem_advise         = uct_ib_mem_advise,
    .mkey_pack          = uct_ib_mkey_pack,
    .detect_memory_type = ucs_empty_function_return_unsupported,
};

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

    return 1;
}

static ucs_status_t uct_ib_query_md_resources(uct_component_t *component,
                                              uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    UCS_MODULE_FRAMEWORK_DECLARE(uct_ib);
    int num_resources = 0;
    uct_md_resource_desc_t *resources;
    struct ibv_device **device_list;
    ucs_status_t status;
    int i, num_devices;

    UCS_MODULE_FRAMEWORK_LOAD(uct_ib, 0);

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    resources = ucs_calloc(num_devices, sizeof(*resources), "ib resources");
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

static ucs_status_t UCS_V_UNUSED
uct_ib_md_global_odp_init(uct_ib_md_t *md, uct_mem_h *memh_p)
{
    uct_ib_verbs_mem_t *global_odp;
    uct_ib_mr_t *mr;
    ucs_status_t status;

    global_odp = (uct_ib_verbs_mem_t *)uct_ib_memh_alloc(md);
    if (global_odp == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    mr = &global_odp->mrs[UCT_IB_MR_DEFAULT];
    status = uct_ib_reg_mr(md->pd, 0, UINT64_MAX,
                           UCT_IB_MEM_ACCESS_FLAGS | IBV_ACCESS_ON_DEMAND,
                           &mr->ib, 1);
    if (status != UCS_OK) {
        ucs_debug("%s: failed to register global mr: %m",
                  uct_ib_device_name(&md->dev));
        goto err;
    }

    global_odp->super.flags = UCT_IB_MEM_FLAG_ODP;
    uct_ib_memh_init_keys(&global_odp->super, mr->ib->lkey, mr->ib->rkey);
    *memh_p = global_odp;
    return UCS_OK;

err:
    uct_ib_memh_free(&global_odp->super);
    return status;
}

static ucs_status_t
uct_ib_md_parse_reg_methods(uct_ib_md_t *md,
                            const uct_ib_md_config_t *md_config)
{
    ucs_rcache_params_t rcache_params;
    ucs_status_t status;
    int i;

    for (i = 0; i < md_config->reg_methods.count; ++i) {
        if (!strcasecmp(md_config->reg_methods.rmtd[i], "rcache")) {
            uct_md_set_rcache_params(&rcache_params, &md_config->rcache);
            rcache_params.region_struct_size = sizeof(ucs_rcache_region_t) +
                                               md->memh_struct_size;
            rcache_params.max_alignment      = ucs_get_page_size();
            rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED;
            if (md->reg_mem_types & ~UCS_BIT(UCS_MEMORY_TYPE_HOST)) {
                rcache_params.ucm_events    |= UCM_EVENT_MEM_TYPE_FREE;
            }
            rcache_params.context            = md;
            rcache_params.ops                = &uct_ib_rcache_ops;

            status = ucs_rcache_create(&rcache_params, uct_ib_device_name(&md->dev),
                                       UCS_STATS_RVAL(md->stats), &md->rcache);
            if (status != UCS_OK) {
                ucs_debug("%s: failed to create registration cache: %s",
                          uct_ib_device_name(&md->dev),
                          ucs_status_string(status));
                continue;
            }

            md->super.ops = &uct_ib_md_rcache_ops;
            md->reg_cost  = ucs_linear_func_make(
                    uct_md_rcache_overhead(&md_config->rcache), 0);
            ucs_debug("%s: using registration cache",
                      uct_ib_device_name(&md->dev));
            return UCS_OK;
#if HAVE_ODP_IMPLICIT
        } else if (!strcasecmp(md_config->reg_methods.rmtd[i], "odp")) {
            if (!(md->dev.flags & UCT_IB_DEVICE_FLAG_ODP_IMPLICIT)) {
                ucs_debug("%s: on-demand-paging with global memory region is "
                          "not supported", uct_ib_device_name(&md->dev));
                continue;
            }

            status = uct_ib_md_global_odp_init(md, &md->global_odp);
            if (status != UCS_OK) {
                continue;
            }

            md->super.ops = &uct_ib_md_global_odp_ops;
            md->reg_cost  = ucs_linear_func_make(10e-9, 0);
            ucs_debug("%s: using odp global key", uct_ib_device_name(&md->dev));
            return UCS_OK;
#endif
        } else if (!strcmp(md_config->reg_methods.rmtd[i], "direct")) {
            md->super.ops = &uct_ib_md_ops;
            md->reg_cost  = md_config->uc_reg_cost;
            ucs_debug("%s: using direct registration",
                      uct_ib_device_name(&md->dev));
            return UCS_OK;
        }
    }

    return UCS_ERR_INVALID_PARAM;
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

static void uct_ib_md_release_reg_method(uct_ib_md_t *md)
{
    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }
    if (md->global_odp != NULL) {
        uct_ib_mem_dereg(&md->super, md->global_odp);
    }
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
            if (UCS_CONFIG_BW_IS_AUTO(md_config->pci_bw.device[i].bw)) {
                break; /* read data from system */
            }

            md->pci_bw = md_config->pci_bw.device[i].bw;
            return;
        }
    }

    /* Did not find a matching configuration - take from underlying device */
    md->pci_bw = md->dev.pci_bw;
}

ucs_status_t uct_ib_md_open(uct_component_t *component, const char *md_name,
                            const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    const uct_ib_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_ib_md_config_t);
    ucs_status_t status = UCS_ERR_UNSUPPORTED;
    uct_ib_md_t *md = NULL;
    struct ibv_device **ib_device_list, *ib_device;
    int i, num_devices, ret, fork_init = 0;

    ucs_trace("opening IB device %s", md_name);

#if !HAVE_DEVX
    if (md_config->devx == UCS_YES) {
        ucs_error("DEVX requested but not supported");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }
#endif

    /* Get device list from driver */
    ib_device_list = ibv_get_device_list(&num_devices);
    if (ib_device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    ib_device = NULL;
    for (i = 0; i < num_devices; ++i) {
        if (!strcmp(ibv_get_device_name(ib_device_list[i]), md_name)) {
            ib_device = ib_device_list[i];
            break;
        }
    }

    if (ib_device == NULL) {
        ucs_debug("IB device %s not found", md_name);
        status = UCS_ERR_NO_DEVICE;
        goto out_free_dev_list;
    }

    if (md_config->fork_init != UCS_NO) {
        ret = ibv_fork_init();
        if (ret) {
            if (md_config->fork_init == UCS_YES) {
                ucs_error("ibv_fork_init() failed: %m");
                status = UCS_ERR_IO_ERROR;
                goto out_free_dev_list;
            }
            ucs_debug("ibv_fork_init() failed: %m, continuing, but fork may be unsafe.");
            uct_ib_fork_warn_enable();
        } else {
            fork_init = 1;
        }
    } else {
        uct_ib_fork_warn_enable();
    }

    for (i = 0; i < ucs_static_array_size(uct_ib_ops); i++) {
        status = uct_ib_ops[i]->ops->open(ib_device, md_config, &md);
        if (status == UCS_OK) {
            ucs_debug("%s: md open by '%s' is successful", md_name,
                      uct_ib_ops[i]->name);
            md->ops = uct_ib_ops[i]->ops;
            break;
        } else if (status != UCS_ERR_UNSUPPORTED) {
            goto out_free_dev_list;
        }
        ucs_debug("%s: md open by '%s' failed, trying next", md_name,
                  uct_ib_ops[i]->name);
    }

    if (status != UCS_OK) {
        ucs_assert(status == UCS_ERR_UNSUPPORTED);
        ucs_debug("Unsupported IB device %s", md_name);
        goto out_free_dev_list;
    }

    /* cppcheck-suppress autoVariables */
    *md_p         = &md->super;
    md->fork_init = fork_init;
    status        = UCS_OK;

out_free_dev_list:
    ibv_free_device_list(ib_device_list);
out:
    return status;
}

void uct_ib_md_parse_relaxed_order(uct_ib_md_t *md,
                                   const uct_ib_md_config_t *md_config)
{
#if HAVE_DECL_IBV_ACCESS_RELAXED_ORDERING
    md->relaxed_order = (md_config->mr_relaxed_order == UCS_CONFIG_ON) ||
                        ((md_config->mr_relaxed_order == UCS_CONFIG_AUTO) &&
                         ucs_cpu_prefer_relaxed_order());
#else
    if (md_config->mr_relaxed_order == UCS_CONFIG_ON) {
        ucs_warn("relaxed order memory access requested but not supported");
    }
#endif
}

static void uct_ib_check_gpudirect_driver(uct_ib_md_t *md, const char *file,
                                          ucs_memory_type_t mem_type)
{
    if (!access(file, F_OK)) {
        md->reg_mem_types |= UCS_BIT(mem_type);
    }

    ucs_debug("%s: %s GPUDirect RDMA is %s", uct_ib_device_name(&md->dev),
              ucs_memory_type_names[mem_type],
              md->reg_mem_types & UCS_BIT(mem_type) ? "enabled" : "disabled");
}

ucs_status_t uct_ib_md_open_common(uct_ib_md_t *md,
                                   struct ibv_device *ib_device,
                                   const uct_ib_md_config_t *md_config)
{
    ucs_status_t status;

    md->super.ops       = &uct_ib_md_ops;
    md->super.component = &uct_ib_component;
    md->config          = md_config->ext;
    md->cap_flags       = UCT_MD_FLAG_REG |
                          UCT_MD_FLAG_NEED_MEMH |
                          UCT_MD_FLAG_NEED_RKEY |
                          UCT_MD_FLAG_ADVISE;

    if (md->config.odp.max_size == UCS_MEMUNITS_AUTO) {
        md->config.odp.max_size = uct_ib_device_odp_max_size(&md->dev);
    }

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

#if HAVE_DECL_IBV_EXP_SETENV
    ibv_exp_setenv(md->dev.ibv_context, "MLX_QP_ALLOC_TYPE", "ANON", 0);
    ibv_exp_setenv(md->dev.ibv_context, "MLX_CQ_ALLOC_TYPE", "ANON", 0);
#endif

    if (strlen(md_config->subnet_prefix) > 0) {
        status = uct_ib_md_parse_subnet_prefix(md_config->subnet_prefix,
                                               &md->subnet_filter);

        if (status != UCS_OK) {
            goto err_cleanup_device;
        }

        md->check_subnet_filter = 1;
    }

    /* Allocate memory domain */
    md->pd = ibv_alloc_pd(md->dev.ibv_context);
    if (md->pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err_cleanup_device;
    }

    /* Check for GPU-direct support */
    md->reg_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    if (md_config->enable_gpudirect_rdma != UCS_NO) {
        /* check if GDR driver is loaded */
        uct_ib_check_gpudirect_driver(
                md, "/sys/kernel/mm/memory_peers/nv_mem/version",
                UCS_MEMORY_TYPE_CUDA);

        /* check if ROCM KFD driver is loaded */
        uct_ib_check_gpudirect_driver(md, "/dev/kfd", UCS_MEMORY_TYPE_ROCM);

        if (!(md->reg_mem_types & ~UCS_MEMORY_TYPES_CPU_ACCESSIBLE) &&
            (md_config->enable_gpudirect_rdma == UCS_YES)) {
            ucs_error("%s: Couldn't enable GPUDirect RDMA. Please make sure"
                      " nv_peer_mem or amdgpu plugin installed correctly.",
                      uct_ib_device_name(&md->dev));
            status = UCS_ERR_UNSUPPORTED;
            goto err_dealloc_pd;
        }
    }

    status = uct_ib_md_parse_reg_methods(md, md_config);
    if (status != UCS_OK) {
        goto err_dealloc_pd;
    }

    md->dev.max_zcopy_log_sge = INT_MAX;
    if (md->reg_mem_types & ~UCS_BIT(UCS_MEMORY_TYPE_HOST)) {
        md->dev.max_zcopy_log_sge = 1;
    }

    uct_ib_md_set_pci_bw(md, md_config);

    return UCS_OK;

err_dealloc_pd:
    ibv_dealloc_pd(md->pd);
err_cleanup_device:
    uct_ib_device_cleanup(&md->dev);
err_release_stats:
    UCS_STATS_NODE_FREE(md->stats);
err:
    return status;
}

void uct_ib_md_close(uct_md_h uct_md)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    /* Must be done before md->ops->cleanup, since it can call functions from
     * md->ops */
    uct_ib_md_release_reg_method(md);
    md->ops->cleanup(md);
    uct_ib_md_release_device_config(md);
    uct_ib_device_cleanup_ah_cached(&md->dev);
    ibv_dealloc_pd(md->pd);
    uct_ib_device_cleanup(&md->dev);
    ibv_close_device(md->dev.ibv_context);
    UCS_STATS_NODE_FREE(md->stats);
    ucs_free(md);
}

static uct_ib_md_ops_t uct_ib_verbs_md_ops;

static ucs_status_t uct_ib_verbs_md_open(struct ibv_device *ibv_device,
                                         const uct_ib_md_config_t *md_config,
                                         uct_ib_md_t **p_md)
{
    uct_ib_device_t *dev;
    ucs_status_t status;
    uct_ib_md_t *md;
    int num_mrs;

    md = ucs_calloc(1, sizeof(*md), "ib_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* Open verbs context */
    dev              = &md->dev;
    dev->ibv_context = ibv_open_device(ibv_device);
    if (dev->ibv_context == NULL) {
        ucs_diag("ibv_open_device(%s) failed: %m",
                 ibv_get_device_name(ibv_device));
        status = UCS_ERR_IO_ERROR;
        goto err;
    }

    status = uct_ib_device_query(dev, ibv_device);
    if (status != UCS_OK) {
        goto err_free_context;
    }

    if (UCT_IB_HAVE_ODP_IMPLICIT(&dev->dev_attr)) {
        md->dev.flags |= UCT_IB_DEVICE_FLAG_ODP_IMPLICIT;
    }

    if (IBV_EXP_HAVE_ATOMIC_HCA(&dev->dev_attr)) {
        dev->atomic_arg_sizes = sizeof(uint64_t);
    }

    md->ops = &uct_ib_verbs_md_ops;
    status = uct_ib_md_parse_device_config(md, md_config);
    if (status != UCS_OK) {
        goto err_free_context;
    }

    uct_ib_md_parse_relaxed_order(md, md_config);
    num_mrs = 1;      /* UCT_IB_MR_DEFAULT */

    if (md->relaxed_order) {
        ++num_mrs;    /* UCT_IB_MR_STRICT_ORDER */
    }

    md->memh_struct_size = sizeof(uct_ib_verbs_mem_t) +
                          (sizeof(uct_ib_mr_t) * num_mrs);

    status = uct_ib_md_open_common(md, ibv_device, md_config);
    if (status != UCS_OK) {
        goto err_dev_cfg;
    }

    md->dev.flags = uct_ib_device_spec(&md->dev)->flags;
    md->name      = UCT_IB_MD_NAME(verbs);

    *p_md = md;
    return UCS_OK;

err_dev_cfg:
    uct_ib_md_release_device_config(md);
err_free_context:
    ibv_close_device(dev->ibv_context);
err:
    ucs_free(md);
    return status;
}

static void uct_ib_md_vfs_init(uct_md_h md)
{
    uct_ib_md_t *ib_md = ucs_derived_of(md, uct_ib_md_t);

    if (ib_md->rcache != NULL) {
        ucs_vfs_obj_add_sym_link(md, ib_md->rcache, "rcache");
    }
}

static uct_ib_md_ops_t uct_ib_verbs_md_ops = {
    .open                = uct_ib_verbs_md_open,
    .cleanup             = (uct_ib_md_cleanup_func_t)ucs_empty_function,
    .reg_key             = uct_ib_verbs_reg_key,
    .reg_indirect_key    = (uct_ib_md_reg_indirect_key_func_t)ucs_empty_function_return_unsupported,
    .dereg_key           = uct_ib_verbs_dereg_key,
    .reg_atomic_key      = uct_ib_verbs_reg_atomic_key,
    .dereg_atomic_key    = (uct_ib_md_dereg_atomic_key_func_t)ucs_empty_function_return_success,
    .reg_multithreaded   = (uct_ib_md_reg_multithreaded_func_t)ucs_empty_function_return_unsupported,
    .dereg_multithreaded = (uct_ib_md_dereg_multithreaded_func_t)ucs_empty_function_return_unsupported,
    .mem_prefetch        = (uct_ib_md_mem_prefetch_func_t)ucs_empty_function_return_success,
    .get_atomic_mr_id    = (uct_ib_md_get_atomic_mr_id_func_t)ucs_empty_function_return_unsupported,
};

static UCT_IB_MD_DEFINE_ENTRY(verbs, uct_ib_verbs_md_ops);

uct_component_t uct_ib_component = {
    .query_md_resources = uct_ib_query_md_resources,
    .md_open            = uct_ib_md_open,
    .cm_open            = ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ib_rkey_unpack,
    .rkey_ptr           = ucs_empty_function_return_unsupported,
    .rkey_release       = ucs_empty_function_return_success,
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
    .md_vfs_init        = uct_ib_md_vfs_init
};

void UCS_F_CTOR uct_ib_init()
{
    int i;

    uct_component_register(&uct_ib_component);

    for (i = 0; i < ucs_static_array_size(uct_ib_tls); i++) {
        uct_tl_register(&uct_ib_component, uct_ib_tls[i]);
    }
}

void UCS_F_DTOR uct_ib_cleanup()
{
    int i;

    for (i = ucs_static_array_size(uct_ib_tls) - 1; i >= 0; i--) {
        uct_tl_unregister(uct_ib_tls[i]);
    }

    uct_component_unregister(&uct_ib_component);
}
