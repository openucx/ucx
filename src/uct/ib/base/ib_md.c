/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) The University of Tennessee and The University
 *               of Tennessee Research Foundation. 2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ib_md.h"
#include "ib_device.h"

#include <ucs/arch/atomic.h>
#include <ucs/debug/profile.h>
#include <pthread.h>


#define UCT_IB_MD_PREFIX         "ib"
#define UCT_IB_MEM_ACCESS_FLAGS  (IBV_ACCESS_LOCAL_WRITE | \
                                  IBV_ACCESS_REMOTE_WRITE | \
                                  IBV_ACCESS_REMOTE_READ | \
                                  IBV_ACCESS_REMOTE_ATOMIC)
#define UCT_IB_MD_RCACHE_DEFAULT_ALIGN 16


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
     ucs_offsetof(uct_ib_md_config_t, uc_reg_cost.overhead), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
     ucs_offsetof(uct_ib_md_config_t, uc_reg_cost.growth), UCS_CONFIG_TYPE_TIME},

    {"FORK_INIT", "try",
     "Initialize a fork-safe IB library with ibv_fork_init().",
     ucs_offsetof(uct_ib_md_config_t, fork_init), UCS_CONFIG_TYPE_TERNARY},

    {"ASYNC_EVENTS", "n",
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
     "  <vendor-id>:<part-id>[:name[:<flags>[:<priority>]]]\n"
     "where:\n"
     "  <vendor-id> - (mandatory) vendor id, integer or hexadecimal.\n"
     "  <part-id>   - (mandatory) vendor part id, integer or hexadecimal.\n"
     "  <name>      - (optional) device name.\n"
     "  <flags>     - (optional) empty, or any of: '4' - mlx4 device, '5' - mlx5 device.\n"
     "  <priority>  - (optional) device priority, integer.\n",
     ucs_offsetof(uct_ib_md_config_t, custom_devices), UCS_CONFIG_TYPE_STRING_ARRAY},

    {"PREFER_NEAREST_DEVICE", "y",
     "Prefer nearest device to cpu when selecting a device from NET_DEVICES list.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.prefer_nearest_device), UCS_CONFIG_TYPE_BOOL},

    {"CONTIG_PAGES", "n",
     "Enable allocation with contiguous pages. Warning: enabling this option may\n"
     "cause stack smashing.\n",
     ucs_offsetof(uct_ib_md_config_t, ext.enable_contig_pages), UCS_CONFIG_TYPE_BOOL},

    {NULL}
};

#if ENABLE_STATS
static ucs_stats_class_t uct_ib_md_stats_class = {
    .name           = "",
    .num_counters   = UCT_IB_MD_STAT_LAST,
    .counter_names = {
        [UCT_IB_MD_STAT_MEM_ALLOC]   = "mem_alloc",
        [UCT_IB_MD_STAT_MEM_REG]     = "mem_reg"
    }
};
#endif

static ucs_status_t uct_ib_md_query(uct_md_h uct_md, uct_md_attr_t *md_attr)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    md_attr->cap.max_alloc = ULONG_MAX; /* TODO query device */
    md_attr->cap.max_reg   = ULONG_MAX; /* TODO query device */
    md_attr->cap.flags     = UCT_MD_FLAG_REG       |
                             UCT_MD_FLAG_NEED_MEMH |
                             UCT_MD_FLAG_NEED_RKEY |
                             UCT_MD_FLAG_ADVISE;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);

#if HAVE_CUDA
    /* check if GDR driver is loaded */
    if (!access("/sys/kernel/mm/memory_peers/nv_mem/version", F_OK)) {
        md_attr->cap.reg_mem_types |= UCS_BIT(UCT_MD_MEM_TYPE_CUDA);
    }
#endif

    md_attr->cap.mem_type     = UCT_MD_MEM_TYPE_HOST;
    md_attr->rkey_packed_size = UCT_IB_MD_PACKED_RKEY_SIZE;

    if (md->config.enable_contig_pages &&
        IBV_EXP_HAVE_CONTIG_PAGES(&md->dev.dev_attr))
    {
        md_attr->cap.flags |= UCT_MD_FLAG_ALLOC;
    }

    md_attr->reg_cost      = md->reg_cost;
    md_attr->local_cpus    = md->dev.local_cpus;
    return UCS_OK;
}

static ucs_status_t uct_ib_md_umr_qp_create(uct_ib_md_t *md)
{
#if HAVE_EXP_UMR
    struct ibv_exp_qp_init_attr qp_init_attr;
    struct ibv_qp_attr qp_attr;
    uint8_t port_num;
    int ret;
    uct_ib_device_t *ibdev;
    struct ibv_exp_port_attr *port_attr;

    ibdev = &md->dev;

    if (!(ibdev->dev_attr.exp_device_cap_flags & IBV_EXP_DEVICE_UMR)) {
        return UCS_ERR_UNSUPPORTED;
    }

    /* TODO: fix port selection. It looks like active port should be used */
    port_num = ibdev->first_port;
    port_attr = uct_ib_device_port_attr(ibdev, port_num);

    memset(&qp_init_attr, 0, sizeof(qp_init_attr));

    md->umr_cq = ibv_create_cq(ibdev->ibv_context, 1, NULL, NULL, 0);
    if (md->umr_cq == NULL) {
        ucs_error("failed to create UMR CQ: %m");
        goto err;
    }

    qp_init_attr.qp_type             = IBV_QPT_RC;
    qp_init_attr.send_cq             = md->umr_cq;
    qp_init_attr.recv_cq             = md->umr_cq;
    qp_init_attr.cap.max_inline_data = 0;
    qp_init_attr.cap.max_recv_sge    = 1;
    qp_init_attr.cap.max_send_sge    = 1;
    qp_init_attr.srq                 = NULL;
    qp_init_attr.cap.max_recv_wr     = 16;
    qp_init_attr.cap.max_send_wr     = 16;
    qp_init_attr.pd                  = md->pd;
    qp_init_attr.comp_mask           = IBV_EXP_QP_INIT_ATTR_PD|IBV_EXP_QP_INIT_ATTR_MAX_INL_KLMS;
    qp_init_attr.max_inl_recv        = 0;
#if (HAVE_IBV_EXP_QP_CREATE_UMR_CAPS || HAVE_EXP_UMR_NEW_API)
    qp_init_attr.max_inl_send_klms   = ibdev->dev_attr.umr_caps.max_send_wqe_inline_klms;
#else
    qp_init_attr.max_inl_send_klms   = ibdev->dev_attr.max_send_wqe_inline_klms;
#endif

#if HAVE_IBV_EXP_QP_CREATE_UMR
    qp_init_attr.comp_mask          |= IBV_EXP_QP_INIT_ATTR_CREATE_FLAGS;
    qp_init_attr.exp_create_flags    = IBV_EXP_QP_CREATE_UMR;
#endif

    md->umr_qp = ibv_exp_create_qp(ibdev->ibv_context, &qp_init_attr);
    if (md->umr_qp == NULL) {
        ucs_error("failed to create UMR QP: %m");
        goto err_destroy_cq;
    }

    memset(&qp_attr, 0, sizeof(qp_attr));

    /* Modify QP to INIT state */
    qp_attr.qp_state                 = IBV_QPS_INIT;
    qp_attr.pkey_index               = 0;
    qp_attr.port_num                 = port_num;
    qp_attr.qp_access_flags          = UCT_IB_MEM_ACCESS_FLAGS;
    ret = ibv_modify_qp(md->umr_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (ret) {
        ucs_error("Failed to modify UMR QP to INIT: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTR */
    qp_attr.qp_state                 = IBV_QPS_RTR;
    qp_attr.dest_qp_num              = md->umr_qp->qp_num;

    memset(&qp_attr.ah_attr, 0, sizeof(qp_attr.ah_attr));
    qp_attr.ah_attr.port_num         = port_num;
    qp_attr.ah_attr.dlid             = port_attr->lid;
    qp_attr.ah_attr.is_global        = 1;
    if (uct_ib_device_query_gid(ibdev, port_num, 0, &qp_attr.ah_attr.grh.dgid) != UCS_OK) {
        goto err_destroy_qp;
    }
    qp_attr.rq_psn                   = 0;
    qp_attr.path_mtu                 = IBV_MTU_512;
    qp_attr.min_rnr_timer            = 7;
    qp_attr.max_dest_rd_atomic       = 1;
    ret = ibv_modify_qp(md->umr_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
                        IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (ret) {
        ucs_error("Failed to modify UMR QP to RTR: %m");
        goto err_destroy_qp;
    }

    /* Modify to RTS */
    qp_attr.qp_state                 = IBV_QPS_RTS;
    qp_attr.sq_psn                   = 0;
    qp_attr.timeout                  = 7;
    qp_attr.rnr_retry                = 7;
    qp_attr.retry_cnt                = 7;
    qp_attr.max_rd_atomic            = 1;
    ret = ibv_modify_qp(md->umr_qp, &qp_attr,
                        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
                        IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                        IBV_QP_MAX_QP_RD_ATOMIC);
    if (ret) {
        ucs_error("Failed to modify UMR QP to RTS: %m");
        goto err_destroy_qp;
    }
    return UCS_OK;

err_destroy_qp:
    ibv_destroy_qp(md->umr_qp);
err_destroy_cq:
    ibv_destroy_cq(md->umr_cq);
err:
    return UCS_ERR_IO_ERROR;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static void uct_ib_md_umr_qp_destroy(uct_ib_md_t *md)
{
#if HAVE_EXP_UMR
    if (md->umr_qp != NULL) {
        ibv_destroy_qp(md->umr_qp);
    }
    if (md->umr_cq != NULL) {
        ibv_destroy_cq(md->umr_cq);
    }
#endif
}

uint8_t uct_ib_md_get_atomic_mr_id(uct_ib_md_t *md)
{
#if HAVE_EXP_UMR
    if ((md->umr_qp == NULL) || (md->umr_cq == NULL)) {
        return 0;
    }
    /* Generate atomic UMR id. We want umrs for same virtual addresses to have
     * different ids across processes.
     *
     * Usually parallel processes running on the same node as part of a single
     * job will have consecutive PIDs. For example MPI ranks, slurm spawned tasks...
     */
    return getpid() % 256;
#else
    return 0;
#endif
}

static ucs_status_t uct_ib_md_reg_mr(uct_ib_md_t *md, void *address,
                                     size_t length, uint64_t exp_access,
                                     int silent, struct ibv_mr **mr_p)
{
    ucs_log_level_t level = silent ? UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR;
    struct ibv_mr *mr;

    if (exp_access) {
#if HAVE_DECL_IBV_EXP_REG_MR
        struct ibv_exp_reg_mr_in in;

        memset(&in, 0, sizeof(in));
        in.pd           = md->pd;
        in.addr         = address;
        in.length       = length;
        in.exp_access   = UCT_IB_MEM_ACCESS_FLAGS | exp_access;

        mr = UCS_PROFILE_CALL(ibv_exp_reg_mr, &in);
        if (mr == NULL) {
            ucs_log(level, "ibv_exp_reg_mr(address=%p, length=%zu, exp_access=0x%lx) failed: %m",
                    in.addr, in.length, in.exp_access);
            return UCS_ERR_IO_ERROR;
        }
#else
        return UCS_ERR_UNSUPPORTED;
#endif
    } else {
        mr = UCS_PROFILE_CALL(ibv_reg_mr, md->pd, address, length,
                              UCT_IB_MEM_ACCESS_FLAGS);
        if (mr == NULL) {
            ucs_log(level, "ibv_reg_mr(address=%p, length=%zu, access=0x%x) failed: %m",
                      address, length, UCT_IB_MEM_ACCESS_FLAGS);
            return UCS_ERR_IO_ERROR;
        }
    }

    *mr_p = mr;
    return UCS_OK;
}

static ucs_status_t uct_ib_md_post_umr(uct_ib_md_t *md, struct ibv_mr *mr,
                                       off_t offset, struct ibv_mr **umr_p)
{
#if HAVE_EXP_UMR
    struct ibv_exp_mem_region mem_reg;
    struct ibv_exp_send_wr wr, *bad_wr;
    struct ibv_exp_create_mr_in mrin;
    ucs_status_t status;
    struct ibv_mr *umr;
    struct ibv_wc wc;
    int ret;

    if (md->umr_qp == NULL) {
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    /* Create memory key */
    memset(&mrin, 0, sizeof(mrin));
    mrin.pd                       = md->pd;

#ifdef HAVE_EXP_UMR_NEW_API
    mrin.attr.create_flags        = IBV_EXP_MR_INDIRECT_KLMS;
    mrin.attr.exp_access_flags    = UCT_IB_MEM_ACCESS_FLAGS;
    mrin.attr.max_klm_list_size   = 1;
#else
    mrin.attr.create_flags        = IBV_MR_NONCONTIG_MEM;
    mrin.attr.access_flags        = UCT_IB_MEM_ACCESS_FLAGS;
    mrin.attr.max_reg_descriptors = 1;
#endif

    umr = ibv_exp_create_mr(&mrin);
    if (!umr) {
        ucs_error("Failed to create modified_mr: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    /* Fill memory list and UMR */
    memset(&wr, 0, sizeof(wr));
    memset(&mem_reg, 0, sizeof(mem_reg));

    mem_reg.base_addr                              = (uintptr_t) mr->addr;
    mem_reg.length                                 = mr->length;

#ifdef HAVE_EXP_UMR_NEW_API
    mem_reg.mr                                     = mr;
    wr.ext_op.umr.umr_type                         = IBV_EXP_UMR_MR_LIST;
    wr.ext_op.umr.mem_list.mem_reg_list            = &mem_reg;
    wr.ext_op.umr.exp_access                       = UCT_IB_MEM_ACCESS_FLAGS;
    wr.ext_op.umr.modified_mr                      = umr;
    wr.ext_op.umr.base_addr                        = (uint64_t) (uintptr_t) mr->addr + offset;
    wr.ext_op.umr.num_mrs                          = 1;
    ucs_trace_data("UMR_FILL qp 0x%x lkey 0x%x base 0x%lx [addr %lx len %zu lkey 0x%x]",
                   md->umr_qp->qp_num, wr.ext_op.umr.modified_mr->lkey,
                   wr.ext_op.umr.base_addr, mem_reg.base_addr, mem_reg.length,
                   mem_reg.mr->lkey);
#else
    mem_reg.m_key                                  = mr;
    wr.ext_op.umr.memory_key.mkey_type             = IBV_EXP_UMR_MEM_LAYOUT_NONCONTIG;
    wr.ext_op.umr.memory_key.mem_list.mem_reg_list = &mem_reg;
    wr.ext_op.umr.memory_key.access                = UCT_IB_MEM_ACCESS_FLAGS;
    wr.ext_op.umr.memory_key.modified_mr           = umr;
    wr.ext_op.umr.memory_key.region_base_addr      = mr->addr + offset;
    wr.num_sge                                     = 1;
#endif

    wr.exp_opcode                                  = IBV_EXP_WR_UMR_FILL;
    wr.exp_send_flags                              = IBV_EXP_SEND_INLINE | IBV_EXP_SEND_SIGNALED;

    /* Post UMR */
    ret = ibv_exp_post_send(md->umr_qp, &wr, &bad_wr);
    if (ret) {
        ucs_error("ibv_exp_post_send(UMR_FILL) failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free_umr;
    }

    /* Wait for send UMR completion */
    for (;;) {
        ret = ibv_poll_cq(md->umr_cq, 1, &wc);
        if (ret < 0) {
            ucs_error("ibv_exp_poll_cq(umr_cq) failed: %m");
            status = UCS_ERR_IO_ERROR;
            goto err_free_umr;
        }
        if (ret == 1) {
            if (wc.status != IBV_WC_SUCCESS) {
                ucs_error("UMR_FILL completed with error: %s vendor_err %d",
                          ibv_wc_status_str(wc.status), wc.vendor_err);
                status = UCS_ERR_IO_ERROR;
                goto err_free_umr;
            }
            break;
        }
    }

    ucs_trace("UMR registered memory %p..%p offset 0x%lx on %s lkey 0x%x rkey 0x%x",
              mr->addr, mr->addr + mr->length, offset, uct_ib_device_name(&md->dev),
              umr->lkey, umr->rkey);
    *umr_p = umr;

    return UCS_OK;

err_free_umr:
    UCS_PROFILE_CALL(ibv_dereg_mr, umr);
err:
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_dereg_mr(struct ibv_mr *mr)
{
    int ret;

    ret = UCS_PROFILE_CALL(ibv_dereg_mr, mr);
    if (ret != 0) {
        ucs_error("ibv_dereg_mr() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t uct_ib_memh_dereg(uct_ib_mem_t *memh)
{
    ucs_status_t s1, s2;

    s1 = s2 = UCS_OK;
    if (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) {
        s2 = uct_ib_dereg_mr(memh->atomic_mr);
        memh->flags &= ~UCT_IB_MEM_FLAG_ATOMIC_MR;
    }
    if (memh->mr != NULL) {
        s1 = uct_ib_dereg_mr(memh->mr);
    }
    return (s1 != UCS_OK) ? s1 : s2;
}

static void uct_ib_memh_free(uct_ib_mem_t *memh)
{
    ucs_free(memh);
}

static uct_ib_mem_t *uct_ib_memh_alloc()
{
    return ucs_calloc(1, sizeof(uct_ib_mem_t), "ib_memh");
}

static uint64_t uct_ib_md_access_flags(uct_ib_md_t *md, unsigned flags,
                                       size_t length)
{
    uint64_t exp_access = 0;

    if ((flags & UCT_MD_MEM_FLAG_NONBLOCK) && (length > 0) &&
        (length <= md->config.odp.max_size)) {
        exp_access |= IBV_EXP_ACCESS_ON_DEMAND;
    }
    return exp_access;
}

#if HAVE_NUMA
static ucs_status_t uct_ib_mem_set_numa_policy(uct_ib_md_t *md, uct_ib_mem_t *memh)
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
        start = ucs_align_down_pow2((uintptr_t)memh->mr->addr, ucs_get_page_size());
        end   = ucs_align_up_pow2((uintptr_t)memh->mr->addr + memh->mr->length,
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
static ucs_status_t uct_ib_mem_set_numa_policy(uct_ib_md_t *md, uct_ib_mem_t *memh)
{
    return UCS_OK;
}
#endif /* UCT_MD_DISABLE_NUMA */

static ucs_status_t 
uct_ib_mem_prefetch_internal(uct_ib_md_t *md, uct_ib_mem_t *memh, void *addr, size_t length)
{
#if HAVE_DECL_IBV_EXP_PREFETCH_MR
    struct ibv_exp_prefetch_attr attr;
    int ret;

    if ((memh->flags & UCT_IB_MEM_FLAG_ODP)) {
        if ((addr < memh->mr->addr) ||
            (addr + length > memh->mr->addr + memh->mr->length)) {
            return UCS_ERR_INVALID_PARAM;
        }
        ucs_debug("memh %p prefetch %p length %llu", memh, addr, 
                  (unsigned long long)length);
        attr.flags     = IBV_EXP_PREFETCH_WRITE_ACCESS;
        attr.addr      = addr;
        attr.length    = length;
        attr.comp_mask = 0;

        ret = UCS_PROFILE_CALL(ibv_exp_prefetch_mr, memh->mr, &attr);
        if (ret) {
            ucs_error("ibv_exp_prefetch_mr(addr=%p length=%zu) returned %d: %m",
                      attr.addr, attr.length, ret);
            return UCS_ERR_IO_ERROR;
        }
    }
#endif
    return UCS_OK;
}

static void uct_ib_mem_init(uct_ib_mem_t *memh, unsigned uct_flags,
                            uint64_t exp_access)
{
    memh->lkey  = memh->mr->lkey;
    memh->flags = 0;

    /* coverity[dead_error_condition] */
    if (exp_access & IBV_EXP_ACCESS_ON_DEMAND) {
        memh->flags |= UCT_IB_MEM_FLAG_ODP;
    }

    if (uct_flags & UCT_MD_MEM_ACCESS_REMOTE_ATOMIC) {
        memh->flags |= UCT_IB_MEM_ACCESS_REMOTE_ATOMIC;
    }
}

static ucs_status_t uct_ib_mem_alloc(uct_md_h uct_md, size_t *length_p,
                                     void **address_p, unsigned flags,
                                     uct_mem_h *memh_p UCS_MEMTRACK_ARG)
{
#if HAVE_DECL_IBV_EXP_ACCESS_ALLOCATE_MR
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_status_t status;
    uint64_t exp_access;
    uct_ib_mem_t *memh;
    size_t length;

    if (!md->config.enable_contig_pages) {
        return UCS_ERR_UNSUPPORTED;
    }

    memh = uct_ib_memh_alloc();
    if (memh == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    length     = ucs_memtrack_adjust_alloc_size(*length_p);
    exp_access = uct_ib_md_access_flags(md, flags, length) |
                 IBV_EXP_ACCESS_ALLOCATE_MR;
    status = uct_ib_md_reg_mr(md, NULL, length, exp_access, 0, &memh->mr);
    if (status != UCS_OK) {
        goto err_free_memh;
    }

    ucs_trace("allocated memory %p..%p on %s lkey 0x%x rkey 0x%x",
              memh->mr->addr, memh->mr->addr + memh->mr->length, uct_ib_device_name(&md->dev),
              memh->mr->lkey, memh->mr->rkey);

    uct_ib_mem_init(memh, flags, exp_access);
    uct_ib_mem_set_numa_policy(md, memh);

    if (md->config.odp.prefetch) {
        uct_ib_mem_prefetch_internal(md, memh, memh->mr->addr, memh->mr->length);
    }

    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_ALLOC, +1);
    *address_p = memh->mr->addr;
    *length_p  = memh->mr->length;
    *memh_p    = memh;
    ucs_memtrack_allocated(address_p, length_p UCS_MEMTRACK_VAL);
    return UCS_OK;

err_free_memh:
    uct_ib_memh_free(memh);
err:
    return status;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

static ucs_status_t uct_ib_mem_free(uct_md_h md, uct_mem_h memh)
{
    uct_ib_mem_t *ib_memh = memh;
    ucs_status_t status;

    ucs_memtrack_releasing_adjusted(ib_memh->mr->addr);

    status = UCS_PROFILE_CALL(uct_ib_memh_dereg, memh);
    if (status != UCS_OK) {
        return status;
    }

    uct_ib_memh_free(ib_memh);
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_reg_internal(uct_md_h uct_md, void *address,
                                            size_t length, unsigned flags,
                                            int silent, uct_ib_mem_t *memh)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    ucs_status_t status;
    uint64_t exp_access;

    exp_access = uct_ib_md_access_flags(md, flags, length);
    status = uct_ib_md_reg_mr(md, address, length, exp_access, silent, &memh->mr);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("registered memory %p..%p on %s lkey 0x%x rkey 0x%x "
              "exp_access 0x%lx flags 0x%x", address, address + length,
              uct_ib_device_name(&md->dev), memh->mr->lkey, memh->mr->rkey,
              exp_access, flags);

    uct_ib_mem_init(memh, flags, exp_access);
    uct_ib_mem_set_numa_policy(md, memh);
    if (md->config.odp.prefetch) {
        uct_ib_mem_prefetch_internal(md, memh, memh->mr->addr, memh->mr->length);
    }

    UCS_STATS_UPDATE_COUNTER(md->stats, UCT_IB_MD_STAT_MEM_REG, +1);
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                   unsigned flags, uct_mem_h *memh_p)
{
    ucs_status_t status;
    uct_ib_mem_t *memh;

    memh = uct_ib_memh_alloc();
    if (memh == NULL) {
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

static ucs_status_t uct_ib_mem_dereg_internal(uct_ib_mem_t *memh)
{
    return uct_ib_memh_dereg(memh);
}

static ucs_status_t uct_ib_mem_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_ib_mem_t *ib_memh = memh;
    ucs_status_t status;

    status = uct_ib_mem_dereg_internal(ib_memh);
    uct_ib_memh_free(ib_memh);
    return status;
}

static ucs_status_t 
uct_ib_mem_advise(uct_md_h uct_md, uct_mem_h memh, void *addr, size_t length,
                  unsigned advice)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    ucs_debug("memh %p advice %d", memh, advice);
    if ((advice == UCT_MADV_WILLNEED) && !md->config.odp.prefetch) {
        return uct_ib_mem_prefetch_internal(md, memh, addr, length);
    }
    return UCS_OK;
}

static ucs_status_t uct_ib_mkey_pack(uct_md_h uct_md, uct_mem_h uct_memh,
                                     void *rkey_buffer)
{
    uct_ib_md_t *md         = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_mem_t *memh      = uct_memh;
    uint32_t atomic_rkey;
    uint16_t umr_offset;
    ucs_status_t status;

    /* create umr only if a user requested atomic access to the
     * memory region and the hardware supports it.
     */
    if ((memh->flags & UCT_IB_MEM_ACCESS_REMOTE_ATOMIC) &&
        !(memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) &&
        (memh != &md->global_odp))
    {
        /* create UMR on-demand */
        ucs_assert(memh->atomic_mr == NULL);
        umr_offset = uct_ib_md_atomic_offset(uct_ib_md_get_atomic_mr_id(md));
        status = UCS_PROFILE_CALL(uct_ib_md_post_umr, md, memh->mr,
                                  umr_offset, &memh->atomic_mr);
        if (status == UCS_OK) {
            memh->flags |= UCT_IB_MEM_FLAG_ATOMIC_MR;
            ucs_trace("created atomic key 0x%x for 0x%x", memh->atomic_mr->rkey,
                      memh->mr->lkey);
        } else if (status != UCS_ERR_UNSUPPORTED) {
            return status;
        }
    }
    if (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) {
        ucs_assert(memh->atomic_mr != NULL);
        atomic_rkey = memh->atomic_mr->rkey;
    } else {
        atomic_rkey = UCT_IB_INVALID_RKEY;
    }

    uct_ib_md_pack_rkey(memh->mr->rkey, atomic_rkey, rkey_buffer);
    return UCS_OK;
}

static ucs_status_t uct_ib_rkey_unpack(uct_md_component_t *mdc,
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

static void uct_ib_md_close(uct_md_h md);

static uct_md_ops_t uct_ib_md_ops = {
    .close             = uct_ib_md_close,
    .query             = uct_ib_md_query,
    .mem_alloc         = uct_ib_mem_alloc,
    .mem_free          = uct_ib_mem_free,
    .mem_reg           = uct_ib_mem_reg,
    .mem_dereg         = uct_ib_mem_dereg,
    .mem_advise        = uct_ib_mem_advise,
    .mkey_pack         = uct_ib_mkey_pack,
    .is_mem_type_owned = (void*)ucs_empty_function_return_zero,
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

static ucs_status_t uct_ib_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);
    uct_ib_rcache_region_t *region = uct_ib_rcache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t uct_ib_md_rcache_ops = {
    .close             = uct_ib_md_close,
    .query             = uct_ib_md_query,
    .mem_alloc         = uct_ib_mem_alloc,
    .mem_free          = uct_ib_mem_free,
    .mem_reg           = uct_ib_mem_rcache_reg,
    .mem_dereg         = uct_ib_mem_rcache_dereg,
    .mem_advise        = uct_ib_mem_advise,
    .mkey_pack         = uct_ib_mkey_pack,
    .is_mem_type_owned = (void*)ucs_empty_function_return_zero,
};

static ucs_status_t uct_ib_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                                             void *arg, ucs_rcache_region_t *rregion)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_md_t *md = context;
    int *flags = arg;
    ucs_status_t status;

    status = uct_ib_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                     region->super.super.end - region->super.super.start,
                                     *flags, 1, &region->memh);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}

static void uct_ib_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                       ucs_rcache_region_t *rregion)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);

    (void)uct_ib_mem_dereg_internal(&region->memh);
}

static void uct_ib_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion, char *buf,
                                         size_t max)
{
    uct_ib_rcache_region_t *region = ucs_derived_of(rregion, uct_ib_rcache_region_t);
    uct_ib_mem_t *memh = &region->memh;

    snprintf(buf, max, "lkey 0x%x rkey 0x%x atomic: lkey 0x%x rkey 0x%x",
             memh->mr->lkey, memh->mr->rkey,
             (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) ? memh->atomic_mr->lkey :
                             UCT_IB_INVALID_RKEY,
             (memh->flags & UCT_IB_MEM_FLAG_ATOMIC_MR) ? memh->atomic_mr->rkey :
                             UCT_IB_INVALID_RKEY
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
    md_attr->cap.reg_mem_types &= UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_global_odp_reg(uct_md_h uct_md, void *address,
                                              size_t length, unsigned flags,
                                              uct_mem_h *memh_p)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    ucs_assert(md->global_odp.mr != NULL);
    if (flags & UCT_MD_MEM_FLAG_LOCK) {
        return uct_ib_mem_reg(uct_md, address, length, flags, memh_p);
    }

    if (md->config.odp.prefetch) {
        uct_ib_mem_prefetch_internal(md, &md->global_odp, address, length);
    }
    *memh_p = &md->global_odp;
    return UCS_OK;
}

static ucs_status_t uct_ib_mem_global_odp_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    if (memh == &md->global_odp) {
        return UCS_OK;
    }

    return uct_ib_mem_dereg(uct_md, memh);
}

static uct_md_ops_t UCS_V_UNUSED uct_ib_md_global_odp_ops = {
    .close             = uct_ib_md_close,
    .query             = uct_ib_md_odp_query,
    .mem_alloc         = uct_ib_mem_alloc,
    .mem_free          = uct_ib_mem_free,
    .mem_reg           = uct_ib_mem_global_odp_reg,
    .mem_dereg         = uct_ib_mem_global_odp_dereg,
    .mem_advise        = uct_ib_mem_advise,
    .mkey_pack         = uct_ib_mkey_pack,
    .is_mem_type_owned = (void*)ucs_empty_function_return_zero,
};

static void uct_ib_make_md_name(char md_name[UCT_MD_NAME_MAX], struct ibv_device *device)
{
    snprintf(md_name, UCT_MD_NAME_MAX, "%s/", UCT_IB_MD_PREFIX);
    strncat(md_name, device->name, UCT_MD_NAME_MAX - strlen(device->name) - 1);
}

static ucs_status_t uct_ib_query_md_resources(uct_md_resource_desc_t **resources_p,
                                              unsigned *num_resources_p)
{
    uct_md_resource_desc_t *resources;
    struct ibv_device **device_list;
    ucs_status_t status;
    int i, num_devices;

    /* Get device list from driver */
    device_list = ibv_get_device_list(&num_devices);
    if (device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    resources = ucs_calloc(num_devices, sizeof(*resources), "ib resources");
    if (resources == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_device_list;
    }

    for (i = 0; i < num_devices; ++i) {
        uct_ib_make_md_name(resources[i].md_name, device_list[i]);
    }

    *resources_p     = resources;
    *num_resources_p = num_devices;
    status = UCS_OK;

out_free_device_list:
    ibv_free_device_list(device_list);
out:
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
uct_ib_md_parse_reg_methods(uct_ib_md_t *md, const uct_ib_md_config_t *md_config)
{
    ucs_rcache_params_t rcache_params;
    ucs_status_t status;
    int i;

    for (i = 0; i < md_config->reg_methods.count; ++i) {
        if (!strcasecmp(md_config->reg_methods.rmtd[i], "rcache")) {
            rcache_params.region_struct_size = sizeof(uct_ib_rcache_region_t);
            rcache_params.alignment          = md_config->rcache.alignment;
            rcache_params.max_alignment      = ucs_get_page_size();
            rcache_params.ucm_event_priority = md_config->rcache.event_prio;
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

            md->super.ops         = &uct_ib_md_rcache_ops;
            md->reg_cost.overhead = md_config->rcache.overhead;
            md->reg_cost.growth   = 0; /* It's close enough to 0 */
            ucs_debug("%s: using registration cache",
                      uct_ib_device_name(&md->dev));
            return UCS_OK;
#if HAVE_DECL_IBV_EXP_REG_MR && HAVE_DECL_IBV_EXP_ODP_SUPPORT_IMPLICIT
        } else if (!strcasecmp(md_config->reg_methods.rmtd[i], "odp")) {
            if (!uct_ib_device_odp_has_global_mr(&md->dev)) {
                ucs_debug("%s: on-demand-paging with global memory region is "
                          "not supported", uct_ib_device_name(&md->dev));
                continue;
            }

            struct ibv_exp_reg_mr_in in;
            memset(&in, 0, sizeof(in));
            in.pd             = md->pd;
            in.length         = IBV_EXP_IMPLICIT_MR_SIZE;
            in.exp_access     = UCT_IB_MEM_ACCESS_FLAGS | IBV_EXP_ACCESS_ON_DEMAND;
            md->global_odp.mr = UCS_PROFILE_CALL(ibv_exp_reg_mr, &in);
            if (md->global_odp.mr == NULL) {
                ucs_debug("%s: failed to register global mr: %m",
                          uct_ib_device_name(&md->dev));
                continue;
            }

            md->global_odp.lkey      = md->global_odp.mr->lkey;
            md->global_odp.flags     = UCT_IB_MEM_FLAG_ODP;
            md->global_odp.atomic_mr = NULL;
            md->super.ops            = &uct_ib_md_global_odp_ops;
            md->reg_cost.overhead    = 10e-9;
            md->reg_cost.growth      = 0;
            uct_ib_mem_init(&md->global_odp, 0, in.exp_access);
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
    uct_ib_device_spec_t *spec;
    ucs_status_t status;
    char *flags_str, *p;
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
                         &spec->vendor_id, &spec->part_id, &spec->name,
                         &flags_str, &spec->priority);
        if (nfields < 2) {
            ucs_error("failed to parse device config '%s' (parsed: %d/%d)",
                      md_config->custom_devices.spec[i], nfields, 5);
            status = UCS_ERR_INVALID_PARAM;
            goto err_free;
        }

        if (nfields >= 4) {
            for (p = flags_str; *p != 0; ++p) {
                if (*p == '4') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_MLX4_PRM;
                } else if (*p == '5') {
                    spec->flags |= UCT_IB_DEVICE_FLAG_MLX5_PRM;
                } else {
                    ucs_error("invalid device flag: '%c'", *p);
                    free(flags_str);
                    status = UCS_ERR_INVALID_PARAM;
                    goto err_free;
                }
            }
            free(flags_str);
        }

        ucs_trace("added device '%s' vendor_id 0x%x part_id %d flags %c%c prio %d",
                  spec->name, spec->vendor_id, spec->part_id,
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
    uct_ib_memh_dereg(&md->global_odp);
}

static ucs_status_t
uct_ib_md_open(const char *md_name, const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    const uct_ib_md_config_t *md_config = ucs_derived_of(uct_md_config, uct_ib_md_config_t);
    struct ibv_device **ib_device_list, *ib_device;
    char tmp_md_name[UCT_MD_NAME_MAX];
    ucs_status_t status;
    int i, num_devices, ret;
    uct_ib_md_t *md;

    /* Get device list from driver */
    ib_device_list = ibv_get_device_list(&num_devices);
    if (ib_device_list == NULL) {
        ucs_debug("Failed to get IB device list, assuming no devices are present");
        status = UCS_ERR_NO_DEVICE;
        goto out;
    }

    ib_device = NULL;
    for (i = 0; i < num_devices; ++i) {
        uct_ib_make_md_name(tmp_md_name, ib_device_list[i]);
        if (!strcmp(tmp_md_name, md_name)) {
            ib_device = ib_device_list[i];
            break;
        }
    }
    if (ib_device == NULL) {
        status = UCS_ERR_NO_DEVICE;
        goto out_free_dev_list;
    }

    md = ucs_calloc(1, sizeof(*md), "ib_md");
    if (md == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_free_dev_list;
    }

    md->super.component       = &uct_ib_mdc;
    md->config                = md_config->ext;

    /* Create statistics */
    status = UCS_STATS_NODE_ALLOC(&md->stats, &uct_ib_md_stats_class,
                                  ucs_stats_get_root(),
                                  "%s-%p", ibv_get_device_name(ib_device), md);
    if (status != UCS_OK) {
        goto err_free_md;
    }

    if (md_config->fork_init != UCS_NO) {
        ret = ibv_fork_init();
        if (ret) {
            if (md_config->fork_init == UCS_YES) {
                ucs_error("ibv_fork_init() failed: %m");
                status = UCS_ERR_IO_ERROR;
                goto err_release_stats;
            }
            ucs_debug("ibv_fork_init() failed: %m, continuing, but fork may be unsafe.");
            uct_ib_fork_warn_enable();
        }
    } else {
        uct_ib_fork_warn_enable();
    }

    status = uct_ib_device_init(&md->dev, ib_device, md_config->async_events
                                UCS_STATS_ARG(md->stats));
    if (status != UCS_OK) {
        goto err_release_stats;
    }

    /* Disable contig pages allocator for IB transport objects */
    if (!md->config.enable_contig_pages) {
        ibv_exp_setenv(md->dev.ibv_context, "MLX_QP_ALLOC_TYPE", "ANON", 0);
        ibv_exp_setenv(md->dev.ibv_context, "MLX_CQ_ALLOC_TYPE", "ANON", 0);
    }

    if (md->config.odp.max_size == UCS_CONFIG_MEMUNITS_AUTO) {
        /* Must be done after we open and query the device */
        md->config.odp.max_size = uct_ib_device_odp_max_size(&md->dev);
    }

    /* Allocate memory domain */
    md->pd = ibv_alloc_pd(md->dev.ibv_context);
    if (md->pd == NULL) {
        ucs_error("ibv_alloc_pd() failed: %m");
        status = UCS_ERR_NO_MEMORY;
        goto err_cleanup_device;
    }

    status = uct_ib_md_umr_qp_create(md);
    if (status == UCS_ERR_UNSUPPORTED) {
        md->umr_qp = NULL;
        md->umr_cq = NULL;
    } else if (status != UCS_OK) {
        goto err_dealloc_pd;
    }

    status = uct_ib_md_parse_reg_methods(md, md_config);
    if (status != UCS_OK) {
        goto err_destroy_umr_qp;
    }

    status = uct_ib_md_parse_device_config(md, md_config);
    if (status != UCS_OK) {
        goto err_release_reg_method;
    }

    *md_p = &md->super;
    status = UCS_OK;

out_free_dev_list:
    ibv_free_device_list(ib_device_list);
out:
    return status;

err_release_reg_method:
    uct_ib_md_release_reg_method(md);
err_destroy_umr_qp:
    uct_ib_md_umr_qp_destroy(md);
err_dealloc_pd:
    ibv_dealloc_pd(md->pd);
err_cleanup_device:
    uct_ib_device_cleanup(&md->dev);
err_release_stats:
    UCS_STATS_NODE_FREE(md->stats);
err_free_md:
    ucs_free(md);
    goto out_free_dev_list;
}

static void uct_ib_md_close(uct_md_h uct_md)
{
    uct_ib_md_t *md = ucs_derived_of(uct_md, uct_ib_md_t);

    uct_ib_md_release_device_config(md);
    uct_ib_md_release_reg_method(md);
    uct_ib_md_umr_qp_destroy(md);
    ibv_dealloc_pd(md->pd);
    uct_ib_device_cleanup(&md->dev);
    UCS_STATS_NODE_FREE(md->stats);
    ucs_free(md);
}

UCT_MD_COMPONENT_DEFINE(uct_ib_mdc, UCT_IB_MD_PREFIX,
                        uct_ib_query_md_resources, uct_ib_md_open, NULL,
                        uct_ib_rkey_unpack,
                        (void*)ucs_empty_function_return_success /* release */,
                        "IB_", uct_ib_md_config_table, uct_ib_md_config_t);
