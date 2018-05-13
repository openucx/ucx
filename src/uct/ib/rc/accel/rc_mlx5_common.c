/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include "rc_mlx5_common.h"

#include <uct/api/uct.h>
#include <uct/ib/rc/base/rc_iface.h>


#if ENABLE_STATS
ucs_stats_class_t uct_rc_mlx5_iface_stats_class = {
    .name = "mlx5",
    .num_counters = UCT_RC_MLX5_IFACE_STAT_LAST,
    .counter_names = {
     [UCT_RC_MLX5_IFACE_STAT_RX_INL_32] = "rx_inl_32",
     [UCT_RC_MLX5_IFACE_STAT_RX_INL_64] = "rx_inl_64"
    }
};
#endif


ucs_config_field_t uct_mlx5_common_config_table[] = {
#if HAVE_IBV_EXP_DM
    {"DM_SIZE", "2k",
     "Device Memory segment size (0 - disabled)",
     ucs_offsetof(uct_common_mlx5_iface_config_t, dm.seg_len), UCS_CONFIG_TYPE_MEMUNITS},
    {"DM_COUNT", "1",
     "Device Memory segments count (0 - disabled)",
     ucs_offsetof(uct_common_mlx5_iface_config_t, dm.count), UCS_CONFIG_TYPE_UINT},
#endif
    {NULL}
};

#if HAVE_IBV_EXP_DM
/* uct_mlx5_dm_va is used to get pointer to DM mapped into process address space */
typedef struct uct_mlx5_dm_va {
    struct ibv_exp_dm  ibv_dm;
    size_t             length;
    uint64_t           *start_va;
} uct_mlx5_dm_va_t;
#endif


void uct_rc_mlx5_common_packet_dump(uct_base_iface_t *iface, uct_am_trace_type_t type,
                                    void *data, size_t length, size_t valid_length,
                                    char *buffer, size_t max)
{
    uct_rc_ep_packet_dump(iface, type, data, length, valid_length, buffer, max, 0);
}

unsigned uct_rc_mlx5_iface_srq_post_recv(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq)
{
    uct_ib_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    uint16_t count, index, next_index;
    uct_rc_hdr_t *hdr;

    /* Make sure the union is right */
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_mlx5_srq_seg_t, mlx5_srq.next_wqe_index) ==
                      ucs_offsetof(uct_ib_mlx5_srq_seg_t, srq.next_wqe_index));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_ib_mlx5_srq_seg_t, dptr) ==
                      sizeof(struct mlx5_wqe_srq_next_seg));

    ucs_assert(UCS_CIRCULAR_COMPARE16(srq->ready_idx, <=, srq->free_idx));

    index = srq->ready_idx;
    for (;;) {
        next_index = index + 1;
        seg = uct_ib_mlx5_srq_get_wqe(srq, next_index & srq->mask);
        if (UCS_CIRCULAR_COMPARE16(next_index, >, srq->free_idx)) {
            if (!seg->srq.free) {
                break;
            }

            ucs_assert(next_index == (uint16_t)(srq->free_idx + 1));
            seg->srq.free  = 0;
            srq->free_idx  = next_index;
        }

        if (seg->srq.desc == NULL) {
            UCT_TL_IFACE_GET_RX_DESC(&iface->super.super, &iface->rx.mp,
                                     desc, break);

            /* Set receive data segment pointer. Length is pre-initialized. */
            hdr            = uct_ib_iface_recv_desc_hdr(&iface->super, desc);
            seg->srq.desc  = desc;
            seg->dptr.lkey = htonl(desc->lkey);
            seg->dptr.addr = htobe64((uintptr_t)hdr);
            VALGRIND_MAKE_MEM_NOACCESS(hdr, iface->super.config.seg_size);
        }

        index = next_index;
    }

    count = index - srq->sw_pi;
    ucs_assert(iface->rx.srq.available >= count);

    if (count > 0) {
        srq->ready_idx           = index;
        srq->sw_pi               = index;
        iface->rx.srq.available -= count;
        ucs_memory_cpu_store_fence();
        *srq->db = htonl(srq->sw_pi);
        ucs_assert(uct_ib_mlx5_srq_get_wqe(srq, srq->mask)->srq.next_wqe_index == 0);
    }
    return count;
}

void uct_rc_mlx5_iface_common_prepost_recvs(uct_rc_iface_t *iface,
                                            uct_rc_mlx5_iface_common_t *mlx5_common)
{
    iface->rx.srq.available = iface->rx.srq.quota;
    iface->rx.srq.quota     = 0;
    uct_rc_mlx5_iface_srq_post_recv(iface, &mlx5_common->rx.srq);
}

#define UCT_RC_MLX5_DEFINE_ATOMIC_LE_HANDLER(_bits) \
    static void \
    uct_rc_mlx5_common_atomic##_bits##_le_handler(uct_rc_iface_send_op_t *op, \
                                                  const void *resp) \
    { \
        uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t); \
        uint##_bits##_t *dest        = desc->super.buffer; \
        const uint##_bits##_t *value = resp; \
        \
        VALGRIND_MAKE_MEM_DEFINED(value, sizeof(*value)); \
        if (resp == (desc + 1)) { \
            *dest = *value; /* response in desc buffer */ \
        } else if (_bits == 32) { \
            *dest = ntohl(*value);  /* response in CQE as 32-bit value */ \
        } else if (_bits == 64) { \
            *dest = be64toh(*value); /* response in CQE as 64-bit value */ \
        } \
        \
        uct_invoke_completion(desc->super.user_comp, UCS_OK); \
        ucs_mpool_put(desc); \
    }

UCT_RC_MLX5_DEFINE_ATOMIC_LE_HANDLER(32)
UCT_RC_MLX5_DEFINE_ATOMIC_LE_HANDLER(64)

ucs_status_t
uct_rc_mlx5_iface_common_tag_init(uct_rc_mlx5_iface_common_t *iface,
                                  uct_rc_iface_t *rc_iface,
                                  uct_rc_iface_config_t *rc_config,
                                  struct ibv_exp_create_srq_attr *srq_init_attr,
                                  unsigned rndv_hdr_len)
{
    ucs_status_t status = UCS_OK;
#if IBV_EXP_HW_TM
    struct ibv_qp *cmd_qp;
    int i;

    if (!UCT_RC_IFACE_TM_ENABLED(rc_iface)) {
        return UCS_OK;
    }

    status = uct_rc_iface_tag_init(rc_iface, rc_config, srq_init_attr,
                                   rndv_hdr_len, 0);
    if (status != UCS_OK) {
        goto err;
    }

    cmd_qp = uct_dv_get_cmd_qp(rc_iface->rx.srq.srq);
    if (!cmd_qp) {
        status = UCS_ERR_NO_DEVICE;
        goto err_tag_cleanup;
    }

    status = uct_ib_mlx5_txwq_init(rc_iface->super.super.worker,
                                   &iface->tm.cmd_wq.super, cmd_qp);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    iface->tm.cmd_wq.qp_num   = cmd_qp->qp_num;
    iface->tm.cmd_wq.ops_mask = rc_iface->tm.cmd_qp_len - 1;
    iface->tm.cmd_wq.ops_head = iface->tm.cmd_wq.ops_tail = 0;
    iface->tm.cmd_wq.ops      = ucs_calloc(rc_iface->tm.cmd_qp_len,
                                           sizeof(uct_rc_mlx5_srq_op_t),
                                           "srq tag ops");
    if (iface->tm.cmd_wq.ops == NULL) {
        ucs_error("Failed to allocate memory for srq tm ops array");
        status = UCS_ERR_NO_MEMORY;
        goto err_tag_cleanup;
    }

    iface->tm.list = ucs_calloc(rc_iface->tm.num_tags + 1,
                                sizeof(uct_rc_mlx5_tag_entry_t), "tm list");
    if (iface->tm.list == NULL) {
        ucs_error("Failed to allocate memory for tag matching list");
        status = UCS_ERR_NO_MEMORY;
        goto err_cmd_wq_free;
    }

    for (i = 0; i < rc_iface->tm.num_tags; ++i) {
        iface->tm.list[i].next = &iface->tm.list[i + 1];
    }

    iface->tm.head = &iface->tm.list[0];
    iface->tm.tail = &iface->tm.list[i];

    return UCS_OK;

err_cmd_wq_free:
    ucs_free(iface->tm.cmd_wq.ops);
err_tag_cleanup:
    uct_rc_iface_tag_cleanup(rc_iface);
err:
#endif

    return status;
}

void uct_rc_mlx5_iface_common_tag_cleanup(uct_rc_mlx5_iface_common_t *iface,
                                          uct_rc_iface_t *rc_iface)
{
#if IBV_EXP_HW_TM
    if (UCT_RC_IFACE_TM_ENABLED(rc_iface)) {
        uct_ib_mlx5_txwq_cleanup(&iface->tm.cmd_wq.super);
        ucs_free(iface->tm.list);
        ucs_free(iface->tm.cmd_wq.ops);
        uct_rc_iface_tag_cleanup(rc_iface);
    }
#endif
}


#if HAVE_IBV_EXP_DM
static ucs_status_t
uct_rc_mlx5_iface_common_dm_mpool_chunk_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucs_status_t status;

    status = ucs_mpool_chunk_malloc(mp, size_p, chunk_p);
    if (status == UCS_OK) {
        memset(*chunk_p, 0, *size_p);
    }

    return status;
}

static void uct_rc_mlx5_iface_common_dm_mp_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_mlx5_dm_data_t *dm         = ucs_container_of(mp, uct_mlx5_dm_data_t, mp);
    uct_rc_iface_send_desc_t* desc = (uct_rc_iface_send_desc_t*)obj;

    ucs_assert(desc->super.buffer == NULL);
    ucs_assert(dm->seg_attached < dm->seg_count);

    desc->lkey          = dm->mr->lkey;
    desc->super.buffer  = UCS_PTR_BYTE_OFFSET(dm->start_va, dm->seg_attached * dm->seg_len);
    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    dm->seg_attached++;
}

static ucs_mpool_ops_t uct_dm_iface_mpool_ops = {
    .chunk_alloc   = uct_rc_mlx5_iface_common_dm_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_rc_mlx5_iface_common_dm_mp_obj_init,
    .obj_cleanup   = NULL
};


static int uct_rc_mlx5_iface_common_dm_device_cmp(uct_mlx5_dm_data_t *dm_data,
                                                  uct_rc_iface_t *iface,
                                                  const uct_common_mlx5_iface_config_t *config)
{
    uct_ib_device_t *dev = uct_ib_iface_device(&iface->super);

    return dm_data->device->ibv_context == dev->ibv_context;
}

static ucs_status_t
uct_rc_mlx5_iface_common_dm_tl_init(uct_mlx5_dm_data_t *data,
                                    uct_rc_iface_t *iface,
                                    const uct_common_mlx5_iface_config_t *config)
{
    ucs_status_t status;
    struct ibv_exp_alloc_dm_attr dm_attr;
    struct ibv_exp_reg_mr_in mr_in;

    data->seg_len      = ucs_min(ucs_align_up(config->dm.seg_len,
                                              sizeof(uct_rc_mlx5_dm_copy_data_t)),
                                 iface->super.config.seg_size);
    data->seg_count    = config->dm.count;
    data->seg_attached = 0;
    data->device       = uct_ib_iface_device(&iface->super);

    dm_attr.length     = data->seg_len * data->seg_count;
    dm_attr.comp_mask  = 0;
    data->dm           = ibv_exp_alloc_dm(data->device->ibv_context, &dm_attr);

    if (data->dm == NULL) {
        /* TODO: prompt warning? */
        ucs_debug("ibv_exp_alloc_dm(dev=%s length=%zu) failed: %m",
                  uct_ib_device_name(data->device), dm_attr.length);
        return UCS_ERR_NO_RESOURCE;
    }

    memset(&mr_in, 0, sizeof(mr_in));
    mr_in.pd           = uct_ib_iface_md(&iface->super)->pd;
    mr_in.comp_mask    = IBV_EXP_REG_MR_DM;
    mr_in.dm           = data->dm;
    mr_in.length       = dm_attr.length;
    data->mr           = ibv_exp_reg_mr(&mr_in);
    if (data->mr == NULL) {
        ucs_warn("ibv_exp_reg_mr() error - On Device Memory registration failed, %d %m", errno);
        status = UCS_ERR_NO_RESOURCE;
        goto failed_mr;
    }

    data->start_va = ((uct_mlx5_dm_va_t*)data->dm)->start_va;

    status = ucs_mpool_init(&data->mp, 0,
                            sizeof(uct_rc_iface_send_desc_t), 0, UCS_SYS_CACHE_LINE_SIZE,
                            data->seg_count, data->seg_count,
                            &uct_dm_iface_mpool_ops, "mlx5_dm_desc");
    if (status != UCS_OK) {
        goto failed_mpool;
    }

    /* DM initialization may fail due to any reason, just
     * free resources & continue without DM */
    return UCS_OK;

failed_mpool:
    ibv_dereg_mr(data->mr);
failed_mr:
    ibv_exp_free_dm(data->dm);
    data->dm = NULL;
    return status;
}

static void uct_rc_mlx5_iface_common_dm_tl_cleanup(uct_mlx5_dm_data_t *data)
{
    ucs_assert(data->dm != NULL);
    ucs_assert(data->mr != NULL);

    ucs_mpool_cleanup(&data->mp, 1);
    ibv_dereg_mr(data->mr);
    ibv_exp_free_dm(data->dm);
}
#endif

static ucs_status_t
uct_rc_mlx5_iface_common_dm_init(uct_rc_mlx5_iface_common_t *iface,
                                 uct_rc_iface_t *rc_iface,
                                 const uct_common_mlx5_iface_config_t *config)
{
#if HAVE_IBV_EXP_DM
    if ((config->dm.seg_len * config->dm.count) == 0) {
        goto fallback;
    }

    iface->dm.dm = uct_worker_tl_data_get(rc_iface->super.super.worker,
                                          UCT_IB_MLX5_WORKER_DM_KEY,
                                          uct_mlx5_dm_data_t,
                                          uct_rc_mlx5_iface_common_dm_device_cmp,
                                          uct_rc_mlx5_iface_common_dm_tl_init,
                                          rc_iface, config);
    if (UCS_PTR_IS_ERR(iface->dm.dm)) {
        goto fallback;
    }

    ucs_assert(iface->dm.dm->dm != NULL);
    iface->dm.seg_len = iface->dm.dm->seg_len;
    return UCS_OK;

fallback:
    iface->dm.dm = NULL;
#endif
    return UCS_OK;
}

static void uct_rc_mlx5_iface_common_dm_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
#if HAVE_IBV_EXP_DM
    if (iface->dm.dm) {
        uct_worker_tl_data_put(iface->dm.dm, uct_rc_mlx5_iface_common_dm_tl_cleanup);
    }
#endif
}

ucs_status_t uct_rc_mlx5_iface_common_init(uct_rc_mlx5_iface_common_t *iface,
                                           uct_rc_iface_t *rc_iface,
                                           uct_rc_iface_config_t *config,
                                           uct_common_mlx5_iface_config_t *common_config)
{
    ucs_status_t status;

    status = uct_ib_mlx5_get_cq(rc_iface->super.cq[UCT_IB_DIR_TX], &iface->cq[UCT_IB_DIR_TX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(rc_iface->super.cq[UCT_IB_DIR_RX], &iface->cq[UCT_IB_DIR_RX]);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_srq_init(&iface->rx.srq, rc_iface->rx.srq.srq,
                                  rc_iface->super.config.seg_size);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_rc_mlx5_iface_common_dm_init(iface, rc_iface, common_config);
    if (status != UCS_OK) {
        return status;
    }

    rc_iface->rx.srq.quota = iface->rx.srq.mask + 1;

    /* By default set to something that is always in cache */
    iface->rx.pref_ptr = iface;

    status = UCS_STATS_NODE_ALLOC(&iface->stats, &uct_rc_mlx5_iface_stats_class,
                                  rc_iface->stats);
    if (status != UCS_OK) {
        goto cleanup_dm;
    }

    status = uct_iface_mpool_init(&rc_iface->super.super,
                                  &iface->tx.atomic_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_RC_MAX_ATOMIC_SIZE,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_RC_MAX_ATOMIC_SIZE,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.tx.mp,
                                  rc_iface->config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_mlx5_atomic_desc");
    if (status != UCS_OK) {
        UCS_STATS_NODE_FREE(iface->stats);
        goto cleanup_dm;
    }

    /* For little-endian atomic reply, override the default functions, to still
     * treat the response as big-endian when it arrives in the CQE.
     */
    if (!uct_ib_atomic_is_be_reply(uct_ib_iface_device(&rc_iface->super), 0, sizeof(uint64_t))) {
        rc_iface->config.atomic64_handler    = uct_rc_mlx5_common_atomic64_le_handler;
    }
    if (!uct_ib_atomic_is_be_reply(uct_ib_iface_device(&rc_iface->super), 1, sizeof(uint32_t))) {
       rc_iface->config.atomic32_ext_handler = uct_rc_mlx5_common_atomic32_le_handler;
    }
    if (!uct_ib_atomic_is_be_reply(uct_ib_iface_device(&rc_iface->super), 1, sizeof(uint64_t))) {
       rc_iface->config.atomic64_ext_handler = uct_rc_mlx5_common_atomic64_le_handler;
    }

    return UCS_OK;

cleanup_dm:
    uct_rc_mlx5_iface_common_dm_cleanup(iface);
    return status;
}

void uct_rc_mlx5_iface_common_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
    UCS_STATS_NODE_FREE(iface->stats);
    ucs_mpool_cleanup(&iface->tx.atomic_desc_mp, 1);
    uct_rc_mlx5_iface_common_dm_cleanup(iface);
}

void uct_rc_mlx5_iface_common_query(uct_ib_iface_t *iface, uct_iface_attr_t *iface_attr)
{
    uct_ib_device_t *dev = uct_ib_iface_device(iface);

    /* Atomics */
    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM;

    if (uct_ib_atomic_is_supported(dev, 0, sizeof(uint64_t))) {
        iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD);
        iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                                              UCS_BIT(UCT_ATOMIC_OP_CSWAP);

        iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;
    }

    if (uct_ib_atomic_is_supported(dev, 1, sizeof(uint64_t))) {
        iface_attr->cap.atomic64.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_AND)  |
                                              UCS_BIT(UCT_ATOMIC_OP_OR)   |
                                              UCS_BIT(UCT_ATOMIC_OP_XOR);
        iface_attr->cap.atomic64.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_AND)  |
                                              UCS_BIT(UCT_ATOMIC_OP_OR)   |
                                              UCS_BIT(UCT_ATOMIC_OP_XOR)  |
                                              UCS_BIT(UCT_ATOMIC_OP_SWAP);
        iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;
    }

    if (uct_ib_atomic_is_supported(dev, 1, sizeof(uint32_t))) {
        iface_attr->cap.atomic32.op_flags  |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                                              UCS_BIT(UCT_ATOMIC_OP_AND)  |
                                              UCS_BIT(UCT_ATOMIC_OP_OR)   |
                                              UCS_BIT(UCT_ATOMIC_OP_XOR);
        iface_attr->cap.atomic32.fop_flags |= UCS_BIT(UCT_ATOMIC_OP_ADD)  |
                                              UCS_BIT(UCT_ATOMIC_OP_AND)  |
                                              UCS_BIT(UCT_ATOMIC_OP_OR)   |
                                              UCS_BIT(UCT_ATOMIC_OP_XOR)  |
                                              UCS_BIT(UCT_ATOMIC_OP_SWAP) |
                                              UCS_BIT(UCT_ATOMIC_OP_CSWAP);
        iface_attr->cap.flags              |= UCT_IFACE_FLAG_ATOMIC_DEVICE;
    }

    /* Software overhead */
    iface_attr->overhead          = 40e-9;

}

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface)
{
#if !HAVE_DECL_MLX5DV_INIT_OBJ
    uct_ib_mlx5_update_cq_ci(ib_iface->cq[UCT_IB_DIR_TX], iface->cq[UCT_IB_DIR_TX].cq_ci);
    uct_ib_mlx5_update_cq_ci(ib_iface->cq[UCT_IB_DIR_RX], iface->cq[UCT_IB_DIR_RX].cq_ci);
#endif
}

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface)
{
#if !HAVE_DECL_MLX5DV_INIT_OBJ
    iface->cq[UCT_IB_DIR_TX].cq_ci = uct_ib_mlx5_get_cq_ci(ib_iface->cq[UCT_IB_DIR_TX]);
    iface->cq[UCT_IB_DIR_RX].cq_ci = uct_ib_mlx5_get_cq_ci(ib_iface->cq[UCT_IB_DIR_RX]);
#endif
}

int uct_rc_mlx5_iface_commom_clean(uct_ib_mlx5_cq_t *mlx5_cq,
                                   uct_ib_mlx5_srq_t *srq, uint32_t qpn)
{
    const size_t cqe_sz       = 1ul << mlx5_cq->cqe_size_log;
    struct mlx5_cqe64 *cqe, *dest;
    uct_ib_mlx5_srq_seg_t *seg;
    unsigned pi, idx;
    uint8_t owner_bit;
    int nfreed;

    pi = mlx5_cq->cq_ci;
    for (;;) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, pi);
        if (((cqe->op_own & MLX5_CQE_OWNER_MASK) == !(pi & mlx5_cq->cq_length)) ||
            ((cqe->op_own >> 4) == MLX5_CQE_INVALID)) {
            break;
        }

        ++pi;
        if (pi == (mlx5_cq->cq_ci + mlx5_cq->cq_length - 1)) {
            break;
        }
    }

    ucs_memory_cpu_load_fence();

    /* Remove CQEs of the destroyed QP, so the driver would not see them and try
     * to remove them itself, creating a mess with the free-list.
     */
    nfreed = 0;
    while ((int)--pi - (int)mlx5_cq->cq_ci >= 0) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, pi);
        if ((ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER)) == qpn) {
            idx = ntohs(cqe->wqe_counter);
            if (srq) {
                seg = uct_ib_mlx5_srq_get_wqe(srq, idx);
                seg->srq.free = 1;
                ucs_trace("cq %p: freed srq seg[%d] of qpn 0x%x",
                          mlx5_cq, idx, qpn);
            }
            ++nfreed;
        } else if (nfreed) {
            dest = uct_ib_mlx5_get_cqe(mlx5_cq, pi + nfreed);
            owner_bit = dest->op_own & MLX5_CQE_OWNER_MASK;
            memcpy((void*)(dest + 1) - cqe_sz, (void*)(cqe + 1) - cqe_sz, cqe_sz);
            dest->op_own = (dest->op_own & ~MLX5_CQE_OWNER_MASK) | owner_bit;
        }
    }

    mlx5_cq->cq_ci             += nfreed;

    return nfreed;
}
