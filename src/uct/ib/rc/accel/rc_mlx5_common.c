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
    struct ibv_srq *srq;
    struct mlx5_srq *msrq;
    int i;

    if (!UCT_RC_IFACE_TM_ENABLED(rc_iface)) {
        return UCS_OK;
    }

    status = uct_rc_iface_tag_init(rc_iface, rc_config, srq_init_attr,
                                   rndv_hdr_len, 0);
    if (status != UCS_OK) {
        goto err;
    }

    srq = rc_iface->rx.srq.srq;
    if (srq->handle == LEGACY_XRC_SRQ_HANDLE) {
        srq = (struct ibv_srq *)(((struct ibv_srq_legacy *)srq)->ibv_srq);
    }

    msrq = ucs_container_of(srq, struct mlx5_srq, vsrq.srq);
    if (msrq->counter != 0) {
        ucs_error("SRQ counter is not 0 (%d)", msrq->counter);
        status = UCS_ERR_NO_DEVICE;
        goto err_tag_cleanup;
    }

    status = uct_ib_mlx5_txwq_init(rc_iface->super.super.worker,
                                   &iface->tm.cmd_wq.super,
                                   &msrq->cmd_qp->verbs_qp.qp);
    if (status != UCS_OK) {
        goto err_tag_cleanup;
    }

    iface->tm.cmd_wq.qp_num   = msrq->cmd_qp->verbs_qp.qp.qp_num;
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

ucs_status_t uct_rc_mlx5_iface_common_init(uct_rc_mlx5_iface_common_t *iface,
                                           uct_rc_iface_t *rc_iface,
                                           uct_rc_iface_config_t *config)
{
    ucs_status_t status;

    status = uct_ib_mlx5_get_cq(rc_iface->super.send_cq, &iface->tx.cq);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(rc_iface->super.recv_cq, &iface->rx.cq);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_srq_init(&iface->rx.srq, rc_iface->rx.srq.srq,
                                  rc_iface->super.config.seg_size);
    if (status != UCS_OK) {
        return status;
    }

    rc_iface->rx.srq.quota = iface->rx.srq.mask + 1;

    /* By default set to something that is always in cache */
    iface->rx.pref_ptr = iface;

    status = UCS_STATS_NODE_ALLOC(&iface->stats, &uct_rc_mlx5_iface_stats_class,
                                  rc_iface->stats);
    if (status != UCS_OK) {
        return status;
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

    return status;
}

void uct_rc_mlx5_iface_common_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
    UCS_STATS_NODE_FREE(iface->stats);
    ucs_mpool_cleanup(&iface->tx.atomic_desc_mp, 1);
}

void uct_rc_mlx5_iface_common_query(uct_iface_attr_t *iface_attr)
{
    /* Atomics */
    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM;

    /* Software overhead */
    iface_attr->overhead          = 40e-9;

}

void uct_rc_mlx5_iface_common_update_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                            uct_ib_iface_t *ib_iface)
{
    uct_ib_mlx5_update_cq_ci(ib_iface->send_cq, iface->tx.cq.cq_ci);
    uct_ib_mlx5_update_cq_ci(ib_iface->recv_cq, iface->rx.cq.cq_ci);
}

void uct_rc_mlx5_iface_common_sync_cqs_ci(uct_rc_mlx5_iface_common_t *iface,
                                          uct_ib_iface_t *ib_iface)
{
    iface->tx.cq.cq_ci = uct_ib_mlx5_get_cq_ci(ib_iface->send_cq);
    iface->rx.cq.cq_ci = uct_ib_mlx5_get_cq_ci(ib_iface->recv_cq);
}

void uct_rc_mlx5_iface_commom_clean_srq(uct_rc_mlx5_iface_common_t *mlx5_common_iface,
                                        uct_rc_iface_t *rc_iface, uint32_t qpn)
{
    uct_ib_mlx5_cq_t *mlx5_cq = &mlx5_common_iface->rx.cq;
    const size_t cqe_sz       = 1ul << mlx5_cq->cqe_size_log;
    struct mlx5_cqe64 *cqe, *dest;
    uct_ib_mlx5_srq_seg_t *seg;
    unsigned ci, pi, idx;
    uint8_t owner_bit;
    int nfreed;

    pi = ci = mlx5_cq->cq_ci;
    while (uct_ib_mlx5_poll_cq(&rc_iface->super, mlx5_cq)) {
        if (pi == ci + mlx5_cq->cq_length - 1) {
            break;
        }
        ++pi;
    }
    ucs_assert(pi == mlx5_cq->cq_ci);

    ucs_memory_cpu_load_fence();

    /* Remove CQEs of the destroyed QP, so the drive would not see them and try
     * to remove them itself, creating a mess with the free-list.
     */
    nfreed = 0;
    while ((int)--pi - (int)ci >= 0) {
        cqe = uct_ib_mlx5_get_cqe(mlx5_cq, pi);
        if ((ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER)) == qpn) {
            idx = ntohs(cqe->wqe_counter);
            seg = uct_ib_mlx5_srq_get_wqe(&mlx5_common_iface->rx.srq, idx);
            seg->srq.free = 1;
            ucs_trace("iface %p: freed srq seg[%d] of qpn 0x%x",
                      mlx5_common_iface, idx, qpn);
            ++nfreed;
        } else if (nfreed) {
            /* push the CQEs we want to keep to cq_ci, and move cq_ci backwards */
            dest = uct_ib_mlx5_get_cqe(mlx5_cq, mlx5_cq->cq_ci);
            owner_bit = dest->op_own & MLX5_CQE_OWNER_MASK;
            memcpy((void*)(dest + 1) - cqe_sz, (void*)(cqe + 1) - cqe_sz, cqe_sz);
            dest->op_own = (dest->op_own & ~MLX5_CQE_OWNER_MASK) | owner_bit;
            --mlx5_cq->cq_ci;
        }
    }

    rc_iface->rx.srq.available += nfreed;
}
