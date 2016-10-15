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
            if (!seg->srq.ooo) {
                break;
            }

            ucs_assert(next_index == (uint16_t)(srq->free_idx + 1));
            seg->srq.ooo   = 0;
            srq->free_idx  = next_index;
        }

        if (seg->srq.desc == NULL) {
            UCT_TL_IFACE_GET_RX_DESC(&iface->super.super, &iface->rx.mp,
                                     desc, break);

            /* Set receive data segment pointer. Length is pre-initialized. */
            hdr            = uct_ib_iface_recv_desc_hdr(&iface->super, desc);
            seg->srq.desc  = desc;
            seg->dptr.lkey = htonl(desc->lkey);
            seg->dptr.addr = htonll((uintptr_t)hdr);
            VALGRIND_MAKE_MEM_NOACCESS(hdr, iface->super.config.seg_size);
        }

        index = next_index;
    }

    count = index - srq->sw_pi;
    if (count > 0) {
        srq->ready_idx        = index;
        srq->sw_pi            = index;
        iface->rx.available  -= count;
        ucs_memory_cpu_store_fence();
        *srq->db = htonl(srq->sw_pi);
        ucs_assert(uct_ib_mlx5_srq_get_wqe(srq, srq->mask)->srq.next_wqe_index == 0);
    }
    return count;
}

ucs_status_t uct_rc_mlx5_iface_srq_init(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq)
{
    uct_ib_mlx5_srq_info_t srq_info;
    uct_ib_mlx5_srq_seg_t *seg;
    ucs_status_t status;
    unsigned i;

    status = uct_ib_mlx5_get_srq_info(iface->rx.srq, &srq_info);
    if (status != UCS_OK) {
        return status;
    }

    if (srq_info.head != 0) {
        ucs_error("SRQ head is not 0 (%d)", srq_info.head);
        return UCS_ERR_NO_DEVICE;
    }

    if (srq_info.stride != UCT_IB_MLX5_SRQ_STRIDE) {
        ucs_error("SRQ stride is not %lu (%d)", UCT_IB_MLX5_SRQ_STRIDE,
                  srq_info.stride);
        return UCS_ERR_NO_DEVICE;
    }

    if (!ucs_is_pow2(srq_info.tail + 1)) {
        ucs_error("SRQ length is not power of 2 (%d)", srq_info.tail + 1);
        return UCS_ERR_NO_DEVICE;
    }

    iface->rx.available = srq_info.tail + 1;
    srq->buf             = srq_info.buf;
    srq->db              = srq_info.dbrec;
    srq->free_idx        = srq_info.tail;
    srq->ready_idx       = -1;
    srq->sw_pi           = -1;
    srq->mask            = srq_info.tail;
    srq->tail            = srq_info.tail;

    for (i = srq_info.head; i <= srq_info.tail; ++i) {
        seg = uct_ib_mlx5_srq_get_wqe(srq, i);
        seg->srq.ooo         = 0;
        seg->srq.desc        = NULL;
        seg->dptr.byte_count = htonl(iface->super.config.seg_size);
    }

    return UCS_OK;
}

void uct_rc_mlx5_iface_srq_cleanup(uct_rc_iface_t *iface, uct_ib_mlx5_srq_t *srq)
{
    uct_ib_mlx5_srq_info_t srq_info;
    uct_ib_mlx5_srq_seg_t *seg;
    ucs_status_t status;
    unsigned index, next;

    status = uct_ib_mlx5_get_srq_info(iface->rx.srq, &srq_info);
    ucs_assert_always(status == UCS_OK);

    /* Restore order of all segments which the driver has put on its free list */
    index = srq->tail;
    while (index != srq_info.tail) {
        seg = uct_ib_mlx5_srq_get_wqe(srq, index);
        next = ntohs(seg->srq.next_wqe_index);
        seg->srq.next_wqe_index = htons((index + 1) & srq->mask);
        index = next;
    }
    srq->tail = index;
}

ucs_status_t uct_rc_mlx5_iface_common_init(uct_rc_mlx5_iface_common_t *iface, uct_rc_iface_t *rc_iface,
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

    status = uct_rc_mlx5_iface_srq_init(rc_iface, &iface->rx.srq);
    if (status != UCS_OK) {
        return status;
    }

    if (uct_rc_mlx5_iface_srq_post_recv(rc_iface, &iface->rx.srq) == 0) {
        ucs_error("Failed to post receives");
        return UCS_ERR_NO_MEMORY;
    }

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
    return status;
}

void uct_rc_mlx5_iface_common_cleanup(uct_rc_mlx5_iface_common_t *iface)
{
    UCS_STATS_NODE_FREE(iface->stats);
    ucs_mpool_cleanup(&iface->tx.atomic_desc_mp, 1);
}

void uct_rc_mlx5_iface_common_query(uct_rc_iface_t *iface,
                                    uct_iface_attr_t *iface_attr, size_t av_size)
{
    /* PUT */
    iface_attr->cap.put.max_short = UCT_RC_MLX5_PUT_MAX_SHORT(av_size);
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.put.max_iov   = 1;

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;
    iface_attr->cap.get.max_iov   = 1;

    /* AM */
    iface_attr->cap.am.max_short  = UCT_RC_MLX5_AM_MAX_SHORT(av_size) - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_hdr    = UCT_RC_MLX5_AM_MAX_HDR(av_size) - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_iov    = 1;

    /* Atomics */
    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF |
                                    UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM;

    /* Error Handling */
    iface_attr->cap.flags        |= UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;

    /* Software overhead */
    iface_attr->overhead          = 40e-9;

}
