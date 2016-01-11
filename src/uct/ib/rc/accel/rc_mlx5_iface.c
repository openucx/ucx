/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/base/ib_device.h>
#include <uct/base/uct_pd.h>
#include <ucs/arch/cpu.h>
#include <ucs/debug/log.h>

#include "rc_mlx5.h"


#define UCT_RC_MLX5_SRQ_STRIDE   (sizeof(struct mlx5_wqe_srq_next_seg) + \
                                  sizeof(struct mlx5_wqe_data_seg))


ucs_config_field_t uct_rc_mlx5_iface_config_table[] = {
  {"RC_", "", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {NULL}
};

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

static inline uct_rc_mlx5_srq_seg_t*
uct_rc_mlx5_iface_get_srq_wqe(uct_rc_mlx5_iface_t *iface, uint16_t index)
{
    ucs_assert(index <= iface->rx.mask);
    return iface->rx.buf + index * UCT_RC_MLX5_SRQ_STRIDE;
}

static UCS_F_NOINLINE unsigned uct_rc_mlx5_iface_post_recv(uct_rc_mlx5_iface_t *iface)
{
    uct_rc_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    uint16_t count, index, next_index;
    uct_rc_hdr_t *hdr;

    /* Make sure the union is right */
    UCS_STATIC_ASSERT(ucs_offsetof(uct_rc_mlx5_srq_seg_t, mlx5_srq.next_wqe_index) ==
                      ucs_offsetof(uct_rc_mlx5_srq_seg_t, srq.next_wqe_index));
    UCS_STATIC_ASSERT(ucs_offsetof(uct_rc_mlx5_srq_seg_t, dptr) ==
                      sizeof(struct mlx5_wqe_srq_next_seg));

    ucs_assert(UCS_CIRCULAR_COMPARE16(iface->rx.ready_idx, <=, iface->rx.free_idx));

    index = iface->rx.ready_idx;
    for (;;) {
        next_index = index + 1;
        seg = uct_rc_mlx5_iface_get_srq_wqe(iface, next_index & iface->rx.mask);
        if (UCS_CIRCULAR_COMPARE16(next_index, >, iface->rx.free_idx)) {
            if (!seg->srq.ooo) {
                break;
            }

            ucs_assert(next_index == (uint16_t)(iface->rx.free_idx + 1));
            seg->srq.ooo   = 0;
            iface->rx.free_idx = next_index;
        }

        if (seg->srq.desc == NULL) {
            UCT_TL_IFACE_GET_RX_DESC(&iface->super.super.super, &iface->super.rx.mp,
                                     desc, break);

            /* Set receive data segment pointer. Length is pre-initialized. */
            hdr            = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
            seg->srq.desc  = desc;
            seg->dptr.lkey = htonl(desc->lkey);
            seg->dptr.addr = htonll((uintptr_t)hdr);
            VALGRIND_MAKE_MEM_NOACCESS(hdr, iface->super.super.config.seg_size);
        }

        index = next_index;
    }

    count = index - iface->rx.sw_pi;
    if (count > 0) {
        iface->rx.ready_idx        = index;
        iface->rx.sw_pi            = index;
        iface->super.rx.available -= count;
        ucs_memory_cpu_store_fence();
        *iface->rx.db = htonl(iface->rx.sw_pi);
    }
    return count;
}

static UCS_F_ALWAYS_INLINE void 
uct_rc_mlx5_iface_poll_tx(uct_rc_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned qp_num;
    uint16_t hw_ci;

    cqe = uct_ib_mlx5_get_cqe(&iface->tx.cq, UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        return;
    }

    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_TX_COMPLETION, 1);

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & UCS_MASK(UCT_IB_QPN_ORDER);
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num), uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);

    hw_ci = ntohs(cqe->wqe_counter);
    ep->super.available = uct_ib_mlx5_txwq_update_bb(&ep->tx.wq, hw_ci);
    ++iface->super.tx.cq_available;

    uct_rc_ep_process_tx_completion(&iface->super, &ep->super, hw_ci);
}

static inline void uct_rc_mlx5_iface_rx_inline(uct_rc_mlx5_iface_t *iface,
                                              uct_ib_iface_recv_desc_t *desc,
                                              int stats_counter, unsigned byte_len)
{
    UCS_STATS_UPDATE_COUNTER(iface->stats, stats_counter, 1);
    VALGRIND_MAKE_MEM_UNDEFINED(uct_ib_iface_recv_desc_hdr(&iface->super.super, desc),
                                byte_len);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_mlx5_iface_poll_rx(uct_rc_mlx5_iface_t *iface)
{
    uct_rc_mlx5_srq_seg_t *seg;
    uct_ib_iface_recv_desc_t *desc;
    uct_rc_hdr_t *hdr;
    struct mlx5_cqe64 *cqe;
    unsigned byte_len;
    uint16_t wqe_ctr;
    uint16_t max_batch;
    ucs_status_t status;
    void *udesc;

    cqe = uct_ib_mlx5_get_cqe(&iface->rx.cq, iface->rx.cq.cqe_size_log);
    if (cqe == NULL) {
        /* If not CQE - post receives */
        status = UCS_ERR_NO_PROGRESS;
        goto done;
    }

    ucs_memory_cpu_load_fence();
    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_RX_COMPLETION, 1);

    byte_len = ntohl(cqe->byte_cnt);
    wqe_ctr  = ntohs(cqe->wqe_counter);
    seg      = uct_rc_mlx5_iface_get_srq_wqe(iface, wqe_ctr);
    desc     = seg->srq.desc;

    /* Get a pointer to AM header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = (uct_rc_hdr_t*)(cqe);
        uct_rc_mlx5_iface_rx_inline(iface, desc, UCT_RC_MLX5_IFACE_STAT_RX_INL_32, byte_len);
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = (uct_rc_hdr_t*)(cqe - 1);
        uct_rc_mlx5_iface_rx_inline(iface, desc, UCT_RC_MLX5_IFACE_STAT_RX_INL_64, byte_len);
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
    }

    uct_ib_mlx5_log_rx(&iface->super.super, IBV_QPT_RC, cqe, hdr,
                       uct_rc_ep_am_packet_dump);

    udesc  = (char*)desc + iface->super.super.config.rx_headroom_offset;
    status = uct_iface_invoke_am(&iface->super.super.super, hdr->am_id, hdr + 1,
                                 byte_len - sizeof(*hdr), udesc);
    if ((status == UCS_OK) && (wqe_ctr == ((iface->rx.ready_idx + 1) & iface->rx.mask))) {
        /* If the descriptor was not used - if there are no "holes", we can just
         * reuse it on the receive queue. Otherwise, ready pointer will stay behind
         * until post_recv allocated more descriptors from the memory pool, fills
         * the holes, and moves it forward.
         */
        ucs_assert(wqe_ctr == ((iface->rx.free_idx + 1) & iface->rx.mask));
        ++iface->rx.ready_idx;
        ++iface->rx.free_idx;
   } else {
        if (status != UCS_OK) {
            uct_recv_desc_iface(udesc) = &iface->super.super.super.super;
            seg->srq.desc              = NULL;
        }
        if (wqe_ctr == ((iface->rx.free_idx + 1) & iface->rx.mask)) {
            ++iface->rx.free_idx;
        } else {
            /* Mark the segment as out-of-order, post_recv will advance free */
            seg->srq.ooo = 1;
        }
    }

    ++iface->super.rx.available;
    status = UCS_OK;

done:
    max_batch = iface->super.config.rx_max_batch;
    if (iface->super.rx.available >= max_batch) {
        uct_rc_mlx5_iface_post_recv(iface);
    }
    return status;
}

static void uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_rc_mlx5_iface_poll_rx(iface);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_rc_mlx5_iface_poll_tx(iface);
    }
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    size_t max_wqe_size;

    uct_rc_iface_query(iface, iface_attr);

    max_wqe_size = MLX5_SEND_WQE_BB * UCT_RC_MLX5_MAX_BB;

    /* PUT */
    iface_attr->cap.put.max_short = max_wqe_size
                                        - sizeof(struct mlx5_wqe_ctrl_seg)
                                        - sizeof(struct mlx5_wqe_raddr_seg)
                                        - sizeof(struct mlx5_wqe_inl_data_seg);
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* GET */
    iface_attr->cap.get.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.get.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* AM */
    iface_attr->cap.am.max_short  = max_wqe_size
                                        - sizeof(struct mlx5_wqe_ctrl_seg)
                                        - sizeof(struct mlx5_wqe_inl_data_seg)
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_hdr    = max_wqe_size
                                        - sizeof(struct mlx5_wqe_ctrl_seg)
                                        - sizeof(struct mlx5_wqe_inl_data_seg)
                                        - sizeof(uct_rc_hdr_t)
                                        - sizeof(struct mlx5_wqe_data_seg);

    /* Atomics */
    iface_attr->cap.flags |= UCT_IFACE_FLAG_ATOMIC_ADD32 |
                             UCT_IFACE_FLAG_ATOMIC_FADD32 |
                             UCT_IFACE_FLAG_ATOMIC_SWAP32 |
                             UCT_IFACE_FLAG_ATOMIC_CSWAP32 |
                             UCT_IFACE_FLAG_ATOMIC_ADD64 |
                             UCT_IFACE_FLAG_ATOMIC_FADD64 |
                             UCT_IFACE_FLAG_ATOMIC_SWAP64 |
                             UCT_IFACE_FLAG_ATOMIC_CSWAP64 |
                             UCT_IFACE_FLAG_ERRHANDLE_ZCOPY_BUF |
                             UCT_IFACE_FLAG_ERRHANDLE_REMOTE_MEM;

    /* Software overhead */
    iface_attr->overhead = 40e-9;

    return UCS_OK;
}

static ucs_status_t uct_rc_mlx5_iface_init_rx(uct_rc_mlx5_iface_t *iface)
{
    uct_ib_mlx5_srq_info_t srq_info;
    uct_rc_mlx5_srq_seg_t *seg;
    ucs_status_t status;
    unsigned i;

    status = uct_ib_mlx5_get_srq_info(iface->super.rx.srq, &srq_info);
    if (status != UCS_OK) {
        return status;
    }

    if (srq_info.head != 0) {
        ucs_error("SRQ head is not 0 (%d)", srq_info.head);
        return UCS_ERR_NO_DEVICE;
    }

    if (srq_info.stride != UCT_RC_MLX5_SRQ_STRIDE) {
        ucs_error("SRQ stride is not %lu (%d)", UCT_RC_MLX5_SRQ_STRIDE,
                  srq_info.stride);
        return UCS_ERR_NO_DEVICE;
    }

    if (!ucs_is_pow2(srq_info.tail + 1)) {
        ucs_error("SRQ length is not power of 2 (%d)", srq_info.tail + 1);
        return UCS_ERR_NO_DEVICE;
    }

    iface->super.rx.available = srq_info.tail + 1;
    iface->rx.buf             = srq_info.buf;
    iface->rx.db              = srq_info.dbrec;
    iface->rx.free_idx        = srq_info.tail;
    iface->rx.ready_idx       = -1;
    iface->rx.sw_pi           = -1;
    iface->rx.mask            = srq_info.tail;

    for (i = srq_info.head; i <= srq_info.tail; ++i) {
        seg = uct_rc_mlx5_iface_get_srq_wqe(iface, i);
        seg->srq.ooo         = 0;
        seg->srq.desc        = NULL;
        seg->dptr.byte_count = htonl(iface->super.super.config.seg_size);
    }

    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config, uct_rc_mlx5_iface_config_t);
    ucs_status_t status;

    extern uct_iface_ops_t uct_rc_mlx5_iface_ops;
    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_mlx5_iface_ops, pd, worker,
                              dev_name, rx_headroom, 0, &config->super);

    status = uct_ib_mlx5_get_cq(self->super.super.send_cq, &self->tx.cq);
    if (status != UCS_OK) {
        goto err;
    }

    if (uct_ib_mlx5_cqe_size(&self->tx.cq) != sizeof(struct mlx5_cqe64)) {
        ucs_error("TX CQE size is not 64");
        goto err;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.recv_cq, &self->rx.cq);
    if (status != UCS_OK) {
        goto err;
    }


    status = uct_iface_mpool_init(&self->super.super.super,
                                  &self->tx.atomic_desc_mp,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_RC_MAX_ATOMIC_SIZE,
                                  sizeof(uct_rc_iface_send_desc_t) + UCT_RC_MAX_ATOMIC_SIZE,
                                  UCS_SYS_CACHE_LINE_SIZE,
                                  &config->super.super.tx.mp,
                                  self->super.config.tx_qp_len,
                                  uct_rc_iface_send_desc_init,
                                  "rc_mlx5_atomic_desc");
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_rc_mlx5_iface_init_rx(self);
    if (status != UCS_OK) {
        goto err;
    }

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_mlx5_iface_stats_class,
                                  self->super.stats);
    if (status != UCS_OK) {
        goto err_destroy_atomic_mp;
    }

    if (uct_rc_mlx5_iface_post_recv(self) == 0) {
        ucs_error("Failed to post receives");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_stats;
    }

    uct_worker_progress_register(worker, uct_rc_mlx5_iface_progress, self);
    return UCS_OK;

err_free_stats:
    UCS_STATS_NODE_FREE(self->stats);
err_destroy_atomic_mp:
    ucs_mpool_cleanup(&self->tx.atomic_desc_mp, 1);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_worker_progress_unregister(self->super.super.super.worker,
                                   uct_rc_mlx5_iface_progress, self);
    UCS_STATS_NODE_FREE(self->stats);
    ucs_mpool_cleanup(&self->tx.atomic_desc_mp, 1);
}


UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char*, size_t,
                                 const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);

uct_iface_ops_t uct_rc_mlx5_iface_ops = {
    .iface_query         = uct_rc_mlx5_iface_query,
    .iface_flush         = uct_rc_iface_flush,
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t),
    .iface_release_am_desc= uct_ib_iface_release_am_desc,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_get_address      = uct_rc_ep_get_address,
    .ep_connect_to_ep    = uct_rc_ep_connect_to_ep,
    .iface_get_address   = uct_ib_iface_get_subnet_address,
    .iface_is_reachable  = uct_ib_iface_is_reachable,
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_put_short        = uct_rc_mlx5_ep_put_short,
    .ep_put_bcopy        = uct_rc_mlx5_ep_put_bcopy,
    .ep_put_zcopy        = uct_rc_mlx5_ep_put_zcopy,
    .ep_get_bcopy        = uct_rc_mlx5_ep_get_bcopy,
    .ep_get_zcopy        = uct_rc_mlx5_ep_get_zcopy,
    .ep_am_short         = uct_rc_mlx5_ep_am_short,
    .ep_am_bcopy         = uct_rc_mlx5_ep_am_bcopy,
    .ep_am_zcopy         = uct_rc_mlx5_ep_am_zcopy,
    .ep_atomic_add64     = uct_rc_mlx5_ep_atomic_add64,
    .ep_atomic_fadd64    = uct_rc_mlx5_ep_atomic_fadd64,
    .ep_atomic_swap64    = uct_rc_mlx5_ep_atomic_swap64,
    .ep_atomic_cswap64   = uct_rc_mlx5_ep_atomic_cswap64,
    .ep_atomic_add32     = uct_rc_mlx5_ep_atomic_add32,
    .ep_atomic_fadd32    = uct_rc_mlx5_ep_atomic_fadd32,
    .ep_atomic_swap32    = uct_rc_mlx5_ep_atomic_swap32,
    .ep_atomic_cswap32   = uct_rc_mlx5_ep_atomic_cswap32,
    .ep_pending_add      = uct_rc_ep_pending_add,
    .ep_pending_purge    = uct_rc_ep_pending_purge,
    .ep_flush            = uct_rc_mlx5_ep_flush
};


static ucs_status_t uct_rc_mlx5_query_resources(uct_pd_h pd,
                                                uct_tl_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(&ucs_derived_of(pd, uct_ib_pd_t)->dev,
                                            "rc_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_rc_mlx5_tl,
                        uct_rc_mlx5_query_resources,
                        uct_rc_mlx5_iface_t,
                        "rc_mlx5",
                        "RC_MLX5_",
                        uct_rc_mlx5_iface_config_table,
                        uct_rc_mlx5_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pdc, &uct_rc_mlx5_tl);
