/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <uct/ib/mlx5/ib_mlx5_log.h>
#include <uct/ib/base/ib_device.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>


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

static inline struct mlx5_wqe_srq_next_seg*
uct_rc_mlx5_iface_get_srq_wqe(uct_rc_mlx5_iface_t *iface, unsigned index)
{
    return iface->rx.buf + index * UCT_RC_MLX5_SRQ_STRIDE;
}

static inline unsigned uct_rc_mlx5_srq_next_wqe_ind(struct mlx5_wqe_srq_next_seg* seg)
{
    return ntohs(seg->next_wqe_index);
}

static unsigned uct_rc_mlx5_iface_post_recv(uct_rc_mlx5_iface_t *iface, unsigned max)
{
    struct mlx5_wqe_srq_next_seg *seg;
    uct_rc_mlx5_recv_desc_t *desc;
    unsigned count, head;
    uct_rc_hdr_t *hdr;
    unsigned length;

    head   = iface->rx.head;
    length = iface->super.super.config.seg_size;
    count = 0;
    while (count < max) {
        ucs_assert(head != iface->rx.tail);

        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super.super, iface->super.rx.mp,
                                 desc, break);

        seg = uct_rc_mlx5_iface_get_srq_wqe(iface, head);

        hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, &desc->super);
        uct_ib_mlx5_wqe_set_data_seg((void*)(seg + 1), hdr,
                                     length, /* TODO pre-init length */
                                     desc->super.lkey);
        VALGRIND_MAKE_MEM_NOACCESS(hdr, length);

        ucs_queue_push(&iface->rx.desc_q, &desc->queue);
        head = uct_rc_mlx5_srq_next_wqe_ind(seg);
        ++count;
    }

    if (count > 0) {
        iface->rx.head             = head;
        iface->rx.sw_pi           += count;
        iface->super.rx.available -= count;
        ucs_memory_cpu_store_fence();
        *iface->rx.db = htonl(iface->rx.sw_pi);
    }

    return count;
}

static inline void uct_rc_mlx5_iface_poll_tx(uct_rc_mlx5_iface_t *iface)
{
    uct_rc_iface_send_op_t *op;
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

    /* "cqe->wqe_counter" is the index of the completed wqe (modulo 2^16).
     * Calculate new max_pi based on that value.
     */
    hw_ci         = ntohs(cqe->wqe_counter);
    ep->tx.max_pi = uct_rc_mlx5_calc_max_pi(iface, hw_ci);
    ++iface->super.tx.cq_available;

    /* Process completions */
    ucs_queue_for_each_extract(op, &ep->super.outstanding, queue,
                               UCS_CIRCULAR_COMPARE16(op->sn, <=, hw_ci)) {
        op->handler(op);
    }
}

static inline void uct_rc_mlx5_iface_poll_rx(uct_rc_mlx5_iface_t *iface)
{
    struct mlx5_wqe_srq_next_seg *seg;
    uct_rc_mlx5_recv_desc_t *desc;
    uct_rc_hdr_t *hdr;
    struct mlx5_cqe64 *cqe;
    unsigned byte_len;
    unsigned max_batch;
    uint16_t wqe_ctr_be;

    cqe = uct_ib_mlx5_get_cqe(&iface->rx.cq, iface->rx.cq.cqe_size_log);
    if (cqe == NULL) {
        /* If not CQE - post receives */
        max_batch = iface->super.config.rx_max_batch;
        if (iface->super.rx.available >= max_batch) {
            uct_rc_mlx5_iface_post_recv(iface, max_batch);
        }
        return;
    }

    UCS_STATS_UPDATE_COUNTER(iface->super.stats, UCT_RC_IFACE_STAT_RX_COMPLETION, 1);

    ucs_assert(!ucs_queue_is_empty(&iface->rx.desc_q));
    ucs_memory_cpu_load_fence();

    desc     = ucs_queue_pull_elem_non_empty(&iface->rx.desc_q, uct_rc_mlx5_recv_desc_t, queue);
    byte_len = ntohl(cqe->byte_cnt);

    uct_ib_iface_desc_received(&iface->super.super, &desc->super, byte_len,
                               !(cqe->op_own & (MLX5_INLINE_SCATTER_32|MLX5_INLINE_SCATTER_64)));

    /* Get a pointer to AM header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        hdr = (uct_rc_hdr_t*)cqe;
        UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_RC_MLX5_IFACE_STAT_RX_INL_32, 1);
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        hdr = (uct_rc_hdr_t*)(cqe - 1);
        UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_RC_MLX5_IFACE_STAT_RX_INL_64, 1)
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, &desc->super);
    }

    uct_ib_mlx5_log_rx(IBV_QPT_RC, cqe, hdr, uct_rc_ep_am_packet_dump);
    uct_rc_iface_invoke_am(&iface->super, hdr, byte_len, &desc->super);

    /* Add completed SRQ WQE to the tail
     * TODO return the descriptor directly to SRQ
     * TODO use SRQ reserved fields for desc pointer
     * */
    seg = uct_rc_mlx5_iface_get_srq_wqe(iface, iface->rx.tail);
    wqe_ctr_be = cqe->wqe_counter;
    seg->next_wqe_index = wqe_ctr_be;
    iface->rx.tail = ntohs(wqe_ctr_be);
    ++iface->super.rx.available;
}

static void uct_rc_mlx5_iface_free_rx_descs(uct_rc_mlx5_iface_t *iface)
{
    uct_rc_mlx5_recv_desc_t *desc;

    while (!ucs_queue_is_empty(&iface->rx.desc_q)) {
        desc = ucs_queue_pull_elem_non_empty(&iface->rx.desc_q,
                                             uct_rc_mlx5_recv_desc_t, queue);
        ucs_mpool_put(desc);
    }
    ucs_mpool_check_empty(iface->super.rx.mp);
}

static void uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;

    uct_rc_mlx5_iface_poll_tx(iface);
    uct_rc_mlx5_iface_poll_rx(iface);
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
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config, uct_rc_mlx5_iface_config_t);
    uct_ib_mlx5_srq_info_t srq_info;
    ucs_status_t status;

    extern uct_iface_ops_t uct_rc_mlx5_iface_ops;
    UCS_CLASS_CALL_SUPER_INIT(uct_rc_iface_t, &uct_rc_mlx5_iface_ops, pd, worker,
                              dev_name, rx_headroom,
                              sizeof(uct_rc_mlx5_recv_desc_t) - sizeof(uct_ib_iface_recv_desc_t),
                              &config->super);

    status = uct_ib_mlx5_get_cq(self->super.super.send_cq, &self->tx.cq);
    if (status != UCS_OK) {
        goto err;
    }

    if (self->tx.cq.cqe_size != sizeof(struct mlx5_cqe64)) {
        ucs_error("TX CQE size is not 64");
        goto err;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.recv_cq, &self->rx.cq);
    if (status != UCS_OK) {
        goto err;
    }

    status = uct_ib_mlx5_get_srq_info(self->super.rx.srq, &srq_info);
    if (status != UCS_OK) {
        goto err;
    }

    if (srq_info.stride != UCT_RC_MLX5_SRQ_STRIDE) {
        ucs_error("SRQ stride is not %lu (%d)", UCT_RC_MLX5_SRQ_STRIDE, srq_info.stride);
        status = UCS_ERR_NO_DEVICE;
        goto err;
    }

    status = uct_iface_mpool_create(&self->super.super.super.super,
                                    sizeof(uct_rc_iface_send_desc_t) + UCT_RC_MAX_ATOMIC_SIZE,
                                    sizeof(uct_rc_iface_send_desc_t) + UCT_RC_MAX_ATOMIC_SIZE,
                                    UCS_SYS_CACHE_LINE_SIZE,
                                    &config->super.super.tx.mp,
                                    self->super.config.tx_qp_len,
                                    uct_rc_iface_send_desc_init,
                                    "rc_mlx5_atomic_desc", &self->tx.atomic_desc_mp);
    if (status != UCS_OK) {
        goto err;
    }

    status = UCS_STATS_NODE_ALLOC(&self->stats, &uct_rc_mlx5_iface_stats_class,
                                  self->super.stats);
    if (status != UCS_OK) {
        goto err_destroy_atomic_mp;
    }

    self->rx.buf   = srq_info.buf;
    self->rx.db    = srq_info.dbrec;
    self->rx.head  = srq_info.head;
    self->rx.tail  = srq_info.tail;
    self->rx.sw_pi = 0;
    ucs_queue_head_init(&self->rx.desc_q);

    if (uct_rc_mlx5_iface_post_recv(self, self->super.rx.available) == 0) {
        ucs_error("Failed to post receives");
        status = UCS_ERR_NO_MEMORY;
        goto err_free_stats;
    }

    ucs_notifier_chain_add(&worker->progress_chain, uct_rc_mlx5_iface_progress,
                           self);
    return UCS_OK;

err_free_stats:
    UCS_STATS_NODE_FREE(self->stats);
err_destroy_atomic_mp:
    ucs_mpool_destroy(self->tx.atomic_desc_mp);
err:
    return status;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    ucs_notifier_chain_remove(&self->super.super.super.worker->progress_chain,
                              uct_rc_mlx5_iface_progress, self);
    uct_rc_mlx5_iface_free_rx_descs(self);
    UCS_STATS_NODE_FREE(self->stats);
    ucs_mpool_destroy(self->tx.atomic_desc_mp);
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
    .ep_flush            = uct_rc_mlx5_ep_flush
};


static ucs_status_t uct_rc_mlx5_query_resources(uct_pd_h pd,
                                                uct_tl_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(ucs_derived_of(pd, uct_ib_device_t),
                                            "rc_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                            ucs_max(sizeof(uct_rc_hdr_t), UCT_IB_RETH_LEN),
                                            40,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_rc_mlx5_tl,
                        uct_rc_mlx5_query_resources,
                        uct_rc_mlx5_iface_t,
                        "rc_mlx5",
                        "RC_MLX5_",
                        uct_rc_mlx5_iface_config_table,
                        uct_rc_mlx5_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pd, &uct_rc_mlx5_tl);
