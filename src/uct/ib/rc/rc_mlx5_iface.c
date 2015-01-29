/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_mlx5.h"

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>


#define UCT_RC_MLX5_SRQ_STRIDE   (sizeof(struct mlx5_wqe_srq_next_seg) + \
                                  sizeof(struct mlx5_wqe_data_seg))


ucs_config_field_t uct_rc_mlx5_iface_config_table[] = {
  {"RC_", "IB_RX_INLINE=32", NULL,
   ucs_offsetof(uct_rc_mlx5_iface_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_rc_iface_config_table)},

  {NULL}
};

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

        desc = ucs_mpool_get(iface->super.rx.mp);
        if (desc == NULL) {
            break;
        }

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
    struct mlx5_cqe64 *cqe;
    uct_rc_mlx5_ep_t *ep;
    unsigned qp_num;
    uint16_t hw_ci, prev_pi, new_pi;

    cqe = uct_ib_mlx5_get_cqe(&iface->tx.cq);
    if (cqe == NULL) {
        return;
    }

    ucs_memory_cpu_load_fence();

    qp_num = ntohl(cqe->sop_drop_qpn) & 0xffffff;
    ep = ucs_derived_of(uct_rc_iface_lookup_ep(&iface->super, qp_num), uct_rc_mlx5_ep_t);
    ucs_assert(ep != NULL);

    /* "cqe->wqe_counter" is the index of the completed wqe (modulo 2^16).
     * So we can post additional "wqe_cnt" WQEs, plus the one just completed.
     */
    prev_pi  = ep->tx.max_pi;
    hw_ci    = ntohs(cqe->wqe_counter);
    new_pi   = hw_ci + ep->tx.wqe_cnt + 1;
    iface->super.tx.outstanding -= (uint16_t)(new_pi - prev_pi);
    ep->tx.max_pi = new_pi;

    /* Process completions */
    ucs_callbackq_pull(&ep->super.tx.comp, hw_ci);
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
    ucs_status_t status;

    cqe = uct_ib_mlx5_get_cqe(&iface->rx.cq);
    if (cqe == NULL) {
        /* If not CQE - post receives */
        max_batch = iface->super.config.rx_max_batch;
        if (iface->super.rx.available >= max_batch) {
            uct_rc_mlx5_iface_post_recv(iface, max_batch);
        }
        return;
    }

    ucs_assert(!ucs_queue_is_empty(&iface->rx.desc_q));

    /* TODO support CQE size of 128 */
    ucs_assertv(iface->rx.cq.cqe_size == 64, "cqe_size=%d", iface->rx.cq.cqe_size);

    ucs_memory_cpu_load_fence();

    desc     = ucs_queue_pull_elem_non_empty(&iface->rx.desc_q, uct_rc_mlx5_recv_desc_t, queue);
    byte_len = ntohl(cqe->byte_cnt);

    /* Get a pointer to AM header (after which comes the payload)
     * Support cases of inline scatter by pointing directly to CQE.
     */
    if (cqe->op_own & MLX5_INLINE_SCATTER_32) {
        status = uct_rc_iface_invoke_am(&iface->super, &desc->super,
                                        (uct_rc_hdr_t*)cqe, byte_len);
    } else if (cqe->op_own & MLX5_INLINE_SCATTER_64) {
        /* TODO support inline scatter of 64b */
        ucs_fatal("inl data @ %p", cqe - 1);
    } else {
        hdr = uct_ib_iface_recv_desc_hdr(&iface->super.super, &desc->super);
        VALGRIND_MAKE_MEM_DEFINED(hdr, byte_len);
        status = uct_rc_iface_invoke_am(&iface->super, &desc->super, hdr,
                                        byte_len);
    }
    if (status == UCS_OK) {
        ucs_mpool_put(desc);
    }

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

static void uct_rc_mlx5_iface_progress(void *arg)
{
    uct_rc_mlx5_iface_t *iface = arg;

    uct_rc_mlx5_iface_poll_tx(iface);
    uct_rc_mlx5_iface_poll_rx(iface);
}

static ucs_status_t uct_rc_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_rc_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_iface_t);
    size_t max_bb = MLX5_SEND_WQE_BB;

    uct_rc_iface_query(iface, iface_attr);

    /* PUT */
    iface_attr->cap.put.max_short = max_bb
                                        - sizeof(struct mlx5_wqe_ctrl_seg)
                                        - sizeof(struct mlx5_wqe_raddr_seg)
                                        - sizeof(struct mlx5_wqe_inl_data_seg);
    iface_attr->cap.put.max_bcopy = iface->super.config.seg_size;
    iface_attr->cap.put.max_zcopy = uct_ib_iface_port_attr(&iface->super)->max_msg_sz;

    /* AM */
    iface_attr->cap.am.max_short  = max_bb
                                        - sizeof(struct mlx5_wqe_ctrl_seg)
                                        - sizeof(struct mlx5_wqe_inl_data_seg)
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_bcopy  = iface->super.config.seg_size
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_zcopy  = iface->super.config.seg_size
                                        - sizeof(uct_rc_hdr_t);
    iface_attr->cap.am.max_hdr    = max_bb
                                        - sizeof(struct mlx5_wqe_ctrl_seg)
                                        - sizeof(struct mlx5_wqe_inl_data_seg)
                                        - sizeof(uct_rc_hdr_t)
                                        - sizeof(struct mlx5_wqe_data_seg);

    /* TODO add get and atomics */
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_mlx5_iface_t, uct_context_h context,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_rc_mlx5_iface_config_t *config = ucs_derived_of(tl_config, uct_rc_mlx5_iface_config_t);
    uct_ib_mlx5_srq_info_t srq_info;
    ucs_status_t status;

    extern uct_iface_ops_t uct_rc_mlx5_iface_ops;
    UCS_CLASS_CALL_SUPER_INIT(&uct_rc_mlx5_iface_ops, context, dev_name,
                              rx_headroom,
                              sizeof(uct_rc_mlx5_recv_desc_t) - sizeof(uct_ib_iface_recv_desc_t),
                              &config->super);

    status = uct_ib_mlx5_get_cq(self->super.super.send_cq, &self->tx.cq);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.recv_cq, &self->rx.cq);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ib_mlx5_get_srq_info(self->super.rx.srq, &srq_info);
    if (status != UCS_OK) {
        return status;
    }

    if (srq_info.stride != UCT_RC_MLX5_SRQ_STRIDE) {
        ucs_error("SRQ stride is not %lu (%d)", UCT_RC_MLX5_SRQ_STRIDE, srq_info.stride);
        return UCS_ERR_NO_DEVICE;
    }

    self->rx.buf   = srq_info.buf;
    self->rx.db    = srq_info.dbrec;
    self->rx.head  = srq_info.head;
    self->rx.tail  = srq_info.tail;
    self->rx.sw_pi = 0;
    ucs_queue_head_init(&self->rx.desc_q);

    if (uct_rc_mlx5_iface_post_recv(self, self->super.rx.available) == 0) {
        ucs_error("Failed to post receives");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_notifier_chain_add(&context->progress_chain, uct_rc_mlx5_iface_progress,
                           self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_mlx5_iface_t)
{
    uct_context_h context = uct_ib_iface_device(&self->super.super)->super.context;
    ucs_notifier_chain_remove(&context->progress_chain, uct_rc_mlx5_iface_progress, self);
}


UCS_CLASS_DEFINE(uct_rc_mlx5_iface_t, uct_rc_iface_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_mlx5_iface_t, uct_iface_t, uct_context_h,
                                 const char*, size_t, uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_mlx5_iface_t, uct_iface_t);

uct_iface_ops_t uct_rc_mlx5_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_iface_t),
    .iface_get_address   = uct_rc_iface_get_address,
    .iface_flush         = uct_rc_iface_flush,
    .ep_get_address      = uct_rc_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_rc_ep_connect_to_ep,
    .iface_query         = uct_rc_mlx5_iface_query,
    .ep_put_short        = uct_rc_mlx5_ep_put_short,
    .ep_put_bcopy        = uct_rc_mlx5_ep_put_bcopy,
    .ep_put_zcopy        = uct_rc_mlx5_ep_put_zcopy,
    .ep_am_short         = uct_rc_mlx5_ep_am_short,
    .ep_am_bcopy         = uct_rc_mlx5_ep_am_bcopy,
    .ep_am_zcopy         = uct_rc_mlx5_ep_am_zcopy,
    .ep_flush            = uct_rc_mlx5_ep_flush,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_rc_mlx5_ep_t),
};


static ucs_status_t uct_rc_mlx5_query_resources(uct_context_h context,
                                                uct_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, UCT_IB_RESOURCE_FLAG_MLX5_PRM,
                                  resources_p, num_resources_p);
}

static uct_tl_ops_t uct_rc_mlx5_tl_ops = {
    .query_resources     = uct_rc_mlx5_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_rc_mlx5_iface_t),
    .rkey_unpack         = uct_ib_rkey_unpack,
};

static void uct_rc_mlx5_register(uct_context_t *context)
{
    uct_register_tl(context, "rc_mlx5", uct_rc_mlx5_iface_config_table,
                    sizeof(uct_rc_mlx5_iface_config_t), "RC_MLX5_", &uct_rc_mlx5_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, rc_mlx5, uct_rc_mlx5_register, ucs_empty_function, 0)
