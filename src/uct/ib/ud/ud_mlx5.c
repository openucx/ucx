/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <uct/api/uct.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/tl/context.h>
#include <uct/tl/tl_log.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */

#include "ud_iface.h"
#include "ud_ep.h"
#include "ud_def.h"
#include "ud_mlx5.h"

#include "ud_inl.h"

static inline void uct_ud_mlx5_post_send(uct_ud_mlx5_iface_t *iface, 
                                         uct_ud_mlx5_ep_t *ep, 
                                         struct uct_ib_mlx5_ctrl_dgram_seg *ctrl,
                                         int wqe_size)
{
    uct_ib_mlx5_set_ctrl_seg(&ctrl->ctrl, iface->tx.wq.sw_pi,
                             MLX5_OPCODE_SEND, 0,
                             iface->super.qp->qp_num,
                             uct_ud_mlx5_tx_moderation(iface),
                             wqe_size);
    uct_ib_mlx5_set_dgram_seg(&ctrl->dgram, &ep->av, 0);
    uct_ib_mlx5_post_send(&iface->tx.wq, &ctrl->ctrl, wqe_size);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_mlx5_iface_post_recv(uct_ud_mlx5_iface_t *iface)
{
    unsigned batch = iface->super.config.rx_max_batch;
    struct mlx5_wqe_data_seg *rx_wqes;
    uct_ib_mlx5_index_t pi, next_pi, count;
    uct_ib_iface_recv_desc_t *desc;

    rx_wqes = iface->rx.wq.wqes;
    pi      = iface->rx.wq.rq_wqe_counter & iface->rx.wq.mask;

    for (count = 0; count < batch; count ++) {
        next_pi = (pi + 1) &  iface->rx.wq.mask;
        ucs_prefetch(rx_wqes + next_pi);
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super.super, iface->super.rx.mp, desc, break);
        rx_wqes[pi].lkey = htonl(desc->lkey);
        rx_wqes[pi].addr = htonll((uintptr_t)uct_ib_iface_recv_desc_hdr(&iface->super.super, desc));
        pi = next_pi;
    }
    if (ucs_unlikely(count == 0)) {
        ucs_error("iface(%p) failed to post receive wqes", iface);
        return;
    }
    pi = iface->rx.wq.rq_wqe_counter + count;
    iface->rx.wq.rq_wqe_counter = pi;
    iface->super.rx.available -= count;
    ucs_memory_cpu_fence();
    *iface->rx.wq.dbrec = htonl(pi);
}

static UCS_CLASS_INIT_FUNC(uct_ud_mlx5_ep_t, uct_iface_h tl_iface)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_mlx5_iface_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_ud_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_ud_mlx5_ep_t, uct_ud_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_ep_t, uct_ep_t, uct_iface_h);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_ep_t, uct_ep_t);


static ucs_status_t uct_ud_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                            const void *buffer, unsigned length)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_mlx5_iface_t);
    struct uct_ib_mlx5_ctrl_dgram_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    uct_ud_am_short_hdr_t *am;
    uct_ud_neth_t *neth;
    unsigned wqe_size;
    uct_ud_send_skb_t *skb;

    /* data a written directly into tx wqe, so it is impossible to use
     * common ud am code
     */
    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + sizeof(hdr) + length,
                    iface->super.config.max_inline, "am_short");

    skb = uct_ud_iface_get_tx_skb2(&iface->super, &ep->super);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = (struct uct_ib_mlx5_ctrl_dgram_seg *)iface->tx.wq.seg;
    /* Set inline segment which has AM id, AM header, and AM payload */
    inl = uct_ib_mlx5_get_next_seg(&iface->tx.wq, (char *)ctrl, sizeof(*ctrl));
    wqe_size = length + sizeof(*am) + sizeof(*neth);
    inl->byte_count = htonl(wqe_size | MLX5_INLINE_SEG);

    /* assume that neth and am header fit into one bb */
    ucs_assert(sizeof(*am) + sizeof(*neth) < MLX5_SEND_WQE_BB);
    neth = (void*)(inl + 1);
    uct_ud_am_neth(neth, &ep->super, id);

    am      = (void*)(neth + 1);
    am->hdr = hdr;
    uct_ib_mlx5_inline_copy(am + 1, buffer, length, &iface->tx.wq);

    wqe_size += sizeof(*ctrl) + sizeof(*inl);
    UCT_CHECK_LENGTH(wqe_size, UCT_UD_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                     "am_short");
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, ctrl, wqe_size);

    skb->len = sizeof(*neth) + sizeof(*am);
    memcpy(skb->neth, neth, skb->len); 
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 (char *)skb->neth + skb->len, buffer, length);
    return UCS_OK;
}

static ucs_status_t uct_ud_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                            uct_pack_callback_t pack_cb,
                                            void *arg, size_t length)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_mlx5_iface_t);
    struct uct_ib_mlx5_ctrl_dgram_seg *ctrl;
    struct mlx5_wqe_data_seg *dptr;
    uct_ud_send_skb_t *skb;
    ucs_status_t status;

    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + length, 4096 /* TODO */, "am_bcopy");
    status = uct_ud_am_common(&iface->super, &ep->super, id, &skb);
    if (status != UCS_OK) {
        return status;
    }

    uct_ud_skb_bcopy(skb, pack_cb, arg, length);

    ctrl = (struct uct_ib_mlx5_ctrl_dgram_seg *)iface->tx.wq.seg;
    dptr = uct_ib_mlx5_get_next_seg(&iface->tx.wq, (char *)ctrl, sizeof(*ctrl));
    uct_ib_mlx5_set_dptr_seg(dptr, skb->neth, skb->len, skb->lkey);

    UCT_UD_EP_HOOK_CALL_TX(&ep->super, skb->neth);
    uct_ud_mlx5_post_send(iface, ep, ctrl, sizeof(*ctrl) + sizeof(*dptr));

    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    return UCS_OK;
}

static ucs_status_t uct_ud_mlx5_ep_put_short(uct_ep_h tl_ep, 
                                             const void *buffer, unsigned length,
                                             uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_mlx5_iface_t);
    struct uct_ib_mlx5_ctrl_dgram_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    unsigned wqe_size;
    uct_ud_put_hdr_t *put_hdr;
    uct_ud_neth_t *neth;
    uct_ud_send_skb_t *skb;

    skb = uct_ud_iface_get_tx_skb2(&iface->super, &ep->super);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = (struct uct_ib_mlx5_ctrl_dgram_seg *)iface->tx.wq.seg;
    /* Set inline segment which has AM id, AM header, and AM payload */
    inl = uct_ib_mlx5_get_next_seg(&iface->tx.wq, (char *)ctrl, sizeof(*ctrl));
    wqe_size = length + sizeof(*put_hdr) + sizeof(*neth);
    inl->byte_count = htonl(wqe_size | MLX5_INLINE_SEG);

    /* assume that neth and am header fit into one bb */
    ucs_assert(sizeof(*put_hdr) + sizeof(*neth) < MLX5_SEND_WQE_BB);
    neth = (void*)(inl + 1);
    uct_ud_neth_init_data(&ep->super, neth);
    uct_ud_neth_set_type_put(&ep->super, neth);
    uct_ud_neth_ack_req(&ep->super, neth);

    put_hdr = (uct_ud_put_hdr_t *)(neth+1);
    put_hdr->rva = remote_addr;

    uct_ib_mlx5_inline_copy(put_hdr + 1, buffer, length, &iface->tx.wq);

    wqe_size += sizeof(*ctrl) + sizeof(*inl);
    UCT_CHECK_LENGTH(wqe_size, UCT_UD_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                     "put_short");
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, ctrl, wqe_size);

    skb->len = sizeof(*neth) + sizeof(*put_hdr);
    memcpy(skb->neth, neth, skb->len); 
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 (char *)skb->neth + skb->len, buffer, length);
    return UCS_OK;
}

static inline void uct_ud_mlx5_iface_poll_rx(uct_ud_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;
    uct_ib_mlx5_index_t ci;
    uct_ib_iface_recv_desc_t *desc;
    uint32_t len;
    char *packet;

    ci     = iface->rx.wq.cq_wqe_counter & iface->rx.wq.mask;
    packet = (char *)ntohll(iface->rx.wq.wqes[ci].addr);
    ucs_prefetch(packet + UCT_IB_GRH_LEN);
    desc   = (uct_ib_iface_recv_desc_t *)(packet - iface->super.super.config.rx_hdr_offset);

    cqe = uct_ib_mlx5_get_cqe(&iface->rx.cq, UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        goto out;
    }
    uct_ib_mlx5_log_cqe(cqe);
    ucs_assert(0 == (cqe->op_own & (MLX5_INLINE_SCATTER_32|MLX5_INLINE_SCATTER_64)));
    ucs_assert(ntohs(cqe->wqe_counter) == iface->rx.wq.cq_wqe_counter);

    iface->super.rx.available++;
    iface->rx.wq.cq_wqe_counter++;

    len = ntohl(cqe->byte_cnt);
    VALGRIND_MAKE_MEM_DEFINED(packet, len);

    uct_ud_ep_process_rx(&iface->super,
                         (uct_ud_neth_t *)(packet + UCT_IB_GRH_LEN),
                         len - UCT_IB_GRH_LEN,
                         (uct_ud_recv_skb_t *)desc);

out:
    if (iface->super.rx.available >= iface->super.config.rx_max_batch) {
        /* we need to try to post buffers always. Otherwise it is possible
         * to run out of rx wqes if receiver is slow and there are always
         * cqe to process
         */
        uct_ud_mlx5_iface_post_recv(iface);
    }
}

static inline void uct_ud_mlx5_iface_poll_tx(uct_ud_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;

    cqe = uct_ib_mlx5_get_cqe(&iface->tx.cq, UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        return;
    }
    uct_ib_mlx5_log_cqe(cqe);
    iface->super.tx.available += UCT_UD_TX_MODERATION + 1;
}


static void uct_ud_mlx5_iface_progress_pending(uct_ud_mlx5_iface_t *iface)
{
    uct_ud_ep_t *ep;
    ucs_status_t status;
    uct_ud_neth_t *neth;
    uct_ud_send_skb_t *skb;
    struct uct_ib_mlx5_ctrl_dgram_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    struct mlx5_wqe_data_seg *dptr;

    while (!ucs_queue_is_empty(&iface->super.tx.pending_ops)) {
        ctrl = (struct uct_ib_mlx5_ctrl_dgram_seg *)iface->tx.wq.seg;
        inl = uct_ib_mlx5_get_next_seg(&iface->tx.wq, (char *)ctrl, sizeof(*ctrl));
        neth = (void*)(inl + 1);

        status = uct_ud_iface_get_next_pending(&iface->super, &ep, neth, &skb);
        if (status == UCS_ERR_NO_RESOURCE) {
            return;
        }
        if (status == UCS_INPROGRESS) {
            continue;
        }

        if (ucs_unlikely(skb != NULL)) {
            dptr = uct_ib_mlx5_get_next_seg(&iface->tx.wq, (char *)ctrl, sizeof(*ctrl));
            uct_ib_mlx5_set_dptr_seg(dptr, skb->neth, skb->len, skb->lkey);
            UCT_UD_EP_HOOK_CALL_TX(ep, skb->neth);
            uct_ud_mlx5_post_send(iface, ucs_derived_of(ep, uct_ud_mlx5_ep_t), 
                                  ctrl, sizeof(*ctrl) + sizeof(*dptr));
            uct_ud_ep_log_tx_tag("PENDING_TX: (skb)", ep, skb->neth, skb->len);
        } 
        else {
            inl->byte_count = htonl(sizeof(*neth) | MLX5_INLINE_SEG);
            UCT_UD_EP_HOOK_CALL_TX(ep, neth);
            uct_ud_mlx5_post_send(iface, ucs_derived_of(ep, uct_ud_mlx5_ep_t), 
                                  ctrl, sizeof(*ctrl) + sizeof(*inl) + sizeof(*neth));
            uct_ud_ep_log_tx_tag("PENDING_TX: (neth)", ep, neth, sizeof(neth));
        }
    }
}


static void uct_ud_mlx5_iface_progress(void *arg)
{
    uct_ud_mlx5_iface_t *iface = arg;

    uct_ud_mlx5_iface_poll_rx(iface);
    uct_ud_mlx5_iface_poll_tx(iface);
    uct_ud_mlx5_iface_progress_pending(iface);
}

static ucs_status_t uct_ud_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");
    uct_ud_iface_query(iface, iface_attr);

    return UCS_OK;
}

static ucs_status_t uct_ud_mlx5_ep_create_ah(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep, uct_sockaddr_ib_t *if_addr)
{
    struct ibv_ah *ah;

    ah = uct_ib_create_ah(&iface->super.super, if_addr->lid);
    if (ah == NULL) {
        ucs_error("failed to create address handle: %m");
        return UCS_ERR_INVALID_ADDR;
    }

    uct_ib_mlx5_get_av(ah, &ep->av);
    mlx5_av_base(&ep->av)->key.qkey.qkey      = htonl(UCT_UD_QKEY);
    mlx5_av_base(&ep->av)->key.qkey.reserved  = iface->super.qp->qp_num;
    mlx5_av_base(&ep->av)->dqp_dct            = htonl(if_addr->qp_num |
                                                      MXM_IB_MLX5_EXTENDED_UD_AV); 
    ibv_destroy_ah(ah);
    return UCS_OK;
}

ucs_status_t uct_ud_mlx5_ep_create_connected(uct_iface_h iface_h, const struct sockaddr *addr, uct_ep_h *new_ep_p)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(iface_h, uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep;
    uct_sockaddr_ib_t *if_addr = (uct_sockaddr_ib_t *)addr;
    uct_ud_send_skb_t *skb;
    ucs_status_t status;
    struct uct_ib_mlx5_ctrl_dgram_seg *ctrl;
    struct mlx5_wqe_data_seg *dptr;

    status = uct_ud_ep_create_connected(&iface->super, 
                                        if_addr, (uct_ud_ep_t **)&ep, &skb);
    if (status != UCS_OK) {
        return status;
    }
    *new_ep_p = &ep->super.super.super;
    if (skb == NULL) {
        return UCS_OK;
    }

    status = uct_ud_mlx5_ep_create_ah(iface, ep, if_addr);
    if (status != UCS_OK) {
        goto err;
    }

    ctrl = (struct uct_ib_mlx5_ctrl_dgram_seg *)iface->tx.wq.seg;
    dptr = uct_ib_mlx5_get_next_seg(&iface->tx.wq, (char *)ctrl, sizeof(*ctrl));
    uct_ib_mlx5_set_dptr_seg(dptr, skb->neth, skb->len, skb->lkey);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, skb->neth);
    uct_ud_mlx5_post_send(iface, ucs_derived_of(ep, uct_ud_mlx5_ep_t), 
                          ctrl, sizeof(*ctrl) + sizeof(*dptr));

    ucs_trace_data("TX: CREQ (qp=%x lid=%d)", if_addr->qp_num, if_addr->lid);
    return UCS_OK;

err:
    uct_ud_ep_destroy_connected(&ep->super, if_addr);
    *new_ep_p = NULL;
    return status;
}

ucs_status_t uct_ud_mlx5_ep_connect_to_ep(uct_ep_h tl_ep,
                                          const struct sockaddr *addr)
{
    ucs_status_t status;
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_mlx5_iface_t);
    uct_sockaddr_ib_t *if_addr = (uct_sockaddr_ib_t *)addr;

    ucs_trace_func("");
    status = uct_ud_ep_connect_to_ep(&ep->super, addr);
    if (status != UCS_OK) {
        return status;
    }

    status = uct_ud_mlx5_ep_create_ah(iface, ep, if_addr);
    if (status != UCS_OK) {
        return status;
    }

    return UCS_OK;
}


static void UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_iface_t)(uct_iface_t*);

uct_iface_ops_t uct_ud_mlx5_iface_ops = {
    .iface_close           = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_iface_t),
    .iface_flush           = uct_ud_iface_flush,
    .iface_release_am_desc = uct_ib_iface_release_am_desc,
    .iface_get_address     = uct_ud_iface_get_address,
    .iface_is_reachable    = uct_ib_iface_is_reachable,
    .iface_query           = uct_ud_mlx5_iface_query,

    .ep_create             = UCS_CLASS_NEW_FUNC_NAME(uct_ud_mlx5_ep_t),
    .ep_destroy            = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_mlx5_ep_t),
    .ep_get_address        = uct_ud_ep_get_address,

    .ep_create_connected   = uct_ud_mlx5_ep_create_connected,
    .ep_connect_to_ep      = uct_ud_mlx5_ep_connect_to_ep, 

    .ep_put_short          = uct_ud_mlx5_ep_put_short,
    .ep_am_short           = uct_ud_mlx5_ep_am_short,
    .ep_am_bcopy           = uct_ud_mlx5_ep_am_bcopy,

    .ep_req_notify         = uct_ud_ep_req_notify,
    .ep_flush              = uct_ud_ep_flush
};


static UCS_CLASS_INIT_FUNC(uct_ud_mlx5_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_ud_iface_config_t *config = ucs_derived_of(tl_config, uct_ud_iface_config_t);
    uct_ib_mlx5_qp_info_t qp_info;
    ucs_status_t status;
    int i;

    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_mlx5_iface_ops, pd, worker,
                              dev_name, rx_headroom, 0, config);

    status = uct_ib_mlx5_get_cq(self->super.super.send_cq, &self->tx.cq);
    if (status != UCS_OK) {
        return status;
    }
    if (self->tx.cq.cqe_size != sizeof(struct mlx5_cqe64)) {
        ucs_error("TX CQE size (%d) is not %d", self->tx.cq.cqe_size, (int)sizeof(struct mlx5_cqe64));
        return UCS_ERR_IO_ERROR;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.recv_cq, &self->rx.cq);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }
    if (self->rx.cq.cqe_size != sizeof(struct mlx5_cqe64)) {
        ucs_error("RX CQE size (%d) is not %d", self->rx.cq.cqe_size, (int)sizeof(struct mlx5_cqe64));
        return UCS_ERR_IO_ERROR;
    }

    /* setup tx work q */
    /* TODO: rc/ud - unify code */
    /* rx/tx wq setup should be moved to separate functions */
    status = uct_ib_mlx5_get_qp_info(self->super.qp, &qp_info);
    if (status != UCS_OK) {
        ucs_error("Failed to get mlx5 QP information");
        return UCS_ERR_IO_ERROR;
    }

    if ((qp_info.bf.size == 0) || !ucs_is_pow2(qp_info.bf.size) ||
            (qp_info.sq.stride != MLX5_SEND_WQE_BB) ||
            !ucs_is_pow2(qp_info.sq.wqe_cnt))
    {
        ucs_error("mlx5 device parameters not suitable for transport");
        return UCS_ERR_IO_ERROR;
    }

    self->tx.wq.qstart     = qp_info.sq.buf;
    self->tx.wq.qend       = qp_info.sq.buf + (qp_info.sq.stride * qp_info.sq.wqe_cnt);
    self->tx.wq.seg        = self->tx.wq.qstart;
    self->tx.wq.sw_pi      = 0;
    self->tx.wq.prev_sw_pi = -1;
    /* todo: ud may need different calc here */
    self->tx.wq.max_pi     = uct_ud_mlx5_calc_max_pi(self, self->tx.wq.prev_sw_pi);
    self->tx.wq.bf_reg     = qp_info.bf.reg;
    self->tx.wq.bf_size    = qp_info.bf.size;
    self->tx.wq.dbrec      = &qp_info.dbrec[MLX5_SND_DBR];
    memset(self->tx.wq.qstart, 0, self->tx.wq.qend - self->tx.wq.qstart); 

    /* setup rx wq */
    if (!ucs_is_pow2(qp_info.rq.wqe_cnt) ||
        qp_info.rq.stride != sizeof(struct mlx5_wqe_data_seg)) {
        ucs_error("mlx5 rx wq [count=%d stride=%d] has invalid parameters", 
                  qp_info.rq.wqe_cnt,
                  qp_info.rq.stride);
        return UCS_ERR_IO_ERROR;
    }
    self->rx.wq.wqes            = qp_info.rq.buf;
    self->rx.wq.rq_wqe_counter  = 0;
    self->rx.wq.cq_wqe_counter  = 0;
    self->rx.wq.mask            = qp_info.rq.wqe_cnt - 1;
    self->rx.wq.dbrec           = &qp_info.dbrec[MLX5_RCV_DBR];
    memset(self->rx.wq.wqes, 0, qp_info.rq.wqe_cnt * sizeof(struct mlx5_wqe_data_seg)); 

    /* write buffer sizes */
    for (i = 0; i <= self->rx.wq.mask; i++) {
        self->rx.wq.wqes[i].byte_count = htonl(self->super.super.config.rx_payload_offset + 
                                               self->super.super.config.seg_size);
    }
    while (self->super.rx.available >= self->super.config.rx_max_batch) {
        uct_ud_mlx5_iface_post_recv(self);
    }

    /* TODO: add progress on first ep creation */
    ucs_notifier_chain_add(&worker->progress_chain, uct_ud_mlx5_iface_progress,
                           self);

    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_iface_t)
{
    ucs_trace_func("");
    ucs_notifier_chain_remove(&self->super.super.super.worker->progress_chain,
                              uct_ud_mlx5_iface_progress, self);
}

UCS_CLASS_DEFINE(uct_ud_mlx5_iface_t, uct_ud_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_iface_t, uct_iface_t, uct_pd_h, uct_worker_h,
                                 const char*, size_t, const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_iface_t, uct_iface_t);

static ucs_status_t uct_ud_mlx5_query_resources(uct_pd_h pd,
                                                uct_tl_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    ucs_trace_func("");
    /* TODO take transport overhead into account */
    return uct_ib_device_query_tl_resources(ucs_derived_of(pd, uct_ib_device_t),
                                            "ud_mlx5",
                                            UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                            UCT_IB_DETH_LEN + sizeof(uct_ud_neth_t),
                                            80,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_ud_mlx5_tl,
                        uct_ud_mlx5_query_resources,
                        uct_ud_mlx5_iface_t,
                        "ud_mlx5",
                        "UD_MLX5_",
                        uct_ud_iface_config_table,
                        uct_ud_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pd, &uct_ud_mlx5_tl);


