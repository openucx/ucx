/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/api/uct.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/base/uct_pd.h>
#include <uct/base/uct_log.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */

#include <uct/ib/mlx5/ib_mlx5_log.h>

#include <uct/ib/ud/base/ud_iface.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_def.h>
#include "ud_mlx5.h"
#include <uct/ib/ud/base/ud_inl.h>

#define UCT_UD_MLX5_WQE_SIZE (sizeof(struct mlx5_wqe_ctrl_seg) + \
                              sizeof(struct mlx5_wqe_datagram_seg))

static UCS_F_ALWAYS_INLINE void
uct_ud_mlx5_post_send(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep, 
                      struct mlx5_wqe_ctrl_seg *ctrl, unsigned wqe_size)
{
    uct_ib_mlx5_set_ctrl_seg(ctrl, iface->tx.wq.sw_pi,
                             MLX5_OPCODE_SEND, 0,
                             iface->super.qp->qp_num,
                             uct_ud_mlx5_tx_moderation(iface),
                             wqe_size);
    uct_ib_mlx5_set_dgram_seg((struct mlx5_wqe_datagram_seg *)(ctrl+1),
                              &ep->av, 0);
    uct_ib_mlx5_log_tx(&iface->super.super, IBV_QPT_UD, ctrl,
                       iface->tx.wq.qstart, iface->tx.wq.qend, NULL);
    iface->super.tx.available -= uct_ib_mlx5_post_send(&iface->tx.wq,
                                                       ctrl, wqe_size);
    ucs_assert((int16_t)iface->tx.wq.bb_max >= iface->super.tx.available);
}

static UCS_F_ALWAYS_INLINE void
uct_ud_mlx5_ep_tx_skb(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                      uct_ud_send_skb_t *skb)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_data_seg *dptr;

    ctrl = iface->tx.wq.curr;
    dptr = uct_ib_mlx5_get_next_seg(&iface->tx.wq, ctrl, UCT_UD_MLX5_WQE_SIZE);
    uct_ib_mlx5_set_data_seg(dptr, skb->neth, skb->len, skb->lkey);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, skb->neth);
    uct_ud_mlx5_post_send(iface, ucs_derived_of(ep, uct_ud_mlx5_ep_t), 
                          ctrl, UCT_UD_MLX5_WQE_SIZE + sizeof(*dptr));
}

static inline void
uct_ud_mlx5_ep_tx_inl(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                      const void *buf, unsigned length)
{
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;

    ctrl = iface->tx.wq.curr;
    inl = uct_ib_mlx5_get_next_seg(&iface->tx.wq, ctrl, UCT_UD_MLX5_WQE_SIZE);
    inl->byte_count = htonl(length | MLX5_INLINE_SEG);
    uct_ib_mlx5_inline_copy(inl + 1, buf, length, &iface->tx.wq);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)buf);
    uct_ud_mlx5_post_send(iface, ep, ctrl,
                          UCT_UD_MLX5_WQE_SIZE + sizeof(*inl) + length);
}


static void 
uct_ud_mlx5_ep_tx_ctl_skb(uct_ud_iface_t *ud_iface, uct_ud_ep_t *ud_ep,
                          uct_ud_send_skb_t *skb)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(ud_iface,
                                                uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(ud_ep, uct_ud_mlx5_ep_t);

    if (skb->len >= iface->super.config.max_inline) {
        uct_ud_mlx5_ep_tx_skb(iface, ep, skb);
    } else {
        uct_ud_mlx5_ep_tx_inl(iface, ep, skb->neth, skb->len);
    }
}

static UCS_F_NOINLINE void
uct_ud_mlx5_iface_post_recv(uct_ud_mlx5_iface_t *iface)
{
    unsigned batch = iface->super.config.rx_max_batch;
    struct mlx5_wqe_data_seg *rx_wqes;
    uint16_t pi, next_pi, count;
    uct_ib_iface_recv_desc_t *desc;

    rx_wqes = iface->rx.wq.wqes;
    pi      = iface->rx.wq.rq_wqe_counter & iface->rx.wq.mask;

    for (count = 0; count < batch; count ++) {
        next_pi = (pi + 1) &  iface->rx.wq.mask;
        ucs_prefetch(rx_wqes + next_pi);
        UCT_TL_IFACE_GET_RX_DESC(&iface->super.super.super, &iface->super.rx.mp,
                                 desc, break);
        rx_wqes[pi].lkey = htonl(desc->lkey);
        rx_wqes[pi].addr = htonll((uintptr_t)uct_ib_iface_recv_desc_hdr(&iface->super.super, desc));
        pi = next_pi;
    }
    if (ucs_unlikely(count == 0)) {
        ucs_debug("iface(%p) failed to post receive wqes", iface);
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
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_ep_t, uct_ep_t);


static ucs_status_t
uct_ud_mlx5_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                        const void *buffer, unsigned length)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    struct mlx5_wqe_ctrl_seg *ctrl;
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

    uct_ud_enter(&iface->super);
    uct_ud_iface_progress_pending_tx(&iface->super);
    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        uct_ud_leave(&iface->super);
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = iface->tx.wq.curr;
    /* Set inline segment which has AM id, AM header, and AM payload */
    inl = uct_ib_mlx5_get_next_seg(&iface->tx.wq, ctrl, UCT_UD_MLX5_WQE_SIZE);
    wqe_size = length + sizeof(*am) + sizeof(*neth);
    inl->byte_count = htonl(wqe_size | MLX5_INLINE_SEG);

    /* assume that neth and am header fit into one bb */
    ucs_assert(sizeof(*am) + sizeof(*neth) < MLX5_SEND_WQE_BB);
    neth = (void*)(inl + 1);
    uct_ud_am_set_neth(neth, &ep->super, id);

    am      = (void*)(neth + 1);
    am->hdr = hdr;
    uct_ib_mlx5_inline_copy(am + 1, buffer, length, &iface->tx.wq);

    wqe_size += UCT_UD_MLX5_WQE_SIZE + sizeof(*inl);
    UCT_CHECK_LENGTH(wqe_size, UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                     "am_short");
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, ctrl, wqe_size);

    skb->len = sizeof(*neth) + sizeof(*am);
    memcpy(skb->neth, neth, skb->len); 
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 (char *)skb->neth + skb->len, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_ud_leave(&iface->super);
    return UCS_OK;
}

static ssize_t uct_ud_mlx5_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                       uct_pack_callback_t pack_cb, void *arg)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    uct_ud_send_skb_t *skb;
    ucs_status_t status;
    size_t length;

    uct_ud_enter(&iface->super);
    uct_ud_iface_progress_pending_tx(&iface->super);
    status = uct_ud_am_common(&iface->super, &ep->super, id, &skb);
    if (status != UCS_OK) {
        uct_ud_leave(&iface->super);
        return status;
    }

    length = uct_ud_skb_bcopy(skb, pack_cb, arg);

    uct_ud_mlx5_ep_tx_skb(iface, ep, skb);
    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_ud_leave(&iface->super);
    return length;
}

static ucs_status_t 
uct_ud_mlx5_ep_put_short(uct_ep_h tl_ep, const void *buffer, unsigned length,
                         uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    struct mlx5_wqe_ctrl_seg *ctrl;
    struct mlx5_wqe_inl_data_seg *inl;
    unsigned wqe_size;
    uct_ud_put_hdr_t *put_hdr;
    uct_ud_neth_t *neth;
    uct_ud_send_skb_t *skb;

    uct_ud_enter(&iface->super);
    uct_ud_iface_progress_pending_tx(&iface->super);
    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        uct_ud_leave(&iface->super);
        return UCS_ERR_NO_RESOURCE;
    }

    ctrl = iface->tx.wq.curr;
    /* Set inline segment which has AM id, AM header, and AM payload */
    inl = uct_ib_mlx5_get_next_seg(&iface->tx.wq, ctrl, UCT_UD_MLX5_WQE_SIZE);
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

    wqe_size += UCT_UD_MLX5_WQE_SIZE + sizeof(*inl);
    UCT_CHECK_LENGTH(wqe_size, UCT_IB_MLX5_MAX_BB * MLX5_SEND_WQE_BB,
                     "put_short");
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, neth);
    uct_ud_mlx5_post_send(iface, ep, ctrl, wqe_size);

    skb->len = sizeof(*neth) + sizeof(*put_hdr);
    memcpy(skb->neth, neth, skb->len); 
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 (char *)skb->neth + skb->len, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_ud_leave(&iface->super);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE 
ucs_status_t uct_ud_mlx5_iface_poll_rx(uct_ud_mlx5_iface_t *iface, int is_async)
{
    struct mlx5_cqe64 *cqe;
    uint16_t ci;
    uct_ib_iface_recv_desc_t *desc;
    uint32_t len;
    void *packet;
    ucs_status_t status;

    ci     = iface->rx.wq.cq_wqe_counter & iface->rx.wq.mask;
    packet = (void *)ntohll(iface->rx.wq.wqes[ci].addr);
    ucs_prefetch(packet + UCT_IB_GRH_LEN);
    desc   = (uct_ib_iface_recv_desc_t *)(packet - iface->super.super.config.rx_hdr_offset);

    cqe = uct_ib_mlx5_get_cqe(&iface->rx.cq, UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        status = UCS_ERR_NO_PROGRESS;
        goto out;
    }
    uct_ib_mlx5_log_cqe(cqe);
    ucs_assert(0 == (cqe->op_own & 
               (MLX5_INLINE_SCATTER_32|MLX5_INLINE_SCATTER_64)));
    ucs_assert(ntohs(cqe->wqe_counter) == iface->rx.wq.cq_wqe_counter);

    iface->super.rx.available++;
    iface->rx.wq.cq_wqe_counter++;

    len = ntohl(cqe->byte_cnt);
    VALGRIND_MAKE_MEM_DEFINED(packet, len);

    uct_ud_ep_process_rx(&iface->super,
                         (uct_ud_neth_t *)(packet + UCT_IB_GRH_LEN),
                         len - UCT_IB_GRH_LEN,
                         (uct_ud_recv_skb_t *)desc, is_async);
    status = UCS_OK;

out:
    if (iface->super.rx.available >= iface->super.config.rx_max_batch) {
        /* we need to try to post buffers always. Otherwise it is possible
         * to run out of rx wqes if receiver is slow and there are always
         * cqe to process
         */
        uct_ud_mlx5_iface_post_recv(iface);
    }
    return status;
}

static UCS_F_ALWAYS_INLINE void 
uct_ud_mlx5_iface_poll_tx(uct_ud_mlx5_iface_t *iface)
{
    struct mlx5_cqe64 *cqe;

    cqe = uct_ib_mlx5_get_cqe(&iface->tx.cq, UCT_IB_MLX5_CQE64_SIZE_LOG);
    if (cqe == NULL) {
        return;
    }
    uct_ib_mlx5_log_cqe(cqe);
    iface->super.tx.available = uct_ib_mlx5_txwq_update_bb(&iface->tx.wq, ntohs(cqe->wqe_counter));
}

static void uct_ud_mlx5_iface_progress(void *arg)
{
    uct_ud_mlx5_iface_t *iface = arg;
    ucs_status_t status;

    uct_ud_enter(&iface->super);
    status = uct_ud_iface_dispatch_pending_rx(&iface->super);
    if (ucs_likely(status == UCS_OK)) {
        status = uct_ud_mlx5_iface_poll_rx(iface, 0);
        if (status == UCS_ERR_NO_PROGRESS) {
            uct_ud_mlx5_iface_poll_tx(iface);
        }
    }
    uct_ud_iface_progress_pending(&iface->super, 0);
    uct_ud_leave(&iface->super);
}

static void uct_ud_mlx5_iface_async_progress(void *arg)
{
    uct_ud_mlx5_iface_t *iface = arg;
    ucs_status_t status;

    do {
        status = uct_ud_mlx5_iface_poll_rx(iface, 1);
    } while (status == UCS_OK);

    uct_ud_mlx5_iface_poll_tx(iface);
    uct_ud_iface_progress_pending(&iface->super, 1);

}

static ucs_status_t 
uct_ud_mlx5_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");
    uct_ud_iface_query(iface, iface_attr);

    return UCS_OK;
}

static ucs_status_t
uct_ud_mlx5_ep_create_ah(uct_ud_mlx5_iface_t *iface, uct_ud_mlx5_ep_t *ep,
                         const uct_sockaddr_ib_t *if_addr)
{
    struct ibv_ah *ah;

    ah = uct_ib_create_ah(&iface->super.super, if_addr->lid);
    if (ah == NULL) {
        ucs_error("failed to create address handle: %m");
        return UCS_ERR_INVALID_ADDR;
    }

    uct_ib_mlx5_get_av(ah, &ep->av);
    mlx5_av_base(&ep->av)->key.qkey.qkey      = htonl(UCT_IB_QKEY);
    mlx5_av_base(&ep->av)->key.qkey.reserved  = iface->super.qp->qp_num;
    mlx5_av_base(&ep->av)->dqp_dct            = htonl(if_addr->qp_num |
                                                      UCT_IB_MLX5_EXTENDED_UD_AV); 
    ibv_destroy_ah(ah);
    return UCS_OK;
}

ucs_status_t
uct_ud_mlx5_ep_create_connected(uct_iface_h iface_h,
                                const struct sockaddr *addr,
                                uct_ep_h *new_ep_p)
{
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(iface_h, uct_ud_mlx5_iface_t);
    uct_ud_mlx5_ep_t *ep;
    uct_ud_ep_t *new_ud_ep;
    const uct_sockaddr_ib_t *if_addr = (const uct_sockaddr_ib_t *)addr;
    uct_ud_send_skb_t *skb;
    ucs_status_t status;

    uct_ud_enter(&iface->super);
    status = uct_ud_ep_create_connected_common(&iface->super, if_addr,
                                               &new_ud_ep, &skb);
    if (status != UCS_OK) {
        uct_ud_leave(&iface->super);
        return status;
    }
    ep = ucs_derived_of(new_ud_ep, uct_ud_mlx5_ep_t);
    *new_ep_p = &ep->super.super.super;
    if (skb == NULL) {
        uct_ud_leave(&iface->super);
        return UCS_OK;
    }

    status = uct_ud_mlx5_ep_create_ah(iface, ep, if_addr);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_trace_data("TX: CREQ (qp=%x lid=%d)", if_addr->qp_num, if_addr->lid);
    uct_ud_mlx5_ep_tx_skb(iface, ep, skb);
    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    uct_ud_leave(&iface->super);
    return UCS_OK;

err:
    uct_ud_ep_destroy_connected(&ep->super, if_addr);
    uct_ud_leave(&iface->super);
    *new_ep_p = NULL;
    return status;
}

ucs_status_t uct_ud_mlx5_ep_connect_to_ep(uct_ep_h tl_ep,
                                          const struct sockaddr *addr)
{
    ucs_status_t status;
    uct_ud_mlx5_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_mlx5_ep_t);
    uct_ud_mlx5_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                uct_ud_mlx5_iface_t);
    const uct_sockaddr_ib_t *if_addr = (const uct_sockaddr_ib_t *)addr;

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
    .iface_release_am_desc = uct_ud_iface_release_am_desc,
    .iface_get_address     = uct_ud_iface_get_address,
    .iface_is_reachable    = uct_ib_iface_is_reachable,
    .iface_query           = uct_ud_mlx5_iface_query,

    .ep_create             = UCS_CLASS_NEW_FUNC_NAME(uct_ud_mlx5_ep_t),
    .ep_destroy            = uct_ud_ep_disconnect, 
    .ep_get_address        = uct_ud_ep_get_address,

    .ep_create_connected   = uct_ud_mlx5_ep_create_connected,
    .ep_connect_to_ep      = uct_ud_mlx5_ep_connect_to_ep, 

    .ep_put_short          = uct_ud_mlx5_ep_put_short,
    .ep_am_short           = uct_ud_mlx5_ep_am_short,
    .ep_am_bcopy           = uct_ud_mlx5_ep_am_bcopy,

    .ep_pending_add        = uct_ud_ep_pending_add,
    .ep_pending_purge      = uct_ud_ep_pending_purge,

    .ep_flush              = uct_ud_ep_flush
};


static UCS_CLASS_INIT_FUNC(uct_ud_mlx5_iface_t,
                           uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_ud_iface_config_t *config = ucs_derived_of(tl_config,
                                                   uct_ud_iface_config_t);
    ucs_status_t status;
    int i;

    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_mlx5_iface_ops,
                              pd, worker,
                              dev_name, rx_headroom, 0, config);

    self->super.ops.async_progress = uct_ud_mlx5_iface_async_progress;
    self->super.ops.tx_skb         = uct_ud_mlx5_ep_tx_ctl_skb;

    status = uct_ib_mlx5_get_cq(self->super.super.send_cq, &self->tx.cq);
    if (status != UCS_OK) {
        return status;
    }
    if (uct_ib_mlx5_cqe_size(&self->tx.cq) != sizeof(struct mlx5_cqe64)) {
        ucs_error("TX CQE size (%d) is not %d", 
                  uct_ib_mlx5_cqe_size(&self->tx.cq),
                  (int)sizeof(struct mlx5_cqe64));
        return UCS_ERR_IO_ERROR;
    }

    status = uct_ib_mlx5_get_cq(self->super.super.recv_cq, &self->rx.cq);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }
    if (uct_ib_mlx5_cqe_size(&self->rx.cq) != sizeof(struct mlx5_cqe64)) {
        ucs_error("RX CQE size (%d) is not %d", 
                  uct_ib_mlx5_cqe_size(&self->rx.cq),
                  (int)sizeof(struct mlx5_cqe64));
        return UCS_ERR_IO_ERROR;
    }

    status = uct_ib_mlx5_get_txwq(self->super.super.super.worker, self->super.qp,
                                  &self->tx.wq);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }
    self->super.tx.available = self->tx.wq.bb_max;
   
    status = uct_ib_mlx5_get_rxwq(self->super.qp, &self->rx.wq); 
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    /* write buffer sizes */
    for (i = 0; i <= self->rx.wq.mask; i++) {
        self->rx.wq.wqes[i].byte_count = htonl(self->super.super.config.rx_payload_offset + 
                                               self->super.super.config.seg_size);
    }
    while (self->super.rx.available >= self->super.config.rx_max_batch) {
        uct_ud_mlx5_iface_post_recv(self);
    }

    /* TODO: add progress on first ep creation */
    uct_worker_progress_register(worker, uct_ud_mlx5_iface_progress, self);

    uct_ud_leave(&self->super);
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ud_mlx5_iface_t)
{
    ucs_trace_func("");
    uct_ud_enter(&self->super);
    uct_worker_progress_unregister(self->super.super.super.worker,
                                   uct_ud_mlx5_iface_progress, self);
    uct_ib_mlx5_put_txwq(self->super.super.super.worker, &self->tx.wq);
    UCT_UD_IFACE_DELETE_EPS(&self->super, uct_ud_mlx5_ep_t);
    uct_ud_enter(&self->super);
}

UCS_CLASS_DEFINE(uct_ud_mlx5_iface_t, uct_ud_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_mlx5_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h,
                                 const char*, size_t, const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_mlx5_iface_t, uct_iface_t);

static ucs_status_t
uct_ud_mlx5_query_resources(uct_pd_h pd,
                            uct_tl_resource_desc_t **resources_p,
                            unsigned *num_resources_p)
{
    ucs_trace_func("");
    /* TODO take transport overhead into account */
    return uct_ib_device_query_tl_resources(&ucs_derived_of(pd, uct_ib_pd_t)->dev,
                                            "ud_mlx5", UCT_IB_DEVICE_FLAG_MLX5_PRM,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_ud_mlx5_tl,
                        uct_ud_mlx5_query_resources,
                        uct_ud_mlx5_iface_t,
                        "ud_mlx5",
                        "UD_MLX5_",
                        uct_ud_iface_config_table,
                        uct_ud_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pdc, &uct_ud_mlx5_tl);
