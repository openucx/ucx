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

#include <uct/ib/base/ib_log.h>

#include "ud_iface.h"
#include "ud_ep.h"
#include "ud_def.h"

#include "ud_verbs.h"

#include "ud_inl.h"

static UCS_F_NOINLINE void
uct_ud_verbs_iface_post_recv_always(uct_ud_verbs_iface_t *iface, int max);

static inline void
uct_ud_verbs_iface_post_recv(uct_ud_verbs_iface_t *iface);

static void uct_ud_verbs_iface_progress_pending(uct_ud_verbs_iface_t *iface);

static UCS_CLASS_INIT_FUNC(uct_ud_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_verbs_iface_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_ud_ep_t, &iface->super);
    self->ah = NULL;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_verbs_ep_t)
{
    ucs_trace_func("");
    if (self->ah) { 
        ibv_destroy_ah(self->ah);
        self->ah = NULL;
    }
}

UCS_CLASS_DEFINE(uct_ud_verbs_ep_t, uct_ud_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_verbs_ep_t, uct_ep_t, uct_iface_h);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_ep_t, uct_ep_t);

static inline void uct_ud_verbs_iface_fill_tx_wr(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep, struct ibv_send_wr *wr, int flags)
{
    if (iface->super.tx.unsignaled >= UCT_UD_TX_MODERATION) {
        wr->send_flags       = (flags|IBV_SEND_SIGNALED);
        iface->super.tx.unsignaled = 0;
    } else {
        wr->send_flags       = flags;
        ++iface->super.tx.unsignaled;
    }
    wr->wr.ud.remote_qpn = ep->super.dest_qpn;
    wr->wr.ud.ah         = ep->ah;
}
    
static inline void uct_ud_verbs_iface_tx_ctl(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep)
{
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    uct_ud_verbs_iface_fill_tx_wr(iface, ep, 
                                  &iface->tx.wr_ctl, IBV_SEND_INLINE);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)iface->tx.sge[0].addr);
    ret = ibv_post_send(iface->super.qp, &iface->tx.wr_ctl, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
    uct_ib_log_post_send(iface->super.qp,  &iface->tx.wr_ctl, NULL);
}

static inline void uct_ud_verbs_iface_tx_inl(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep, const void *buffer, unsigned length)
{
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    iface->tx.sge[1].addr   = (uintptr_t)buffer;
    iface->tx.sge[1].length = length;
    uct_ud_verbs_iface_fill_tx_wr(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)iface->tx.sge[0].addr);
    ret = ibv_post_send(iface->super.qp, &iface->tx.wr_inl, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
    uct_ib_log_post_send(iface->super.qp, &iface->tx.wr_inl, NULL);
}

static inline void uct_ud_verbs_iface_tx_data(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep)
{
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    uct_ud_verbs_iface_fill_tx_wr(iface, ep, 
                                  &iface->tx.wr_bcp, 0);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)iface->tx.sge[0].addr);
    ret = ibv_post_send(iface->super.qp, &iface->tx.wr_bcp, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
    uct_ib_log_post_send(iface->super.qp, &iface->tx.wr_bcp, NULL);
}



static void uct_ud_verbs_iface_progress_pending(uct_ud_verbs_iface_t *iface)
{
    uct_ud_ep_t *ep;
    ucs_status_t status;
    uct_ud_neth_t neth;
    uct_ud_send_skb_t *skb;

    while (!ucs_queue_is_empty(&iface->super.tx.pending_ops)) {
        status = uct_ud_iface_get_next_pending(&iface->super, &ep, &neth, &skb);
        if (status == UCS_ERR_NO_RESOURCE) {
            return;
        }
        if (status == UCS_INPROGRESS) {
            continue;
        }

        if (ucs_unlikely(skb != NULL)) {
            /* TODO: not every skb is inline */
            iface->tx.sge[0].addr   = (uintptr_t) (skb->neth);
            iface->tx.sge[0].length = skb->len;
            uct_ud_verbs_iface_tx_ctl(iface, ucs_derived_of(ep, uct_ud_verbs_ep_t));
            uct_ud_ep_log_tx_tag("PENDING_TX: (skb)", ep, skb->neth, skb->len);
        } 
        else {
            iface->tx.sge[0].addr   = (uintptr_t)&neth;
            iface->tx.sge[0].length = sizeof(neth);
            UCT_UD_EP_HOOK_CALL_TX(ep, &neth);
            uct_ud_verbs_iface_tx_ctl(iface, ucs_derived_of(ep, uct_ud_verbs_ep_t));
            uct_ud_ep_log_tx_tag("PENDING_TX: (neth)", ep, &neth, sizeof(neth));
        }
    }
}

static inline ucs_status_t uct_ud_verbs_am_common(uct_ud_verbs_iface_t *iface,
                                                  uct_ud_verbs_ep_t *ep,
                                                  uint8_t id,
                                                  uct_ud_send_skb_t **skb_p)
{
    uct_ud_send_skb_t *skb;
    uct_ud_neth_t *neth;

    UCT_CHECK_AM_ID(id);

    if (!uct_ud_ep_is_connected(&ep->super)) {
        return UCS_ERR_NO_RESOURCE;
    }

    skb = uct_ud_iface_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }
    VALGRIND_MAKE_MEM_DEFINED(skb, sizeof *skb);

    neth = skb->neth;
    uct_ud_neth_init_data(&ep->super, neth);
    uct_ud_neth_set_type_am(&ep->super, neth, id);
    uct_ud_neth_ack_req(&ep->super, neth);

    iface->tx.sge[0].addr   = (uintptr_t)neth;
    *skb_p = skb;
    return UCS_OK;
}

static ucs_status_t uct_ud_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                              const void *buffer, unsigned length)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_verbs_iface_t);
    uct_ud_send_skb_t *skb;
    uct_ud_am_short_hdr_t *am_hdr;
    ucs_status_t status;

    status = uct_ud_verbs_am_common(iface, ep, id, &skb);
    if (status != UCS_OK) {
        return status;
    }

    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + sizeof(hdr) + length,
                     iface->super.config.max_inline, "am_short");

    am_hdr = (uct_ud_am_short_hdr_t *)(skb->neth+1);
    am_hdr->hdr = hdr;
    iface->tx.sge[0].length = sizeof(uct_ud_neth_t) + sizeof(*am_hdr);

    uct_ud_verbs_iface_tx_inl(iface, ep, buffer, length);
    ucs_trace_data("TX: AM [%d] buf=%p len=%u", id, buffer, length);

    skb->len = iface->tx.sge[0].length;

    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 am_hdr+1, buffer, length);
    return UCS_OK;
}

static ucs_status_t uct_ud_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id, 
                                             uct_pack_callback_t pack_cb,
                                             void *arg, size_t length)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_verbs_iface_t);
    uct_ud_send_skb_t *skb;
    char *data;
    ucs_status_t status;

    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + length, 4096 /* TODO */, "am_bcopy");

    status = uct_ud_verbs_am_common(iface, ep, id, &skb);
    if (status != UCS_OK) {
        return status;
    }

    data = (char *)(skb->neth+1);
    pack_cb(data, arg, length);

    iface->tx.sge[0].lkey               = skb->lkey;
    skb->len  = iface->tx.sge[0].length = sizeof(uct_ud_neth_t) + length;

    uct_ud_verbs_iface_tx_data(iface, ep);
    ucs_trace_data("TX(iface=%p): AM_BCOPY [%d] skb=%p buf=%p len=%u", iface, id, skb, arg, (int)length);

    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    return UCS_OK;
}

static ucs_status_t uct_ud_verbs_ep_put_short(uct_ep_h tl_ep, 
                                              const void *buffer, unsigned length,
                                              uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_verbs_iface_t);
    uct_ud_send_skb_t *skb;
    uct_ud_put_hdr_t *put_hdr;
    uct_ud_neth_t *neth;

    if (!uct_ud_ep_is_connected(&ep->super)) {
        return UCS_ERR_NO_RESOURCE;
    }
    /* TODO: UCT_CHECK_LENGTH(length <= iface->config.max_inline, "put_short"); */
    skb = uct_ud_iface_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    skb = uct_ud_iface_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    neth = skb->neth;
    uct_ud_neth_init_data(&ep->super, neth);
    uct_ud_neth_set_type_put(&ep->super, neth);
    uct_ud_neth_ack_req(&ep->super, neth);
    
    put_hdr = (uct_ud_put_hdr_t *)(neth+1);
    put_hdr->rva = remote_addr;
    iface->tx.sge[0].addr   = (uintptr_t)neth;
    iface->tx.sge[0].length = sizeof(*neth) + sizeof(*put_hdr);

    uct_ud_verbs_iface_tx_inl(iface, ep, buffer, length);
    ucs_trace_data("TX: PUT [%0llx] buf=%p len=%u", (unsigned long long)remote_addr, buffer, length);

    skb->len = iface->tx.sge[0].length;
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 put_hdr+1, buffer, length);
    return UCS_OK;
}


static inline void uct_ud_verbs_iface_poll_tx(uct_ud_verbs_iface_t *iface)
{
    struct ibv_wc wc;
    int ret;

    ret = ibv_poll_cq(iface->super.super.send_cq, 1, &wc);
    if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll send CQ");
        return;
    }

    if (ret == 0) {
        return;
    }

    if (ucs_unlikely(wc.status != IBV_WC_SUCCESS)) {
        ucs_fatal("Send completion (wr_id=0x%0X with error: %s ", (unsigned)wc.wr_id, ibv_wc_status_str(wc.status));
        return;
    }

    iface->super.tx.available += UCT_UD_TX_MODERATION + 1;
}

static inline ucs_status_t uct_ud_verbs_iface_poll_rx(uct_ud_verbs_iface_t *iface)
{
    uct_ib_iface_recv_desc_t *desc;
    struct ibv_wc wc[UCT_IB_MAX_WC];
    int i, ret;
    char *packet;


    ret = ibv_poll_cq(iface->super.super.recv_cq, UCT_IB_MAX_WC, wc);
    if (ret == 0) {
        return UCS_ERR_NO_PROGRESS;
    } 
    if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll receive CQ");
    }

    for (i = 0; i < ret; ++i) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            ucs_fatal("Receive completion with error: %s", ibv_wc_status_str(wc[i].status));
        }

        desc = (void*)wc[i].wr_id;
        ucs_trace_data("pkt rcvd: buf=%p len=%d", desc, wc[i].byte_len);
        packet = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
        VALGRIND_MAKE_MEM_DEFINED(packet, wc[i].byte_len);

        uct_ud_ep_process_rx(&iface->super, 
                             (uct_ud_neth_t *)(packet + UCT_IB_GRH_LEN),
                             wc[i].byte_len - UCT_IB_GRH_LEN,
                             (uct_ud_recv_skb_t *)desc); 
    }
    iface->super.rx.available += ret;
    uct_ud_verbs_iface_post_recv(iface);
    return UCS_OK;
}

static void uct_ud_verbs_iface_progress(void *arg)
{
    uct_ud_verbs_iface_t *iface = arg;
    ucs_status_t status;

    status = uct_ud_verbs_iface_poll_rx(iface);
    if (status == UCS_ERR_NO_PROGRESS) {
        uct_ud_verbs_iface_poll_tx(iface);
    }
    uct_ud_verbs_iface_progress_pending(iface);
}

static ucs_status_t uct_ud_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");
    uct_ud_iface_query(iface, iface_attr);

    return UCS_OK;
}

ucs_status_t uct_ud_verbs_ep_create_connected(uct_iface_h iface_h, const struct sockaddr *addr, uct_ep_h *new_ep_p)
{
    uct_ud_ep_t *ready_ep;
    uct_ud_verbs_ep_t *ep;
    uct_ud_verbs_iface_t *iface = ucs_derived_of(iface_h, uct_ud_verbs_iface_t);
    uct_sockaddr_ib_t *if_addr = (uct_sockaddr_ib_t *)addr;
    uct_ud_send_skb_t *skb;
    struct ibv_ah *ah;
    ucs_status_t status;


    /* check if we can reuse half duplex ep */
    ready_ep = uct_ud_iface_cep_lookup(&iface->super, if_addr, UCT_UD_EP_CONN_ID_MAX);
    if (ready_ep) {
        *new_ep_p = &ready_ep->super.super;
        return UCS_OK;
    }

    status = iface_h->ops.ep_create(iface_h, new_ep_p);
    if (status != UCS_OK) {
        return status;
    }
    ep = ucs_derived_of(*new_ep_p, uct_ud_verbs_ep_t);

    status = uct_ud_ep_connect_to_iface(&ep->super, addr);
    if (status != UCS_OK) {
        return status;
    }
    ucs_assert_always(ep->ah == NULL);
    ah = uct_ib_create_ah(&iface->super.super, if_addr->lid);
    if (ah == NULL) {
        ucs_error("failed to create address handle: %m");
        status = UCS_ERR_INVALID_ADDR;
        goto err1;
    }
    ep->ah = ah;
    
    status = uct_ud_iface_cep_insert(&iface->super, if_addr, &ep->super, UCT_UD_EP_CONN_ID_MAX);
    if (status != UCS_OK) {
        goto err2;
    }

    skb = uct_ud_ep_prepare_creq(&ep->super);
    if (!skb) {
        status = UCS_ERR_NO_RESOURCE;
        goto err3;
    }

    iface->tx.sge[0].addr   = (uintptr_t)skb->neth;
    iface->tx.sge[0].length = skb->len;
    uct_ud_verbs_iface_tx_ctl(iface, ep);
    ucs_trace_data("TX: CREQ (qp=%x lid=%d)", if_addr->qp_num, if_addr->lid);
    return UCS_OK;

err3:
    uct_ud_iface_cep_rollback(&iface->super, if_addr, &ep->super);
err2:
    ibv_destroy_ah(ep->ah);
    ep->ah = NULL;
err1:
    uct_ud_ep_disconnect_from_iface(*new_ep_p);
    *new_ep_p = NULL;
    return status;
}


ucs_status_t uct_ud_verbs_ep_connect_to_ep(uct_ep_h tl_ep,
                                           const struct sockaddr *addr)
{
    ucs_status_t status;
    struct ibv_ah *ah;
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ib_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    uct_sockaddr_ib_t *if_addr = (uct_sockaddr_ib_t *)addr;

    status = uct_ud_ep_connect_to_ep(&ep->super, addr);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert_always(ep->ah == NULL);
    ah = uct_ib_create_ah(iface, if_addr->lid);
    if (ah == NULL) {
        ucs_error("failed to create address handle: %m");
        return UCS_ERR_INVALID_ADDR;
    }
    ep->ah = ah;
    return UCS_OK;
}


static void UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_iface_t)(uct_iface_t*);

uct_iface_ops_t uct_ud_verbs_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_iface_t),
    .iface_flush         = uct_ud_iface_flush,
    .iface_release_am_desc=uct_ib_iface_release_am_desc,
    .ep_get_address      = uct_ud_ep_get_address,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_ud_verbs_ep_t),
    .ep_create_connected = uct_ud_verbs_ep_create_connected,
    .ep_connect_to_ep    = uct_ud_verbs_ep_connect_to_ep, 
    .iface_get_address   = uct_ud_iface_get_address,
    .iface_is_reachable  = uct_ib_iface_is_reachable,
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_ep_t),
    .iface_query         = uct_ud_verbs_iface_query,
    .ep_put_short        = uct_ud_verbs_ep_put_short,
    .ep_am_short         = uct_ud_verbs_ep_am_short,
    .ep_am_bcopy         = uct_ud_verbs_ep_am_bcopy,
    .ep_pending_add      = (void*)ucs_empty_function_return_success, /* TODO */
    .ep_pending_purge    = (void*)ucs_empty_function_return_success,
    .ep_flush            = uct_ud_ep_flush
};

static UCS_F_NOINLINE void
uct_ud_verbs_iface_post_recv_always(uct_ud_verbs_iface_t *iface, int max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = ucs_alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super.super,
                                        iface->super.rx.mp, wrs, max);
    if (count == 0) {
        return;
    }

    ret = ibv_post_recv(iface->super.qp, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_recv() returned %d: %m", ret);
    }
    iface->super.rx.available -= count;
}

static inline void
uct_ud_verbs_iface_post_recv(uct_ud_verbs_iface_t *iface)
{
    unsigned batch = iface->super.config.rx_max_batch;

    if (iface->super.rx.available < batch) 
        return;
    
    uct_ud_verbs_iface_post_recv_always(iface, batch);
}

static UCS_CLASS_INIT_FUNC(uct_ud_verbs_iface_t, uct_pd_h pd, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           const uct_iface_config_t *tl_config)
{
    uct_ud_iface_config_t *config = ucs_derived_of(tl_config, uct_ud_iface_config_t);
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_verbs_iface_ops, pd,
                              worker, dev_name, rx_headroom, 0, config);

    while (self->super.rx.available >= self->super.config.rx_max_batch) {
        uct_ud_verbs_iface_post_recv(self);
    }
    
    memset(&self->tx.wr_inl, 0, sizeof(self->tx.wr_inl));
    self->tx.wr_inl.opcode            = IBV_WR_SEND;
    self->tx.wr_inl.wr_id             = 0xBEEBBEEB;
    self->tx.wr_inl.wr.ud.remote_qkey = UCT_UD_QKEY;
    self->tx.wr_inl.imm_data          = 0;
    self->tx.wr_inl.next              = 0;
    self->tx.wr_inl.sg_list           = self->tx.sge;
    self->tx.wr_inl.num_sge           = UCT_UD_MAX_SGE;

    memset(&self->tx.wr_bcp, 0, sizeof(self->tx.wr_bcp));
    self->tx.wr_bcp.opcode            = IBV_WR_SEND;
    self->tx.wr_bcp.wr_id             = 0xFAAFFAAF;
    self->tx.wr_bcp.wr.ud.remote_qkey = UCT_UD_QKEY;
    self->tx.wr_bcp.imm_data          = 0;
    self->tx.wr_bcp.next              = 0;
    self->tx.wr_bcp.sg_list           = self->tx.sge;
    self->tx.wr_bcp.num_sge           = 1;

    memset(&self->tx.wr_ctl, 0, sizeof(self->tx.wr_ctl));
    self->tx.wr_ctl.opcode            = IBV_WR_SEND;
    self->tx.wr_ctl.wr_id             = 0xCCCCCCCC;
    self->tx.wr_ctl.wr.ud.remote_qkey = UCT_UD_QKEY;
    self->tx.wr_ctl.imm_data          = 0;
    self->tx.wr_ctl.next              = 0;
    self->tx.wr_ctl.sg_list           = self->tx.sge;
    self->tx.wr_ctl.num_sge           = 1;
    /* TODO: add progress on first ep creation */
    ucs_notifier_chain_add(&worker->progress_chain, uct_ud_verbs_iface_progress,
                           self);
    return UCS_OK;
}


static UCS_CLASS_CLEANUP_FUNC(uct_ud_verbs_iface_t)
{
    ucs_trace_func("");
    ucs_notifier_chain_remove(&self->super.super.super.worker->progress_chain,
                              uct_ud_verbs_iface_progress, self);
}

UCS_CLASS_DEFINE(uct_ud_verbs_iface_t, uct_ud_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_verbs_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char*, size_t,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_iface_t, uct_iface_t);

static ucs_status_t uct_ud_verbs_query_resources(uct_pd_h pd,
                                                 uct_tl_resource_desc_t **resources_p,
                                                 unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(ucs_derived_of(pd, uct_ib_device_t),
                                            "ud",
                                            0,
                                            UCT_IB_DETH_LEN + sizeof(uct_ud_neth_t),
                                            80,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_ud_verbs_tl,
                        uct_ud_verbs_query_resources,
                        uct_ud_verbs_iface_t,
                        "ud",
                        "UD_VERBS_",
                        uct_ud_iface_config_table,
                        uct_ud_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pd, &uct_ud_verbs_tl);
