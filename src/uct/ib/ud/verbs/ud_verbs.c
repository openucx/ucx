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

#include <uct/ib/base/ib_log.h>

#include <uct/ib/ud/base/ud_iface.h>
#include <uct/ib/ud/base/ud_ep.h>
#include <uct/ib/ud/base/ud_def.h>

#include "ud_verbs.h"

#include <uct/ib/ud/base/ud_inl.h>

static UCS_F_NOINLINE void
uct_ud_verbs_iface_post_recv_always(uct_ud_verbs_iface_t *iface, int max);

static inline void
uct_ud_verbs_iface_post_recv(uct_ud_verbs_iface_t *iface);

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
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_ep_t, uct_ep_t);

static inline void 
uct_ud_verbs_iface_fill_tx_wr(uct_ud_verbs_iface_t *iface,
                              uct_ud_verbs_ep_t *ep,
                              struct ibv_send_wr *wr, unsigned flags)
{
    if (iface->super.tx.unsignaled >= UCT_UD_TX_MODERATION) {
        wr->send_flags       = (flags|IBV_SEND_SIGNALED);
        iface->super.tx.unsignaled = 0;
    } else {
        wr->send_flags       = flags;
        ++iface->super.tx.unsignaled;
    }
    wr->wr.ud.remote_qpn = ep->dest_qpn;
    wr->wr.ud.ah         = ep->ah;
}
    
static inline void
uct_ud_verbs_ep_tx_inlv(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep,
                        const void *buffer, unsigned length)
{
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    iface->tx.sge[1].addr   = (uintptr_t)buffer;
    iface->tx.sge[1].length = length;
    uct_ud_verbs_iface_fill_tx_wr(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)iface->tx.sge[0].addr);
    ret = ibv_post_send(iface->super.qp, &iface->tx.wr_inl, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
    uct_ib_log_post_send(&iface->super.super, iface->super.qp, &iface->tx.wr_inl, NULL);
    --iface->super.tx.available;
}

static inline void 
uct_ud_verbs_ep_tx_skb(uct_ud_verbs_iface_t *iface,
                          uct_ud_verbs_ep_t *ep, uct_ud_send_skb_t *skb, unsigned flags)
{
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    iface->tx.sge[0].lkey   = skb->lkey;
    iface->tx.sge[0].length = skb->len;
    iface->tx.sge[0].addr   = (uintptr_t)skb->neth;
    uct_ud_verbs_iface_fill_tx_wr(iface, ep, &iface->tx.wr_skb, flags);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)iface->tx.sge[0].addr);
    ret = ibv_post_send(iface->super.qp, &iface->tx.wr_skb, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
    uct_ib_log_post_send(&iface->super.super, iface->super.qp, &iface->tx.wr_skb, NULL);
    --iface->super.tx.available;
}

static void 
uct_ud_verbs_ep_tx_ctl_skb(uct_ud_iface_t *ud_iface, uct_ud_ep_t *ud_ep,
                           uct_ud_send_skb_t *skb)
{
    uct_ud_verbs_iface_t *iface = ucs_derived_of(ud_iface,
                                                 uct_ud_verbs_iface_t);
    uct_ud_verbs_ep_t *ep = ucs_derived_of(ud_ep, uct_ud_verbs_ep_t);
    unsigned flags = 0;

    if (skb->len < iface->super.config.max_inline) {
        flags = IBV_SEND_INLINE;
    }
    uct_ud_verbs_ep_tx_skb(iface, ep, skb, flags);
}

static
ucs_status_t uct_ud_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_ud_verbs_iface_t);
    uct_ud_send_skb_t *skb;
    uct_ud_am_short_hdr_t *am_hdr;
    ucs_status_t status;

    UCT_CHECK_LENGTH(sizeof(uct_ud_neth_t) + sizeof(hdr) + length,
                     iface->super.config.max_inline, "am_short");

    uct_ud_enter(&iface->super);
    uct_ud_iface_progress_pending_tx(&iface->super);
    status = uct_ud_am_common(&iface->super, &ep->super, id, &skb);
    if (status != UCS_OK) {
        uct_ud_leave(&iface->super);
        return status;
    }

    am_hdr = (uct_ud_am_short_hdr_t *)(skb->neth+1);
    am_hdr->hdr = hdr;
    iface->tx.sge[0].length = sizeof(uct_ud_neth_t) + sizeof(*am_hdr);
    iface->tx.sge[0].addr   = (uintptr_t)skb->neth;

    uct_ud_verbs_ep_tx_inlv(iface, ep, buffer, length);
    ucs_trace_data("TX: AM [%d] buf=%p len=%u", id, buffer, length);

    skb->len = iface->tx.sge[0].length;

    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 am_hdr+1, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_ud_leave(&iface->super);
    return UCS_OK;
}

static ssize_t uct_ud_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                        uct_pack_callback_t pack_cb, void *arg)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_ud_verbs_iface_t);
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

    uct_ud_verbs_ep_tx_skb(iface, ep, skb, 0);
    ucs_trace_data("TX(iface=%p): AM_BCOPY [%d] skb=%p buf=%p len=%u", iface, id,
                   skb, arg, skb->len);

    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_ud_leave(&iface->super);
    return length;
}

static
ucs_status_t uct_ud_verbs_ep_put_short(uct_ep_h tl_ep, 
                                       const void *buffer, unsigned length,
                                       uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_ud_verbs_iface_t);
    uct_ud_send_skb_t *skb;
    uct_ud_put_hdr_t *put_hdr;
    uct_ud_neth_t *neth;

    /* TODO: UCT_CHECK_LENGTH(length <= iface->config.max_inline, "put_short"); */
    uct_ud_enter(&iface->super);
    uct_ud_iface_progress_pending_tx(&iface->super);
    skb = uct_ud_ep_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        uct_ud_leave(&iface->super);
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

    uct_ud_verbs_ep_tx_inlv(iface, ep, buffer, length);
    ucs_trace_data("TX: PUT [%0llx] buf=%p len=%u", 
                   (unsigned long long)remote_addr, buffer, length);

    skb->len = iface->tx.sge[0].length;
    uct_ud_iface_complete_tx_inl(&iface->super, &ep->super, skb,
                                 put_hdr+1, buffer, length);
    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_ud_leave(&iface->super);
    return UCS_OK;
}


static UCS_F_ALWAYS_INLINE void 
uct_ud_verbs_iface_poll_tx(uct_ud_verbs_iface_t *iface)
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
        ucs_fatal("Send completion (wr_id=0x%0X with error: %s ",
                  (unsigned)wc.wr_id, ibv_wc_status_str(wc.status));
        return;
    }

    iface->super.tx.available += UCT_UD_TX_MODERATION + 1;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_ud_verbs_iface_poll_rx(uct_ud_verbs_iface_t *iface, int is_async)
{
    uct_ib_iface_recv_desc_t *desc;
    struct ibv_wc wc[UCT_IB_MAX_WC];
    int i, ret;
    char *packet;
    ucs_status_t status;


    ret = ibv_poll_cq(iface->super.super.recv_cq, UCT_IB_MAX_WC, wc);
    if (ret == 0) {
        status = UCS_ERR_NO_PROGRESS;
        goto out;
    } 
    if (ucs_unlikely(ret < 0)) {
        ucs_fatal("Failed to poll receive CQ");
    }

    for (i = 0; i < ret; ++i) {
        if (ucs_unlikely(wc[i].status != IBV_WC_SUCCESS)) {
            ucs_fatal("Receive completion with error: %s", 
                      ibv_wc_status_str(wc[i].status));
        }

        desc = (void*)wc[i].wr_id;
        ucs_trace_data("pkt rcvd: buf=%p len=%d", desc, wc[i].byte_len);
        packet = uct_ib_iface_recv_desc_hdr(&iface->super.super, desc);
        VALGRIND_MAKE_MEM_DEFINED(packet, wc[i].byte_len);

        uct_ud_ep_process_rx(&iface->super, 
                             (uct_ud_neth_t *)(packet + UCT_IB_GRH_LEN),
                             wc[i].byte_len - UCT_IB_GRH_LEN,
                             (uct_ud_recv_skb_t *)desc, is_async); 
    }
    iface->super.rx.available += ret;
    status = UCS_OK;
out:
    uct_ud_verbs_iface_post_recv(iface);
    return status;
}

static void uct_ud_verbs_iface_async_progress(void *arg)
{
    uct_ud_verbs_iface_t *iface = arg;
    ucs_status_t status;
    do {
        status = uct_ud_verbs_iface_poll_rx(iface, 1);
    } while (status == UCS_OK);
    uct_ud_verbs_iface_poll_tx(iface);
    uct_ud_iface_progress_pending(&iface->super, 1);
}

static void uct_ud_verbs_iface_progress(void *arg)
{
    uct_ud_verbs_iface_t *iface = arg;
    ucs_status_t status;

    uct_ud_enter(&iface->super);
    status = uct_ud_iface_dispatch_pending_rx(&iface->super);
    if (status == UCS_OK) {
        status = uct_ud_verbs_iface_poll_rx(iface, 0);
        if (status == UCS_ERR_NO_PROGRESS) {
            uct_ud_verbs_iface_poll_tx(iface);
        }
    }
    uct_ud_iface_progress_pending(&iface->super, 0);
    uct_ud_leave(&iface->super);
}

static ucs_status_t 
uct_ud_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");
    uct_ud_iface_query(iface, iface_attr);

    return UCS_OK;
}

ucs_status_t
uct_ud_verbs_ep_create_connected(uct_iface_h iface_h, 
                                 const uct_iface_addr_t *addr, uct_ep_h *new_ep_p)
{
    uct_ud_verbs_iface_t *iface = ucs_derived_of(iface_h, uct_ud_verbs_iface_t);
    uct_ud_verbs_ep_t *ep;
    uct_ud_ep_t *new_ud_ep; 
    const uct_sockaddr_ib_t *if_addr = (const uct_sockaddr_ib_t *)addr;
    uct_ud_send_skb_t *skb;
    struct ibv_ah *ah;
    ucs_status_t status;

    uct_ud_enter(&iface->super);
    status = uct_ud_ep_create_connected_common(&iface->super, if_addr,
                                               &new_ud_ep, &skb);
    if (status != UCS_OK) {
        return status;
    }
    ep = ucs_derived_of(new_ud_ep, uct_ud_verbs_ep_t);
    *new_ep_p = &ep->super.super.super;
    if (skb == NULL) {
        uct_ud_leave(&iface->super);
        return UCS_OK;
    }

    ucs_assert_always(ep->ah == NULL);
    ah = uct_ib_create_ah(&iface->super.super, if_addr->lid);
    if (ah == NULL) {
        ucs_error("failed to create address handle: %m");
        status = UCS_ERR_INVALID_ADDR;
        goto err;
    }
    ep->ah = ah;
    ep->dest_qpn = if_addr->qp_num;

    ucs_trace_data("TX: CREQ (qp=%x lid=%d)", if_addr->qp_num, if_addr->lid);
    uct_ud_verbs_ep_tx_skb(iface, ep, skb, IBV_SEND_INLINE);
    uct_ud_iface_complete_tx_skb(&iface->super, &ep->super, skb);
    uct_ud_leave(&iface->super);
    return UCS_OK;

err:
    uct_ud_ep_destroy_connected(&ep->super, if_addr);
    uct_ud_leave(&iface->super);
    *new_ep_p = NULL;
    return status;
}


ucs_status_t uct_ud_verbs_ep_connect_to_ep(uct_ep_h tl_ep,
                                           const uct_ep_addr_t *addr)
{
    ucs_status_t status;
    struct ibv_ah *ah;
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ib_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    const uct_sockaddr_ib_t *if_addr = (const uct_sockaddr_ib_t *)addr;

    status = uct_ud_ep_connect_to_ep(&ep->super, addr);
    if (status != UCS_OK) {
        return status;
    }
    ucs_assert_always(ep->ah == NULL);
    ep->dest_qpn = if_addr->qp_num;
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
    .iface_close           = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_iface_t),
    .iface_flush           = uct_ud_iface_flush,
    .iface_release_am_desc = uct_ud_iface_release_am_desc,
    .iface_get_address     = uct_ud_iface_get_address,
    .iface_is_reachable    = uct_ib_iface_is_reachable,
    .iface_query           = uct_ud_verbs_iface_query,

    .ep_create             = UCS_CLASS_NEW_FUNC_NAME(uct_ud_verbs_ep_t),
    .ep_destroy            = uct_ud_ep_disconnect,
    .ep_get_address        = uct_ud_ep_get_address,

    .ep_create_connected   = uct_ud_verbs_ep_create_connected,
    .ep_connect_to_ep      = uct_ud_verbs_ep_connect_to_ep, 

    .ep_put_short          = uct_ud_verbs_ep_put_short,
    .ep_am_short           = uct_ud_verbs_ep_am_short,
    .ep_am_bcopy           = uct_ud_verbs_ep_am_bcopy,

    .ep_pending_add        = uct_ud_ep_pending_add,
    .ep_pending_purge      = uct_ud_ep_pending_purge,

    .ep_flush              = uct_ud_ep_flush
};

static UCS_F_NOINLINE void
uct_ud_verbs_iface_post_recv_always(uct_ud_verbs_iface_t *iface, int max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = ucs_alloca(sizeof *wrs  * max);

    count = uct_ib_iface_prepare_rx_wrs(&iface->super.super, &iface->super.rx.mp,
                                        wrs, max);
    if (count == 0) {
        return;
    }

    ret = ibv_post_recv(iface->super.qp, &wrs[0].ibwr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_recv() returned %d: %m", ret);
    }
    iface->super.rx.available -= count;
}

static UCS_F_ALWAYS_INLINE void
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
    uct_ud_iface_config_t *config = ucs_derived_of(tl_config,
                                                   uct_ud_iface_config_t);
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_verbs_iface_ops, pd,
                              worker, dev_name, rx_headroom, 0, config);

    self->super.ops.async_progress = uct_ud_verbs_iface_async_progress;
    self->super.ops.tx_skb         = uct_ud_verbs_ep_tx_ctl_skb;

    if (self->super.config.rx_max_batch < UCT_IB_MAX_WC) {
        ucs_warn("max batch is too low (%d < %d), performance may be impacted",
                self->super.config.rx_max_batch,
                UCT_IB_MAX_WC);
    }

    while (self->super.rx.available >= self->super.config.rx_max_batch) {
        uct_ud_verbs_iface_post_recv(self);
    }
    
    memset(&self->tx.wr_inl, 0, sizeof(self->tx.wr_inl));
    self->tx.wr_inl.opcode            = IBV_WR_SEND;
    self->tx.wr_inl.wr_id             = 0xBEEBBEEB;
    self->tx.wr_inl.wr.ud.remote_qkey = UCT_IB_QKEY;
    self->tx.wr_inl.imm_data          = 0;
    self->tx.wr_inl.next              = 0;
    self->tx.wr_inl.sg_list           = self->tx.sge;
    self->tx.wr_inl.num_sge           = UCT_UD_MAX_SGE;

    memset(&self->tx.wr_skb, 0, sizeof(self->tx.wr_skb));
    self->tx.wr_skb.opcode            = IBV_WR_SEND;
    self->tx.wr_skb.wr_id             = 0xFAAFFAAF;
    self->tx.wr_skb.wr.ud.remote_qkey = UCT_IB_QKEY;
    self->tx.wr_skb.imm_data          = 0;
    self->tx.wr_skb.next              = 0;
    self->tx.wr_skb.sg_list           = self->tx.sge;
    self->tx.wr_skb.num_sge           = 1;

    uct_ud_iface_complete_init(&self->super, uct_ud_verbs_iface_progress);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_verbs_iface_t)
{
    ucs_trace_func("");
    uct_ud_enter(&self->super);
    uct_worker_progress_unregister(self->super.super.super.worker,
                                   uct_ud_verbs_iface_progress, self);
    UCT_UD_IFACE_DELETE_EPS(&self->super, uct_ud_verbs_ep_t);
    uct_ud_leave(&self->super);
}

UCS_CLASS_DEFINE(uct_ud_verbs_iface_t, uct_ud_iface_t);

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_verbs_iface_t, uct_iface_t, uct_pd_h,
                                 uct_worker_h, const char*, size_t,
                                 const uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_iface_t, uct_iface_t);

static
ucs_status_t uct_ud_verbs_query_resources(uct_pd_h pd,
                                          uct_tl_resource_desc_t **resources_p,
                                          unsigned *num_resources_p)
{
    return uct_ib_device_query_tl_resources(&ucs_derived_of(pd, uct_ib_pd_t)->dev,
                                            "ud", 0,
                                            resources_p, num_resources_p);
}

UCT_TL_COMPONENT_DEFINE(uct_ud_verbs_tl,
                        uct_ud_verbs_query_resources,
                        uct_ud_verbs_iface_t,
                        "ud",
                        "UD_VERBS_",
                        uct_ud_iface_config_table,
                        uct_ud_iface_config_t);
UCT_PD_REGISTER_TL(&uct_ib_pdc, &uct_ud_verbs_tl);
