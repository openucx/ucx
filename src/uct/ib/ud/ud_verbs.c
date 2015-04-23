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
    }
}

UCS_CLASS_DEFINE(uct_ud_verbs_ep_t, uct_ud_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_verbs_ep_t, uct_ep_t, uct_iface_h);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_ep_t, uct_ep_t);

static inline void uct_ud_verbs_iface_fill_tx_wr(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep, struct ibv_send_wr *wr)
{
    if (iface->super.tx.unsignaled >= UCT_UD_TX_MODERATION) {
        wr->send_flags       = IBV_SEND_INLINE|IBV_SEND_SIGNALED;
        iface->super.tx.unsignaled = 0;
    } else {
        wr->send_flags       = IBV_SEND_INLINE;
        ++iface->super.tx.unsignaled;
    }
    wr->wr.ud.remote_qpn = ep->super.dest_qpn;
    wr->wr.ud.ah         = ep->ah;
}
    

static void uct_ud_verbs_iface_progress_pending(uct_ud_verbs_iface_t *iface)
{
    uct_ud_ep_t *ep;
    ucs_status_t status;
    uct_ud_neth_t neth;
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    while (!ucs_queue_is_empty(&iface->super.tx.pending_ops)) {
        status = uct_ud_iface_get_next_pending(&iface->super, &ep, &neth);
        if (status == UCS_ERR_NO_RESOURCE) {
            return;
        }
        if (status == UCS_INPROGRESS) {
            continue;
        }

        iface->tx.sge[0].addr   = (uintptr_t)&neth;
        iface->tx.sge[0].length = sizeof(neth);
        uct_ud_verbs_iface_fill_tx_wr(iface, 
                                      ucs_derived_of(ep, uct_ud_verbs_ep_t),
                                      &iface->tx.ctl_wr);
        UCT_UD_EP_HOOK_CALL_TX(ep, &neth);
        ret = ibv_post_send(iface->super.qp, &iface->tx.ctl_wr, &bad_wr);
        ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
    }
}

static inline void uct_ud_verbs_iface_tx_data(uct_ud_verbs_iface_t *iface, uct_ud_verbs_ep_t *ep, const void *buffer, unsigned length)
{
    int UCS_V_UNUSED ret;
    struct ibv_send_wr *bad_wr;

    iface->tx.sge[1].addr   = (uintptr_t)buffer;
    iface->tx.sge[1].length = length;
    uct_ud_verbs_iface_fill_tx_wr(iface, ep, &iface->tx.wr);
    UCT_UD_EP_HOOK_CALL_TX(&ep->super, (uct_ud_neth_t *)iface->tx.sge[0].addr);
    ret = ibv_post_send(iface->super.qp, &iface->tx.wr, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);
}

static ucs_status_t uct_ud_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                              const void *buffer, unsigned length)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_verbs_iface_t);
    uct_ud_send_skb_t *skb;
    uct_ud_am_short_hdr_t *am_hdr;
    uct_ud_neth_t *neth;

    UCT_CHECK_AM_ID(id);
    /* TODO: UCT_CHECK_LENGTH(sizeof(am_hdr) + length <= iface->super.config.max_inline, "am_short"); */
    skb = uct_ud_iface_get_tx_skb(&iface->super, &ep->super);
    if (!skb) {
        return UCS_ERR_NO_RESOURCE;
    }

    neth = skb->neth;
    uct_ud_neth_init_data(&ep->super, neth);
    uct_ud_neth_set_type_am(&ep->super, neth, id);
    uct_ud_neth_ack_req(&ep->super, neth);
    
    am_hdr = (uct_ud_am_short_hdr_t *)(neth+1);
    am_hdr->hdr = hdr;
    iface->tx.sge[0].addr   = (uintptr_t)neth;
    iface->tx.sge[0].length = sizeof(*neth) + sizeof(*am_hdr);

    uct_ud_verbs_iface_tx_data(iface, ep, buffer, length);
    ucs_trace_data("TX: AM [%d] buf=%p len=%u", id, buffer, length);

    uct_ud_iface_complete_tx(&iface->super, &ep->super, skb,
                             am_hdr+1, buffer, length);
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

    /* TODO: UCT_CHECK_LENGTH(length <= iface->config.max_inline, "put_short"); */

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

    ucs_trace_data("TX: neth->type=0x%08x", neth->packet_type);

    uct_ud_verbs_iface_tx_data(iface, ep, buffer, length);
    ucs_trace_data("TX: PUT [%0llx] buf=%p len=%u", (unsigned long long)remote_addr, buffer, length);

    uct_ud_iface_complete_tx(&iface->super, &ep->super, skb,
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
        ucs_fatal("Send completion with error: %s", ibv_wc_status_str(wc.status));
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

ucs_status_t uct_ud_verbs_ep_connect_to_ep(uct_ep_h tl_ep, const struct sockaddr *addr)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ib_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ib_iface_t);
    const uct_sockaddr_ib_t *ib_addr = (uct_sockaddr_ib_t *)addr;
    struct ibv_ah_attr ah_attr;
    ucs_status_t status;
    struct ibv_ah *ah;

    status = uct_ud_ep_connect_to_ep(&ep->super, addr);
    if (status != UCS_OK) {
        return status;
    }

    memset(&ah_attr, 0, sizeof(ah_attr));
    ah_attr.port_num  = iface->port_num;
    ah_attr.sl        = 0; /* TODO: sl */
    ah_attr.is_global = 0;
    ah_attr.dlid      = ib_addr->lid;

    ah = ibv_create_ah(uct_ib_iface_device(iface)->pd, &ah_attr);
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
    .ep_connect_to_ep    = uct_ud_verbs_ep_connect_to_ep, 
    .iface_get_address   = uct_ib_iface_get_subnet_address,
    .iface_is_reachable  = uct_ib_iface_is_reachable,
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_ep_t),
    .iface_query         = uct_ud_verbs_iface_query,
    .ep_put_short        = uct_ud_verbs_ep_put_short,
    .ep_am_short         = uct_ud_verbs_ep_am_short,
};

static UCS_F_NOINLINE void
uct_ud_verbs_iface_post_recv_always(uct_ud_verbs_iface_t *iface, int max)
{
    struct ibv_recv_wr *bad_wr;
    uct_ib_recv_wr_t *wrs;
    unsigned count;
    int ret;

    wrs  = alloca(sizeof *wrs  * max);

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

    uct_ud_verbs_iface_post_recv_always(self, self->super.rx.available);
    
    memset(&self->tx.wr, 0, sizeof(self->tx.wr));
    self->tx.wr.opcode            = IBV_WR_SEND;
    self->tx.wr.wr_id             = 0;
    self->tx.wr.wr.ud.remote_qkey = UCT_UD_QKEY;
    self->tx.wr.imm_data          = 0;
    self->tx.wr.next              = 0;
    self->tx.wr.sg_list           = self->tx.sge;
    self->tx.wr.num_sge           = UCT_UD_MAX_SGE;

    memset(&self->tx.ctl_wr, 0, sizeof(self->tx.ctl_wr));
    self->tx.ctl_wr.opcode            = IBV_WR_SEND;
    self->tx.ctl_wr.wr_id             = 0;
    self->tx.ctl_wr.wr.ud.remote_qkey = UCT_UD_QKEY;
    self->tx.ctl_wr.imm_data          = 0;
    self->tx.ctl_wr.next              = 0;
    self->tx.ctl_wr.sg_list           = self->tx.sge;
    self->tx.ctl_wr.num_sge           = 1;
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

UCT_TL_COMPONENT_DEFINE(&uct_ib_pd, uct_ud_verbs_tl,
                        uct_ud_verbs_query_resources,
                        uct_ud_verbs_iface_t,
                        "ud",
                        "UD_VERBS_",
                        uct_ud_iface_config_table,
                        uct_ud_iface_config_t);
