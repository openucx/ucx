/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <uct/api/uct.h>
#include <uct/ib/base/ib_context.h>
#include <uct/ib/base/ib_iface.h>
#include <uct/tl/context.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <string.h>
#include <arpa/inet.h> /* For htonl */

#include "ud_iface.h"
#include "ud_ep.h"
#include "ud_def.h"

#include "ud_verbs.h"

static UCS_F_NOINLINE void
uct_ud_verbs_iface_post_recv_always(uct_ud_verbs_iface_t *iface, int max);

static inline void
uct_ud_verbs_iface_post_recv(uct_ud_verbs_iface_t *iface);

static ucs_status_t uct_ud_verbs_query_resources(uct_context_h context,
        uct_resource_desc_t **resources_p,
        unsigned *num_resources_p)
{
    ucs_trace_func("");
    /* TODO take transport overhead into account */
    return uct_ib_query_resources(context, 0,
                                  UCT_IB_DETH_LEN + sizeof(uct_ud_neth_t),
                                  80,
                                  resources_p, num_resources_p);
}

static UCS_CLASS_INIT_FUNC(uct_ud_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_verbs_iface_t);
    ucs_trace_func("");
    UCS_CLASS_CALL_SUPER_INIT(uct_ud_ep_t, &iface->super);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ud_verbs_ep_t)
{
    ucs_trace_func("");
}

UCS_CLASS_DEFINE(uct_ud_verbs_ep_t, uct_ud_ep_t);
static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_verbs_ep_t, uct_ep_t, uct_iface_h);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_ep_t, uct_ep_t);


static ucs_status_t uct_ud_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                              void *buffer, unsigned length)
{
    uct_ud_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_ud_verbs_ep_t);
    uct_ud_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ud_verbs_iface_t);
    struct ibv_send_wr *bad_wr;
    int UCS_V_UNUSED ret;
    uct_ud_send_skb_t *skb;
    uct_ud_am_short_hdr_t *am_hdr;
    uct_ud_neth_t *neth;

    /* TODO:
          check tx/congestion window
          move common code to inline functions
     */
    if (iface->super.tx.available == 0) {
        ucs_trace_data("iface=%p out of tx wqe", iface);
        return UCS_ERR_NO_RESOURCE;
    }

    skb = ucs_mpool_get(iface->super.tx.mp);
    if (!skb) {
        ucs_trace_data("iface=%p out of tx skbs", iface);
        return UCS_ERR_NO_RESOURCE;
    }

    if (ep->super.tx.psn == ep->super.tx.max_psn) {
        ucs_trace_data("iface=%p ep=%p (%d->%d) tx window full (max_psn=%u)",
                       iface, ep, ep->super.ep_id, ep->super.dest_ep_id, (unsigned)ep->super.tx.max_psn);
        return UCS_ERR_NO_RESOURCE;
    } 
    neth = skb->neth;
    neth->psn = ep->super.tx.psn++;
    neth->ack_psn = ep->super.rx.acked_psn = ucs_frag_list_sn(&ep->super.rx.ooo_pkts);
    uct_ud_neth_set_type_am(neth, &ep->super, id);
    
    am_hdr = (uct_ud_am_short_hdr_t *)(neth+1);
    am_hdr->hdr = hdr;

    if (iface->super.tx.unsignaled >= UCT_UD_TX_MODERATION) {
        iface->tx.wr.send_flags       = IBV_SEND_INLINE|IBV_SEND_SIGNALED;
        iface->super.tx.unsignaled = 0;
    } else {
        iface->tx.wr.send_flags       = IBV_SEND_INLINE;
        ++iface->super.tx.unsignaled;
    }

    iface->tx.wr.wr.ud.remote_qpn = ep->super.dest_qpn;
    iface->tx.wr.wr.ud.ah         = ep->super.ah;

    iface->tx.sge[0].addr   = (uintptr_t)neth;
    iface->tx.sge[0].length = sizeof(*neth) + sizeof(*am_hdr);
    iface->tx.sge[1].addr   = (uintptr_t)buffer;
    iface->tx.sge[1].length = length;

    ret = ibv_post_send(iface->super.qp, &iface->tx.wr, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);

    ucs_trace_data("TX: AM [%d] buf=%p len=%u", id, buffer, length);
    --iface->super.tx.available;

    skb->len = length;
    memcpy((char *)(am_hdr+1), buffer, length);
    ucs_queue_push(&ep->super.tx.window, &skb->queue);

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
}

static ucs_status_t uct_ud_verbs_iface_query(uct_iface_h tl_iface, uct_iface_attr_t *iface_attr)
{
    uct_ud_iface_t *iface = ucs_derived_of(tl_iface, uct_ud_iface_t);

    ucs_trace_func("");
    uct_ud_iface_query(iface, iface_attr);

    return UCS_OK;
}

static void UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_iface_t)(uct_iface_t*);

uct_iface_ops_t uct_ud_verbs_iface_ops = {
    .iface_close         = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_iface_t),
    .iface_get_address   = uct_ud_iface_get_address,
    .iface_flush         = uct_ud_iface_flush,
    .ep_get_address      = uct_ud_ep_get_address,
    .ep_connect_to_iface = NULL,
    .ep_connect_to_ep    = uct_ud_ep_connect_to_ep, 
    .iface_query         = uct_ud_verbs_iface_query,
    .ep_put_short        = NULL,
    .ep_am_short         = uct_ud_verbs_ep_am_short,
    .ep_create           = UCS_CLASS_NEW_FUNC_NAME(uct_ud_verbs_ep_t),
    .ep_destroy          = UCS_CLASS_DELETE_FUNC_NAME(uct_ud_verbs_ep_t),
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

static UCS_CLASS_INIT_FUNC(uct_ud_verbs_iface_t, uct_worker_h worker,
                           const char *dev_name, size_t rx_headroom,
                           uct_iface_config_t *tl_config)
{
    uct_ud_iface_config_t *config = ucs_derived_of(tl_config, uct_ud_iface_config_t);
    ucs_trace_func("");

    UCS_CLASS_CALL_SUPER_INIT(uct_ud_iface_t, &uct_ud_verbs_iface_ops, worker,
                              dev_name, rx_headroom, 0, config);

    uct_ud_verbs_iface_post_recv_always(self, self->super.rx.available);
    
    memset(&self->tx.wr, 0, sizeof(self->tx.wr));
    self->tx.wr.opcode            = IBV_WR_SEND;
    self->tx.wr.wr_id             = 0;
    self->tx.wr.wr.ud.remote_qkey = UCT_UD_QKEY;
    self->tx.wr.imm_data          = 0;
    self->tx.wr.next              = 0;
    self->tx.wr.sg_list           = self->tx.sge;
    self->tx.wr.num_sge           = UCT_UD_MAX_SGE;

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

static UCS_CLASS_DEFINE_NEW_FUNC(uct_ud_verbs_iface_t, uct_iface_t, uct_worker_h,
                                 const char*, size_t, uct_iface_config_t*);

static UCS_CLASS_DEFINE_DELETE_FUNC(uct_ud_verbs_iface_t, uct_iface_t);

static uct_tl_ops_t uct_ud_verbs_tl_ops = {
    .query_resources     = uct_ud_verbs_query_resources,
    .iface_open          = UCS_CLASS_NEW_FUNC_NAME(uct_ud_verbs_iface_t),
};

static void uct_ud_verbs_register(uct_context_t *context)
{
    uct_register_tl(context, "ud_verbs", uct_ud_iface_config_table,
                    sizeof(uct_ud_iface_config_t), "UD_VERBS_", &uct_ud_verbs_tl_ops);
}

UCS_COMPONENT_DEFINE(uct_context_t, ud_verbs, uct_ud_verbs_register, ucs_empty_function, 0)

