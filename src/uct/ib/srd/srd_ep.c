/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_ep.h"
#include "srd_iface.h"
#include "srd_inl.h"
#include "srd_def.h"

#include <uct/api/uct_def.h>
#include <uct/ib/base/ib_verbs.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/time/time.h>


static void uct_srd_peer_name(uct_srd_peer_name_t *peer)
{
    ucs_strncpy_zero(peer->name, ucs_get_host_name(), sizeof(peer->name));
    peer->pid = getpid();
}

static void uct_srd_ep_set_state(uct_srd_ep_t *ep, uint32_t state)
{
    ep->flags |= state;
}

#if ENABLE_DEBUG_DATA
static void uct_srd_peer_copy(uct_srd_peer_name_t *dst,
                              uct_srd_peer_name_t *src)
{
    memcpy(dst, src, sizeof(*src));
}

#else
#define  uct_srd_peer_copy(dst, src)
#endif


static void uct_srd_ep_reset(uct_srd_ep_t *ep)
{
    ep->tx.psn             = UCT_SRD_INITIAL_PSN;
    ep->tx.pending.ops     = UCT_SRD_EP_OP_NONE;
    ep->rx_creq_count      = 0;
    ucs_queue_head_init(&ep->tx.outstanding_q);
    ucs_frag_list_init(ep->tx.psn - 1, &ep->rx.ooo_pkts, -1
                       UCS_STATS_ARG(ep->super.stats));
}

static void uct_srd_ep_purge(uct_srd_ep_t *ep, ucs_status_t status)
{
    while (ucs_queue_pull(&ep->tx.outstanding_q));
}

static UCS_F_ALWAYS_INLINE int
uct_srd_ep_is_last_pending_elem(uct_srd_ep_t *ep, ucs_arbiter_elem_t *elem)
{
    return (/* this is the only one pending element in the group */
            (ucs_arbiter_elem_is_only(elem)) ||
            (/* the next element in the group is control operation */
             (elem->next == &ep->tx.pending.elem) &&
             /* only two elements are in the group (the 1st element is the
              * current one, the 2nd (or the last) element is the control one) */
             (ucs_arbiter_group_tail(&ep->tx.pending.group) == &ep->tx.pending.elem)));
}

static ucs_arbiter_cb_result_t
uct_srd_ep_pending_cancel_cb(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                            ucs_arbiter_elem_t *elem, void *arg)
{
    uct_srd_ep_t *ep = ucs_container_of(group, uct_srd_ep_t, tx.pending.group);
    uct_pending_req_t *req;

    /* we may have pending op on ep */
    if (&ep->tx.pending.elem == elem) {
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    /* uct user should not have anything pending */
    req = ucs_container_of(elem, uct_pending_req_t, priv);
    ucs_warn("ep=%p removing user pending req=%p", ep, req);

    if (uct_srd_ep_is_last_pending_elem(ep, elem)) {
        uct_srd_ep_remove_has_pending_flag(ep);
    }

    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}

void uct_srd_ep_clone(uct_srd_ep_t *old_ep, uct_srd_ep_t *new_ep)
{
    uct_ep_t *ep_h = &old_ep->super.super;
    uct_iface_t *iface_h = ep_h->iface;

    uct_srd_iface_replace_ep(ucs_derived_of(iface_h, uct_srd_iface_t), old_ep, new_ep);
    memcpy(new_ep, old_ep, sizeof(uct_srd_ep_t));
}

ucs_status_t uct_srd_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr)
{
    uct_srd_ep_t *ep = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    uct_srd_ep_addr_t *ep_addr = (uct_srd_ep_addr_t *)addr;

    uct_ib_pack_uint24(ep_addr->iface_addr.qp_num, iface->qp->qp_num);
    uct_ib_pack_uint24(ep_addr->ep_id, ep->ep_id);
    return UCS_OK;
}

static ucs_status_t uct_srd_ep_connect_to_iface(uct_srd_ep_t *ep,
                                                const uct_ib_address_t *ib_addr,
                                                const uct_srd_iface_addr_t *if_addr)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    uct_ib_device_t UCS_V_UNUSED *dev = uct_ib_iface_device(&iface->super);
    char buf[128];

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_srd_ep_reset(ep);

    ucs_debug(UCT_IB_IFACE_FMT" lid %d qpn 0x%x epid %u ep %p connected to "
              "IFACE %s qpn 0x%x", UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id, ep,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(if_addr->qp_num));

    return UCS_OK;
}

static ucs_status_t uct_srd_ep_disconnect_from_iface(uct_ep_h tl_ep)
{
    uct_srd_ep_t *ep = ucs_derived_of(tl_ep, uct_srd_ep_t);

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_srd_ep_reset(ep);

    ep->dest_ep_id = UCT_SRD_EP_NULL_ID;
    ep->flags     &= ~UCT_SRD_EP_FLAG_CONNECTED;

    return UCS_OK;
}

void *uct_srd_ep_get_peer_address(uct_srd_ep_t *srd_ep)
{
    uct_srd_ep_t *ep = ucs_derived_of(srd_ep, uct_srd_ep_t);
    return &ep->peer_address;
}

ucs_status_t uct_srd_ep_create_connected(const uct_ep_params_t *ep_params,
                                         uct_ep_h *new_ep_p)
{
    uct_srd_iface_t *iface              = ucs_derived_of(ep_params->iface,
                                                        uct_srd_iface_t);
    const uct_ib_address_t *ib_addr     = (const uct_ib_address_t*)
                                          ep_params->dev_addr;
    const uct_srd_iface_addr_t *if_addr = (const uct_srd_iface_addr_t*)
                                           ep_params->iface_addr;
    int path_index                      = UCT_EP_PARAMS_GET_PATH_INDEX(ep_params);
    void *peer_address;
    uct_srd_send_skb_t *skb;
    uct_srd_ep_conn_sn_t conn_sn;
    uct_ep_params_t params;
    ucs_status_t status;
    uct_srd_ep_t *ep;
    uct_ep_h new_ep_h;

    uct_srd_enter(iface);

    *new_ep_p = NULL;

    conn_sn = uct_srd_iface_cep_get_conn_sn(iface, ib_addr, if_addr, path_index);
    ep      = uct_srd_iface_cep_get_ep(iface, ib_addr, if_addr, path_index,
                                       conn_sn, 1);
    if (ep != NULL) {
        uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREQ_NOTSENT);
        ep->flags &= ~UCT_SRD_EP_FLAG_PRIVATE;
        status     = UCS_OK;
        uct_srd_iface_cep_insert_ep(iface, ib_addr, if_addr, path_index,
                                    conn_sn, ep);
        goto out_set_ep;
    }

    params.field_mask = UCT_EP_PARAM_FIELD_IFACE |
                        UCT_EP_PARAM_FIELD_PATH_INDEX;
    params.iface      = &iface->super.super.super;
    params.path_index = path_index;

    status = uct_ep_create(&params, &new_ep_h);
    if (status != UCS_OK) {
        goto out;
    }

    ep          = ucs_derived_of(new_ep_h, uct_srd_ep_t);
    ep->conn_sn = conn_sn;

    status = uct_srd_ep_connect_to_iface(ep, ib_addr, if_addr);
    if (status != UCS_OK) {
        goto out;
    }

    uct_srd_iface_cep_insert_ep(iface, ib_addr, if_addr, path_index, conn_sn, ep);
    peer_address = uct_srd_ep_get_peer_address(ep);

    status = uct_srd_iface_unpack_peer_address(iface, ib_addr, if_addr,
                                               ep->path_index, peer_address);
    if (status != UCS_OK) {
        uct_srd_ep_disconnect_from_iface(&ep->super.super);
        goto out;
    }

    skb = uct_srd_ep_prepare_creq(ep);
    if (skb != NULL) {
        uct_srd_ep_tx_skb(iface, ep, skb, 0, 1);
        uct_srd_iface_complete_tx(iface, ep, skb);
        uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREQ_SENT);
    } else {
        uct_srd_ep_ctl_op_add(iface, ep, UCT_SRD_EP_OP_CREQ);
    }

out_set_ep:
    /* cppcheck-suppress autoVariables */
    *new_ep_p = &ep->super.super;
out:
    uct_srd_leave(iface);
    return status;
}

ucs_status_t uct_srd_ep_connect_to_ep(uct_ep_h tl_ep,
                                      const uct_device_addr_t *dev_addr,
                                      const uct_ep_addr_t *uct_ep_addr)
{
    uct_srd_ep_t *ep                   = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface             = ucs_derived_of(ep->super.super.iface,
                                                        uct_srd_iface_t);
    const uct_ib_address_t *ib_addr    = (const uct_ib_address_t*)dev_addr;
    const uct_srd_ep_addr_t *ep_addr   = (const uct_srd_ep_addr_t*)uct_ep_addr;
    uct_ib_device_t UCS_V_UNUSED *dev = uct_ib_iface_device(&iface->super);
    void *peer_address;
    char buf[128];

    ucs_assert_always(ep->dest_ep_id == UCT_SRD_EP_NULL_ID);
    ucs_trace_func("");

    uct_srd_ep_set_dest_ep_id(ep, uct_ib_unpack_uint24(ep_addr->ep_id));

    ucs_frag_list_cleanup(&ep->rx.ooo_pkts);
    uct_srd_ep_reset(ep);

    ucs_debug(UCT_IB_IFACE_FMT" slid %d qpn 0x%x epid %u connected to %s "
              "qpn 0x%x epid %u", UCT_IB_IFACE_ARG(&iface->super),
              dev->port_attr[iface->super.config.port_num - dev->first_port].lid,
              iface->qp->qp_num, ep->ep_id,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(ep_addr->iface_addr.qp_num),
              ep->dest_ep_id);

    peer_address = uct_srd_ep_get_peer_address(ep);
    return uct_srd_iface_unpack_peer_address(iface, ib_addr,
                                             &ep_addr->iface_addr,
                                             ep->path_index, peer_address);
}

static uct_srd_ep_t *uct_srd_ep_create_passive(uct_srd_iface_t *iface, uct_srd_ctl_hdr_t *ctl)
{
    uct_ep_params_t params;
    uct_srd_ep_t *ep;
    ucs_status_t status;
    uct_ep_t *ep_h;

    /* create new endpoint */
    params.field_mask = UCT_EP_PARAM_FIELD_IFACE;
    params.iface      = &iface->super.super.super;
    status            = uct_ep_create(&params, &ep_h);
    ucs_assert_always(status == UCS_OK);
    ep = ucs_derived_of(ep_h, uct_srd_ep_t);

    status = uct_ep_connect_to_ep(ep_h, (void*)uct_srd_creq_ib_addr(ctl),
                                  (void*)&ctl->conn_req.ep_addr);
    ucs_assert_always(status == UCS_OK);

    ep->path_index = ctl->conn_req.path_index;

    uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_PRIVATE);

    ep->conn_sn = ctl->conn_req.conn_sn;
    uct_srd_iface_cep_insert_ep(iface, uct_srd_creq_ib_addr(ctl),
                                &ctl->conn_req.ep_addr.iface_addr,
                                ep->path_index, ctl->conn_req.conn_sn, ep);
    return ep;
}

static uct_srd_ep_t *uct_srd_ep_rx_creq(uct_srd_iface_t *iface,
                                        uct_srd_neth_t *neth)
{
    uct_srd_ctl_hdr_t *ctl = (uct_srd_ctl_hdr_t *)(neth + 1);
    uct_srd_ep_t *ep;

    ucs_assert_always(ctl->type == UCT_SRD_PACKET_CREQ);

    ep = uct_srd_iface_cep_get_ep(iface, uct_srd_creq_ib_addr(ctl),
                                  &ctl->conn_req.ep_addr.iface_addr,
                                  ctl->conn_req.path_index,
                                  ctl->conn_req.conn_sn, 0);
    if (ep == NULL) {
        ep = uct_srd_ep_create_passive(iface, ctl);
        ucs_assert_always(ep != NULL);
        ep->rx.ooo_pkts.head_sn = neth->psn;
        uct_srd_peer_copy(&ep->peer, ucs_unaligned_ptr(&ctl->peer));
    } else if (ep->dest_ep_id == UCT_SRD_EP_NULL_ID) {
        /* simultaneuous CREQ */
        uct_srd_ep_set_dest_ep_id(ep, uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id));
        /* creq must always be the next in-order packet, i.e.,
         * there can't be any packets or holes before it. */
        ucs_assertv_always(ep->rx.ooo_pkts.head_sn + 1 == neth->psn,
                           "iface=%p ep=%p conn_sn=%d ep_id=%d, dest_ep_id=%d rx_psn=%u "
                           "neth_psn=%u ep_flags=0x%x ctl_ops=0x%x rx_creq_count=%d",
                           iface, ep, ep->conn_sn, ep->ep_id, ep->dest_ep_id,
                           ep->rx.ooo_pkts.head_sn, neth->psn, ep->flags,
                           ep->tx.pending.ops, ep->rx_creq_count);
        ep->rx.ooo_pkts.head_sn = neth->psn;
        uct_srd_peer_copy(&ep->peer, ucs_unaligned_ptr(&ctl->peer));
        ucs_debug("simultaneuous CREQ ep=%p"
                  "(iface=%p conn_sn=%d ep_id=%d, dest_ep_id=%d rx_psn=%u)",
                  ep, iface, ep->conn_sn, ep->ep_id,
                  ep->dest_ep_id, ep->rx.ooo_pkts.head_sn);
    }
    uct_srd_ep_ctl_op_add(iface, ep, UCT_SRD_EP_OP_CREP);

    ++ep->rx_creq_count;

    ucs_assertv_always(ctl->conn_req.conn_sn == ep->conn_sn,
                       "creq->conn_sn=%d ep->conn_sn=%d",
                       ctl->conn_req.conn_sn, ep->conn_sn);

    ucs_assertv_always(ctl->conn_req.path_index == ep->path_index,
                       "creq->path_index=%d ep->path_index=%d",
                       ctl->conn_req.path_index, ep->path_index);

    ucs_assertv_always(uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id) ==
                       ep->dest_ep_id,
                       "creq->ep_addr.ep_id=%d ep->dest_ep_id=%d",
                       uct_ib_unpack_uint24(ctl->conn_req.ep_addr.ep_id),
                       ep->dest_ep_id);

    ucs_assertv_always(ep->rx.ooo_pkts.head_sn == neth->psn,
                       "iface=%p ep=%p conn_sn=%d ep_id=%d, dest_ep_id=%d rx_psn=%u "
                       "neth_psn=%u ep_flags=0x%x ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->ep_id, ep->dest_ep_id,
                       ep->rx.ooo_pkts.head_sn, neth->psn, ep->flags,
                       ep->tx.pending.ops, ep->rx_creq_count);

    /* schedule connection reply op */
    if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ)) {
        uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREQ_NOTSENT);
    }
    uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREQ);
    uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREQ_RCVD);
    return ep;
}

static void uct_srd_ep_rx_crep(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                               uct_srd_neth_t *neth, uct_srd_recv_skb_t *skb)
{
    uct_srd_ctl_hdr_t *ctl = (uct_srd_ctl_hdr_t*)(neth + 1);

    ucs_trace_func("");
    ucs_assert_always(ctl->type == UCT_SRD_PACKET_CREP);

    if (uct_srd_ep_is_connected(ep)) {
        ucs_assertv_always(ep->dest_ep_id == ctl->conn_rep.src_ep_id,
                           "ep=%p [id=%d dest_ep_id=%d flags=0x%x] "
                           "crep [neth->dest=%d dst_ep_id=%d src_ep_id=%d]",
                           ep, ep->ep_id, ep->dest_ep_id, ep->path_index, ep->flags,
                           uct_srd_neth_get_dest_id(neth), ctl->conn_rep.src_ep_id);
    }

    uct_srd_ep_set_dest_ep_id(ep, ctl->conn_rep.src_ep_id);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    uct_srd_peer_copy(&ep->peer, ucs_unaligned_ptr(&ctl->peer));
    uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREP_RCVD);
}

uct_srd_send_skb_t *uct_srd_ep_prepare_creq(uct_srd_ep_t *ep)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    uct_srd_ctl_hdr_t *creq;
    uct_srd_send_skb_t *skb;
    uct_srd_neth_t *neth;
    ucs_status_t status;

    ucs_assert_always(ep->dest_ep_id == UCT_SRD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_SRD_EP_NULL_ID);

    /* CREQ should not be sent if CREP for the counter CREQ is scheduled
     * (or sent already) */
    ucs_assertv_always(!uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREP) &&
                       !(ep->flags & UCT_SRD_EP_FLAG_CREP_SENT),
                       "iface=%p ep=%p conn_sn=%d rx_psn=%u ep_flags=0x%x "
                       "ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->rx.ooo_pkts.head_sn,
                       ep->flags, ep->tx.pending.ops, ep->rx_creq_count);

    skb = uct_srd_iface_get_tx_skb(iface, ep);
    if (!skb) {
        return NULL;
    }

    neth = skb->neth;
    uct_srd_neth_init_data(ep, neth);

    neth->packet_type  = UCT_SRD_EP_NULL_ID;
    neth->packet_type |= UCT_SRD_PACKET_FLAG_CTLX;

    creq = (uct_srd_ctl_hdr_t *)(neth + 1);

    creq->type                = UCT_SRD_PACKET_CREQ;
    creq->conn_req.conn_sn    = ep->conn_sn;
    creq->conn_req.path_index = ep->path_index;

    status = uct_srd_ep_get_address(&ep->super.super,
                                   (void*)&creq->conn_req.ep_addr);
    if (status != UCS_OK) {
        return NULL;
    }

    status = uct_ib_iface_get_device_address(&iface->super.super.super,
                                             (uct_device_addr_t*)uct_srd_creq_ib_addr(creq));
    if (status != UCS_OK) {
        return NULL;
    }

    uct_srd_peer_name(ucs_unaligned_ptr(&creq->peer));

    skb->len = sizeof(*neth) + sizeof(*creq) + iface->super.addr_size;
    return skb;
}

static void inline
uct_srd_ep_process_rx_skb(uct_srd_iface_t *iface, uct_srd_ep_t *ep, int is_am,
                          uct_srd_neth_t *neth, uct_srd_recv_skb_t *skb)
{
    if (ucs_likely(is_am)) {
        uct_ib_iface_invoke_am_desc(&iface->super, uct_srd_neth_get_am_id(neth),
                                    neth + 1, skb->am.len, &skb->super);
    } else {
        /* must be connection reply packet */
        uct_srd_ep_rx_crep(iface, ep, neth, skb);
        ucs_mpool_put(skb);
    }
}

void uct_srd_ep_process_rx(uct_srd_iface_t *iface, uct_srd_neth_t *neth,
                           unsigned byte_len, uct_srd_recv_skb_t *skb)
{
    uint32_t dest_id;
    uint32_t is_am;
    uct_srd_ep_t *ep = 0;
    ucs_frag_list_elem_t *elem;
    ucs_frag_list_ooo_type_t ooo_type;

    ucs_trace_func("");

    dest_id = uct_srd_neth_get_dest_id(neth);
    is_am   = neth->packet_type & UCT_SRD_PACKET_FLAG_AM;

    if (ucs_unlikely(dest_id == UCT_SRD_EP_NULL_ID)) {
        /* must be connection request packet */
        ep = uct_srd_ep_rx_creq(iface, neth);
        ucs_mpool_put(skb);
        /* In case of simultaneous CREQ, other packets
         * might have been received before CREQ. */
        goto pull_ooo_pkts;
    }

    if (ucs_unlikely(!ucs_ptr_array_lookup(&iface->eps, dest_id, ep) ||
                     (ep->ep_id != dest_id)))
    {
        /* Drop the packet because it is
         * allowed to do disconnect without flush/barrier. So it
         * is possible to get packet for the ep that has been destroyed
         */
        ucs_trace("RX: failed to find ep %d, dropping packet", dest_id);
        UCS_STATS_UPDATE_COUNTER(iface->stats, UCT_SRD_IFACE_STAT_RX_DROP, 1);
        goto out;
    }

    ucs_assert(ep->ep_id != UCT_SRD_EP_NULL_ID);

    if (ucs_likely(is_am)) {
        skb->am.len = byte_len - sizeof(*neth);
    }

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &skb->ooo.elem, neth->psn);
    ucs_assert(ooo_type != UCS_FRAG_LIST_INSERT_DUP);
    if (ucs_unlikely(ooo_type == UCS_FRAG_LIST_INSERT_FAIL)) {
        ucs_fatal("failed to insert SRD packet (psn %u) into rx frag list %p",
                  neth->psn, &ep->rx.ooo_pkts);
        goto out;
    }

    if (ooo_type == UCS_FRAG_LIST_INSERT_FAST ||
        ooo_type == UCS_FRAG_LIST_INSERT_FIRST) {
        /* skb has not been inserted into the frag list */
        uct_srd_ep_process_rx_skb(iface, ep, is_am, neth, skb);
    }

pull_ooo_pkts:
    /* it might now be possible to pull (in order) some old elements */
    while ((elem = ucs_frag_list_pull(&ep->rx.ooo_pkts))) {
        skb   = ucs_container_of(elem, typeof(*skb), ooo.elem);
        neth  = (typeof(neth))uct_ib_iface_recv_desc_hdr(&iface->super,
                                                        (uct_ib_iface_recv_desc_t*)skb);
        is_am = neth->packet_type & UCT_SRD_PACKET_FLAG_AM;
        uct_srd_ep_process_rx_skb(iface, ep, is_am, neth, skb);
    }

    return;

out:
    ucs_mpool_put(skb);
}

ucs_status_t uct_srd_ep_flush_nolock(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                                     uct_completion_t *comp)
{
    uct_srd_send_skb_t *skb;

    if (ucs_unlikely(!uct_srd_ep_is_connected(ep))) {
        /* check for CREQ either being scheduled or sent and waiting for CREP ack */
        if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ) ||
            !ucs_queue_is_empty(&ep->tx.outstanding_q)) {

            return UCS_ERR_NO_RESOURCE; /* connection in progress */
        }

        return UCS_OK; /* Nothing was ever sent */
    }

    if (!uct_srd_iface_can_tx(iface) || !uct_srd_iface_has_skbs(iface)) {
        /* iface has no resources, prevent reordering with possible pending
         * operations by not starting the flush.
         */
        return UCS_ERR_NO_RESOURCE;
    }

    if (ucs_queue_is_empty(&ep->tx.outstanding_q)) {
        /* No outstanding operations */
        return UCS_OK;
    }

    /* If the user requested a callback, allocate a dummy skb which
     * will be released and call user completion callback when all
     * the sequence numbers posted so far by this ep are completed.
     */
    if (comp != NULL) {
        ucs_assert(comp->count > 0);

        skb = ucs_mpool_get(&iface->tx.mp);
        if (skb == NULL) {
            return UCS_ERR_NO_RESOURCE;
        }

        /* Add dummy skb to the flush queue */
        skb->flags                  = UCT_SRD_SEND_SKB_FLAG_COMP |
                                      UCT_SRD_SEND_SKB_FLAG_FLUSH;
        skb->len                    = sizeof(skb->neth[0]);
        skb->neth->packet_type      = 0;
        skb->neth->psn              = ep->tx.psn - 1;
        uct_srd_neth_set_dest_id(skb->neth, UCT_SRD_EP_NULL_ID);
        uct_srd_comp_desc(skb)->comp = comp;

        ucs_queue_push(&ep->tx.outstanding_q, &skb->out_queue);

        ucs_trace_data("added dummy flush skb %p psn %d user_comp %p",
                       skb, skb->neth->psn, comp);
    }

    return UCS_INPROGRESS;
}

ucs_status_t uct_srd_ep_flush(uct_ep_h ep_h, unsigned flags,
                              uct_completion_t *comp)
{
    uct_srd_ep_t *ep = ucs_derived_of(ep_h, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_srd_iface_t);
    ucs_status_t status;

    uct_srd_enter(iface);

    if (ucs_unlikely(flags & UCT_FLUSH_FLAG_CANCEL)) {
        uct_ep_pending_purge(ep_h, NULL, 0);
        uct_srd_ep_purge(ep, UCS_ERR_CANCELED);
        /* FIXME make flush(CANCEL) operation truly non-blocking and wait until
         * all of the outstanding sends are completed. Without this, zero-copy
         * sends which are still on the QP could be reported as completed which
         * can lead to sending corrupt data, or local access error. */
        status = UCS_OK;
        goto out;
    }

    status = uct_srd_ep_flush_nolock(iface, ep, comp);
    if (status == UCS_OK) {
        UCT_TL_EP_STAT_FLUSH(&ep->super);
    } else if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super);
    }

out:
    uct_srd_leave(iface);
    return status;
}

ucs_status_t uct_srd_ep_check(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    UCT_EP_KEEPALIVE_CHECK_PARAM(flags, comp);

    /* FIXME: implement ep check for srd */
    return UCS_OK;
}

static uct_srd_send_skb_t *uct_srd_ep_prepare_crep(uct_srd_ep_t *ep)
{
    uct_srd_send_skb_t *skb;
    uct_srd_neth_t *neth;
    uct_srd_ctl_hdr_t *crep;
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);

    ucs_assert_always(ep->dest_ep_id != UCT_SRD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_SRD_EP_NULL_ID);

    /* Check that CREQ is not sheduled */
    ucs_assertv_always(!uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ),
                       "iface=%p ep=%p conn_sn=%d ep_id=%d, dest_ep_id=%d "
                       "rx_psn=%u ep_flags=0x%x ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->ep_id, ep->dest_ep_id,
                       ep->rx.ooo_pkts.head_sn, ep->flags, ep->tx.pending.ops,
                       ep->rx_creq_count);

    skb = uct_srd_iface_get_tx_skb(iface, ep);
    if (!skb) {
        return NULL;
    }

    neth = skb->neth;
    uct_srd_neth_init_data(ep, neth);

    neth->packet_type  = ep->dest_ep_id;
    neth->packet_type |= UCT_SRD_PACKET_FLAG_CTLX;

    crep = (uct_srd_ctl_hdr_t *)(neth + 1);

    crep->type               = UCT_SRD_PACKET_CREP;
    crep->conn_rep.src_ep_id = ep->ep_id;

    uct_srd_peer_name(ucs_unaligned_ptr(&crep->peer));

    skb->len = sizeof(*neth) + sizeof(*crep);
    uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREP);
    return skb;
}

static void uct_srd_ep_send_creq_crep(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                                      uct_srd_send_skb_t *skb)
{
    uct_srd_ep_tx_skb(iface, ep, skb, 0, 1);
    uct_srd_iface_complete_tx(iface, ep, skb);
}

static void uct_srd_ep_do_pending_ctl(uct_srd_ep_t *ep, uct_srd_iface_t *iface)
{
    uct_srd_send_skb_t *skb;

    if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ)) {
        skb = uct_srd_ep_prepare_creq(ep);
        if (skb) {
            uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREQ_SENT);
            uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREQ);
            uct_srd_ep_send_creq_crep(iface, ep, skb);
        }
    } else if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREP)) {
        skb = uct_srd_ep_prepare_crep(ep);
        if (skb) {
            uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREP_SENT);
            uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREP);
            uct_srd_ep_send_creq_crep(iface, ep, skb);
        }
    } else {
        ucs_assertv(!uct_srd_ep_ctl_op_isany(ep),
                    "unsupported pending op mask: %x", ep->tx.pending.ops);
    }
}

static inline ucs_arbiter_cb_result_t
uct_srd_ep_ctl_op_next(uct_srd_ep_t *ep)
{
    if (uct_srd_ep_ctl_op_isany(ep)) {
        /* can send more control - come here later */
        return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
    }
    /* no more control - nothing to do in
     * this dispatch cycle. */
    return UCS_ARBITER_CB_RESULT_RESCHED_GROUP;
}

/**
 * pending operations are processed according to priority:
 * - control:
 *   - creq request
 *   - crep reply
 * - pending uct requests
 */
ucs_arbiter_cb_result_t
uct_srd_ep_do_pending(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                      ucs_arbiter_elem_t *elem, void *arg)
{
    uct_srd_ep_t *ep       = ucs_container_of(group, uct_srd_ep_t,
                                              tx.pending.group);
    uct_srd_iface_t *iface = ucs_container_of(arbiter, uct_srd_iface_t,
                                              tx.pending_q);
    uct_pending_req_t *req;
    ucs_status_t status;
    int is_last_pending_elem;

    ucs_assert(arg == NULL);

    /* check if we have global resources
     * - tx_wqe
     * - skb
     * control messages does not need skb.
     */
    if (!uct_srd_iface_can_tx(iface)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    /* here we rely on the fact that arbiter
     * will start next dispatch cycle from the
     * next group.
     * So it is ok to stop if there is no ctl.
     * However in worst case only one ctl per
     * dispatch cycle will be sent.
     */
    if (!uct_srd_iface_has_skbs(iface) && !uct_srd_ep_ctl_op_isany(ep)) {
        return UCS_ARBITER_CB_RESULT_STOP;
    }

    /* we can desched group: iff
     * - no control
     * - no connect
     */
    if (!uct_srd_ep_ctl_op_isany(ep) && !uct_srd_ep_is_connected(ep)) {
        return UCS_ARBITER_CB_RESULT_DESCHED_GROUP;
    }

    if (&ep->tx.pending.elem == elem) {
        uct_srd_ep_do_pending_ctl(ep, iface);
        if (uct_srd_ep_ctl_op_isany(ep)) {
            /* there is still some ctl left. go to next group */
            return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
        } else {
            /* no more ctl - dummy elem can be removed */
            return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
        }
    }

    /* user pending can be send iff there are
     * no high priority pending control messages
     */
    req = ucs_container_of(elem, uct_pending_req_t, priv);
    if (!uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CTL_HI_PRIO)) {
        ucs_assert(!(ep->flags & UCT_SRD_EP_FLAG_IN_PENDING));
        ep->flags |= UCT_SRD_EP_FLAG_IN_PENDING;
        /* temporary reset `UCT_SRD_EP_HAS_PENDING` flag to unblock sends */
        uct_srd_ep_remove_has_pending_flag(ep);

        is_last_pending_elem = uct_srd_ep_is_last_pending_elem(ep, elem);

        status = req->func(req);
#if UCS_ENABLE_ASSERT
        /* do not touch the request (or the arbiter element) after
         * calling the callback if UCS_OK is returned from the callback */
        if (status == UCS_OK) {
            req  = NULL;
            elem = NULL;
        }
#endif

        uct_srd_ep_set_has_pending_flag(ep);
        ep->flags &= ~UCT_SRD_EP_FLAG_IN_PENDING;

        if (status == UCS_INPROGRESS) {
            return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
        } else if (status != UCS_OK) {
            /* avoid deadlock: send low priority ctl if user cb failed
             * no need to check for low prio here because we
             * already checked above.
             */
            uct_srd_ep_do_pending_ctl(ep, iface);
            return uct_srd_ep_ctl_op_next(ep);
        }

        if (is_last_pending_elem) {
            uct_srd_ep_remove_has_pending_flag(ep);
        }

        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    /* try to send ctl messages */
    uct_srd_ep_do_pending_ctl(ep, iface);

    /* we still didn't process the current pending request because of hi-prio
     * control messages, so cannot stop sending yet. If we stop, not all
     * resources will be exhausted and out-of-order with pending can occur.
     * (pending control ops may be cleared by uct_srd_ep_do_pending_ctl)
     */
    return UCS_ARBITER_CB_RESULT_NEXT_GROUP;
}

ucs_status_t uct_srd_ep_pending_add(uct_ep_h ep_h, uct_pending_req_t *req,
                                    unsigned flags)
{
    uct_srd_ep_t *ep       = ucs_derived_of(ep_h, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_srd_iface_t);

    uct_srd_enter(iface);

    if (uct_srd_iface_can_tx(iface) &&
        uct_srd_iface_has_skbs(iface) &&
        uct_srd_ep_is_connected_and_no_pending(ep)) {

        uct_srd_leave(iface);
        return UCS_ERR_BUSY;
    }

    UCS_STATIC_ASSERT(sizeof(uct_srd_pending_req_priv_t) <=
                      UCT_PENDING_REQ_PRIV_LEN);
    uct_srd_pending_req_priv(req)->flags = flags;
    uct_srd_ep_set_has_pending_flag(ep);
    uct_pending_req_arb_group_push(&ep->tx.pending.group, req);
    ucs_arbiter_group_schedule(&iface->tx.pending_q, &ep->tx.pending.group);
    ucs_trace_data("srd ep %p: added pending req %p tx_psn %d",
                   ep, req, ep->tx.psn);
    UCT_TL_EP_STAT_PEND(&ep->super);

    uct_srd_leave(iface);
    return UCS_OK;
}

static ucs_arbiter_cb_result_t
uct_srd_ep_pending_purge_cb(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                            ucs_arbiter_elem_t *elem, void *arg)
{
    uct_srd_ep_t *ep                = ucs_container_of(group, uct_srd_ep_t,
                                                       tx.pending.group);
    uct_purge_cb_args_t *cb_args    = arg;
    uct_pending_purge_callback_t cb = cb_args->cb;
    uct_pending_req_t *req;
    int is_last_pending_elem;

    if (&ep->tx.pending.elem == elem) {
        /* return ignored by arbiter */
        return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
    }

    is_last_pending_elem = uct_srd_ep_is_last_pending_elem(ep, elem);

    req = ucs_container_of(elem, uct_pending_req_t, priv);
    if (cb) {
        cb(req, cb_args->arg);
    } else {
        ucs_debug("ep=%p cancelling user pending request %p", ep, req);
    }

    if (is_last_pending_elem) {
        uct_srd_ep_remove_has_pending_flag(ep);
    }

    /* return ignored by arbiter */
    return UCS_ARBITER_CB_RESULT_REMOVE_ELEM;
}


void uct_srd_ep_pending_purge(uct_ep_h ep_h, uct_pending_purge_callback_t cb,
                              void *arg)
{
    uct_srd_ep_t *ep         = ucs_derived_of(ep_h, uct_srd_ep_t);
    uct_srd_iface_t *iface   = ucs_derived_of(ep->super.super.iface,
                                              uct_srd_iface_t);
    uct_purge_cb_args_t args = {cb, arg};

    uct_srd_enter(iface);
    ucs_arbiter_group_purge(&iface->tx.pending_q, &ep->tx.pending.group,
                            uct_srd_ep_pending_purge_cb, &args);
    uct_srd_leave(iface);
}

void uct_srd_ep_disconnect(uct_ep_h tl_ep)
{
    uct_srd_ep_t    *ep    = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);

    ucs_debug("ep %p: disconnect", ep);

    uct_srd_enter(iface);

    /* cancel user pending */
    uct_srd_ep_pending_purge(tl_ep, NULL, NULL);

    /* schedule flush */
    /* FIXME: shouldn't this be ep_purge instead? */
    uct_srd_ep_flush(tl_ep, 0, NULL);

    /* the EP will be destroyed by interface destroy */
    ep->flags |= UCT_SRD_EP_FLAG_DISCONNECTED;

    uct_srd_leave(iface);
}

ucs_status_t uct_srd_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p)
{
    if (ucs_test_all_flags(params->field_mask, UCT_EP_PARAM_FIELD_DEV_ADDR |
                                               UCT_EP_PARAM_FIELD_IFACE_ADDR)) {
        return uct_srd_ep_create_connected(params, ep_p);
    }

    return UCS_CLASS_NEW_FUNC_NAME(uct_srd_ep_t)(params, ep_p);
}


static UCS_CLASS_INIT_FUNC(uct_srd_ep_t, const uct_ep_params_t* params)
{
    uct_srd_iface_t *iface = ucs_derived_of(params->iface, uct_srd_iface_t);

    ucs_trace_func("");

    memset(self, 0, sizeof(*self));
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    uct_srd_enter(iface);

    self->dest_ep_id         = UCT_SRD_EP_NULL_ID;
    self->path_index         = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    self->peer_address.ah    = NULL;
    uct_srd_ep_reset(self);
    uct_srd_iface_add_ep(iface, self);
    ucs_arbiter_group_init(&self->tx.pending.group);
    ucs_arbiter_elem_init(&self->tx.pending.elem);

    ucs_debug("created ep ep=%p iface=%p id=%d", self, iface, self->ep_id);

    uct_srd_leave(iface);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_ep_t)
{
    uct_srd_iface_t *iface = ucs_derived_of(self->super.super.iface, uct_srd_iface_t);

    ucs_trace_func("ep=%p id=%d conn_sn=%d", self, self->ep_id, self->conn_sn);

    uct_srd_enter(iface);

    uct_srd_ep_purge(self, UCS_ERR_CANCELED);

    uct_srd_iface_remove_ep(iface, self);
    uct_srd_iface_cep_remove_ep(iface, self);
    ucs_frag_list_cleanup(&self->rx.ooo_pkts);

    ucs_arbiter_group_purge(&iface->tx.pending_q, &self->tx.pending.group,
                            uct_srd_ep_pending_cancel_cb, 0);

    ucs_arbiter_group_cleanup(&self->tx.pending.group);
    uct_srd_leave(iface);
}

UCS_CLASS_DEFINE(uct_srd_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);
