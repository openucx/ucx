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


#define UCT_SRD_CHECK_FC_WND(_fc, _stats)\
    if ((_fc)->fc_wnd == 0) { \
        UCS_STATS_UPDATE_COUNTER((_fc)->stats, UCT_SRD_FC_STAT_NO_CRED, 1); \
        UCS_STATS_UPDATE_COUNTER(_stats, UCT_EP_STAT_NO_RES, 1); \
        return UCS_ERR_NO_RESOURCE; \
    } \

#define UCT_SRD_CHECK_FC(_iface, _ep, _fc_hdr) \
    { \
        if (ucs_unlikely((_ep)->fc.fc_wnd <= (_iface)->config.fc_soft_thresh)) { \
            UCT_SRD_CHECK_FC_WND(&(_ep)->fc, (_ep)->super.stats); \
            (_fc_hdr) = uct_srd_ep_fc_get_request(_ep, _iface); \
        } \
        if ((_ep)->flags & UCT_SRD_EP_FLAG_FC_GRANT) { \
            (_fc_hdr) |= UCT_SRD_PACKET_FLAG_FC_GRANT; \
        } \
    }

#define UCT_SRD_UPDATE_FC_WND(_iface, _fc) \
    { \
        ucs_assert((_fc)->fc_wnd > 0); \
        (_fc)->fc_wnd--; \
        UCS_STATS_SET_COUNTER((_fc)->stats, UCT_SRD_FC_STAT_FC_WND, (_fc)->fc_wnd); \
    }

#define UCT_SRD_UPDATE_FC(_iface, _ep, _fc_hdr) \
    { \
        if ((_fc_hdr) & UCT_SRD_PACKET_FLAG_FC_GRANT) { \
            UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, UCT_SRD_FC_STAT_TX_GRANT, 1); \
        } \
        if ((_fc_hdr) & UCT_SRD_PACKET_FLAG_FC_SREQ) { \
            UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, UCT_SRD_FC_STAT_TX_SOFT_REQ, 1); \
        } else if ((_fc_hdr) & UCT_SRD_PACKET_FLAG_FC_HREQ) { \
            UCS_STATS_UPDATE_COUNTER((_ep)->fc.stats, UCT_SRD_FC_STAT_TX_HARD_REQ, 1); \
        } \
        \
        (_ep)->flags &= ~UCT_SRD_EP_FLAG_FC_GRANT; \
        \
        UCT_SRD_UPDATE_FC_WND(_iface, &(_ep)->fc) \
    }

#define UCT_SRD_EP_ASSERT_PENDING(_ep) \
    ucs_assertv(((_ep)->flags & UCT_SRD_EP_FLAG_IN_PENDING) ||  \
                !uct_srd_ep_has_pending(_ep),                   \
                "out-of-order send detected for"                \
                " ep %p ep_pending %d arbelem %p",              \
                _ep, ((_ep)->flags & UCT_SRD_EP_FLAG_IN_PENDING), \
                &(_ep)->tx.pending.elem);

#define UCT_SRD_AM_COMMON(_iface, _ep, _id, _neth) \
    { \
        /* either we are executing pending operations, \
         * or there are no pending elements. */ \
        UCT_SRD_EP_ASSERT_PENDING(_ep); \
        \
        UCT_SRD_CHECK_FC(_iface, _ep, (_neth)->fc); \
        uct_srd_neth_set_type_am(_ep, _neth, _id); \
        uct_srd_neth_set_psn(_ep, _neth); \
    }

#ifdef ENABLE_STATS
static ucs_stats_class_t uct_srd_fc_stats_class = {
    .name = "srd_fc",
    .num_counters = UCT_SRD_FC_STAT_LAST,
    .counter_names = {
        [UCT_SRD_FC_STAT_NO_CRED]            = "no_cred",
        [UCT_SRD_FC_STAT_TX_GRANT]           = "tx_grant",
        [UCT_SRD_FC_STAT_TX_PURE_GRANT]      = "tx_pure_grant",
        [UCT_SRD_FC_STAT_TX_SOFT_REQ]        = "tx_soft_req",
        [UCT_SRD_FC_STAT_TX_HARD_REQ]        = "tx_hard_req",
        [UCT_SRD_FC_STAT_RX_GRANT]           = "rx_grant",
        [UCT_SRD_FC_STAT_RX_PURE_GRANT]      = "rx_pure_grant",
        [UCT_SRD_FC_STAT_RX_SOFT_REQ]        = "rx_soft_req",
        [UCT_SRD_FC_STAT_RX_HARD_REQ]        = "rx_hard_req",
        [UCT_SRD_FC_STAT_FC_WND]             = "fc_wnd"
    }
};
#endif

static UCS_F_ALWAYS_INLINE uint8_t
uct_srd_ep_fc_get_request(uct_srd_ep_t *ep, uct_srd_iface_t *iface)
{
    return (ep->fc.fc_wnd == iface->config.fc_hard_thresh) ?
            UCT_SRD_PACKET_FLAG_FC_HREQ:
           (ep->fc.fc_wnd == iface->config.fc_soft_thresh) ?
            UCT_SRD_PACKET_FLAG_FC_SREQ: 0;
}

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
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);

    ep->tx.psn             = UCT_SRD_INITIAL_PSN;
    ep->tx.pending.ops     = UCT_SRD_EP_OP_NONE;
    ep->rx_creq_count      = 0;
    ep->fc.fc_wnd          = iface->config.fc_wnd_size;
    ucs_queue_head_init(&ep->tx.outstanding_q);
    ucs_frag_list_init(ep->tx.psn - 1, &ep->rx.ooo_pkts, -1
                       UCS_STATS_ARG(ep->super.stats));
}

static void uct_srd_ep_purge(uct_srd_ep_t *ep, ucs_status_t status)
{
    uct_srd_send_op_t *send_op;
    ucs_queue_for_each(send_op, &ep->tx.outstanding_q, out_queue) {
        /* Do not release send op here because it might be reused
         * for another send op while the old send request is still
         * pending for completion in the QP.
         */
        send_op->flags |= UCT_SRD_SEND_OP_FLAG_PURGED;
    }
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

static uct_srd_send_desc_t *uct_srd_ep_prepare_creq(uct_srd_ep_t *ep)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    uct_srd_ctl_hdr_t *creq;
    uct_srd_send_desc_t *desc;
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

    desc = uct_srd_iface_get_send_desc(iface);
    if (!desc) {
        return NULL;
    }

    neth               = uct_srd_send_desc_neth(desc);
    neth->packet_type  = UCT_SRD_EP_NULL_ID;
    neth->packet_type |= UCT_SRD_PACKET_FLAG_CTLX;
    uct_srd_neth_set_psn(ep, neth);

    creq                      = (uct_srd_ctl_hdr_t *)(neth + 1);
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

    desc->super.ep           = ep;
    desc->super.len          = sizeof(*neth) + sizeof(*creq) + iface->super.addr_size;
    desc->super.comp_handler = uct_srd_iface_send_op_release;

    return desc;
}

static uct_srd_send_desc_t *uct_srd_ep_prepare_crep(uct_srd_ep_t *ep)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    uct_srd_ctl_hdr_t *crep;
    uct_srd_send_desc_t *desc;
    uct_srd_neth_t *neth;

    ucs_assert_always(ep->dest_ep_id != UCT_SRD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_SRD_EP_NULL_ID);

    /* Check that CREQ is not sheduled */
    ucs_assertv_always(!uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ),
                       "iface=%p ep=%p conn_sn=%d ep_id=%d, dest_ep_id=%d "
                       "rx_psn=%u ep_flags=0x%x ctl_ops=0x%x rx_creq_count=%d",
                       iface, ep, ep->conn_sn, ep->ep_id, ep->dest_ep_id,
                       ep->rx.ooo_pkts.head_sn, ep->flags, ep->tx.pending.ops,
                       ep->rx_creq_count);

    desc = uct_srd_iface_get_send_desc(iface);
    if (!desc) {
        return NULL;
    }

    neth               = uct_srd_send_desc_neth(desc);
    neth->packet_type  = ep->dest_ep_id;
    neth->packet_type |= UCT_SRD_PACKET_FLAG_CTLX;
    uct_srd_neth_set_psn(ep, neth);

    crep                     = (uct_srd_ctl_hdr_t *)(neth + 1);
    crep->type               = UCT_SRD_PACKET_CREP;
    crep->conn_rep.src_ep_id = ep->ep_id;

    uct_srd_peer_name(ucs_unaligned_ptr(&crep->peer));

    uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREP);

    desc->super.ep           = ep;
    desc->super.len          = sizeof(*neth) + sizeof(*crep);
    desc->super.comp_handler = uct_srd_iface_send_op_release;

    return desc;
}

static uct_srd_send_desc_t *uct_srd_ep_prepare_fc_pgrant(uct_srd_ep_t *ep)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    uct_srd_neth_t *neth;

    ucs_assert_always(ep->dest_ep_id != UCT_SRD_EP_NULL_ID);
    ucs_assert_always(ep->ep_id != UCT_SRD_EP_NULL_ID);

    desc = uct_srd_iface_get_send_desc(iface);
    if (!desc) {
        return NULL;
    }

    neth               = uct_srd_send_desc_neth(desc);
    neth->packet_type  = ep->dest_ep_id;
    neth->packet_type |= UCT_SRD_PACKET_FLAG_FC_PGRANT;
    uct_srd_neth_set_psn(ep, neth);

    desc->super.ep           = ep;
    desc->super.len          = sizeof(*neth);
    desc->super.comp_handler = uct_srd_iface_send_op_release;

    uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_FC_PGRANT);

    return desc;
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
    uct_srd_send_desc_t *desc;
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

    desc = uct_srd_ep_prepare_creq(ep);
    if (desc) {
        uct_srd_ep_tx_desc(iface, ep, desc, 0, 1);
        uct_srd_iface_complete_tx_desc(iface, ep, desc);
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
                               uct_srd_neth_t *neth, uct_srd_recv_desc_t *desc)
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

static void inline
uct_srd_ep_fc_handler(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                      uct_srd_neth_t *neth)
{
    uint8_t fc      = neth->fc;
    int16_t old_wnd = ep->fc.fc_wnd;
    uct_srd_send_desc_t *desc;

    if (fc & UCT_SRD_PACKET_FLAG_FC_GRANT) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_SRD_FC_STAT_RX_GRANT, 1);
        ep->fc.fc_wnd = iface->config.fc_wnd_size;

        /* To preserve ordering we have to dispatch
         * all pending operations if fc_wnd was 0
         * (otherwise it will be dispatched by tx progress) */
        if (old_wnd == 0) {
            uct_srd_iface_progress_pending(iface);
        }
    }

    if (fc & UCT_SRD_PACKET_FLAG_FC_SREQ) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_SRD_FC_STAT_RX_SOFT_REQ, 1);

        /* Got soft credit request. Mark ep that it needs to grant
         * credits to the peer in outgoing AM/PUT (if any). */
        uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_FC_GRANT);
    } else if (fc & UCT_SRD_PACKET_FLAG_FC_HREQ) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_SRD_FC_STAT_RX_HARD_REQ, 1);
        desc = uct_srd_ep_prepare_fc_pgrant(ep);
        if (desc) {
            uct_srd_ep_tx_desc(iface, ep, desc, 0, 1);
            uct_srd_iface_complete_tx_desc(iface, ep, desc);
        } else {
            uct_srd_ep_ctl_op_add(iface, ep, UCT_SRD_EP_OP_FC_PGRANT);
        }
    }
}

static inline void uct_srd_ep_rx_put(uct_srd_neth_t *neth, uint32_t byte_len)
{
    uct_srd_put_hdr_t *put_hdr = (uct_srd_put_hdr_t*)neth;

    memcpy((void *)put_hdr->rva, put_hdr+1, byte_len - sizeof(put_hdr->rva));
}

static void inline
uct_srd_ep_process_rx_desc(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                          uct_srd_neth_t *neth, uct_srd_recv_desc_t *desc)
{
    if (ucs_likely(uct_srd_neth_is_am(neth))) {
        uct_srd_ep_fc_handler(iface, ep, neth);
        uct_ib_iface_invoke_am_desc(&iface->super, uct_srd_neth_get_am_id(neth),
                                    neth + 1, desc->data_len, &desc->super);
    } else if (uct_srd_neth_is_put(neth)) {
        uct_srd_ep_fc_handler(iface, ep, neth);
        uct_srd_ep_rx_put(neth, desc->data_len);
        ucs_mpool_put(desc);
    } else if (uct_srd_neth_is_pure_grant(neth)) {
        UCS_STATS_UPDATE_COUNTER(ep->fc.stats, UCT_SRD_FC_STAT_RX_PURE_GRANT, 1);
        ep->fc.fc_wnd = iface->config.fc_wnd_size;
        ucs_mpool_put(desc);
    } else {
        /* must be connection reply packet */
        uct_srd_ep_rx_crep(iface, ep, neth, desc);
        ucs_mpool_put(desc);
    }
}

void uct_srd_ep_process_rx(uct_srd_iface_t *iface, uct_srd_neth_t *neth,
                           unsigned byte_len, uct_srd_recv_desc_t *desc)
{
    uint32_t dest_id;
    uct_srd_ep_t *ep = 0;
    ucs_frag_list_elem_t *elem;
    ucs_frag_list_ooo_type_t ooo_type;

    ucs_trace_func("");

    dest_id = uct_srd_neth_get_dest_id(neth);

    if (ucs_unlikely(dest_id == UCT_SRD_EP_NULL_ID)) {
        /* must be connection request packet */
        ep = uct_srd_ep_rx_creq(iface, neth);
        ucs_mpool_put(desc);
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

    desc->data_len = byte_len - sizeof(*neth);

    ooo_type = ucs_frag_list_insert(&ep->rx.ooo_pkts, &desc->ooo.elem, neth->psn);
    ucs_assert(ooo_type != UCS_FRAG_LIST_INSERT_DUP);
    if (ucs_unlikely(ooo_type == UCS_FRAG_LIST_INSERT_FAIL)) {
        ucs_fatal("failed to insert SRD packet (psn %u) into rx frag list %p",
                  neth->psn, &ep->rx.ooo_pkts);
        goto out;
    }

    if (ooo_type == UCS_FRAG_LIST_INSERT_FAST ||
        ooo_type == UCS_FRAG_LIST_INSERT_FIRST) {
        /* desc has not been inserted into the frag list */
        uct_srd_ep_process_rx_desc(iface, ep, neth, desc);
    }

pull_ooo_pkts:
    /* it might now be possible to pull (in order) some old elements */
    while ((elem = ucs_frag_list_pull(&ep->rx.ooo_pkts))) {
        desc  = ucs_container_of(elem, typeof(*desc), ooo.elem);
        neth  = (typeof(neth))uct_ib_iface_recv_desc_hdr(&iface->super,
                                                        (uct_ib_iface_recv_desc_t*)desc);
        uct_srd_ep_process_rx_desc(iface, ep, neth, desc);
    }

    return;

out:
    ucs_mpool_put(desc);
}

ucs_status_t uct_srd_ep_flush_nolock(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                                     uct_completion_t *comp)
{
    uct_srd_send_op_t *send_op;

    if (ucs_unlikely(!uct_srd_ep_is_connected(ep))) {
        /* check for CREQ either being scheduled or sent and waiting for CREP ack */
        if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ) ||
            !ucs_queue_is_empty(&ep->tx.outstanding_q)) {

            return UCS_ERR_NO_RESOURCE; /* connection in progress */
        }

        return UCS_OK; /* Nothing was ever sent */
    }

    if (!uct_srd_iface_has_all_tx_resources(iface) ||
        !uct_srd_ep_has_fc_resources(ep)) {
        /* iface/ep does not have all tx resources. Prevent reordering
         * with possible pending operations by not starting the flush.
         */
        return UCS_ERR_NO_RESOURCE;
    }

    if (ucs_queue_is_empty(&ep->tx.outstanding_q)) {
        /* No outstanding operations */
        return UCS_OK;
    }

    /* If the user requested a callback, allocate a dummy send_op which
     * will be released and will call the user completion callback when
     * all the sequence numbers posted before it are completed.
     */
    if (comp != NULL) {
        ucs_assert(comp->count > 0);

        /* Add a dummy send op to the outstanding desc queue */
        send_op = ucs_mpool_get(&iface->tx.send_op_mp);
        ucs_assert(send_op != NULL);

        send_op->ep           = ep;
        send_op->flags        = UCT_SRD_SEND_OP_FLAG_FLUSH;
        send_op->user_comp    = comp;
        send_op->comp_handler = uct_srd_iface_send_op_ucomp_release;

        ucs_queue_push(&ep->tx.outstanding_q, &send_op->out_queue);

        ucs_trace_data("added dummy flush send op %p with user_comp %p",
                       send_op, comp);
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
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_srd_iface_t);
    char dummy = 0;

    UCT_EP_KEEPALIVE_CHECK_PARAM(flags, comp);

    uct_srd_enter(iface);

    if (!uct_srd_ep_is_connected(ep) ||
        !uct_srd_ep_has_fc_resources(ep)) {
        uct_srd_leave(iface);
        return UCS_OK;
    }
    uct_srd_leave(iface);

    return uct_srd_ep_put_short(tl_ep, &dummy, 0, 0, 0);
}

static void uct_srd_ep_do_pending_ctl(uct_srd_ep_t *ep, uct_srd_iface_t *iface)
{
    uct_srd_send_desc_t* desc;

    if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREQ)) {
        desc = uct_srd_ep_prepare_creq(ep);
        if (desc) {
            uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREQ_SENT);
            uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREQ);
            uct_srd_ep_tx_desc(iface, ep, desc, 0, 1);
            uct_srd_iface_complete_tx_desc(iface, ep, desc);
        }
    } else if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_CREP)) {
        desc = uct_srd_ep_prepare_crep(ep);
        if (desc) {
            uct_srd_ep_set_state(ep, UCT_SRD_EP_FLAG_CREP_SENT);
            uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_CREP);
            uct_srd_ep_tx_desc(iface, ep, desc, 0, 1);
            uct_srd_iface_complete_tx_desc(iface, ep, desc);
        }
    } else if (uct_srd_ep_ctl_op_check(ep, UCT_SRD_EP_OP_FC_PGRANT)) {
        desc = uct_srd_ep_prepare_fc_pgrant(ep);
        if (desc) {
            uct_srd_ep_ctl_op_del(ep, UCT_SRD_EP_OP_FC_PGRANT);
            uct_srd_ep_tx_desc(iface, ep, desc, 0, 1);
            uct_srd_iface_complete_tx_desc(iface, ep, desc);
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

    /* we should check for all iface/ep tx resources
     * because we do not know what the pending op is */
    if (!uct_srd_iface_has_all_tx_resources(iface) ||
        !uct_srd_ep_has_fc_resources(ep)) {
        /* FIXME: should move to the next element if ep does not
         * have fc credits. Next element might correspond to another ep */
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

    /* user pending can be sent iff there are
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

    if (uct_srd_iface_has_all_tx_resources(iface)  &&
        uct_srd_ep_is_connected_and_no_pending(ep) &&
        uct_srd_ep_has_fc_resources(ep)) {

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

void uct_srd_ep_send_completion(uct_srd_send_op_t *send_op)
{
    uct_srd_ep_t *ep = send_op->ep;
    uct_srd_send_op_t *q_send_op;
    ucs_queue_iter_t iter;

    /* TODO:Do a linear search in the queue to find the completed send op.
     * We can do better by using a hash table, and use a frag_list
     * instead of the queue to keep track of order for flush/rma.
     */
    ucs_queue_for_each_safe(q_send_op, iter, &ep->tx.outstanding_q, out_queue) {
        if (q_send_op == send_op) {
            ucs_queue_del_iter(&ep->tx.outstanding_q, iter);
            send_op->comp_handler(send_op);
            break;
        }
    }
    /* The send op must be found in the queue and released */
    ucs_assert(send_op->flags & UCT_SRD_SEND_OP_FLAG_INVALID);

    /* while queue head is flush desc, remove it and call user callback */
    ucs_queue_for_each_extract(q_send_op, &ep->tx.outstanding_q, out_queue,
                               q_send_op->flags & UCT_SRD_SEND_OP_FLAG_FLUSH) {
        /* outstanding flush must have completion callback. */
        ucs_assert(q_send_op->comp_handler == uct_srd_iface_send_op_ucomp_release);
        q_send_op->comp_handler(q_send_op);
    }
}

ucs_status_t uct_srd_ep_create(const uct_ep_params_t *params, uct_ep_h *ep_p)
{
    if (ucs_test_all_flags(params->field_mask, UCT_EP_PARAM_FIELD_DEV_ADDR |
                                               UCT_EP_PARAM_FIELD_IFACE_ADDR)) {
        return uct_srd_ep_create_connected(params, ep_p);
    }

    return UCS_CLASS_NEW_FUNC_NAME(uct_srd_ep_t)(params, ep_p);
}

ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length)
{
    uct_srd_ep_t *ep           = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface     = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_am_short_hdr_t *am = &iface->tx.am_inl_hdr;
    uct_srd_send_op_t *send_op;

    UCT_SRD_CHECK_AM_SHORT(iface, id, sizeof(am->hdr), length, "am_short");

    uct_srd_enter(iface);

    /* Because send completions can be out of order on EFA, we
     * need a dummy send op for am short to track completion order */
    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (!send_op) {
        uct_srd_leave(iface);
        return UCS_ERR_NO_RESOURCE;
    }

    UCT_SRD_AM_COMMON(iface, ep, id, &am->neth);

    am->hdr = hdr;

    send_op->comp_handler = uct_srd_iface_send_op_release;

    iface->tx.sge[0].addr    = (uintptr_t)am;
    iface->tx.sge[0].length  = sizeof(*am);
    iface->tx.sge[1].addr    = (uintptr_t)buffer;
    iface->tx.sge[1].length  = length;
    iface->tx.wr_inl.num_sge = 2;
    iface->tx.wr_inl.wr_id   = (uintptr_t)send_op;

    uct_srd_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE, 2);
    ucs_queue_push(&ep->tx.outstanding_q, &send_op->out_queue);
    uct_srd_iface_complete_tx_op(iface, ep, send_op);

    UCT_SRD_UPDATE_FC(iface, ep, am->neth.fc);

    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(hdr) + length);
    uct_srd_leave(iface);
    return UCS_OK;
}

ucs_status_t uct_srd_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                     const uct_iov_t *iov, size_t iovcnt)
{
    uct_srd_ep_t *ep           = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface     = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_am_short_hdr_t *am = &iface->tx.am_inl_hdr;
    uct_srd_send_op_t *send_op;

    UCT_SRD_CHECK_AM_SHORT(iface, id, 0, uct_iov_total_length(iov, iovcnt),
                           "am_short_iov");
    UCT_CHECK_IOV_SIZE(iovcnt, iface->config.max_send_sge - 1,
                       "uct_srd_ep_am_short_iov");

    uct_srd_enter(iface);

    /* Because send completions can be out of order on EFA, we
     * need a dummy send op for am short to track completion order */
    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (!send_op) {
        uct_srd_leave(iface);
        return UCS_ERR_NO_RESOURCE;
    }

    UCT_SRD_AM_COMMON(iface, ep, id, &am->neth);

    send_op->comp_handler = uct_srd_iface_send_op_release;

    iface->tx.sge[0].length  = sizeof(am->neth);
    iface->tx.sge[0].addr    = (uintptr_t)&am->neth;
    iface->tx.wr_inl.wr_id   = (uintptr_t)send_op;
    iface->tx.wr_inl.num_sge =
        uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1, iov, iovcnt) + 1;

    uct_srd_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE,
                      iface->tx.wr_inl.num_sge);
    ucs_queue_push(&ep->tx.outstanding_q, &send_op->out_queue);
    uct_srd_iface_complete_tx_op(iface, ep, send_op);

    UCT_SRD_UPDATE_FC(iface, ep, am->neth.fc);

    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, uct_iov_total_length(iov, iovcnt));
    uct_srd_leave(iface);
    return UCS_OK;
}

ssize_t uct_srd_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    uct_srd_neth_t *neth;
    size_t length;

    uct_srd_enter(iface);

    desc = uct_srd_ep_get_send_desc(iface, ep);
    if (!desc) {
        uct_srd_leave(iface);
        return UCS_ERR_NO_RESOURCE;
    }

    neth = uct_srd_send_desc_neth(desc);

    UCT_SRD_AM_COMMON(iface, ep, id, neth);

    length = pack_cb(neth + 1, arg);

    UCT_SRD_CHECK_AM_BCOPY(iface, id, length);

    desc->super.comp_handler = uct_srd_iface_send_op_release;
    desc->super.len          = sizeof(*neth) + length;

    ucs_assert(iface->tx.wr_desc.num_sge == 1);
    uct_srd_ep_tx_desc(iface, ep, desc, 0, INT_MAX);
    uct_srd_iface_complete_tx_desc(iface, ep, desc);

    UCT_SRD_UPDATE_FC(iface, ep, neth->fc);

    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
    uct_srd_leave(iface);
    return length;
}

ucs_status_t
uct_srd_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                    unsigned header_length, const uct_iov_t *iov,
                    size_t iovcnt, unsigned flags, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    uct_srd_neth_t *neth;

    UCT_CHECK_IOV_SIZE(iovcnt, iface->config.max_send_sge - 1,
                       "uct_srd_ep_am_zcopy");

    UCT_SRD_CHECK_AM_ZCOPY(iface, id, header_length,
                           uct_iov_total_length(iov, iovcnt));

    uct_srd_enter(iface);

    desc = uct_srd_ep_get_send_desc(iface, ep);
    if (!desc) {
        uct_srd_leave(iface);
        return UCS_ERR_NO_RESOURCE;
    }

    neth = uct_srd_send_desc_neth(desc);

    UCT_SRD_AM_COMMON(iface, ep, id, neth);

    memcpy(neth + 1, header, header_length);

    uct_srd_zcopy_op_set_comp(&desc->super, comp);
    desc->super.len = sizeof(*neth) + header_length;

    iface->tx.wr_desc.num_sge =
        uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1, iov, iovcnt) + 1;

    uct_srd_ep_tx_desc(iface, ep, desc, 0,
                       UCT_IB_MAX_ZCOPY_LOG_SGE(&iface->super));
    uct_srd_iface_complete_tx_desc(iface, ep, desc);
    iface->tx.wr_desc.num_sge = 1;

    UCT_SRD_UPDATE_FC(iface, ep, neth->fc);

    UCT_TL_EP_STAT_OP(&ep->super, AM, ZCOPY, header_length +
                      uct_iov_total_length(iov, iovcnt));
    uct_srd_leave(iface);
    return UCS_INPROGRESS;
}

#ifdef HAVE_DECL_EFA_DV_RDMA_READ
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_srd_ep_rma(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
               uct_srd_send_op_t *send_op, const struct ibv_sge *sge,
               size_t num_sge, uint64_t remote_addr, uct_rkey_t rkey)
{
    struct ibv_qp_ex *qp_ex = iface->qp_ex;

    send_op->flags |= UCT_SRD_SEND_OP_FLAG_RMA;

    ibv_wr_start(qp_ex);
    qp_ex->wr_id = (uintptr_t)send_op;
    ibv_wr_rdma_read(qp_ex, rkey, remote_addr);
    ibv_wr_set_sge_list(qp_ex, num_sge, sge);
    ibv_wr_set_ud_addr(qp_ex, ep->peer_address.ah,
                       ep->peer_address.dest_qpn, UCT_IB_KEY);
    if (ibv_wr_complete(qp_ex)) {
        ucs_fatal("ibv_wr_complete failed %m");
        return UCS_ERR_IO_ERROR;
    }

    iface->tx.available--;
    ucs_queue_push(&ep->tx.outstanding_q, &send_op->out_queue);
    return UCS_INPROGRESS;
}

static void
uct_srd_ep_get_bcopy_comp_handler_ucomp(uct_srd_send_op_t *send_op)

{
    uct_srd_send_desc_t *desc = ucs_derived_of(send_op, uct_srd_send_desc_t);

    desc->unpack_cb(desc->unpack_arg, desc + 1, desc->super.len);
    if (!(send_op->flags & UCT_SRD_SEND_OP_FLAG_PURGED)) {
        uct_invoke_completion(desc->super.user_comp, UCS_OK);
    }
    uct_srd_iface_send_op_release(send_op);
}

static void
uct_srd_ep_get_bcopy_comp_handler(uct_srd_send_op_t *send_op)

{
    uct_srd_send_desc_t *desc = ucs_derived_of(send_op, uct_srd_send_desc_t);

    desc->unpack_cb(desc->unpack_arg, desc + 1, desc->super.len);
    uct_srd_iface_send_op_release(send_op);
}

ucs_status_t uct_srd_ep_get_bcopy(uct_ep_h tl_ep,
                                  uct_unpack_callback_t unpack_cb,
                                  void *arg, size_t length,
                                  uint64_t remote_addr, uct_rkey_t rkey,
                                  uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    ucs_status_t status;

    UCT_CHECK_LENGTH(length, 0, iface->config.max_get_bcopy, "get_bcopy");

    uct_srd_enter(iface);

    desc = uct_srd_ep_get_send_desc(iface, ep);
    if (!desc) {
        uct_srd_leave(iface);
        return UCS_ERR_NO_RESOURCE;
    }

    desc->super.len  = length;
    desc->unpack_arg = arg;
    desc->unpack_cb  = unpack_cb;

    if (comp) {
        desc->super.user_comp    = comp;
        desc->super.comp_handler = uct_srd_ep_get_bcopy_comp_handler_ucomp;
    } else {
        desc->super.comp_handler = uct_srd_ep_get_bcopy_comp_handler;
    }

    iface->tx.sge[0].lkey   = desc->lkey;
    iface->tx.sge[0].length = length;
    iface->tx.sge[0].addr   = (uintptr_t)(desc + 1);

    status = uct_srd_ep_rma(iface, ep, &desc->super, iface->tx.sge, 1,
                            remote_addr, uct_ib_md_direct_rkey(rkey));
    uct_srd_iface_complete_tx_desc(iface, ep, desc);

    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, GET, BCOPY, length);
    }

    uct_srd_leave(iface);
    return UCS_INPROGRESS;
}

ucs_status_t uct_srd_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    size_t total_length    = uct_iov_total_length(iov, iovcnt);
    size_t num_sge;
    uct_srd_send_op_t *send_op;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, iface->config.max_send_sge,
                       "uct_srd_ep_get_zcopy");

    UCT_CHECK_LENGTH(total_length,
                     iface->super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1,
                     iface->config.max_get_zcopy, "get_zcopy");

    uct_srd_enter(iface);

    UCT_SRD_EP_ASSERT_PENDING(ep);

    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (!send_op) {
        return UCS_ERR_NO_RESOURCE;
    }

    uct_srd_zcopy_op_set_comp(send_op, comp);
    send_op->len = 0;

    num_sge = uct_ib_verbs_sge_fill_iov(iface->tx.sge, iov, iovcnt);
    status  = uct_srd_ep_rma(iface, ep, send_op, iface->tx.sge, num_sge,
                             remote_addr, uct_ib_md_direct_rkey(rkey));
    uct_srd_iface_complete_tx_op(iface, ep, send_op);

    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, GET, ZCOPY, total_length);
    }

    uct_srd_leave(iface);
    return status;
}
#endif /* HAVE_DECL_EFA_DV_RDMA_READ */

ucs_status_t uct_srd_ep_put_short(uct_ep_h tl_ep,
                                  const void *buffer, unsigned length,
                                  uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_srd_ep_t *ep           = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface     = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_put_hdr_t *put_hdr = &iface->tx.put_hdr;
    uct_srd_neth_t *neth       = &put_hdr->neth;
    uct_srd_send_op_t *send_op;

    UCT_CHECK_LENGTH(sizeof(*put_hdr) + length,
                     0, iface->config.max_inline, "put_short");

    uct_srd_enter(iface);

    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (!send_op) {
        uct_srd_leave(iface);
        return UCS_ERR_NO_RESOURCE;
    }

    put_hdr->rva          = remote_addr;
    send_op->comp_handler = uct_srd_iface_send_op_release;

    UCT_SRD_EP_ASSERT_PENDING(ep);
    UCT_SRD_CHECK_FC(iface, ep, neth->fc);
    uct_srd_neth_set_type_put(ep, neth);
    uct_srd_neth_set_psn(ep, neth);

    iface->tx.sge[0].addr    = (uintptr_t)put_hdr;
    iface->tx.sge[0].length  = sizeof(*put_hdr);
    iface->tx.sge[1].addr    = (uintptr_t)buffer;
    iface->tx.sge[1].length  = length;
    iface->tx.wr_inl.num_sge = 2;
    iface->tx.wr_inl.wr_id   = (uintptr_t)send_op;

    uct_srd_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE, 2);
    ucs_queue_push(&ep->tx.outstanding_q, &send_op->out_queue);
    uct_srd_iface_complete_tx_op(iface, ep, send_op);

    UCT_SRD_UPDATE_FC(iface, ep, neth->fc);

    UCT_TL_EP_STAT_OP(&ep->super, PUT, SHORT, length);
    uct_srd_leave(iface);
    return UCS_OK;
}

static ucs_status_t
uct_srd_ep_fc_init(uct_srd_ep_t *ep, int16_t wnd_size
                   UCS_STATS_ARG(ucs_stats_node_t* stats_parent))
{
    ucs_status_t status;

    ep->fc.fc_wnd = wnd_size;

    status = UCS_STATS_NODE_ALLOC(&ep->fc.stats, &uct_srd_fc_stats_class,
                                  stats_parent, "");
    if (status != UCS_OK) {
       return status;
    }

    UCS_STATS_SET_COUNTER(ep->fc.stats, UCT_SRD_FC_STAT_FC_WND, ep->fc.fc_wnd);

    return UCS_OK;
}

static void uct_srd_ep_fc_cleanup(uct_srd_ep_t *ep)
{
    UCS_STATS_NODE_FREE(ep->fc.stats);
}

static UCS_CLASS_INIT_FUNC(uct_srd_ep_t, const uct_ep_params_t* params)
{
    uct_srd_iface_t *iface = ucs_derived_of(params->iface, uct_srd_iface_t);
    ucs_status_t status;

    ucs_trace_func("");

    memset(self, 0, sizeof(*self));
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    uct_srd_enter(iface);

    self->dest_ep_id         = UCT_SRD_EP_NULL_ID;
    self->path_index         = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    self->peer_address.ah    = NULL;

    status = uct_srd_ep_fc_init(self, iface->config.fc_wnd_size
                                UCS_STATS_ARG(self->super.stats));
    if (status != UCS_OK) {
       return status;
    }

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

    uct_srd_ep_fc_cleanup(self);

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
