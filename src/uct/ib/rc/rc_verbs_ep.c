/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_verbs.h"


ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    struct ibv_send_wr *bad_wr;
    int UCS_V_UNUSED ret;

    if (ep->tx.available == 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    iface->inl_rwrite_wr.wr_id               = ep->tx.unsignaled + 1;
    iface->inl_rwrite_wr.wr.rdma.remote_addr = remote_addr;
    iface->inl_rwrite_wr.wr.rdma.rkey        = ntohl(rkey);
    iface->inl_sge[0].addr                   = (uintptr_t)buffer;
    iface->inl_sge[0].length                 = length;

    ret = ibv_post_send(ep->super.qp, &iface->inl_rwrite_wr, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);

    ep->tx.unsignaled = 0;
    --ep->tx.available;
    ++iface->super.tx.outstanding;
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      void *buffer, unsigned length)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    struct ibv_send_wr *bad_wr;
    int UCS_V_UNUSED ret;
    struct {
        uct_rc_hdr_t rc;
        uint64_t     u;
    } UCS_S_PACKED h;

    if (ep->tx.available == 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    h.rc.am_id                = id;
    h.u                       = hdr;
    iface->inl_am_wr.wr_id    = ep->tx.unsignaled + 1;
    if (ep->tx.unsignaled >= iface->super.config.tx_moderation) {
        iface->inl_am_wr.send_flags = IBV_SEND_INLINE|IBV_SEND_SIGNALED;
        ep->tx.unsignaled = 0;
    } else {
        iface->inl_am_wr.send_flags = IBV_SEND_INLINE;
        ++ep->tx.unsignaled;
    }
    iface->inl_sge[0].addr    = (uintptr_t)&h;
    iface->inl_sge[0].length  = sizeof(h);
    iface->inl_sge[1].addr    = (uintptr_t)buffer;
    iface->inl_sge[1].length  = length;

    ret = ibv_post_send(ep->super.qp, &iface->inl_am_wr, &bad_wr);
    ucs_assertv(ret == 0, "ibv_post_send() returned %d (%m)", ret);

    ucs_trace_data("TX: AM [%d]", id);

    --ep->tx.available;
    ++iface->super.tx.outstanding;
    return UCS_OK;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(tl_iface);

    self->tx.available  = iface->super.config.tx_qp_len;
    self->tx.unsignaled = 0;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

