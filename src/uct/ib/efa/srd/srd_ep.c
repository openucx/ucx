/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_ep.h"
#include "srd_iface.h"


static UCS_CLASS_INIT_FUNC(uct_srd_ep_t, const uct_ep_params_t *params)
{
    uct_srd_iface_t *iface              = ucs_derived_of(params->iface,
                                                         uct_srd_iface_t);
    const uct_ib_address_t *ib_addr     = (const uct_ib_address_t*)
                                                  params->dev_addr;
    const uct_srd_iface_addr_t *if_addr = (const uct_srd_iface_addr_t*)
                                                  params->iface_addr;
    char str[128], buf[128];
    ucs_status_t status;
    enum ibv_mtu path_mtu;
    struct ibv_ah_attr ah_attr;

    ucs_trace_func("");

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->ep_uuid    = ucs_generate_uuid((uintptr_t)self);
    self->path_index = UCT_EP_PARAMS_GET_PATH_INDEX(params);
    self->psn        = UCT_SRD_INITIAL_PSN;
    self->inflight   = 0;

    uct_ib_iface_fill_ah_attr_from_addr(&iface->super, ib_addr,
                                        self->path_index, &ah_attr, &path_mtu);
    status = uct_ib_iface_create_ah(&iface->super, &ah_attr, "SRD AH",
                                    &self->ah);
    if (status != UCS_OK) {
        return status;
    }

    self->dest_qpn = uct_ib_unpack_uint24(if_addr->qp_num);

    ucs_debug(UCT_IB_IFACE_FMT
              " ep=%p gid=%s qpn=0x%x ep_uuid=0x%"PRIx64" connected "
              "to iface %s qpn=0x%x",
              UCT_IB_IFACE_ARG(&iface->super),
              self, uct_ib_gid_str(&iface->super.gid_info.gid, str, sizeof(str)),
              iface->qp->qp_num, self->ep_uuid,
              uct_ib_address_str(ib_addr, buf, sizeof(buf)),
              uct_ib_unpack_uint24(if_addr->qp_num));
    return UCS_OK;
}

void uct_srd_ep_send_op_completion(uct_srd_send_op_t *send_op)
{
    if (send_op->ep != NULL) {
        send_op->ep->inflight--;
    }

    ucs_list_del(&send_op->list);
    ucs_mpool_put(send_op);
}

static void uct_srd_ep_send_op_purge(uct_srd_ep_t *ep)
{
    uct_srd_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                            uct_srd_iface_t);
    uct_srd_send_op_t *send_op;

    ucs_list_for_each(send_op, &iface->tx.outstanding_list, list) {
        if (send_op->ep == ep) {
            /*
             * Make ep invalid, as ibv_poll_cq() will return this
             * send_op after it has been released.
             */
            send_op->ep = NULL;
            if (--ep->inflight == 0) {
                break;
            }
        }
    }
}

static UCS_CLASS_CLEANUP_FUNC(uct_srd_ep_t)
{
    ucs_trace_func("");

    if (self->inflight != 0) {
        uct_srd_ep_send_op_purge(self);
        ucs_assertv(self->inflight == 0,
                    "ep=%p failed to complete %u send operations",
                    self, self->inflight);
    }
}

UCS_CLASS_DEFINE(uct_srd_ep_t, uct_base_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_srd_ep_t, uct_ep_t, const uct_ep_params_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_srd_ep_t, uct_ep_t);

static UCS_F_ALWAYS_INLINE
void uct_srd_post_send(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                       struct ibv_send_wr *wr, unsigned send_flags,
                       unsigned max_log_sge)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    wr->wr.ud.remote_qpn = ep->dest_qpn;
    wr->wr.ud.ah         = ep->ah;
    wr->send_flags       = send_flags;

    ret = ibv_post_send(iface->qp, wr, &bad_wr);
    if (ucs_unlikely(ret != 0)) {
        ucs_fatal("ibv_post_send(iface=%p) returned %d (%m)", iface, ret);
    }

    iface->tx.available--;
    ep->psn++;
}

static UCS_F_ALWAYS_INLINE uct_srd_send_op_t *
uct_srd_ep_get_send_op(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    uct_srd_send_op_t *send_op = uct_srd_iface_get_send_op(iface);

    if (ucs_unlikely(send_op == NULL)) {
        ucs_trace_poll("iface=%p ep=%p has no send_op resource (psn=%u)",
                       iface, ep, ep->psn);
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return NULL;
    }

    send_op->ep = ep;
    return send_op;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_hdr_set(const uct_srd_ep_t *ep, uct_srd_hdr_t *neth, uint8_t id)
{
    neth->ep_uuid = ep->ep_uuid;
    neth->psn     = ep->psn;
    neth->id      = id;
}

ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length)
{
    uct_srd_ep_t *ep           = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface     = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_am_short_hdr_t *am = &iface->tx.am_inl_hdr;
    uct_srd_send_op_t *send_op;

    UCT_SRD_CHECK_AM_SHORT(iface, id, sizeof(am->am_hdr), length);

    /* Use an internal send_op for am_short to track completion ordering */
    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (send_op == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    uct_srd_hdr_set(ep, &am->srd_hdr, id);
    am->am_hdr = hdr;

    iface->tx.sge[0].addr    = (uintptr_t)am;
    iface->tx.sge[0].length  = sizeof(*am);
    iface->tx.sge[1].addr    = (uintptr_t)buffer;
    iface->tx.sge[1].length  = length;
    iface->tx.wr_inl.num_sge = 2;
    iface->tx.wr_inl.wr_id   = (uintptr_t)send_op;

    uct_srd_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE, 2);
    ucs_list_add_tail(&iface->tx.outstanding_list, &send_op->list);
    ep->inflight++;

    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, sizeof(am->am_hdr) + length);
    return UCS_OK;
}
