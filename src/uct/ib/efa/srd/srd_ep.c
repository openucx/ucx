/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "srd_ep.h"
#include "srd_log.h"
#include "srd_iface.h"

#include <uct/ib/base/ib_log.h>


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
        send_op->comp_cb(send_op, UCS_OK);
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
            send_op->comp_cb(send_op, UCS_ERR_CANCELED);
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

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_posted(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    iface->tx.available--;
    ep->inflight++;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_post_send(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                     struct ibv_send_wr *wr, unsigned send_flags)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    wr->wr.ud.remote_qpn = ep->dest_qpn;
    wr->wr.ud.ah         = ep->ah;
    wr->send_flags       = send_flags;

    uct_ib_log_post_send(&iface->super, iface->qp, wr, 2, uct_srd_dump_packet);

    ret = ibv_post_send(iface->qp, wr, &bad_wr);
    if (ucs_unlikely(ret != 0)) {
        ucs_fatal("ibv_post_send(iface=%p) returned %d bad_wr=%p (%m)", iface,
                  ret, bad_wr);
    }

    uct_srd_ep_posted(iface, ep);
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

static UCS_F_ALWAYS_INLINE uct_srd_send_desc_t *
uct_srd_ep_get_send_desc(uct_srd_iface_t *iface, uct_srd_ep_t *ep)
{
    uct_srd_send_desc_t *send_desc = uct_srd_iface_get_send_desc(iface);

    if (ucs_unlikely(send_desc == NULL)) {
        ucs_trace_poll("iface=%p ep=%p has no send_desc resource (psn=%u)",
                       iface, ep, ep->psn);
        UCS_STATS_UPDATE_COUNTER(ep->super.stats, UCT_EP_STAT_NO_RES, 1);
        return NULL;
    }

    send_desc->super.ep = ep;
    return send_desc;
}

static UCS_F_ALWAYS_INLINE void
uct_srd_ep_hdr_set(const uct_srd_ep_t *ep, uct_srd_hdr_t *neth, uint8_t id)
{
    neth->ep_uuid = ep->ep_uuid;
    neth->psn     = ep->psn;
    neth->id      = id;
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_srd_ep_am_short_prepare(
        uct_srd_iface_t *iface, uct_srd_ep_t *ep, uint8_t id, size_t am_length)
{
    uct_srd_am_short_hdr_t *am = &iface->tx.am_inl_hdr;
    uct_srd_send_op_t *send_op;

    /* Use an internal send_op for am_short to track completion ordering */
    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (send_op == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    uct_srd_ep_hdr_set(ep, &am->srd_hdr, id);
    send_op->comp_cb         = (uct_srd_send_op_comp_t)ucs_empty_function;
    iface->tx.sge[0].addr    = (uintptr_t)am;
    iface->tx.sge[0].length  = am_length;
    iface->tx.wr_inl.wr_id   = (uintptr_t)send_op;

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void uct_srd_ep_am_short_post(uct_srd_iface_t *iface,
                                                         uct_srd_ep_t *ep,
                                                         size_t length)
{
    uct_srd_send_op_t *send_op = (uct_srd_send_op_t*)iface->tx.wr_inl.wr_id;

    uct_srd_ep_post_send(iface, ep, &iface->tx.wr_inl, IBV_SEND_INLINE);
    ucs_list_add_tail(&iface->tx.outstanding_list, &send_op->list);

    UCT_TL_EP_STAT_OP(&ep->super, AM, SHORT, length);
}

ucs_status_t uct_srd_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                 const void *buffer, unsigned length)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    ucs_status_t status;

    UCT_SRD_CHECK_AM_SHORT(iface, id, sizeof(hdr), length);

    status = uct_srd_ep_am_short_prepare(iface, ep, id,
                                         sizeof(uct_srd_am_short_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    iface->tx.am_inl_hdr.am_hdr = hdr;

    iface->tx.sge[1].addr    = (uintptr_t)buffer;
    iface->tx.sge[1].length  = length;
    iface->tx.wr_inl.num_sge = 2;

    uct_srd_ep_am_short_post(iface, ep, sizeof(hdr) + length);
    return UCS_OK;
}

ucs_status_t uct_srd_ep_am_short_iov(uct_ep_h tl_ep, uint8_t id,
                                     const uct_iov_t *iov, size_t iovcnt)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    size_t length          = uct_iov_total_length(iov, iovcnt);
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt,
                       ucs_min(UCT_IB_MAX_IOV, iface->config.max_send_sge) - 1,
                       "uct_srd_ep_am_short_iov");
    UCT_SRD_CHECK_AM_SHORT(iface, id, 0, length);

    status = uct_srd_ep_am_short_prepare(iface, ep, id, sizeof(uct_srd_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    iface->tx.wr_inl.num_sge = 1 + uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1,
                                                             iov, iovcnt);

    uct_srd_ep_am_short_post(iface, ep, length);
    return UCS_OK;
}

static void uct_srd_ep_send_op_user_completion(uct_srd_send_op_t *send_op,
                                               ucs_status_t status)
{
    if (send_op->user_comp != NULL) {
        uct_invoke_completion(send_op->user_comp, status);
    }
}

static void uct_srd_ep_send_desc_tx(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                                    uct_srd_send_desc_t *desc, size_t length)
{
    iface->tx.sge[0].addr   = (uintptr_t)desc->hdr;
    iface->tx.sge[0].length = sizeof(*desc->hdr) + length;
    iface->tx.sge[0].lkey   = desc->lkey;
    iface->tx.wr_desc.wr_id = (uintptr_t)&desc->super;

    uct_srd_ep_post_send(iface, ep, &iface->tx.wr_desc, 0);

    ucs_list_add_tail(&iface->tx.outstanding_list, &desc->super.list);
}

ucs_status_t uct_srd_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                 unsigned header_length, const uct_iov_t *iov,
                                 size_t iovcnt, unsigned flags,
                                 uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    size_t length;
    uct_srd_send_desc_t *desc;

    UCT_CHECK_IOV_SIZE(iovcnt,
                       ucs_min(UCT_IB_MAX_IOV, iface->config.max_send_sge) - 1,
                       "uct_srd_ep_am_zcopy");
    length = uct_iov_total_length(iov, iovcnt);
    UCT_SRD_CHECK_AM_ZCOPY(iface, id, header_length, length);

    desc = uct_srd_ep_get_send_desc(iface, ep);
    if (desc == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    desc->super.user_comp = comp;
    desc->super.comp_cb   = uct_srd_ep_send_op_user_completion;

    uct_srd_ep_hdr_set(ep, desc->hdr, id);
    memcpy(desc->hdr + 1, header, header_length);
    iface->tx.wr_desc.num_sge = 1 + uct_ib_verbs_sge_fill_iov(iface->tx.sge + 1,
                                                              iov, iovcnt);
    uct_srd_ep_send_desc_tx(iface, ep, desc, header_length);

    UCT_TL_EP_STAT_OP(&ep->super, AM, ZCOPY, header_length + length);
    return UCS_INPROGRESS;
}

ssize_t uct_srd_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                            uct_pack_callback_t pack_cb, void *arg,
                            unsigned flags)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    size_t length;

    UCT_CHECK_AM_ID(id);

    desc = uct_srd_ep_get_send_desc(iface, ep);
    if (desc == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    length = pack_cb(desc->hdr + 1, arg);
    ucs_assertv((sizeof(uct_srd_hdr_t) + length) <=
                    iface->super.config.seg_size,
                "ep=%p am_bcopy_length=%zu seg_size=%d", ep,
                sizeof(uct_srd_hdr_t) + length, iface->super.config.seg_size);

    desc->super.comp_cb = (uct_srd_send_op_comp_t)ucs_empty_function;

    uct_srd_ep_hdr_set(ep, desc->hdr, id);
    iface->tx.wr_desc.num_sge = 1;
    uct_srd_ep_send_desc_tx(iface, ep, desc, length);

    UCT_TL_EP_STAT_OP(&ep->super, AM, BCOPY, length);
    return length;
}

#define UCT_SRD_EP_LOG_RMA(_iface, _ep, _send_op, _sge, _num_sge, \
                           _remote_addr, _rkey) \
    if (ucs_log_is_enabled(UCS_LOG_LEVEL_TRACE_DATA)) { \
        struct ibv_send_wr __wr = { \
            .wr_id               = (uintptr_t)(_send_op), \
            .send_flags          = 0, \
            .opcode              = IBV_WR_RDMA_READ, \
            .wr.rdma.remote_addr = (_remote_addr), \
            .wr.rdma.rkey        = (_rkey), \
            .sg_list             = (struct ibv_sge*)(_sge), \
            .num_sge             = (_num_sge), \
            .next                = NULL \
        }; \
        __uct_ib_log_post_send_one(__FILE__, __LINE__, __func__, \
                                   &(_iface)->super, (_iface)->qp, &__wr, \
                                   (_ep)->ah, (_ep)->dest_qpn, 2, NULL); \
    }

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_srd_ep_post_rma(uct_srd_iface_t *iface, uct_srd_ep_t *ep,
                    uct_srd_send_op_t *send_op, const struct ibv_sge *sge,
                    size_t num_sge, uint64_t remote_addr, uct_rkey_t rkey)
{
#ifdef HAVE_DECL_EFADV_DEVICE_ATTR_CAPS_RDMA_READ
    struct ibv_qp_ex *qp_ex = iface->qp_ex;

    UCT_SRD_EP_LOG_RMA(iface, ep, send_op, sge, num_sge, remote_addr, rkey);

    ibv_wr_start(qp_ex);
    qp_ex->wr_id = (uintptr_t)send_op;
    ibv_wr_rdma_read(qp_ex, rkey, remote_addr);
    ibv_wr_set_sge_list(qp_ex, num_sge, sge);
    ibv_wr_set_ud_addr(qp_ex, ep->ah, ep->dest_qpn, UCT_IB_KEY);
    if (ibv_wr_complete(qp_ex)) {
        ucs_fatal("ibv_wr_complete failed %m");
        return UCS_ERR_IO_ERROR;
    }

    uct_srd_ep_posted(iface, ep);
    ucs_list_add_tail(&iface->tx.outstanding_list, &send_op->list);
    return UCS_INPROGRESS;
#else
    ucs_mpool_put(send_op);
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_srd_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                  size_t iovcnt, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    size_t length          = uct_iov_total_length(iov, iovcnt);
    size_t num_sge;
    uct_srd_send_op_t *send_op;
    ucs_status_t status;

    UCT_CHECK_IOV_SIZE(iovcnt, iface->config.max_recv_sge,
                       "uct_srd_ep_get_zcopy");

    UCT_CHECK_LENGTH(length, iface->super.config.max_inl_cqe[UCT_IB_DIR_TX] + 1,
                     iface->config.max_get_zcopy, "get_zcopy");

    send_op = uct_srd_ep_get_send_op(iface, ep);
    if (send_op == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    send_op->user_comp = comp;
    send_op->comp_cb   = uct_srd_ep_send_op_user_completion;

    num_sge = uct_ib_verbs_sge_fill_iov(iface->tx.sge, iov, iovcnt);
    status  = uct_srd_ep_post_rma(iface, ep, send_op, iface->tx.sge, num_sge,
                                  remote_addr, uct_ib_md_direct_rkey(rkey));
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, GET, ZCOPY, length);
    }

    return status;
}

static void uct_srd_send_op_get_bcopy_completion(uct_srd_send_op_t *send_op,
                                                 ucs_status_t status)
{
    uct_srd_send_desc_t *desc = ucs_derived_of(send_op, uct_srd_send_desc_t);

    if (status == UCS_OK) {
        desc->unpack_cb(desc->unpack_arg, desc + 1, desc->length);
    }

    uct_srd_ep_send_op_user_completion(send_op, status);
}

ucs_status_t uct_srd_ep_get_bcopy(uct_ep_h tl_ep,
                                  uct_unpack_callback_t unpack_cb, void *arg,
                                  size_t length, uint64_t remote_addr,
                                  uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_srd_ep_t *ep       = ucs_derived_of(tl_ep, uct_srd_ep_t);
    uct_srd_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_srd_iface_t);
    uct_srd_send_desc_t *desc;
    ucs_status_t status;

    UCT_CHECK_LENGTH(length, 0, iface->config.max_get_bcopy, "get_bcopy");

    desc = uct_srd_ep_get_send_desc(iface, ep);
    if (desc == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    desc->unpack_arg      = arg;
    desc->unpack_cb       = unpack_cb;
    desc->length          = length;
    desc->super.user_comp = comp;
    desc->super.comp_cb   = uct_srd_send_op_get_bcopy_completion;

    iface->tx.sge[0].lkey   = desc->lkey;
    iface->tx.sge[0].length = length;
    iface->tx.sge[0].addr   = (uintptr_t)(desc + 1);

    status = uct_srd_ep_post_rma(iface, ep, &desc->super, iface->tx.sge, 1,
                                 remote_addr, uct_ib_md_direct_rkey(rkey));
    if (!UCS_STATUS_IS_ERR(status)) {
        UCT_TL_EP_STAT_OP(&ep->super, GET, BCOPY, length);
    }

    return UCS_INPROGRESS;
}
