/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_verbs.h"


static inline ucs_status_t
uct_rc_verbs_ep_post_send(uct_rc_verbs_iface_t* iface, uct_rc_verbs_ep_t* ep,
                          struct ibv_send_wr *wr, int signaled)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    if (ep->tx.available == 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    wr->wr_id = ep->super.tx.unsignaled;

    ret = ibv_post_send(ep->super.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_error("ibv_post_send() returned %d (%m)", ret);
        return UCS_ERR_IO_ERROR;
    }

    if (signaled) {
        ep->super.tx.unsignaled = 0;
    } else {
        ++ep->super.tx.unsignaled;
    }

    --ep->tx.available;
    ++ep->tx.pi;
    ++iface->super.tx.outstanding;
    return UCS_OK;
}

static inline ucs_status_t
uct_rc_verbs_ep_post_send_desc(uct_rc_verbs_ep_t* ep, struct ibv_send_wr *wr,
                               uct_rc_iface_send_desc_t *desc, unsigned send_flags)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_verbs_iface_t);
    struct ibv_sge *sge;
    ucs_status_t status;

    desc->queue.sn = ep->tx.pi;
    wr->send_flags = send_flags;
    wr->next       = NULL;
    sge            = wr->sg_list;
    sge->addr      = (uintptr_t)(desc + 1);
    sge->lkey      = desc->lkey;

    status = uct_rc_verbs_ep_post_send(iface, ep, wr, send_flags & IBV_SEND_SIGNALED);
    if (status != UCS_OK) {
        ucs_mpool_put(desc);
        return status;
    }

    ucs_callbackq_push(&ep->super.tx.comp, &desc->queue);
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);

    iface->inl_rwrite_wr.wr.rdma.remote_addr = remote_addr;
    iface->inl_rwrite_wr.wr.rdma.rkey        = ntohl(rkey);
    iface->inl_sge[0].addr                   = (uintptr_t)buffer;
    iface->inl_sge[0].length                 = length;
    return uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_rwrite_wr, 1);
}

ucs_status_t uct_rc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                       void *arg, size_t length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    desc = ucs_mpool_get(iface->super.tx.mp);
    if (desc == NULL) {
        return UCS_ERR_WOULD_BLOCK;
    }

    desc->queue.super.func = (void*)ucs_mpool_put;

    ucs_assert(length <= iface->super.super.config.seg_size);
    pack_cb(desc + 1, arg, length);

    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey        = ntohl(rkey);
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE;
    sge.length             = length;
    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED);
}

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uct_lkey_t lkey, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    uint16_t sn;
    ucs_status_t status;

    wr.wr.rdma.remote_addr = remote_addr;
    wr.wr.rdma.rkey        = ntohl(rkey);
    wr.sg_list             = &sge;
    wr.num_sge             = 1;
    wr.opcode              = IBV_WR_RDMA_WRITE;
    wr.next                = NULL;
    wr.send_flags          = IBV_SEND_SIGNALED;
    sge.addr               = (uintptr_t)buffer;
    sge.length             = length;
    sge.lkey               = (lkey == UCT_INVALID_MEM_KEY) ? 0 : uct_ib_lkey_mr(lkey)->lkey;
    sn                     = ep->tx.pi;

    status = uct_rc_verbs_ep_post_send(iface, ep, &wr, 1);
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_ep_add_user_completion(&ep->super, comp, sn);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      void *buffer, unsigned length)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_am_short_hdr_t am;
    int send_flags;

    send_flags = uct_rc_iface_tx_moderation(&iface->super, &ep->super,IBV_SEND_SIGNALED) |
                 IBV_SEND_INLINE;

    am.rc_hdr.am_id             = id;
    am.am_hdr                   = hdr;
    iface->inl_sge[0].addr      = (uintptr_t)&am;
    iface->inl_sge[0].length    = sizeof(am);
    iface->inl_sge[1].addr      = (uintptr_t)buffer;
    iface->inl_sge[1].length    = length;
    iface->inl_am_wr.send_flags = send_flags;
    return uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr,
                                     send_flags & IBV_SEND_SIGNALED);
}

ucs_status_t uct_rc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                      uct_pack_callback_t pack_cb, void *arg,
                                      size_t length)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    uct_rc_hdr_t *rch;
    int send_flags;

    desc = ucs_mpool_get(iface->super.tx.mp);
    if (desc == NULL) {
        return UCS_ERR_WOULD_BLOCK;
    }

    desc->queue.super.func = (void*)ucs_mpool_put;

    ucs_assert(sizeof(*rch) + length <= iface->super.super.config.seg_size);
    rch = (void*)(desc + 1);
    rch->am_id = id;
    pack_cb(rch + 1, arg, length);

    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode  = IBV_WR_SEND;
    sge.length = sizeof(*rch) + length;
    send_flags = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                            IBV_SEND_SIGNALED);
    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags);
}

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, void *header,
                                      unsigned header_length, void *payload,
                                      size_t length, uct_lkey_t lkey,
                                      uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge[2];
    uct_rc_hdr_t *rch;
    unsigned send_flags;
    ucs_status_t status;

    desc = ucs_mpool_get(iface->am_hdr_mp);
    if (desc == NULL) {
        return UCS_ERR_WOULD_BLOCK;
    }

    send_flags = (comp == NULL) ?
                 uct_rc_iface_tx_moderation(&iface->super, &ep->super,IBV_SEND_SIGNALED) :
                 IBV_SEND_SIGNALED;

    rch = (void*)(desc + 1);

    /* Header buffer: active message ID + user header */
    rch->am_id = id;
    ucs_assert(sizeof(*rch) + header_length <= iface->super.config.tx_min_inline);
    memcpy(rch + 1, header, header_length);

    wr.sg_list    = sge;
    wr.num_sge    = 2;
    wr.opcode     = IBV_WR_SEND;
    sge[0].length = sizeof(*rch) + header_length;
    sge[1].addr   = (uintptr_t)payload;
    sge[1].length = length;
    sge[1].lkey   = (lkey == UCT_INVALID_MEM_KEY) ? 0 : uct_ib_lkey_mr(lkey)->lkey;

    status = uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags);
    if (status != UCS_OK) {
        return status;;
    }

    uct_rc_ep_add_user_completion(&ep->super, comp, desc->queue.sn);
    return UCS_INPROGRESS; /* No local completion yet */
}

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    if (ep->super.tx.unsignaled == 0) {
        return UCS_OK;
    }

    /* TODO use NOP */
    return uct_rc_verbs_ep_put_short(tl_ep, NULL, 0, 0, 0);
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(tl_iface);

    self->tx.available = iface->super.config.tx_qp_len;
    self->tx.pi        = 0;
    self->tx.ci        = 0;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

