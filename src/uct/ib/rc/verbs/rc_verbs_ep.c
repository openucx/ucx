/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rc_verbs.h"

#include <ucs/arch/bitops.h>
#include <uct/ib/base/ib_log.h>


static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_posted(uct_rc_verbs_ep_t* ep, int signaled)
{
    uct_rc_ep_tx_posted(&ep->super, signaled);
    --ep->super.available;
    ++ep->tx.post_count;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_post_send(uct_rc_verbs_iface_t* iface, uct_rc_verbs_ep_t* ep,
                          struct ibv_send_wr *wr, int send_flags)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    if (!(send_flags & IBV_SEND_SIGNALED)) {
        send_flags |= uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                                 IBV_SEND_SIGNALED);
    }
    wr->send_flags = send_flags;
    wr->wr_id      = ep->super.unsignaled;

    uct_ib_log_post_send(&iface->super.super, ep->super.qp, wr,
                         (wr->opcode == IBV_WR_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    ret = ibv_post_send(ep->super.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_ep_posted(ep, send_flags & IBV_SEND_SIGNALED);
}

#if HAVE_DECL_IBV_EXP_POST_SEND
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_exp_post_send(uct_rc_verbs_ep_t *ep, struct ibv_exp_send_wr *wr,
                           int signal)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_exp_send_wr *bad_wr;
    int ret;

    if (!signal) {
         signal = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                             IBV_EXP_SEND_SIGNALED);
    }
    wr->exp_send_flags |= signal;
    wr->wr_id          = ep->super.unsignaled;

    uct_ib_log_exp_post_send(&iface->super.super, ep->super.qp, wr,
                             (wr->exp_opcode == IBV_EXP_WR_SEND) ? uct_rc_ep_am_packet_dump : NULL);

    ret = ibv_exp_post_send(ep->super.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_fatal("ibv_exp_post_send() returned %d (%m)", ret);
    }

    uct_rc_verbs_ep_posted(ep, signal);
}
#endif

static UCS_F_ALWAYS_INLINE void uct_rc_verbs_ep_push_desc(uct_rc_verbs_ep_t* ep,
                                                          uct_rc_iface_send_desc_t *desc)
{
    /* NOTE: We insert the descriptor with the sequence number after the post,
     * because when polling completions, we get the number of completions (rather
     * than completion zero-based index).
     */
    desc->super.sn = ep->tx.post_count;
    ucs_queue_push(&ep->super.outstanding, &desc->super.queue);
}

/*
 * Helper function for posting sends with a descriptor.
 * User needs to fill: wr.opcode, wr.sg_list, wr.num_sge, first sge length, and
 * operation-specific fields (e.g rdma).
 */
static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_post_send_desc(uct_rc_verbs_ep_t* ep, struct ibv_send_wr *wr,
                               uct_rc_iface_send_desc_t *desc, int send_flags)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_sge *sge;

    wr->next       = NULL;
    sge            = wr->sg_list;
    sge->addr      = (uintptr_t)(desc + 1);
    sge->lkey      = desc->lkey;

    uct_rc_verbs_ep_post_send(iface, ep, wr, send_flags);
    uct_rc_verbs_ep_push_desc(ep, desc);
}

static inline void uct_rc_verbs_fill_rdma_wr(struct ibv_send_wr *wr, int opcode,
                                             struct ibv_sge *sge, size_t length,
                                             uint64_t remote_addr, uct_rkey_t rkey)
{
    wr->wr.rdma.remote_addr = remote_addr;
    wr->wr.rdma.rkey        = rkey;
    wr->sg_list             = sge;
    wr->num_sge             = 1;
    wr->opcode              = opcode;
    sge->length             = length;
}

static inline ucs_status_t
uct_rc_verbs_ep_rdma_zcopy(uct_rc_verbs_ep_t *ep, const void *buffer, size_t length,
                           struct ibv_mr *mr, uint64_t remote_addr,
                           uct_rkey_t rkey, uct_completion_t *comp,
                           int opcode)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    UCT_SKIP_ZERO_LENGTH(length);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    uct_rc_verbs_fill_rdma_wr(&wr, opcode, &sge, length, remote_addr, rkey);
    wr.next                = NULL;
    sge.addr               = (uintptr_t)buffer;
    sge.lkey               = (mr == UCT_INVALID_MEM_HANDLE) ? 0 : mr->lkey;

    uct_rc_verbs_ep_post_send(iface, ep, &wr, IBV_SEND_SIGNALED);
    uct_rc_ep_add_send_comp(&iface->super, &ep->super, comp, ep->tx.post_count);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_atomic_post(uct_rc_verbs_ep_t *ep, int opcode, uint64_t compare_add,
                            uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                            uct_rc_iface_send_desc_t *desc, int force_sig)
{
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    wr.sg_list               = &sge;
    wr.num_sge               = 1;
    wr.opcode                = opcode;
    wr.wr.atomic.compare_add = compare_add;
    wr.wr.atomic.swap        = swap;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey        = rkey;
    sge.length               = sizeof(uint64_t);

    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, force_sig);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_ep_atomic(uct_rc_verbs_ep_t *ep, int opcode, void *result,
                       uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_PARAM(comp != NULL, "completion must be non-NULL");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->short_desc_mp, desc);

    desc->super.handler   = iface->config.atomic64_handler;
    desc->super.buffer    = result;
    desc->super.user_comp = comp;

    uct_rc_verbs_ep_atomic_post(ep, opcode, compare_add, swap, remote_addr,
                                rkey, desc, IBV_SEND_SIGNALED);
    return UCS_INPROGRESS;
}

#if HAVE_IB_EXT_ATOMICS
static inline ucs_status_t
uct_rc_verbs_ext_atomic_post(uct_rc_verbs_ep_t *ep, int opcode, uint32_t length,
                             uint64_t compare_mask, uint64_t compare_add,
                             uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                             uct_rc_iface_send_desc_t *desc, int force_sig,
                             ucs_status_t success)
{
    struct ibv_exp_send_wr wr;
    struct ibv_sge sge;

    sge.addr          = (uintptr_t)(desc + 1);
    sge.lkey          = desc->lkey;
    sge.length        = length;
    wr.next           = NULL;
    wr.sg_list        = &sge;
    wr.num_sge        = 1;
    wr.exp_opcode     = opcode;
    wr.exp_send_flags = IBV_EXP_SEND_EXT_ATOMIC_INLINE;
    wr.comp_mask      = 0;

    wr.ext_op.masked_atomics.log_arg_sz  = ucs_ilog2(length);
    wr.ext_op.masked_atomics.remote_addr = remote_addr;
    wr.ext_op.masked_atomics.rkey        = rkey;

    switch (opcode) {
    case IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP:
        wr.ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_mask = compare_mask;
        wr.ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.compare_val  = compare_add;
        wr.ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_mask    = (uint64_t)(-1);
        wr.ext_op.masked_atomics.wr_data.inline_data.op.cmp_swap.swap_val     = swap;
        break;
    case IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD:
        wr.ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.add_val        = compare_add;
        wr.ext_op.masked_atomics.wr_data.inline_data.op.fetch_add.field_boundary = 0;
        break;
    }

    UCT_TL_EP_STAT_ATOMIC(&ep->super.super);
    uct_rc_verbs_exp_post_send(ep, &wr, force_sig);
    uct_rc_verbs_ep_push_desc(ep, desc);
    return success;
}

static inline ucs_status_t
uct_rc_verbs_ext_atomic(uct_rc_verbs_ep_t *ep, int opcode, void *result,
                        uint32_t length, uint64_t compare_mask,
                        uint64_t compare_add, uint64_t swap, uint64_t remote_addr,
                        uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_CHECK_PARAM(comp != NULL, "completion must be non-NULL");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->short_desc_mp, desc);

    switch (length) {
    case sizeof(uint32_t):
        desc->super.handler = iface->config.atomic32_handler;
        break;
    case sizeof(uint64_t):
        desc->super.handler = iface->config.atomic64_handler;
        break;
    }

    desc->super.buffer    = result;
    desc->super.user_comp = comp;
    return uct_rc_verbs_ext_atomic_post(ep, opcode, length, compare_mask, compare_add,
                                        swap, remote_addr, rkey, desc,
                                        IBV_EXP_SEND_SIGNALED, UCS_INPROGRESS);
}
#endif

ucs_status_t uct_rc_verbs_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                       unsigned length, uint64_t remote_addr,
                                       uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    UCT_CHECK_LENGTH(length, iface->config.max_inline, "put_short");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    iface->inl_rwrite_wr.wr.rdma.remote_addr = remote_addr;
    iface->inl_rwrite_wr.wr.rdma.rkey        = rkey;
    iface->inl_sge[0].addr                   = (uintptr_t)buffer;
    iface->inl_sge[0].length                 = length;

    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, SHORT, length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_rwrite_wr,
                              IBV_SEND_INLINE | IBV_SEND_SIGNALED);
    return UCS_OK;
}

ssize_t uct_rc_verbs_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                  void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    size_t length;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    length = pack_cb(desc + 1, arg);
    UCT_SKIP_ZERO_LENGTH(length, desc);

    uct_rc_verbs_fill_rdma_wr(&wr, IBV_WR_RDMA_WRITE, &sge, length, remote_addr,
                              rkey);

    UCT_TL_EP_STAT_OP(&ep->super.super, PUT, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED);
    return length;
}

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                       uct_mem_h memh, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    status = uct_rc_verbs_ep_rdma_zcopy(ep, buffer, length, memh, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_WRITE);
    UCT_TL_EP_STAT_OP_IF_SUCCESS(status, &ep->super.super, PUT, ZCOPY, length);
    return status;
}

ucs_status_t uct_rc_verbs_ep_get_bcopy(uct_ep_h tl_ep,
                                       uct_unpack_callback_t unpack_cb,
                                       void *arg, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    UCT_CHECK_LENGTH(length, iface->super.super.config.seg_size, "get_bcopy");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);

    ucs_assert(length <= iface->super.super.config.seg_size);

    desc->super.handler     = (comp == NULL) ?
                                uct_rc_ep_get_bcopy_handler_no_completion :
                                uct_rc_ep_get_bcopy_handler;
    desc->super.unpack_arg  = arg;
    desc->super.user_comp   = comp;
    desc->super.length      = length;
    desc->unpack_cb         = unpack_cb;

    uct_rc_verbs_fill_rdma_wr(&wr, IBV_WR_RDMA_READ, &sge, length, remote_addr,
                              rkey);

    UCT_TL_EP_STAT_OP(&ep->super.super, GET, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uct_mem_h memh, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    status = uct_rc_verbs_ep_rdma_zcopy(ep, buffer, length, memh, remote_addr,
                                        rkey, comp, IBV_WR_RDMA_READ);
    if (status == UCS_INPROGRESS) {
        UCT_TL_EP_STAT_OP(&ep->super.super, GET, ZCOPY, length);
    }
    return status;
}

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      const void *buffer, unsigned length)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_am_short_hdr_t am;

    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(sizeof(am) + length, iface->config.max_inline, "am_short");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);

    am.rc_hdr.am_id             = id;
    am.am_hdr                   = hdr;
    iface->inl_sge[0].addr      = (uintptr_t)&am;
    iface->inl_sge[0].length    = sizeof(am);
    iface->inl_sge[1].addr      = (uintptr_t)buffer;
    iface->inl_sge[1].length    = length;

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, SHORT, sizeof(hdr) + length);
    uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr, IBV_SEND_INLINE);
    return UCS_OK;
}

ssize_t uct_rc_verbs_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    uct_rc_hdr_t *rch;
    size_t length;

    UCT_CHECK_AM_ID(id);
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->super.tx.mp, desc);

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;

    rch = (void*)(desc + 1);
    rch->am_id = id;
    length = pack_cb(rch + 1, arg);

    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode  = IBV_WR_SEND;
    sge.length = sizeof(*rch) + length;

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, BCOPY, length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, 0);
    return length;
}

static void uct_rc_verbs_ep_am_zcopy_handler(uct_rc_iface_send_op_t *op)
{
    uct_rc_iface_send_desc_t *desc = ucs_derived_of(op, uct_rc_iface_send_desc_t);
    uct_invoke_completion(desc->super.user_comp);
    ucs_mpool_put(desc);
}

ucs_status_t uct_rc_verbs_ep_am_zcopy(uct_ep_h tl_ep, uint8_t id, const void *header,
                                      unsigned header_length, const void *payload,
                                      size_t length, uct_mem_h memh,
                                      uct_completion_t *comp)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    struct ibv_mr *mr = memh;
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge[2];
    uct_rc_hdr_t *rch;
    int send_flags;

    UCT_CHECK_AM_ID(id);
    UCT_CHECK_LENGTH(sizeof(*rch) + header_length, iface->config.short_desc_size,
                     "am_zcopy header");
    UCT_CHECK_LENGTH(header_length + length, iface->super.super.config.seg_size,
                     "am_zcopy payload");
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->short_desc_mp, desc);

    if (comp == NULL) {
        desc->super.handler   = (uct_rc_send_handler_t)ucs_mpool_put;
        send_flags            = 0;
    } else {
        desc->super.handler   = uct_rc_verbs_ep_am_zcopy_handler;
        desc->super.user_comp = comp;
        send_flags            = IBV_SEND_SIGNALED;
    }

    /* Header buffer: active message ID + user header */
    rch = (void*)(desc + 1);
    rch->am_id = id;
    memcpy(rch + 1, header, header_length);

    wr.sg_list    = sge;
    wr.opcode     = IBV_WR_SEND;
    sge[0].length = sizeof(*rch) + header_length;

    if (ucs_unlikely(length == 0)) {
        wr.num_sge    = 1;
    } else {
        wr.num_sge    = 2;
        sge[1].addr   = (uintptr_t)payload;
        sge[1].length = length;
        sge[1].lkey   = (mr == UCT_INVALID_MEM_HANDLE) ? 0 : mr->lkey;
    }

    UCT_TL_EP_STAT_OP(&ep->super.super, AM, ZCOPY, header_length + length);
    uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags);
    return UCS_INPROGRESS;
}

ucs_status_t uct_rc_verbs_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    /* TODO don't allocate descriptor - have dummy buffer */
    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->short_desc_mp, desc);

    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    uct_rc_verbs_ep_atomic_post(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                IBV_WR_ATOMIC_FETCH_AND_ADD, add, 0,
                                remote_addr, rkey, desc,
                                IBV_SEND_SIGNALED);
    return UCS_OK;
}

ucs_status_t uct_rc_verbs_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_FETCH_AND_ADD, result, add, 0,
                                  remote_addr, rkey, comp);
}

ucs_status_t uct_rc_verbs_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint64_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   result, sizeof(uint64_t), 0, 0, swap, remote_addr,
                                   rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint64_t *result, uct_completion_t *comp)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_CMP_AND_SWP, result, compare, swap,
                                  remote_addr, rkey, comp);
}

ucs_status_t uct_rc_verbs_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
#if HAVE_IB_EXT_ATOMICS
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    UCT_RC_IFACE_GET_TX_DESC(&iface->super, &iface->short_desc_mp, desc);

    /* TODO don't allocate descriptor - have dummy buffer */
    desc->super.handler = (uct_rc_send_handler_t)ucs_mpool_put;
    return uct_rc_verbs_ext_atomic_post(ep, IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                        sizeof(uint32_t), 0, add, 0, remote_addr,
                                        rkey, desc, IBV_EXP_SEND_SIGNALED, UCS_OK);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                   result, sizeof(uint32_t), 0, add, 0,
                                   remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   result, sizeof(uint32_t), 0, 0, swap,
                                   remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uint32_t *result, uct_completion_t *comp)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   result, sizeof(uint32_t), (uint32_t)(-1),
                                   compare, swap, remote_addr, rkey, comp);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_nop(uct_rc_verbs_ep_t *ep)
{
#if HAVE_DECL_IBV_EXP_WR_NOP
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_exp_send_wr wr;

    wr.next           = NULL;
    wr.num_sge        = 0;
    wr.exp_opcode     = IBV_EXP_WR_NOP;
    wr.exp_send_flags = IBV_EXP_SEND_FENCE;
    wr.comp_mask      = 0;

    UCT_RC_CHECK_RES(&iface->super, &ep->super);
    uct_rc_verbs_exp_post_send(ep, &wr, IBV_EXP_SEND_SIGNALED);
    return UCS_OK;
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    if (ep->super.available == iface->super.config.tx_qp_len) {
        UCT_TL_EP_STAT_FLUSH(&ep->super.super);
        return UCS_OK;
    }

    if (ep->super.unsignaled != 0) {
        if (IBV_DEVICE_HAS_NOP(&uct_ib_iface_device(&iface->super.super)->dev_attr)) {
            status = uct_rc_verbs_ep_nop(ep);
        } else {
            status = uct_rc_verbs_ep_put_short(tl_ep, NULL, 0, 0, 0);
        }
        if (status != UCS_OK) {
            return status;
        }
    }
    UCT_TL_EP_STAT_FLUSH_WAIT(&ep->super.super);
    return UCS_INPROGRESS;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_rc_ep_t, &iface->super);

    self->super.available     = iface->super.config.tx_qp_len;
    self->tx.post_count       = 0;
    self->tx.completion_count = 0;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_rc_verbs_ep_t)
{
}

UCS_CLASS_DEFINE(uct_rc_verbs_ep_t, uct_rc_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_rc_verbs_ep_t, uct_ep_t, uct_iface_h);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_rc_verbs_ep_t, uct_ep_t);

