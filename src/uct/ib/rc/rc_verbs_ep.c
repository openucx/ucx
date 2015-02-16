/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "rc_verbs.h"


static UCS_F_ALWAYS_INLINE void
uct_rc_verbs_ep_posted(uct_rc_verbs_iface_t* iface, uct_rc_verbs_ep_t* ep,
                       int signaled)
{
    if (signaled) {
        ep->super.tx.unsignaled = 0;
    } else {
        ++ep->super.tx.unsignaled;
        --iface->super.tx.cq_available;
    }

    --ep->tx.available;
    ++ep->tx.post_count;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_ep_post_send(uct_rc_verbs_iface_t* iface, uct_rc_verbs_ep_t* ep,
                          struct ibv_send_wr *wr, int send_flags)
{
    struct ibv_send_wr *bad_wr;
    int ret;

    if (ep->tx.available == 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    if (!(send_flags & IBV_SEND_SIGNALED)) {
        send_flags |= uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                                 IBV_SEND_SIGNALED);
    }

    if ((send_flags & IBV_SEND_SIGNALED) && !uct_rc_iface_have_tx_cqe_avail(&iface->super)) {
        return UCS_ERR_WOULD_BLOCK;
    }

    wr->send_flags = send_flags;
    wr->wr_id      = ep->super.tx.unsignaled;

    ret = ibv_post_send(ep->super.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_error("ibv_post_send() returned %d (%m)", ret);
        return UCS_ERR_IO_ERROR;
    }

    uct_rc_verbs_ep_posted(iface, ep, send_flags & IBV_SEND_SIGNALED);
    return UCS_OK;
}

#if HAVE_DECL_IBV_EXP_POST_SEND
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_exp_post_send(uct_rc_verbs_ep_t *ep, struct ibv_exp_send_wr *wr,
                           int signal)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_exp_send_wr *bad_wr;
    int ret;

    if (ep->tx.available == 0) {
        return UCS_ERR_WOULD_BLOCK;
    }

    if (!signal) {
        signal = uct_rc_iface_tx_moderation(&iface->super, &ep->super,
                                            IBV_EXP_SEND_SIGNALED);
    }

    if (signal && !uct_rc_iface_have_tx_cqe_avail(&iface->super)) {
        return UCS_ERR_WOULD_BLOCK;
    }

    wr->exp_send_flags |= signal;
    ret = ibv_exp_post_send(ep->super.qp, wr, &bad_wr);
    if (ret != 0) {
        ucs_error("ibv_exp_post_send() returned %d (%m)", ret);
        return UCS_ERR_IO_ERROR;
    }

    uct_rc_verbs_ep_posted(iface, ep, signal);
    return UCS_OK;
}
#endif

static UCS_F_ALWAYS_INLINE void uct_rc_verbs_ep_push_desc(uct_rc_verbs_ep_t* ep,
                                                          uct_rc_iface_send_desc_t *desc)
{
    /* NOTE: We insert the descriptor with the sequence number after the post,
     * because when polling completions, we get the number of completions (rather
     * than completion zero-based index).
     */
    desc->queue.sn = ep->tx.post_count;
    ucs_callbackq_push(&ep->super.tx.comp, &desc->queue);
}

/*
 * Helper function for posting sends with a descriptor.
 * User needs to fill: wr.opcode, wr.sg_list, wr.num_sge, first sge length, and
 * operation-specific fields (e.g rdma).
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_ep_post_send_desc(uct_rc_verbs_ep_t* ep, struct ibv_send_wr *wr,
                               uct_rc_iface_send_desc_t *desc, int send_flags,
                               ucs_status_t success)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.iface, uct_rc_verbs_iface_t);
    struct ibv_sge *sge;
    ucs_status_t status;

    wr->next       = NULL;
    sge            = wr->sg_list;
    sge->addr      = (uintptr_t)(desc + 1);
    sge->lkey      = desc->lkey;

    status = uct_rc_verbs_ep_post_send(iface, ep, wr, send_flags);
    if (status != UCS_OK) {
        ucs_mpool_put(desc);
        return status;
    }

    uct_rc_verbs_ep_push_desc(ep, desc);
    return success;
}

static inline void uct_rc_verbs_fill_rdma_wr(struct ibv_send_wr *wr, int opcode,
                                             struct ibv_sge *sge, size_t length,
                                             uint64_t remote_addr, uct_rkey_t rkey)
{
    wr->wr.rdma.remote_addr = remote_addr;
    wr->wr.rdma.rkey        = ntohl(rkey);
    wr->sg_list             = sge;
    wr->num_sge             = 1;
    wr->opcode              = opcode;
    sge->length             = length;
}

static inline ucs_status_t
uct_rc_verbs_ep_rdma_zcopy(uct_rc_verbs_ep_t *ep, void *buffer, size_t length,
                           uct_lkey_t lkey, uint64_t remote_addr,
                           uct_rkey_t rkey, uct_completion_t *comp,
                           int opcode)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_verbs_iface_t);
    struct ibv_send_wr wr;
    struct ibv_sge sge;
    ucs_status_t status;

    uct_rc_verbs_fill_rdma_wr(&wr, opcode, &sge, length, remote_addr, rkey);
    wr.next                = NULL;
    sge.addr               = (uintptr_t)buffer;
    sge.lkey               = (lkey == UCT_INVALID_MEM_KEY) ? 0 : uct_ib_lkey_mr(lkey)->lkey;

    status = uct_rc_verbs_ep_post_send(iface, ep, &wr, IBV_SEND_SIGNALED);
    if (status != UCS_OK) {
        return status;
    }

    uct_rc_ep_add_user_completion(&ep->super, comp, ep->tx.post_count);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_ep_atomic_post(uct_rc_verbs_ep_t *ep, int opcode, uint64_t compare_add,
                            uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                            uct_rc_iface_send_desc_t *desc, int force_sig,
                            ucs_status_t success)
{
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    wr.sg_list               = &sge;
    wr.num_sge               = 1;
    wr.opcode                = opcode;
    wr.wr.atomic.compare_add = compare_add;
    wr.wr.atomic.swap        = swap;
    wr.wr.atomic.remote_addr = remote_addr;
    wr.wr.atomic.rkey        = ntohl(rkey);
    sge.length               = sizeof(uint64_t);

    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, force_sig, success);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_rc_verbs_ep_atomic(uct_rc_verbs_ep_t *ep, int opcode, uint64_t compare_add,
                       uint64_t swap, uint64_t remote_addr, uct_rkey_t rkey,
                       uct_imm_recv_callback_t cb, void *arg)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_TL_IFACE_GET_TX_DESC(iface->short_desc_mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = iface->config.atomic64_completoin;
    desc->imm_recv.cb      = cb;
    desc->imm_recv.arg     = arg;
    return uct_rc_verbs_ep_atomic_post(ep, opcode, compare_add, swap, remote_addr,
                                       rkey, desc, IBV_SEND_SIGNALED, UCS_INPROGRESS);
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
    ucs_status_t status;
    struct ibv_sge sge;

    sge.addr          = (uintptr_t)(desc + 1);
    sge.lkey          = desc->lkey;
    sge.length        = length;
    wr.wr_id          = ep->super.tx.unsignaled;
    wr.next           = NULL;
    wr.sg_list        = &sge;
    wr.num_sge        = 1;
    wr.exp_opcode     = opcode;
    wr.exp_send_flags = IBV_EXP_SEND_EXT_ATOMIC_INLINE;
    wr.comp_mask      = 0;

    wr.ext_op.masked_atomics.log_arg_sz  = ucs_ilog2(length);
    wr.ext_op.masked_atomics.remote_addr = remote_addr;
    wr.ext_op.masked_atomics.rkey        = htonl(rkey);

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

    status = uct_rc_verbs_exp_post_send(ep, &wr, force_sig);
    if (status != UCS_OK) {
        ucs_mpool_put(desc);
        return status;
    }

    uct_rc_verbs_ep_push_desc(ep, desc);
    return success;
}

static inline ucs_status_t
uct_rc_verbs_ext_atomic(uct_rc_verbs_ep_t *ep, int opcode,uint32_t length,
                        uint64_t compare_mask, uint64_t compare_add, uint64_t swap,
                        uint64_t remote_addr, uct_rkey_t rkey,
                        uct_imm_recv_callback_t cb, void *arg)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(ep->super.super.iface,
                                                 uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_TL_IFACE_GET_TX_DESC(iface->short_desc_mp, desc, UCS_ERR_WOULD_BLOCK);

    switch (length) {
    case sizeof(uint32_t):
        desc->queue.super.func = iface->config.atomic32_completoin;
        break;
    case sizeof(uint64_t):
        desc->queue.super.func = iface->config.atomic64_completoin;
        break;
    }

    desc->imm_recv.cb  = cb;
    desc->imm_recv.arg = arg;
    return uct_rc_verbs_ext_atomic_post(ep, opcode, length, compare_mask, compare_add,
                                        swap, remote_addr, rkey, desc,
                                        IBV_EXP_SEND_SIGNALED, UCS_INPROGRESS);
}
#endif

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
    return uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_rwrite_wr,
                                     IBV_SEND_INLINE | IBV_SEND_SIGNALED);
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

    UCT_TL_IFACE_GET_TX_DESC(iface->super.tx.mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = (ucs_callback_func_t)ucs_mpool_put;

    ucs_assert(length <= iface->super.super.config.seg_size);
    pack_cb(desc + 1, arg, length);

    uct_rc_verbs_fill_rdma_wr(&wr, IBV_WR_RDMA_WRITE, &sge, length, remote_addr,
                              rkey);
    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED, UCS_OK);
}

ucs_status_t uct_rc_verbs_ep_put_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uct_lkey_t lkey, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    return uct_rc_verbs_ep_rdma_zcopy(ep, buffer, length, lkey, remote_addr,
                                      rkey, comp, IBV_WR_RDMA_WRITE);
}

ucs_status_t uct_rc_verbs_ep_get_bcopy(uct_ep_h tl_ep, size_t length,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_bcopy_recv_callback_t cb, void *arg)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;
    struct ibv_send_wr wr;
    struct ibv_sge sge;

    UCT_TL_IFACE_GET_TX_DESC(iface->super.tx.mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = uct_rc_ep_get_bcopy_completion;
    desc->bcopy_recv.cb     = cb;
    desc->bcopy_recv.arg    = arg;
    desc->bcopy_recv.length = length;

    ucs_assert(length <= iface->super.super.config.seg_size);

    uct_rc_verbs_fill_rdma_wr(&wr, IBV_WR_RDMA_READ, &sge, length, remote_addr,
                              rkey);
    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, IBV_SEND_SIGNALED,
                                          UCS_INPROGRESS);
}

ucs_status_t uct_rc_verbs_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                       uct_lkey_t lkey, uint64_t remote_addr,
                                       uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);

    return uct_rc_verbs_ep_rdma_zcopy(ep, buffer, length, lkey, remote_addr,
                                      rkey, comp, IBV_WR_RDMA_READ);
}

ucs_status_t uct_rc_verbs_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t hdr,
                                      void *buffer, unsigned length)
{
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_am_short_hdr_t am;

    am.rc_hdr.am_id             = id;
    am.am_hdr                   = hdr;
    iface->inl_sge[0].addr      = (uintptr_t)&am;
    iface->inl_sge[0].length    = sizeof(am);
    iface->inl_sge[1].addr      = (uintptr_t)buffer;
    iface->inl_sge[1].length    = length;
    return uct_rc_verbs_ep_post_send(iface, ep, &iface->inl_am_wr, IBV_SEND_INLINE);
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

    UCT_TL_IFACE_GET_TX_DESC(iface->super.tx.mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = (void*)ucs_mpool_put;

    ucs_assert(sizeof(*rch) + length <= iface->super.super.config.seg_size);
    rch = (void*)(desc + 1);
    rch->am_id = id;
    pack_cb(rch + 1, arg, length);

    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode  = IBV_WR_SEND;
    sge.length = sizeof(*rch) + length;
    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, 0, UCS_OK);
}

static void uct_rc_verbs_ep_am_zcopy_completion(ucs_callback_t *self)
{
    uct_rc_iface_send_desc_t *desc = ucs_container_of(self, uct_rc_iface_send_desc_t,
                                                      queue.super);
    ucs_callback_t *cb = desc->callback.cb;

    cb->func(cb);
    ucs_mpool_put(desc);
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
    int send_flags;

    UCT_TL_IFACE_GET_TX_DESC(iface->short_desc_mp, desc, UCS_ERR_WOULD_BLOCK);
    if (comp == NULL) {
        desc->queue.super.func = (ucs_callback_func_t)ucs_mpool_put;
        send_flags             = 0;
    } else {
        desc->queue.super.func = uct_rc_verbs_ep_am_zcopy_completion;
        desc->callback.cb      = &comp->super;
        send_flags             = IBV_SEND_SIGNALED;
    }

    /* Header buffer: active message ID + user header */
    rch = (void*)(desc + 1);
    rch->am_id = id;
    memcpy(rch + 1, header, header_length);

    wr.sg_list    = sge;
    wr.num_sge    = 2;
    wr.opcode     = IBV_WR_SEND;
    sge[0].length = sizeof(*rch) + header_length;
    sge[1].addr   = (uintptr_t)payload;
    sge[1].length = length;
    sge[1].lkey   = (lkey == UCT_INVALID_MEM_KEY) ? 0 : uct_ib_lkey_mr(lkey)->lkey;

    return uct_rc_verbs_ep_post_send_desc(ep, &wr, desc, send_flags, UCS_INPROGRESS);
}

ucs_status_t uct_rc_verbs_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_iface_send_desc_t *desc;

    /* TODO don't allocate descriptor - have dummy buffer */
    UCT_TL_IFACE_GET_TX_DESC(iface->short_desc_mp, desc, UCS_ERR_WOULD_BLOCK);
    desc->queue.super.func = (ucs_callback_func_t)ucs_mpool_put;

    return uct_rc_verbs_ep_atomic_post(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                       IBV_WR_ATOMIC_FETCH_AND_ADD, add, 0,
                                       remote_addr, rkey, desc,
                                       IBV_SEND_SIGNALED, UCS_OK);
}

ucs_status_t uct_rc_verbs_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_FETCH_AND_ADD, add, 0,
                                  remote_addr, rkey, cb, arg);
}

ucs_status_t uct_rc_verbs_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_imm_recv_callback_t cb, void *arg)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   sizeof(uint64_t), 0, 0, swap, remote_addr,
                                   rkey, cb, arg);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uct_imm_recv_callback_t cb, void *arg)
{
    return uct_rc_verbs_ep_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                  IBV_WR_ATOMIC_CMP_AND_SWP, compare, swap,
                                  remote_addr, rkey, cb, arg);
}

ucs_status_t uct_rc_verbs_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey)
{
#if HAVE_IB_EXT_ATOMICS
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    uct_rc_iface_send_desc_t *desc;

    UCT_TL_IFACE_GET_TX_DESC(iface->short_desc_mp, desc, UCS_ERR_WOULD_BLOCK);

    /* TODO don't allocate descriptor - have dummy buffer */
    desc->queue.super.func = (ucs_callback_func_t)ucs_mpool_put;
    return uct_rc_verbs_ext_atomic_post(ep, IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                        sizeof(uint32_t), 0, add, 0, remote_addr,
                                        rkey, desc, IBV_EXP_SEND_SIGNALED, UCS_OK);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_imm_recv_callback_t cb, void *arg)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_FETCH_AND_ADD,
                                   sizeof(uint32_t), 0, add, 0, remote_addr,
                                   rkey, cb, arg);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_imm_recv_callback_t cb, void *arg)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   sizeof(uint32_t), 0, 0, swap, remote_addr,
                                   rkey, cb, arg);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                            uint64_t remote_addr, uct_rkey_t rkey,
                                            uct_imm_recv_callback_t cb, void *arg)
{
#if HAVE_IB_EXT_ATOMICS
    return uct_rc_verbs_ext_atomic(ucs_derived_of(tl_ep, uct_rc_verbs_ep_t),
                                   IBV_EXP_WR_EXT_MASKED_ATOMIC_CMP_AND_SWP,
                                   sizeof(uint32_t), (uint32_t)(-1), compare, swap,
                                   remote_addr, rkey, cb, arg);
#else
    return UCS_ERR_UNSUPPORTED;
#endif
}

ucs_status_t uct_rc_verbs_ep_flush(uct_ep_h tl_ep)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_rc_verbs_iface_t);
    uct_rc_verbs_ep_t *ep = ucs_derived_of(tl_ep, uct_rc_verbs_ep_t);
    ucs_status_t status;

    if (ep->tx.available == iface->super.config.tx_qp_len) {
        return UCS_OK;
    }

    if (ep->super.tx.unsignaled != 0) {
#if HAVE_DECL_IBV_EXP_WR_NOP
        if (uct_ib_iface_device(&iface->super.super)->dev_attr.exp_device_cap_flags
                        & IBV_EXP_DEVICE_NOP)
        {
            struct ibv_exp_send_wr wr = {
                .wr_id          = ep->super.tx.unsignaled,
                .next           = NULL,
                .num_sge        = 0,
                .exp_opcode     = IBV_EXP_WR_NOP,
                .exp_send_flags = IBV_EXP_SEND_FENCE,
                .comp_mask      = 0
            };
            status = uct_rc_verbs_exp_post_send(ep, &wr, IBV_EXP_SEND_SIGNALED);
        } else
#endif
        {
            status = uct_rc_verbs_ep_put_short(tl_ep, NULL, 0, 0, 0);
        }
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_INPROGRESS;
}

static UCS_CLASS_INIT_FUNC(uct_rc_verbs_ep_t, uct_iface_h tl_iface)
{
    uct_rc_verbs_iface_t *iface = ucs_derived_of(tl_iface, uct_rc_verbs_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(tl_iface);

    self->tx.available        = iface->super.config.tx_qp_len;
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

