/**
* Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ugni_rdma_ep.h"
#include "ugni_rdma_iface.h"
#include <uct/ugni/base/ugni_device.h>

#define UCT_CHECK_PARAM_IOV(_iov, _iovcnt, _buffer, _length, _memh) \
    void     *_buffer; \
    size_t    _length; \
    uct_mem_h _memh; \
    \
    UCT_CHECK_PARAM(1 == _iovcnt, "iov[iovcnt] has to be 1 at this time"); \
    _buffer = _iov[0].buffer; \
    _length = _iov[0].length; \
    _memh   = _iov[0].memh;

/* Endpoint operations */
static inline void uct_ugni_invoke_orig_comp(uct_ugni_rdma_fetch_desc_t *fma_desc, ucs_status_t status)
{
    if (ucs_likely(NULL != fma_desc->orig_comp_cb)) {
        uct_invoke_completion(fma_desc->orig_comp_cb, status);
    }
}

static inline void uct_ugni_format_fma(uct_ugni_base_desc_t *fma_desc, gni_post_type_t type,
                                       const void *buffer, uint64_t remote_addr,
                                       uct_rkey_t rkey, unsigned length, uct_ugni_ep_t *ep,
                                       uct_completion_t *comp,
                                       uct_unpack_callback_t unpack_cb)
{
    fma_desc->desc.type            = type;
    fma_desc->desc.cq_mode         = GNI_CQMODE_GLOBAL_EVENT;
    fma_desc->desc.dlvr_mode       = GNI_DLVMODE_PERFORMANCE;
    fma_desc->desc.local_addr      = (uint64_t)buffer;
    fma_desc->desc.remote_addr     = remote_addr;
    fma_desc->desc.remote_mem_hndl = *(gni_mem_handle_t *)rkey;
    fma_desc->desc.length          = length;
    fma_desc->flush_group          = ep->flush_group;
    fma_desc->comp_cb              = comp;
    fma_desc->unpack_cb            = unpack_cb;
}

static inline void uct_ugni_format_fma_amo(uct_ugni_rdma_fetch_desc_t *amo, gni_post_type_t type,
                                           gni_fma_cmd_type_t amo_op,
                                           uint64_t first_operand, uint64_t second_operand,
                                           void *buffer, uint64_t remote_addr,
                                           uct_rkey_t rkey, unsigned length, uct_ugni_ep_t *ep,
                                           uct_completion_t *comp,
                                           uct_completion_callback_t unpack_cb, void *arg)
{
    if (NULL != comp) {
        amo->orig_comp_cb = comp;
        comp              = &amo->tmp;
        amo->tmp.func     = unpack_cb;
        amo->tmp.count    = 1;
        amo->tmp.status   = UCS_OK;
    }

    uct_ugni_format_fma(&amo->super, GNI_POST_AMO, buffer, remote_addr,
                        rkey, length, ep, comp, NULL);

    amo->super.desc.amo_cmd = amo_op;
    amo->super.desc.first_operand = first_operand;
    amo->super.desc.second_operand = second_operand;
    amo->user_buffer = arg;
}

static inline void uct_ugni_format_rdma(uct_ugni_base_desc_t *rdma, gni_post_type_t type,
                                        const void *buffer, uint64_t remote_addr,
                                        uct_mem_h memh, uct_rkey_t rkey,
                                        unsigned length, uct_ugni_ep_t *ep,
                                        gni_cq_handle_t cq,
                                        uct_completion_t *comp)
{
    rdma->desc.type = type;
    rdma->desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
    rdma->desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE;
    rdma->desc.local_addr = (uint64_t) buffer;
    rdma->desc.local_mem_hndl = *(gni_mem_handle_t *)memh;
    rdma->desc.remote_addr = remote_addr;
    rdma->desc.remote_mem_hndl = *(gni_mem_handle_t *)rkey;
    rdma->desc.length = length;
    rdma->desc.src_cq_hndl = cq;
    rdma->flush_group = ep->flush_group;
    rdma->comp_cb = comp;
}

static inline ucs_status_t uct_ugni_post_rdma(uct_ugni_rdma_iface_t *iface,
                                              uct_ugni_ep_t *ep,
                                              uct_ugni_base_desc_t *rdma)
{
    gni_return_t ugni_rc;

    if (ucs_unlikely(!uct_ugni_ep_can_send(ep))) {
        ucs_mpool_put(rdma);
        return UCS_ERR_NO_RESOURCE;
    }
    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_PostRdma(ep->ep, &rdma->desc);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    if (ucs_unlikely(GNI_RC_SUCCESS != ugni_rc)) {
        ucs_mpool_put(rdma);
        if(GNI_RC_ERROR_RESOURCE == ugni_rc || GNI_RC_ERROR_NOMEM == ugni_rc) {
            ucs_debug("GNI_PostRdma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_NO_RESOURCE;
        } else {
            ucs_error("GNI_PostRdma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_IO_ERROR;
        }
    }

    ++rdma->flush_group->flush_comp.count;
    ++iface->super.outstanding;

    return UCS_INPROGRESS;
}

static inline ssize_t uct_ugni_post_fma(uct_ugni_rdma_iface_t *iface,
                                        uct_ugni_ep_t *ep,
                                        uct_ugni_base_desc_t *fma_desc,
                                        ssize_t ok_status)
{
    gni_return_t ugni_rc;

    if (ucs_unlikely(!uct_ugni_ep_can_send(ep))) {
        ucs_mpool_put(fma_desc);
        return UCS_ERR_NO_RESOURCE;
    }
    uct_ugni_cdm_lock(&iface->super.cdm);
    ugni_rc = GNI_PostFma(ep->ep, &fma_desc->desc);
    uct_ugni_cdm_unlock(&iface->super.cdm);
    if (ucs_unlikely(GNI_RC_SUCCESS != ugni_rc)) {
        ucs_mpool_put(fma_desc);
        if(GNI_RC_ERROR_RESOURCE == ugni_rc || GNI_RC_ERROR_NOMEM == ugni_rc) {
            ucs_debug("GNI_PostFma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_NO_RESOURCE;
        } else {
            ucs_error("GNI_PostFma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_IO_ERROR;
        }
    }

    ++fma_desc->flush_group->flush_comp.count;
    ++iface->super.outstanding;

    return ok_status;
}

ucs_status_t uct_ugni_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                   unsigned length, uint64_t remote_addr,
                                   uct_rkey_t rkey)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_base_desc_t *fma_desc;

    UCT_SKIP_ZERO_LENGTH(length);
    UCT_CHECK_LENGTH(length, 0, iface->config.fma_seg_size, "put_short");
    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc,
                             fma_desc, return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma(fma_desc, GNI_POST_FMA_PUT, buffer,
                        remote_addr, rkey, length, ep, NULL, NULL);
    ucs_trace_data("Posting PUT Short, GNI_PostFma of size %"PRIx64" from %p to "
                   "%p, with [%"PRIx64" %"PRIx64"]",
                   fma_desc->desc.length,
                   (void *)fma_desc->desc.local_addr,
                   (void *)fma_desc->desc.remote_addr,
                   fma_desc->desc.remote_mem_hndl.qword1,
                   fma_desc->desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    return uct_ugni_post_fma(iface, ep, fma_desc, UCS_OK);
}

ssize_t uct_ugni_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                              void *arg, uint64_t remote_addr, uct_rkey_t rkey)
{
    /* Since custom pack function is used
     * we have to allocate separate memory to pack
     * the info and pass it to FMA
     * something like:
     * pack_cb(desc + 1, arg, length); */
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_base_desc_t *fma_desc;
    size_t length;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_buffer,
                             fma_desc, return UCS_ERR_NO_RESOURCE);

    length = pack_cb(fma_desc + 1, arg);
    UCT_SKIP_ZERO_LENGTH(length, fma_desc);
    UCT_CHECK_LENGTH(length, 0, iface->config.fma_seg_size, "put_bcopy");
    uct_ugni_format_fma(fma_desc, GNI_POST_FMA_PUT, fma_desc + 1,
                        remote_addr, rkey, length, ep, NULL, NULL);
    ucs_trace_data("Posting PUT BCOPY, GNI_PostFma of size %"PRIx64" from %p to "
                   "%p, with [%"PRIx64" %"PRIx64"]",
                   fma_desc->desc.length,
                   (void *)fma_desc->desc.local_addr,
                   (void *)fma_desc->desc.remote_addr,
                   fma_desc->desc.remote_mem_hndl.qword1,
                   fma_desc->desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, BCOPY, length);
    return uct_ugni_post_fma(iface, ep, fma_desc, length);
}

ucs_status_t uct_ugni_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_base_desc_t *rdma;

    UCT_CHECK_PARAM_IOV(iov, iovcnt, buffer, length, memh);
    UCT_SKIP_ZERO_LENGTH(length);
    UCT_CHECK_LENGTH(length, 0, iface->config.rdma_max_size, "put_zcopy");
    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc, rdma,
                             return UCS_ERR_NO_RESOURCE);
    /* Setup Callback */
    uct_ugni_format_rdma(rdma, GNI_POST_RDMA_PUT, buffer, remote_addr, memh,
                         rkey, length, ep, iface->super.local_cq, comp);

    ucs_trace_data("Posting PUT ZCOPY, GNI_PostRdma of size %"PRIx64" from %p to %p, with [%"PRIx64" %"PRIx64"]",
                   rdma->desc.length,
                   (void *)rdma->desc.local_addr,
                   (void *)rdma->desc.remote_addr,
                   rdma->desc.remote_mem_hndl.qword1,
                   rdma->desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY, length);
    return uct_ugni_post_rdma(iface, ep, rdma);
}

static void uct_ugni_amo_unpack64(uct_completion_t *self)
{
    uct_ugni_rdma_fetch_desc_t *fma_desc = (uct_ugni_rdma_fetch_desc_t *)
        ucs_container_of(self, uct_ugni_rdma_fetch_desc_t, tmp);

    /* Call the original callback and skip padding */
    *(uint64_t *)fma_desc->user_buffer = *(uint64_t *)(fma_desc + 1);
    uct_ugni_invoke_orig_comp(fma_desc, self->status);
}

ucs_status_t uct_ugni_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uint64_t *result, uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_famo, fma_desc,
                             return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma_amo(fma_desc, GNI_POST_AMO, GNI_FMA_ATOMIC_CSWAP,
                            compare, swap, fma_desc + 1, remote_addr,
                            rkey, LEN_64, ep, comp, uct_ugni_amo_unpack64, (void *)result);
    ucs_trace_data("Posting AMO CSWAP, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx64" compare %"PRIx64" to %p, with [%"PRIx64" %"PRIx64"]",
                   fma_desc->super.desc.length, swap, compare,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_INPROGRESS);
}

static void uct_ugni_amo_unpack32(uct_completion_t *self)
{
    uct_ugni_rdma_fetch_desc_t *fma_desc = (uct_ugni_rdma_fetch_desc_t *)
        ucs_container_of(self, uct_ugni_rdma_fetch_desc_t, tmp);

    /* Call the original callback and skip padding */
    *(uint32_t *)fma_desc->user_buffer = *(uint32_t *)(fma_desc + 1);
    uct_ugni_invoke_orig_comp(fma_desc, self->status);
}

ucs_status_t uct_ugni_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uint32_t *result, uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_famo, fma_desc,
                             return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma_amo(fma_desc, GNI_POST_AMO, GNI_FMA_ATOMIC2_FCSWAP_S,
                            (uint64_t)compare, (uint64_t)swap, fma_desc + 1, remote_addr,
                            rkey, LEN_32, ep, comp, uct_ugni_amo_unpack32, (void *)result);
    ucs_trace_data("Posting AMO CSWAP, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx32" compare %"PRIx32" to %p, with [%"PRIx64" %"PRIx64"]",
                   fma_desc->super.desc.length, swap, compare,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_INPROGRESS);
}

ucs_status_t uct_ugni_ep_atomic_op32(uct_ep_h tl_ep, uint32_t op,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     gni_fma_cmd_type_t op_type, char *op_str)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_famo, fma_desc,
                             return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma_amo(fma_desc, GNI_POST_AMO, op_type,
                            (uint64_t)op, 0, NULL, remote_addr,
                            rkey, LEN_32, ep, NULL, NULL, NULL);
    ucs_trace_data("Posting AMO %s, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx32" to %p, with [%"PRIx64" %"PRIx64"]",
                   op_str, fma_desc->super.desc.length, op,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_OK);
}

ucs_status_t uct_ugni_ep_atomic32_post(uct_ep_h ep, unsigned opcode, uint32_t value,
                                       uint64_t remote_addr, uct_rkey_t rkey)
{
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        return uct_ugni_ep_atomic_op32(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC2_IADD_S, "ADD");
    case UCT_ATOMIC_OP_XOR:
        return uct_ugni_ep_atomic_op32(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC2_XOR_S, "XOR");
    case UCT_ATOMIC_OP_AND:
        return uct_ugni_ep_atomic_op32(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC2_AND_S, "AND");
    case UCT_ATOMIC_OP_OR:
        return uct_ugni_ep_atomic_op32(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC2_OR_S, "OR");
    default:
        ucs_assertv(0, "incorrect opcode for atomic: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }
}

ucs_status_t uct_ugni_ep_atomic_op64(uct_ep_h tl_ep, uint64_t op,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     gni_fma_cmd_type_t op_type, char *op_str)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_famo, fma_desc,
                             return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma_amo(fma_desc, GNI_POST_AMO, op_type,
                            op, 0, NULL, remote_addr,
                            rkey, LEN_64, ep, NULL, NULL, NULL);
    ucs_trace_data("Posting AMO %s, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx64" to %p, with [%"PRIx64" %"PRIx64"]",
                   op_str, fma_desc->super.desc.length, op,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_OK);
}

ucs_status_t uct_ugni_ep_atomic64_post(uct_ep_h ep, unsigned opcode, uint64_t value,
                                       uint64_t remote_addr, uct_rkey_t rkey)
{
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        return uct_ugni_ep_atomic_op64(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC_ADD, "ADD");
    case UCT_ATOMIC_OP_XOR:
        return uct_ugni_ep_atomic_op64(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC_XOR, "XOR");
   case UCT_ATOMIC_OP_AND:
        return uct_ugni_ep_atomic_op64(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC_AND, "AND"); 
   case UCT_ATOMIC_OP_OR:
        return uct_ugni_ep_atomic_op64(ep, value, remote_addr, rkey,
                                       GNI_FMA_ATOMIC_OR, "OR");
    default:
        ucs_assertv(0, "incorrect opcode for atomic: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }
}

ucs_status_t uct_ugni_ep_atomic_fop64(uct_ep_h tl_ep, uint64_t op,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp,
                                     gni_fma_cmd_type_t op_type, char *op_str)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_famo, fma_desc,
                             return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma_amo(fma_desc, GNI_POST_AMO, op_type,
                            op, 0, fma_desc + 1, remote_addr,
                            rkey, LEN_64, ep, comp, uct_ugni_amo_unpack64, (void *)result);
    ucs_trace_data("Posting AMO %s, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx64" to %p, with [%"PRIx64" %"PRIx64"]",
                   op_str, fma_desc->super.desc.length, op,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_INPROGRESS);
}

ucs_status_t uct_ugni_ep_atomic64_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                        uint64_t value, uint64_t *result,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        return uct_ugni_ep_atomic_fop64(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC_FADD, "FADD");
    case UCT_ATOMIC_OP_SWAP:
        return uct_ugni_ep_atomic_fop64(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC2_FSWAP, "FSWAP");
    case UCT_ATOMIC_OP_XOR:
        return uct_ugni_ep_atomic_fop64(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC_FXOR, "FXOR");
    case UCT_ATOMIC_OP_AND:
        return uct_ugni_ep_atomic_fop64(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC_FAND, "FAND");
    case UCT_ATOMIC_OP_OR:
        return uct_ugni_ep_atomic_fop64(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC_FOR, "FOR");
    default:
        ucs_assertv(0, "incorrect opcode for atomic: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }
}

ucs_status_t uct_ugni_ep_atomic_fop32(uct_ep_h tl_ep, uint32_t op,
                                      uint64_t remote_addr, uct_rkey_t rkey,
                                      uint32_t *result, uct_completion_t *comp,
                                      gni_fma_cmd_type_t op_type, char *op_str)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_famo, fma_desc,
                             return UCS_ERR_NO_RESOURCE);
    uct_ugni_format_fma_amo(fma_desc, GNI_POST_AMO, op_type,
                            (uint64_t)op, 0, fma_desc + 1, remote_addr,
                            rkey, LEN_32, ep, comp, uct_ugni_amo_unpack32, (void *)result);
    ucs_trace_data("Posting AMO %s, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx32" to %p, with [%"PRIx64" %"PRIx64"]",
                   op_str, fma_desc->super.desc.length, op,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_ATOMIC(ucs_derived_of(tl_ep, uct_base_ep_t));
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_INPROGRESS);
}

ucs_status_t uct_ugni_ep_atomic32_fetch(uct_ep_h ep, uct_atomic_op_t opcode,
                                        uint32_t value, uint32_t *result,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{
    switch (opcode) {
    case UCT_ATOMIC_OP_ADD:
        return uct_ugni_ep_atomic_fop32(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC2_FIADD_S, "ADD");
    case UCT_ATOMIC_OP_SWAP:
        return uct_ugni_ep_atomic_fop32(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC2_FSWAP_S, "SWAP");
    case UCT_ATOMIC_OP_XOR:
        return uct_ugni_ep_atomic_fop32(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC2_FXOR_S, "XOR");
    case UCT_ATOMIC_OP_AND:
        return uct_ugni_ep_atomic_fop32(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC2_FAND_S, "AND"); 
    case UCT_ATOMIC_OP_OR:
        return uct_ugni_ep_atomic_fop32(ep, value, remote_addr, rkey, result, comp,
                                        GNI_FMA_ATOMIC2_FOR_S, "OR");
    default:
        ucs_assertv(0, "incorrect opcode for atomic: %d", opcode);
        return UCS_ERR_UNSUPPORTED;
    }
}

static void uct_ugni_unalign_fma_get_cb(uct_completion_t *self)
{
    uct_ugni_rdma_fetch_desc_t *fma_desc = (uct_ugni_rdma_fetch_desc_t *)
        ucs_container_of(self, uct_ugni_rdma_fetch_desc_t, tmp);

    /* Call the original callback and skip padding */
    fma_desc->super.unpack_cb(fma_desc->user_buffer, (char *)(fma_desc + 1) + fma_desc->padding,
                              fma_desc->super.desc.length - fma_desc->padding - fma_desc->tail);

    uct_ugni_invoke_orig_comp(fma_desc, self->status);
}

static inline void uct_ugni_format_get_fma(uct_ugni_rdma_fetch_desc_t *fma_desc,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           unsigned length, uct_ugni_ep_t *ep,
                                           uct_completion_t *user_comp,
                                           uct_completion_t *internal_comp,
                                           uct_unpack_callback_t unpack_cb,
                                           void *arg)
{
    uint64_t addr;
    void *buffer;
    unsigned align_length;

    fma_desc->padding      = ucs_padding_pow2(remote_addr, UGNI_GET_ALIGN);
    fma_desc->orig_comp_cb = user_comp;
    /* Make sure that the address is always aligned */
    addr                   = remote_addr - fma_desc->padding;
    buffer                 = (fma_desc + 1);
    fma_desc->user_buffer  = arg;
    /* Make sure that the length is always aligned */
    align_length = ucs_check_if_align_pow2(length + fma_desc->padding, UGNI_GET_ALIGN) ?
        ucs_align_up_pow2((length + fma_desc->padding), UGNI_GET_ALIGN):length + fma_desc->padding;
    fma_desc->tail = align_length - length - fma_desc->padding;
    ucs_assert(ucs_check_if_align_pow2(addr, UGNI_GET_ALIGN)==0);
    ucs_assert(ucs_check_if_align_pow2(align_length, UGNI_GET_ALIGN)==0);
    uct_ugni_format_fma(&fma_desc->super, GNI_POST_FMA_GET, buffer, addr, rkey, align_length,
                        ep, internal_comp, unpack_cb);
}

static inline void uct_ugni_format_unaligned_rdma(uct_ugni_rdma_fetch_desc_t *rdma,
                                                  const void *buffer, uint64_t remote_addr,
                                                  uct_mem_h memh, uct_rkey_t rkey,
                                                  unsigned length, uct_ugni_ep_t *ep,
                                                  gni_cq_handle_t cq,
                                                  uct_completion_t *composed_comp)
{
    uint64_t addr;
    unsigned align_len;
    char *local_buffer;
    size_t local_padding, remote_padding;

    addr = ucs_align_up_pow2((uint64_t)buffer, UGNI_GET_ALIGN);
    local_padding = addr - (uint64_t)buffer;
    local_buffer = (char *)addr;

    addr = ucs_align_down(remote_addr, UGNI_GET_ALIGN);
    remote_padding = remote_addr - addr;

    rdma->padding = local_padding + remote_padding;
    align_len =  ucs_align_up(length + rdma->padding, UGNI_GET_ALIGN);
    rdma->tail = align_len - (length + rdma->padding);

    uct_ugni_format_rdma(&(rdma->super), GNI_POST_RDMA_GET, local_buffer, addr, memh, rkey,
                         align_len, ep, cq, composed_comp);
}

ucs_status_t uct_ugni_ep_get_bcopy(uct_ep_h tl_ep,
                                   uct_unpack_callback_t unpack_cb,
                                   void *arg, size_t length,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_rdma_fetch_desc_t *fma_desc;

    UCT_SKIP_ZERO_LENGTH(length);
    UCT_CHECK_LENGTH(ucs_align_up_pow2(length, UGNI_GET_ALIGN), 0,
                     iface->config.fma_seg_size, "get_bcopy");
    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_get_buffer,
                             fma_desc, return UCS_ERR_NO_RESOURCE);

    fma_desc->tmp.func  = uct_ugni_unalign_fma_get_cb;
    fma_desc->tmp.count = 1;
    fma_desc->tmp.status = UCS_OK;

    uct_ugni_format_get_fma(fma_desc,
                            remote_addr, rkey, length,
                            ep, comp,
                            &fma_desc->tmp,
                            unpack_cb, arg);

    ucs_trace_data("Posting GET BCOPY, GNI_PostFma of size %"PRIx64" (%lu) from %p to "
                   "%p, with [%"PRIx64" %"PRIx64"]",
                   fma_desc->super.desc.length, length,
                   (void *)fma_desc->super.desc.local_addr,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, BCOPY, length);
    return uct_ugni_post_fma(iface, ep, &fma_desc->super, UCS_INPROGRESS);
}

static void assemble_composed_unaligned(uct_completion_t *self)
{
    uct_ugni_rdma_fetch_desc_t *fma_head = (uct_ugni_rdma_fetch_desc_t *)
        ucs_container_of(self, uct_ugni_rdma_fetch_desc_t, tmp);
    void *buffer = fma_head->user_buffer;
    uct_ugni_rdma_fetch_desc_t *rdma = fma_head->head;

    if(fma_head->head == NULL){
        memcpy(buffer, (char *)(fma_head + 1) + fma_head->padding,
               fma_head->super.desc.length - fma_head->padding - fma_head->tail);
    } else {
        memmove(buffer, buffer + rdma->padding, rdma->super.desc.length);
        memcpy(buffer + rdma->super.desc.length - rdma->padding,
               (char *)(fma_head + 1) + rdma->tail,
               fma_head->super.desc.length - (fma_head->tail + rdma->tail));
    }
    uct_ugni_invoke_orig_comp(fma_head, self->status);
}

static void free_composed_desc(void *arg)
{
    uct_ugni_rdma_fetch_desc_t *desc = (uct_ugni_rdma_fetch_desc_t*)arg;
    uct_ugni_rdma_fetch_desc_t *fma_desc = ucs_container_of(desc->super.comp_cb, uct_ugni_rdma_fetch_desc_t, tmp);
    uct_ugni_rdma_fetch_desc_t *rdma = fma_desc->head;

    if (0 == --rdma->tmp.count) {
        fma_desc->super.free_cb = rdma->super.free_cb = ucs_mpool_put;
        ucs_mpool_put(fma_desc);
        ucs_mpool_put(rdma);
    }
}

static ucs_status_t uct_ugni_ep_get_composed_fma_rdma(uct_ep_h tl_ep, void *buffer, size_t length,
                                                      uct_mem_h memh, uint64_t remote_addr,
                                                      uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_ugni_rdma_fetch_desc_t *fma_desc = NULL;
    uct_ugni_rdma_fetch_desc_t *rdma = NULL;
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    size_t fma_length, rdma_length, aligned_fma_remote_start;
    uint64_t fma_remote_start, rdma_remote_start;
    ucs_status_t post_result;


    rdma_length = length - iface->config.fma_seg_size;
    fma_length = iface->config.fma_seg_size;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_get_buffer,
                             fma_desc, return UCS_ERR_NO_RESOURCE);
    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc_get,
                             rdma, {ucs_mpool_put(fma_desc);return UCS_ERR_NO_RESOURCE;});

    rdma_remote_start = remote_addr;
    fma_remote_start = rdma_remote_start + rdma_length;
    aligned_fma_remote_start = ucs_align_up_pow2(fma_remote_start, UGNI_GET_ALIGN);
    /* The FMA completion is used to signal when both descs have completed. */
    fma_desc->tmp.count  = 2;
    fma_desc->tmp.status = UCS_OK;
    fma_desc->tmp.func   = assemble_composed_unaligned;
    /* The RDMA completion is used to signal when both descs have been freed */
    rdma->tmp.count  = 2;
    rdma->tmp.status = UCS_OK;
    uct_ugni_format_get_fma(fma_desc, aligned_fma_remote_start, rkey, fma_length, ep, comp, &fma_desc->tmp, NULL, NULL);
    fma_desc->tail = aligned_fma_remote_start - fma_remote_start;
    uct_ugni_format_unaligned_rdma(rdma, buffer, rdma_remote_start, memh, rkey,
                                   rdma_length+fma_desc->tail, ep, iface->super.local_cq,
                                   &fma_desc->tmp);
    fma_desc->head = rdma;
    rdma->head = fma_desc;
    fma_desc->user_buffer = rdma->user_buffer = buffer;
    fma_desc->super.free_cb = rdma->super.free_cb = free_composed_desc;

    ucs_trace_data("Posting split GET ZCOPY, GNI_PostFma of size %"PRIx64" (%lu) from %p to "
                   "%p, with [%"PRIx64" %"PRIx64"] and GNI_PostRdma of size %"PRIx64" (%lu)"
                   " from %p to %p, with [%"PRIx64" %"PRIx64"]",
                   fma_desc->super.desc.length, length,
                   (void *)fma_desc->super.desc.local_addr,
                   (void *)fma_desc->super.desc.remote_addr,
                   fma_desc->super.desc.remote_mem_hndl.qword1,
                   fma_desc->super.desc.remote_mem_hndl.qword2,
                   rdma->super.desc.length, length,
                   (void *)rdma->super.desc.local_addr,
                   (void *)rdma->super.desc.remote_addr,
                   rdma->super.desc.remote_mem_hndl.qword1,
                   rdma->super.desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY, length);
    post_result = uct_ugni_post_fma(iface, ep, &(fma_desc->super), UCS_INPROGRESS);
    if(post_result != UCS_OK && post_result != UCS_INPROGRESS){
        ucs_mpool_put(rdma);
        return post_result;
    }
    return uct_ugni_post_rdma(iface, ep, &(rdma->super));
}

static ucs_status_t uct_ugni_ep_get_composed(uct_ep_h tl_ep, void *buffer, size_t length,
                                             uct_mem_h memh, uint64_t remote_addr,
                                             uct_rkey_t rkey, uct_completion_t *comp)
{
    uint64_t aligned_remote = ucs_align_down(remote_addr, UGNI_GET_ALIGN);
    uint64_t remote_padding = remote_addr - aligned_remote;
    uint64_t fetch_length = length + remote_padding;
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);

    if(fetch_length < iface->config.fma_seg_size) {
        return uct_ugni_ep_get_bcopy(tl_ep,
                                     (uct_unpack_callback_t)memcpy,
                                     buffer, length,
                                     remote_addr, rkey,
                                     comp);
    }

    return uct_ugni_ep_get_composed_fma_rdma(tl_ep, buffer, length, memh,
                                             remote_addr, rkey, comp);
}

ucs_status_t uct_ugni_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_rdma_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_rdma_iface_t);
    uct_ugni_base_desc_t *rdma;

    UCT_CHECK_PARAM_IOV(iov, iovcnt, buffer, length, memh);
    UCT_SKIP_ZERO_LENGTH(length);
    UCT_CHECK_LENGTH(ucs_align_up_pow2(length, UGNI_GET_ALIGN), 0,
                     iface->config.rdma_max_size, "get_zcopy");

    /* Special flow for an unalign data */
    if (ucs_unlikely((uct_ugni_check_device_type(&iface->super, GNI_DEVICE_GEMINI) && 
                      ucs_check_if_align_pow2((uintptr_t)buffer, UGNI_GET_ALIGN)) ||
                      ucs_check_if_align_pow2(remote_addr, UGNI_GET_ALIGN)        ||
                      ucs_check_if_align_pow2(length, UGNI_GET_ALIGN))) {
        return uct_ugni_ep_get_composed(tl_ep, buffer, length, memh,
                                        remote_addr, rkey, comp);
    }

    /* Everything is perfectly aligned */
    UCT_TL_IFACE_GET_TX_DESC(&iface->super.super, &iface->free_desc, rdma,
                             return UCS_ERR_NO_RESOURCE);

    /* Setup Callback */
    uct_ugni_format_rdma(rdma, GNI_POST_RDMA_GET, buffer, remote_addr, memh, rkey,
                         ucs_align_up_pow2(length, UGNI_GET_ALIGN), ep, iface->super.local_cq, comp);

    ucs_trace_data("Posting GET ZCOPY, GNI_PostRdma of size %"PRIx64" (%lu) "
                   "from %p to %p, with [%"PRIx64" %"PRIx64"]",
                   rdma->desc.length, length,
                   (void *)rdma->desc.local_addr,
                   (void *)rdma->desc.remote_addr,
                   rdma->desc.remote_mem_hndl.qword1,
                   rdma->desc.remote_mem_hndl.qword2);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY, length);
    return uct_ugni_post_rdma(iface, ep, rdma);
}

UCS_CLASS_INIT_FUNC(uct_ugni_rdma_ep_t, const uct_ep_params_t *params)
{
    ucs_status_t rc;
    uct_ugni_iface_t *iface;
    const uct_sockaddr_ugni_t *iface_addr;
    const uct_devaddr_ugni_t *ugni_dev_addr;

    UCS_CLASS_CALL_SUPER_INIT(uct_ugni_ep_t, params);
    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);

    iface = ucs_derived_of(params->iface, uct_ugni_iface_t);
    iface_addr = (const uct_sockaddr_ugni_t*)params->iface_addr;
    ugni_dev_addr = (const uct_devaddr_ugni_t *)params->dev_addr;
    ucs_debug("Connecting RDMA ep %p", self);
    rc = ugni_connect_ep(&self->super, iface, iface_addr, ugni_dev_addr);

    if (UCS_OK != rc) {
        ucs_error("Could not connect ep %p", self);
        return rc;
    }

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_rdma_ep_t)
{
    return;
}

UCS_CLASS_DEFINE(uct_ugni_rdma_ep_t, uct_ugni_ep_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_rdma_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_rdma_ep_t, uct_ep_t);

ucs_status_t uct_ugni_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}
