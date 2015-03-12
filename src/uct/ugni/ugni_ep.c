/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>
#include <uct/tl/tl_log.h>

#include "ugni_ep.h"
#include "ugni_iface.h"
#include "ugni_device.h"

static inline ptrdiff_t uct_ugni_ep_compare(uct_ugni_ep_t *ep1, 
                                            uct_ugni_ep_t *ep2)
{
    return ep1->hash_key - ep2->hash_key;
}

static inline unsigned uct_ugni_ep_hash(uct_ugni_ep_t *ep)
{
    return ep->hash_key;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, 
                                         uct_ugni_ep_hash);
SGLIB_DEFINE_LIST_FUNCTIONS(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, 
                                        uct_ugni_ep_hash);

static UCS_CLASS_INIT_FUNC(uct_ugni_ep_t, uct_iface_t *tl_iface)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_iface, uct_ugni_iface_t);
    gni_return_t ugni_rc;

    UCS_CLASS_CALL_SUPER_INIT(tl_iface)

    ugni_rc = GNI_EpCreate(iface->nic_handle, iface->local_cq, &self->ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_CdmCreate failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
        return UCS_ERR_NO_DEVICE;
    }

    self->outstanding = 0;
    self->hash_key = (uintptr_t)&self->ep;
    sglib_hashed_uct_ugni_ep_t_add(iface->eps, self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_ugni_ep_t)
{
    uct_ugni_iface_t *iface = ucs_derived_of(self->super.iface, uct_ugni_iface_t);
    gni_return_t ugni_rc;

    ugni_rc = GNI_EpDestroy(self->ep);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpDestroy failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    sglib_hashed_uct_ugni_ep_t_delete(iface->eps, self);
}
UCS_CLASS_DEFINE(uct_ugni_ep_t, uct_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

uct_ugni_ep_t *uct_ugni_iface_lookup_ep(uct_ugni_iface_t *iface, 
                                        uintptr_t hash_key)
{
    uct_ugni_ep_t tmp;
    tmp.hash_key = hash_key;
    return sglib_hashed_uct_ugni_ep_t_find_member(iface->eps, &tmp);
}

ucs_status_t uct_ugni_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    ((uct_ugni_ep_addr_t*)ep_addr)->ep_id = iface->domain_id;
    return UCS_OK;
}

ucs_status_t uct_ugni_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                       uct_ep_addr_t *tl_ep_addr)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_addr_t *iface_addr = ucs_derived_of(tl_iface_addr, 
                                                       uct_ugni_iface_addr_t);
    uct_ugni_ep_addr_t *ep_addr = ucs_derived_of(tl_ep_addr, uct_ugni_ep_addr_t);
    gni_return_t ugni_rc;

    ugni_rc = GNI_EpBind(ep->ep, iface_addr->nic_addr, ep_addr->ep_id);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_warn("GNI_EpBind failed, Error status: %s %d",
                  gni_err_str[ugni_rc], ugni_rc);
    }
    ucs_debug("Binding ep %p to address (%d %d)",
              ep, iface_addr->nic_addr,
              ep_addr->ep_id);
    return UCS_OK;
}

static inline void uct_ugni_format_fma(uct_ugni_base_desc_t *fma, gni_post_type_t type,
                                       void *buffer, uint64_t remote_addr,
                                       uct_rkey_t rkey, unsigned length, uct_ugni_ep_t *ep,
                                       uct_completion_t *comp)
{
    fma->desc.type = type;
    fma->desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
    fma->desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE;
    fma->desc.local_addr = (uint64_t)buffer;
    fma->desc.remote_addr = remote_addr;
    fma->desc.remote_mem_hndl = *(gni_mem_handle_t *)rkey;
    fma->desc.length = length;
    fma->ep = ep;
    fma->comp_cb = comp;
}

static inline void uct_ugni_format_fma_amo(uct_ugni_base_desc_t *amo, gni_post_type_t type,
                                           gni_fma_cmd_type_t amo_op, 
                                           uint64_t first_operand, uint64_t second_operand,
                                           void *buffer, uint64_t remote_addr,
                                           uct_rkey_t rkey, unsigned length, uct_ugni_ep_t *ep,
                                           uct_completion_t *comp)
{
    uct_ugni_format_fma(amo, GNI_POST_AMO, buffer, remote_addr, 
                        rkey, length, ep, comp);
    amo->desc.amo_cmd = amo_op;
    amo->desc.first_operand = first_operand;
    amo->desc.second_operand = second_operand;
}

static inline void uct_ugni_format_rdma(uct_ugni_base_desc_t *rdma, 
                                        void *buffer, uint64_t remote_addr,
                                        uct_mem_h memh, uct_rkey_t rkey,
                                        unsigned length, uct_ugni_ep_t *ep,
                                        gni_cq_handle_t cq,
                                        uct_completion_t *comp)
{
    rdma->desc.type = GNI_POST_RDMA_PUT;
    rdma->desc.cq_mode = GNI_CQMODE_GLOBAL_EVENT;
    rdma->desc.dlvr_mode = GNI_DLVMODE_PERFORMANCE;
    rdma->desc.local_addr = (uint64_t) buffer;
    rdma->desc.local_mem_hndl = *(gni_mem_handle_t *)memh;
    rdma->desc.remote_addr = remote_addr;
    rdma->desc.remote_mem_hndl = *(gni_mem_handle_t *)rkey;
    rdma->desc.length = length;
    rdma->desc.src_cq_hndl = cq;
    rdma->ep = ep;
    rdma->comp_cb = comp;
}

static inline ucs_status_t uct_ugni_post_rdma(uct_ugni_iface_t *iface,
                                              uct_ugni_ep_t *ep,
                                              uct_ugni_base_desc_t *rdma)
{
    gni_return_t ugni_rc;

    ugni_rc = GNI_PostRdma(ep->ep, &rdma->desc);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_mpool_put(rdma);
        if(GNI_RC_ERROR_RESOURCE == ugni_rc || GNI_RC_ERROR_NOMEM == ugni_rc) {
            ucs_debug("GNI_PostRdma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_WOULD_BLOCK;
        } else {
            ucs_error("GNI_PostRdma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_IO_ERROR;
        }
    }

    ++ep->outstanding;
    ++iface->outstanding;

    return UCS_INPROGRESS;
}

static inline ucs_status_t uct_ugni_post_fma(uct_ugni_iface_t *iface,
                                             uct_ugni_ep_t *ep,
                                             uct_ugni_base_desc_t *fma,
                                             ucs_status_t ok_status)
{
    gni_return_t ugni_rc;

    ugni_rc = GNI_PostFma(ep->ep, &fma->desc);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_mpool_put(fma);
        if(GNI_RC_ERROR_RESOURCE == ugni_rc || GNI_RC_ERROR_NOMEM == ugni_rc) {
            ucs_debug("GNI_PostFma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_WOULD_BLOCK;
        } else {
            ucs_error("GNI_PostFma failed, Error status: %s %d",
                      gni_err_str[ugni_rc], ugni_rc);
            return UCS_ERR_IO_ERROR;
        }
    }

    ++ep->outstanding;
    ++iface->outstanding;

    return ok_status;
}

#define UCT_UGNI_ZERO_LENGTH_POST(len)              \
if (0 == len) {                                     \
    ucs_trace_data("Zero length request: skip it"); \
    return UCS_OK;                                  \
}

ucs_status_t uct_ugni_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                   unsigned length, uint64_t remote_addr,
                                   uct_rkey_t rkey)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_base_desc_t *fma;

    UCT_UGNI_ZERO_LENGTH_POST(length);
    UCT_CHECK_LENGTH(length <= iface->config.fma_seg_size, "put_short");
    UCT_TL_IFACE_GET_TX_DESC(&iface->super, iface->free_desc, 
                             fma, return UCS_ERR_WOULD_BLOCK);
    uct_ugni_format_fma(fma, GNI_POST_FMA_PUT, buffer, 
                        remote_addr, rkey, length, ep, NULL);
    ucs_trace_data("Posting PUT Short, GNI_PostFma of size %"PRIx64" from %p to "
                   "%p, with [%"PRIx64" %"PRIx64"]",
                   fma->desc.length,
                   (void *)fma->desc.local_addr,
                   (void *)fma->desc.remote_addr,
                   fma->desc.remote_mem_hndl.qword1,
                   fma->desc.remote_mem_hndl.qword2);
    return uct_ugni_post_fma(iface, ep, fma, UCS_OK);
}

ucs_status_t uct_ugni_ep_put_bcopy(uct_ep_h tl_ep, uct_pack_callback_t pack_cb,
                                   void *arg, size_t length, uint64_t remote_addr,
                                   uct_rkey_t rkey)
{
    /* Since custom pack function is used
     * we have to allocate separate memory to pack
     * the info and pass it to FMA 
     * something like:
     * pack_cb(desc + 1, arg, length); */
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_base_desc_t *fma;

    UCT_UGNI_ZERO_LENGTH_POST(length);
    UCT_CHECK_LENGTH(length <= iface->config.fma_seg_size, "put_bcopy");
    UCT_TL_IFACE_GET_TX_DESC(&iface->super, iface->free_desc_buffer,
                             fma, return UCS_ERR_WOULD_BLOCK);
    ucs_assert(length <= iface->config.fma_seg_size);
    pack_cb(fma + 1, arg, length);
    uct_ugni_format_fma(fma, GNI_POST_FMA_PUT, fma + 1,
                        remote_addr, rkey, length, ep, NULL);
    ucs_trace_data("Posting PUT BCOPY, GNI_PostFma of size %"PRIx64" from %p to"
                   "%p, with [%"PRIx64" %"PRIx64"]",
                   fma->desc.length,
                   (void *)fma->desc.local_addr,
                   (void *)fma->desc.remote_addr,
                   fma->desc.remote_mem_hndl.qword1,
                   fma->desc.remote_mem_hndl.qword2);
    return uct_ugni_post_fma(iface, ep, fma, UCS_OK);
}

ucs_status_t uct_ugni_ep_put_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_base_desc_t *rdma;

    UCT_UGNI_ZERO_LENGTH_POST(length);
    UCT_CHECK_LENGTH(length <= iface->config.rdma_max_size, "put_zcopy");
    UCT_TL_IFACE_GET_TX_DESC(&iface->super, iface->free_desc, rdma, return UCS_ERR_WOULD_BLOCK);
    /* Setup Callback */
    uct_ugni_format_rdma(rdma, buffer, remote_addr, memh, rkey, length, ep,
                         iface->local_cq, comp);

    ucs_trace_data("Posting PUT ZCOPY, GNI_PostRdma of size %"PRIx64" from %p to %p, with [%"PRIx64" %"PRIx64"]",
                   rdma->desc.length,
                   (void *)rdma->desc.local_addr,
                   (void *)rdma->desc.remote_addr,
                   rdma->desc.remote_mem_hndl.qword1,
                   rdma->desc.remote_mem_hndl.qword2);
    return uct_ugni_post_rdma(iface, ep, rdma);
}

#define LEN_64 (sizeof(uint64_t)) /* Length fetch and add for 64 bit */
#define LEN_32 (sizeof(uint32_t)) /* Length fetch and add for 32 bit */

ucs_status_t uct_ugni_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_base_desc_t *fma;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super, iface->free_desc_famo, fma, return UCS_ERR_WOULD_BLOCK);
    uct_ugni_format_fma_amo(fma, GNI_POST_AMO, GNI_FMA_ATOMIC_ADD,
                            add, 0, NULL, remote_addr,
                            rkey, LEN_64, ep, NULL);
    ucs_trace_data("Posting AMO ADD, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx64" to %p, with [%"PRIx64" %"PRIx64"]",
                   fma->desc.length, add,
                   (void *)fma->desc.remote_addr,
                   fma->desc.remote_mem_hndl.qword1,
                   fma->desc.remote_mem_hndl.qword2);
    return uct_ugni_post_fma(iface, ep, fma, UCS_OK);
}

ucs_status_t uct_ugni_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_base_desc_t *fma;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super, iface->free_desc_famo, fma, return UCS_ERR_WOULD_BLOCK);
    uct_ugni_format_fma_amo(fma, GNI_POST_AMO, GNI_FMA_ATOMIC_FADD,
                            add, 0, fma + 1, remote_addr,
                            rkey, LEN_64, ep, comp);
    ucs_trace_data("Posting AMO FADD, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx64" to %p, with [%"PRIx64" %"PRIx64"]",
                   fma->desc.length, add,
                   (void *)fma->desc.remote_addr,
                   fma->desc.remote_mem_hndl.qword1,
                   fma->desc.remote_mem_hndl.qword2);
    return uct_ugni_post_fma(iface, ep, fma, UCS_INPROGRESS);
}

ucs_status_t uct_ugni_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_ugni_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    uct_ugni_ep_t *ep = ucs_derived_of(tl_ep, uct_ugni_ep_t);
    uct_ugni_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_ugni_iface_t);
    uct_ugni_base_desc_t *fma;

    UCT_TL_IFACE_GET_TX_DESC(&iface->super, iface->free_desc_famo, fma, return UCS_ERR_WOULD_BLOCK);
    uct_ugni_format_fma_amo(fma, GNI_POST_AMO, GNI_FMA_ATOMIC_CSWAP,
                            compare, swap, fma + 1, remote_addr,
                            rkey, LEN_64, ep, comp);
    ucs_trace_data("Posting AMO CSWAP, GNI_PostFma of size %"PRIx64" value"
                   "%"PRIx64" compare %"PRIx64" to %p, with [%"PRIx64" %"PRIx64"]",
                   fma->desc.length, swap, compare,
                   (void *)fma->desc.remote_addr,
                   fma->desc.remote_mem_hndl.qword1,
                   fma->desc.remote_mem_hndl.qword2);
    return uct_ugni_post_fma(iface, ep, fma, UCS_INPROGRESS);
}

ucs_status_t uct_ugni_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                         uint64_t remote_addr, uct_rkey_t rkey)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_ugni_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_ugni_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                          uint64_t remote_addr, uct_rkey_t rkey,
                                          uct_completion_t *comp)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_ugni_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare, uint32_t swap,
                                           uint64_t remote_addr, uct_rkey_t rkey,
                                           uct_completion_t *comp)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_ugni_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}
