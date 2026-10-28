/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_SGL_INL_
#define UCP_RMA_SGL_INL_

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_multi.inl>

/* Post a batch of SGL elements on a single lane */
typedef ucs_status_t (*ucp_proto_rma_sgl_post_func_t)(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        void *const *buffers, const size_t *lengths, uct_mem_h const *memhs,
        const uint64_t *remote_addrs, uct_rkey_t const *rkeys, size_t count);

/* Adjust a single SGL element fragment before it is posted. May be NULL */
typedef ucs_status_t (*ucp_proto_rma_sgl_adjust_func_t)(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        void **buffer_p, size_t *length_p, uint64_t *remote_addr_p);

static UCS_F_ALWAYS_INLINE int
ucp_proto_rma_sgl_elem_fits(size_t length, size_t min_frag_length,
                            size_t max_frag_length)
{
    return (length >= ucs_max(min_frag_length, 1)) &&
           (length <= max_frag_length);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rma_sgl_max_elem_count(const ucp_proto_multi_lane_priv_t *lpriv,
                                 size_t count)
{
    return ucs_min(lpriv->max_sgl_zcopy_count,
                   ucp_proto_multi_scaled_length(lpriv->weight, count));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rma_sgl_offload_send_frag(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        size_t max_frag_length, ucp_datatype_iter_t *next_iter,
        ucp_proto_rma_sgl_post_func_t post_func,
        ucp_proto_rma_sgl_adjust_func_t adjust_func)
{
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_rsc_index_t md_index     = lpriv->super.md_index;
    ucp_mem_h *sgl_memhs         = dt_iter->type.sgl.memhs;
    void *buffer                 = NULL;
    size_t length                = 0;
    size_t elem_index            = 0;
    uint64_t remote_addr         = 0;
    ucs_status_t status;
    uct_mem_h uct_memh;
    uct_rkey_t uct_rkey;

    if (ucp_datatype_iter_next_sgl_frags(dt_iter,
                                         req->send.rma.sgl.remote_addrs, 1,
                                         max_frag_length, next_iter, &buffer,
                                         &length, &remote_addr,
                                         &elem_index) == 0) {
        return UCS_OK;
    }

    if (adjust_func != NULL) {
        status = adjust_func(req, lpriv, &buffer, &length, &remote_addr);
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }
    }

    uct_memh = (sgl_memhs != NULL) ?
                       ucp_datatype_iter_uct_memh(sgl_memhs[elem_index],
                                                  md_index) :
                       UCT_MEM_HANDLE_NULL;
    uct_rkey = ucp_rkey_get_tl_rkey(req->send.rma.sgl.rkeys[elem_index],
                                    lpriv->super.rkey_index);

    return post_func(req, lpriv, &buffer, &length, &uct_memh, &remote_addr,
                     &uct_rkey, 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rma_sgl_offload_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter,
        ucp_proto_rma_sgl_post_func_t post_func,
        ucp_proto_rma_sgl_adjust_func_t adjust_func)
{
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_rsc_index_t md_index     = lpriv->super.md_index;
    ucp_rsc_index_t rkey_index   = lpriv->super.rkey_index;
    size_t max_frag_length       = lpriv->max_frag;
    size_t min_frag_length       = lpriv->min_sgl_frag;
    ucp_mem_h *sgl_memhs         = dt_iter->type.sgl.memhs;
    ucp_rkey_h const *sgl_rkeys  = req->send.rma.sgl.rkeys;
    void *const *buffers         = dt_iter->type.sgl.buffers;
    const size_t *lengths        = dt_iter->type.sgl.lengths;
    const uint64_t *remote_addrs = req->send.rma.sgl.remote_addrs;
    size_t start_index           = dt_iter->offset;
    size_t max_elem_count        = ucs_min(
            ucp_proto_rma_sgl_max_elem_count(lpriv, dt_iter->length),
            dt_iter->length - start_index);
    size_t uct_rkeys_size        = max_elem_count * sizeof(uct_rkey_t);
    size_t uct_memhs_size        = max_elem_count * sizeof(uct_mem_h);
    uct_rkey_t *uct_rkeys;
    uct_mem_h *uct_memhs;
    ucs_status_t status;
    size_t elem_count, idx;

    ucs_assert(max_frag_length > 0);
    ucs_assert(max_elem_count > 0);

    /* Silence compiler warning, in case of an early return below */
    next_iter->offset               = start_index;
    next_iter->type.sgl.frag_offset = dt_iter->type.sgl.frag_offset;

    if (ucs_unlikely((dt_iter->type.sgl.frag_offset != 0) ||
                     !ucp_proto_rma_sgl_elem_fits(lengths[start_index],
                                                  min_frag_length,
                                                  max_frag_length))) {
        return ucp_proto_rma_sgl_offload_send_frag(req, lpriv, max_frag_length,
                                                   next_iter, post_func,
                                                   adjust_func);
    }

    uct_rkeys = ucs_alloc_on_stack(uct_rkeys_size, "uct_sgl_rkeys");
    if (uct_rkeys == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    uct_memhs = ucs_alloc_on_stack(uct_memhs_size, "uct_sgl_memhs");
    if (uct_memhs == NULL) {
        ucs_free_on_stack(uct_rkeys, uct_rkeys_size);
        return UCS_ERR_NO_MEMORY;
    }

    for (elem_count = 0; elem_count < max_elem_count; elem_count++) {
        idx = start_index + elem_count;
        if (!ucp_proto_rma_sgl_elem_fits(lengths[idx], min_frag_length,
                                         max_frag_length)) {
            break;
        }

        uct_memhs[elem_count] = (sgl_memhs != NULL) ?
                                ucp_datatype_iter_uct_memh(sgl_memhs[idx],
                                                           md_index) :
                                UCT_MEM_HANDLE_NULL;
        uct_rkeys[elem_count] = ucp_rkey_get_tl_rkey(sgl_rkeys[idx],
                                                     rkey_index);
    }

    next_iter->offset               = start_index + elem_count;
    next_iter->type.sgl.frag_offset = 0;

    status = post_func(req, lpriv, &buffers[start_index], &lengths[start_index],
                       uct_memhs, &remote_addrs[start_index], uct_rkeys,
                       elem_count);

    ucs_free_on_stack(uct_memhs, uct_memhs_size);
    ucs_free_on_stack(uct_rkeys, uct_rkeys_size);

    return status;
}

#endif
