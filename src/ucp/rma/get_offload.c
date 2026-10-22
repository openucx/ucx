/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020-2026. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_multi.inl>


static void ucp_proto_get_offload_bcopy_unpack(void *arg, const void *data,
                                               size_t length)
{
    void *dest = arg;
    ucs_memcpy_relaxed(dest, data, length, UCS_ARCH_MEMCPY_NT_SOURCE, length);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_offload_bcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);
    size_t max_length, length;
    void *dest;

    max_length = ucp_proto_multi_max_payload(req, lpriv, 0);
    length     = ucp_datatype_iter_next_ptr(&req->send.state.dt_iter,
                                            max_length, next_iter, &dest);
    return uct_ep_get_bcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            ucp_proto_get_offload_bcopy_unpack, dest, length,
                            req->send.rma.remote_addr +
                                    req->send.state.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static void ucp_proto_get_offload_bcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                              UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_proto_request_restart_or_complete(req, req->send.state.uct_comp.status,
                                          NULL);
}

static ucs_status_t ucp_proto_get_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                  = ucs_container_of(self, ucp_request_t,
                                                           send.uct);
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_multi_request_init(req);

        status = ucp_ep_rma_handle_fence(req->send.ep, req, mpriv->lane_map);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_get_offload_bcopy_completion);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_progress(req, req->send.proto_config->priv,
                                    ucp_proto_get_offload_bcopy_send_func,
                                    ucp_request_invoke_uct_completion_success,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

static void
ucp_proto_get_offload_bcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_GET_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING |
                               UCP_PROTO_COMMON_INIT_FLAG_FAILOVER,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .max_lanes           = UCP_PROTO_RMA_MAX_BCOPY_LANES,
        .min_chunk           = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_GET_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_GET_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_GET))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

static void ucp_proto_get_offload_reset(ucp_request_t *req)
{
    /* get_am protocol will use this field after reset, so it must be
     * initialized */
    req->send.state.completed_size = 0;
}

static ucs_status_t ucp_proto_get_offload_bcopy_reset(ucp_request_t *req)
{
    ucp_proto_get_offload_reset(req);
    /* GET is idempotent, so a restarted request replays from the beginning
     * instead of resuming from an offset which counts posted rather than
     * completed fragments */
    ucp_datatype_iter_rewind(&req->send.state.dt_iter, UCP_DT_MASK_ALL);
    return ucp_proto_request_bcopy_reset(req);
}

ucp_proto_t ucp_get_offload_bcopy_proto = {
    .name     = "get/bcopy",
    .desc     = UCP_PROTO_COPY_OUT_DESC,
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_CONTIG),
    .probe    = ucp_proto_get_offload_bcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_get_offload_bcopy_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_get_offload_bcopy_reset
};

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);
    size_t offset      = req->send.state.dt_iter.offset;
    const ucp_proto_multi_priv_t *mpriv;
    uct_iov_t iov;

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               lpriv->super.md_index, UCP_DT_MASK_CONTIG_IOV,
                               next_iter, &iov, 1);

    mpriv = req->send.proto_config->priv;
    ucp_proto_common_zcopy_adjust_min_frag(req, mpriv->min_frag, iov.length,
                                           &iov, 1, &offset);
    return uct_ep_get_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            &iov, 1, req->send.rma.remote_addr + offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t ucp_proto_get_offload_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_WRITE, UCP_DT_MASK_CONTIG_IOV,
            ucp_proto_get_offload_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

static void
ucp_proto_get_offload_zcopy_probe_param(
        const ucp_proto_init_params_t *init_params, unsigned cfg_priority,
        uint64_t extra_flags, uint64_t tl_v2_cap_flags,
        ptrdiff_t min_frag_offs)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = cfg_priority,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = min_frag_offs,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.get.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.get.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_GET_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RESPONSE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING |
                               extra_flags,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        .max_lanes                  = context->config.ext.max_rma_lanes,
        .min_chunk                  = context->config.ext.min_rma_chunk_size,
        .use_single_lane_min_length = (min_frag_offs !=
                                       UCP_PROTO_COMMON_OFFSET_INVALID),
        .initial_reg_md_map         = 0,
        .first.tl_cap_flags         = UCT_IFACE_FLAG_GET_ZCOPY,
        .first.tl_v2_cap_flags      = tl_v2_cap_flags,
        .first.lane_type            = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags        = UCT_IFACE_FLAG_GET_ZCOPY,
        .middle.tl_v2_cap_flags     = tl_v2_cap_flags,
        .middle.lane_type           = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs             = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_GET))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

static void
ucp_proto_get_offload_zcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_get_offload_zcopy_probe_param(
            init_params, 30,
            UCP_PROTO_COMMON_INIT_FLAG_MIN_FRAG |
                    UCP_PROTO_COMMON_INIT_FLAG_FAILOVER,
            0, ucs_offsetof(uct_iface_attr_t, cap.get.min_zcopy));
}

static ucs_status_t ucp_proto_get_offload_zcopy_reset(ucp_request_t *req)
{
    ucp_proto_get_offload_reset(req);
    return ucp_proto_offload_zcopy_reset(req);
}

ucp_proto_t ucp_get_offload_zcopy_proto = {
    .name           = "get/zcopy",
    .desc           = UCP_PROTO_ZCOPY_DESC,
    .flags          = 0,
    .fallback_class = UCP_PROTO_CLASS_RMA_RNDV,
    .dt_mask        = UCP_DT_MASK_CONTIG_IOV,
    .probe          = ucp_proto_get_offload_zcopy_probe,
    .query          = ucp_proto_multi_query,
    .progress       = {ucp_proto_get_offload_zcopy_progress},
    .abort          = ucp_proto_request_zcopy_abort,
    .reset          = ucp_proto_get_offload_zcopy_reset
};

static void
ucp_proto_get_sgl_offload_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_get_offload_zcopy_probe_param(init_params, 30, 0,
                                            UCT_IFACE_FLAG_V2_GET_SGL_ZCOPY,
                                            UCP_PROTO_COMMON_OFFSET_INVALID);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_get_sgl_adjust_min_frag(ucp_request_t *req,
                                  const ucp_proto_multi_lane_priv_t *lpriv,
                                  void **buffer_p, size_t *length_p,
                                  uint64_t *remote_addr_p)
{
    ucp_ep_t *ep             = req->send.ep;
    ucp_rsc_index_t rsc_idx  = ucp_ep_get_rsc_index(ep, lpriv->super.lane);
    size_t min_zcopy         = ucp_worker_iface(ep->worker,
                                                rsc_idx)->attr.cap.get.min_zcopy;
    size_t frag_offset       = req->send.state.dt_iter.type.sgl.frag_offset;
    size_t diff;

    if ((*length_p >= min_zcopy) || (frag_offset == 0)) {
        return;
    }

    diff            = ucs_min(min_zcopy - *length_p, frag_offset);
    *buffer_p       = UCS_PTR_BYTE_OFFSET(*buffer_p, -diff);
    *remote_addr_p -= diff;
    *length_p      += diff;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_sgl_offload_post(ucp_request_t *req,
                               const ucp_proto_multi_lane_priv_t *lpriv,
                               void *const *buffers, const size_t *lengths,
                               uct_mem_h const *memhs,
                               const uint64_t *remote_addrs,
                               uct_rkey_t const *rkeys, size_t count)
{
    uct_ep_h uct_ep = ucp_ep_get_lane(req->send.ep, lpriv->super.lane);

    return uct_ep_get_sgl_zcopy(uct_ep, buffers, lengths, memhs, remote_addrs,
                                rkeys, NULL, NULL, count,
                                &req->send.state.uct_comp);
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_get_sgl_elem_fits(size_t length, size_t max_frag_length)
{
    return (length > 0) && (length <= max_frag_length);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_get_sgl_max_elem_count(const ucp_proto_multi_lane_priv_t *lpriv,
                                 size_t count)
{
    return ucs_min(lpriv->max_sgl_zcopy_count,
                   ucp_proto_multi_scaled_length(lpriv->weight, count));
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_sgl_offload_send_frag(ucp_request_t *req,
                                    const ucp_proto_multi_lane_priv_t *lpriv,
                                    size_t max_frag_length,
                                    ucp_datatype_iter_t *next_iter)
{
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_rsc_index_t md_index     = lpriv->super.md_index;
    ucp_mem_h *sgl_memhs         = dt_iter->type.sgl.memhs;
    void *buffer                 = NULL;
    size_t length                = 0;
    size_t elem_index            = 0;
    uint64_t remote_addr         = 0;
    uct_mem_h uct_memh;
    uct_rkey_t uct_rkey;

    if (ucp_datatype_iter_next_sgl_frags(dt_iter,
                                         req->send.rma.sgl.remote_addrs, 1,
                                         max_frag_length, next_iter, &buffer,
                                         &length, &remote_addr,
                                         &elem_index) == 0) {
        return UCS_OK;
    }

    ucp_proto_get_sgl_adjust_min_frag(req, lpriv, &buffer, &length,
                                      &remote_addr);

    uct_memh = (sgl_memhs != NULL) ?
                       ucp_datatype_iter_uct_memh(sgl_memhs[elem_index],
                                                  md_index) :
                       UCT_MEM_HANDLE_NULL;
    uct_rkey = ucp_rkey_get_tl_rkey(req->send.rma.sgl.rkeys[elem_index],
                                    lpriv->super.rkey_index);

    return ucp_proto_get_sgl_offload_post(req, lpriv, &buffer, &length,
                                          &uct_memh, &remote_addr, &uct_rkey,
                                          1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_sgl_offload_send_func(ucp_request_t *req,
                                    const ucp_proto_multi_lane_priv_t *lpriv,
                                    ucp_datatype_iter_t *next_iter,
                                    ucp_lane_index_t *lane_shift)
{
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_rsc_index_t md_index     = lpriv->super.md_index;
    ucp_rsc_index_t rkey_index   = lpriv->super.rkey_index;
    size_t max_frag_length       = lpriv->max_frag;
    ucp_mem_h *sgl_memhs         = dt_iter->type.sgl.memhs;
    ucp_rkey_h const *sgl_rkeys  = req->send.rma.sgl.rkeys;
    void *const *buffers         = dt_iter->type.sgl.buffers;
    const size_t *lengths        = dt_iter->type.sgl.lengths;
    const uint64_t *remote_addrs = req->send.rma.sgl.remote_addrs;
    size_t start_index           = dt_iter->offset;
    size_t max_elem_count        = ucs_min(
            ucp_proto_get_sgl_max_elem_count(lpriv, dt_iter->length),
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
                     !ucp_proto_get_sgl_elem_fits(lengths[start_index],
                                                  max_frag_length))) {
        return ucp_proto_get_sgl_offload_send_frag(req, lpriv, max_frag_length,
                                                   next_iter);
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
        if (!ucp_proto_get_sgl_elem_fits(lengths[idx], max_frag_length)) {
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

    status = ucp_proto_get_sgl_offload_post(req, lpriv, &buffers[start_index],
                                            &lengths[start_index], uct_memhs,
                                            &remote_addrs[start_index],
                                            uct_rkeys, elem_count);

    ucs_free_on_stack(uct_memhs, uct_memhs_size);
    ucs_free_on_stack(uct_rkeys, uct_rkeys_size);

    return status;
}

static ucs_status_t
ucp_proto_get_sgl_offload_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_WRITE, UCS_BIT(UCP_DATATYPE_SGL),
            ucp_proto_get_sgl_offload_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

ucp_proto_t ucp_get_sgl_offload_proto = {
    .name     = "get/sgl/offload",
    .desc     = "sgl",
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_SGL),
    .probe    = ucp_proto_get_sgl_offload_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_get_sgl_offload_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_get_offload_zcopy_reset
};

static void
ucp_proto_get_sgl_offload_sw_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_get_offload_zcopy_probe_param(init_params, 20, 0, 0,
                                            UCP_PROTO_COMMON_OFFSET_INVALID);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_sgl_offload_sw_send_func(ucp_request_t *req,
                                       const ucp_proto_multi_lane_priv_t *lpriv,
                                       ucp_datatype_iter_t *next_iter,
                                       ucp_lane_index_t *lane_shift)
{
    ucp_ep_t *ep                 = req->send.ep;
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_lane_index_t lane        = lpriv->super.lane;
    uct_ep_h uct_ep              = ucp_ep_get_lane(ep, lane);
    ucp_rsc_index_t md_index     = lpriv->super.md_index;
    ucp_rsc_index_t rkey_index   = lpriv->super.rkey_index;
    size_t max_frag_length       = lpriv->max_frag;
    ucp_mem_h *sgl_memhs         = dt_iter->type.sgl.memhs;
    ucp_rkey_h const *sgl_rkeys  = req->send.rma.sgl.rkeys;
    void *buffer                 = NULL;
    size_t length                = 0;
    size_t elem_index            = 0;
    uint64_t remote_addr         = 0;
    size_t UCS_V_UNUSED desc_count;
    uct_rkey_t tl_rkey;
    uct_iov_t iov;

    ucs_assert(max_frag_length > 0);

    desc_count = ucp_datatype_iter_next_sgl_frags(
            dt_iter, req->send.rma.sgl.remote_addrs, 1, max_frag_length,
            next_iter, &buffer, &length, &remote_addr, &elem_index);
    if (desc_count == 0) {
        return UCS_OK;
    }

    ucs_assertv(desc_count == 1,
                "dt_iter=%p offset=%zu length=%zu frag_offset=%zu "
                "max_frag_length=%zu",
                dt_iter, dt_iter->offset, dt_iter->length,
                dt_iter->type.sgl.frag_offset, max_frag_length);

    ucp_proto_get_sgl_adjust_min_frag(req, lpriv, &buffer, &length,
                                      &remote_addr);

    tl_rkey    = ucp_rkey_get_tl_rkey(sgl_rkeys[elem_index], rkey_index);
    iov.buffer = buffer;
    iov.length = length;
    iov.memh   = (sgl_memhs != NULL) ?
                         ucp_datatype_iter_uct_memh(sgl_memhs[elem_index],
                                                    md_index) :
                         UCT_MEM_HANDLE_NULL;
    iov.stride = 0;
    iov.count  = 1;

    return uct_ep_get_zcopy(uct_ep, &iov, 1, remote_addr, tl_rkey,
                            &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_get_sgl_offload_sw_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_WRITE, UCS_BIT(UCP_DATATYPE_SGL),
            ucp_proto_get_sgl_offload_sw_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

ucp_proto_t ucp_get_sgl_offload_sw_proto = {
    .name     = "get/sgl/offload_sw",
    .desc     = "sgl " UCP_PROTO_RMA_EMULATION_DESC,
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_SGL),
    .probe    = ucp_proto_get_sgl_offload_sw_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_get_sgl_offload_sw_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_get_offload_zcopy_reset
};
