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
        uint64_t extra_flags, uint64_t tl_v2_cap_flags, int is_sgl)
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
        .super.min_frag_offs = is_sgl ? UCP_PROTO_COMMON_OFFSET_INVALID :
                               ucs_offsetof(uct_iface_attr_t,
                                            cap.get.min_zcopy),
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
        .use_single_lane_min_length = 1,
        .sgl_min_get_zcopy          = is_sgl,
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
            0, 0);
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
                                            UCT_IFACE_FLAG_V2_GET_SGL_ZCOPY, 1);
}

static void
ucp_proto_get_sgl_short_elem_completion(void *request, ucs_status_t status,
                                        void *user_data)
{
    ucp_request_t *req = user_data;

    ucp_request_free(request);
    ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
}

/* Get an element which is shorter than the minimal zero-copy get size, using
 * a regular get sub-request. The sub-request holds a reference on the parent
 * request completion, which is taken before the sub-request is created, so the
 * parent can't be completed before the sub-request. Returns UCS_INPROGRESS if
 * the element was handled (the reference is not counted by the caller) */
static ucs_status_t
ucp_proto_get_sgl_short_elem(ucp_request_t *req, size_t elem_index,
                             void *buffer, size_t length, uint64_t remote_addr)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_USER_DATA |
                        UCP_OP_ATTR_FIELD_MEMORY_TYPE,
        .cb.send      = ucp_proto_get_sgl_short_elem_completion,
        .user_data    = req,
        .memory_type  = req->send.state.dt_iter.mem_info.type
    };
    ucs_status_ptr_t status_ptr;
    ucs_status_t status;

    ++req->send.state.uct_comp.count;
    status_ptr = ucp_get_nbx(req->send.ep, buffer, length, remote_addr,
                             req->send.rma.sgl.rkeys[elem_index], &param);
    if (UCS_PTR_IS_PTR(status_ptr)) {
        return UCS_INPROGRESS;
    }

    /* Completed immediately, the callback will not be called. The parent holds
       an extra reference until all elements are posted, so it can't drop to
       0 */
    --req->send.state.uct_comp.count;
    status = UCS_PTR_STATUS(status_ptr);
    return (status == UCS_OK) ? UCS_INPROGRESS : status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_sgl_adjust_min_frag(ucp_request_t *req,
                                  const ucp_proto_multi_lane_priv_t *lpriv,
                                  size_t elem_index, void **buffer_p,
                                  size_t *length_p, uint64_t *remote_addr_p)
{
    size_t min_frag    = lpriv->min_sgl_frag;
    size_t frag_offset = req->send.state.dt_iter.type.sgl.frag_offset;
    size_t diff;

    if (*length_p >= min_frag) {
        return UCS_OK;
    }

    if (frag_offset == 0) {
        return ucp_proto_get_sgl_short_elem(req, elem_index, *buffer_p,
                                            *length_p, *remote_addr_p);
    }

    diff = min_frag - *length_p;
    ucs_assertv(diff <= frag_offset, "diff=%zu frag_offset=%zu", diff,
                frag_offset);
    *buffer_p       = UCS_PTR_BYTE_OFFSET(*buffer_p, -diff);
    *remote_addr_p -= diff;
    *length_p      += diff;
    return UCS_OK;
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

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_get_sgl_offload_send_func(ucp_request_t *req,
                                    const ucp_proto_multi_lane_priv_t *lpriv,
                                    ucp_datatype_iter_t *next_iter,
                                    ucp_lane_index_t *lane_shift)
{
    return ucp_proto_rma_sgl_offload_send_func(
            req, lpriv, next_iter, ucp_proto_get_sgl_offload_post,
            ucp_proto_get_sgl_adjust_min_frag);
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
    ucp_proto_get_offload_zcopy_probe_param(init_params, 20, 0, 0, 1);
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
    ucs_status_t status;
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

    status = ucp_proto_get_sgl_adjust_min_frag(req, lpriv, elem_index, &buffer,
                                               &length, &remote_addr);
    if (ucs_unlikely(status != UCS_OK)) {
        return (status == UCS_INPROGRESS) ? UCS_OK : status;
    }

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
    .desc     = "sgl per-element " UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_SGL),
    .probe    = ucp_proto_get_sgl_offload_sw_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_get_sgl_offload_sw_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_get_offload_zcopy_reset
};
