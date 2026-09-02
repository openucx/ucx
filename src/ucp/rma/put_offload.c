/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_single.inl>

static ucs_status_t ucp_proto_put_offload_short_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    ucp_ep_t *ep                         = req->send.ep;
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;
    uct_rkey_t tl_rkey;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_ep_rma_handle_fence(ep, req, UCS_BIT(spriv->super.lane));
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey, spriv->super.rkey_index);
    status  = uct_ep_put_short(ucp_ep_get_fast_lane(ep, spriv->super.lane),
                               req->send.state.dt_iter.type.contig.buffer,
                               req->send.state.dt_iter.length,
                               req->send.rma.remote_addr, tl_rkey);
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req->send.lane = spriv->super.lane; /* for pending add */
        return status;
    }

    /* UCS_INPROGRESS is not expected */
    ucs_assert((status == UCS_OK) || UCS_STATUS_IS_ERR(status));

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                              UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_request_complete_send(req, status);
    return UCS_OK;
}

static void
ucp_proto_put_offload_short_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_short),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_SHORT,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG   |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = ~UCP_MAX_FAST_PATH_LANES_MASK,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = UCP_LANE_TYPE_RMA,
        .tl_cap_flags        = UCT_IFACE_FLAG_PUT_SHORT
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT)) ||
        !ucp_proto_is_short_supported(init_params->select_param)) {
        return;
    }

    if ((init_params->rkey_config_key != NULL) &&
        ucp_rkey_need_remote_flush(init_params->rkey_config_key)) {
        return;
    }

    ucp_proto_single_probe(&params);
}

ucp_proto_t ucp_put_offload_short_proto = {
    .name     = "put/offload/short",
    .desc     = UCP_PROTO_SHORT_DESC,
    .flags    = UCP_PROTO_FLAG_PUT_SHORT,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_CONTIG),
    .probe    = ucp_proto_put_offload_short_probe,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_put_offload_short_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static size_t ucp_proto_put_offload_bcopy_pack(void *dest, void *arg)
{
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;

    return ucp_proto_multi_data_pack(pack_ctx, dest);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_put_offload_update_remote_flush(ucp_ep_h ep,
                                          ucp_sys_dev_map_t flush_sys_dev_mask,
                                          uct_rkey_t tl_rkey, uct_ep_h uct_ep,
                                          uint64_t address)
{
    if (ucs_test_all_flags(ep->ext->flush_sys_dev_map, flush_sys_dev_mask)) {
        return;
    }

    ucp_worker_remote_flush_hash_put(&ep->worker->remote_flush_hash, ep,
                                     ucs_ffs64_safe(flush_sys_dev_mask),
                                     tl_rkey, uct_ep, address);
    ep->ext->flush_sys_dev_map |= flush_sys_dev_mask;
}

static UCS_F_INLINE_OPTIMIZED ucs_status_t
ucp_proto_put_offload_bcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    ucp_ep_h ep        = req->send.ep;
    uct_ep_h uct_ep    = ucp_ep_get_lane(ep, lpriv->super.lane);
    uint64_t address   = req->send.rma.remote_addr +
                         req->send.state.dt_iter.offset;
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);

    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req         = req,
        .max_payload = ucp_proto_multi_max_payload(req, lpriv, 0),
        .next_iter   = next_iter
    };
    ssize_t packed_size;
    ucs_status_t status;

    packed_size = uct_ep_put_bcopy(uct_ep, ucp_proto_put_offload_bcopy_pack,
                                   &pack_ctx, address, tl_rkey);
    status      = ucp_proto_bcopy_send_func_status(packed_size);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucp_proto_put_offload_update_remote_flush(ep, lpriv->flush_sys_dev_mask,
                                                  tl_rkey, uct_ep, address);
    }

    return status;
}

static ucs_status_t ucp_proto_put_offload_bcopy_progress(uct_pending_req_t *self)
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

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_progress(req, req->send.proto_config->priv,
                                    ucp_proto_put_offload_bcopy_send_func,
                                    ucp_proto_request_bcopy_complete_success,
                                    UCP_DT_MASK_ALL);
}

static void
ucp_proto_put_offload_bcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_multi,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                           cap.put.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .max_lanes           = UCP_PROTO_RMA_MAX_BCOPY_LANES,
        .min_chunk           = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

ucp_proto_t ucp_put_offload_bcopy_proto = {
    .name     = "put/offload/bcopy",
    .desc     = UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .dt_mask  = UCP_PROTO_DT_MASK_DEFAULT,
    .probe    = ucp_proto_put_offload_bcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_offload_bcopy_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_multi_lane_priv_t *lpriv,
                                      ucp_datatype_iter_t *next_iter,
                                      ucp_lane_index_t *lane_shift)
{
    ucp_ep_h ep        = req->send.ep;
    uct_ep_h uct_ep    = ucp_ep_get_lane(ep, lpriv->super.lane);
    uint64_t address   = req->send.rma.remote_addr +
                         req->send.state.dt_iter.offset;
    uct_rkey_t tl_rkey = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                              lpriv->super.rkey_index);
    uct_iov_t iov;
    ucs_status_t status;

    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               ucp_proto_multi_max_payload(req, lpriv, 0),
                               lpriv->super.md_index, UCP_DT_MASK_CONTIG_IOV,
                               next_iter, &iov, 1);
    status = uct_ep_put_zcopy(uct_ep, &iov, 1, address, tl_rkey,
                              &req->send.state.uct_comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucp_proto_put_offload_update_remote_flush(ep, lpriv->flush_sys_dev_mask,
                                                  tl_rkey, uct_ep, address);
    }

    return status;
}

static ucs_status_t
ucp_proto_put_offload_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCP_DT_MASK_CONTIG_IOV,
            ucp_proto_put_offload_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

static void
ucp_proto_put_offload_zcopy_probe_param(
        const ucp_proto_init_params_t *init_params, unsigned cfg_priority,
        uint64_t extra_flags, uint64_t tl_v2_cap_flags)
{
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super            = *init_params,
        .super.latency          = 0,
        .super.overhead         = context->config.ext.proto_overhead_multi,
        .super.cfg_thresh       = context->config.ext.zcopy_thresh,
        .super.cfg_priority     = cfg_priority,
        .super.min_length       = 0,
        .super.max_length       = SIZE_MAX,
        .super.min_iov          = 1,
        .super.min_frag_offs    = ucs_offsetof(uct_iface_attr_t,
                                               cap.put.min_zcopy),
        .super.max_frag_offs    = ucs_offsetof(uct_iface_attr_t,
                                               cap.put.max_zcopy),
        .super.max_iov_offs     = ucs_offsetof(uct_iface_attr_t,
                                               cap.put.max_iov),
        .super.hdr_size         = 0,
        .super.send_op          = UCT_EP_OP_PUT_ZCOPY,
        .super.memtype_op       = UCT_EP_OP_LAST,
        .super.flags            = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY    |
                                  UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY    |
                                  UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                                  UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING  |
                                  extra_flags,
        .super.exclude_map      = 0,
        .super.reg_mem_info     = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        .max_lanes              = context->config.ext.max_rma_lanes,
        .min_chunk              = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map     = 0,
        .first.tl_cap_flags     = UCT_IFACE_FLAG_PUT_ZCOPY,
        .first.tl_v2_cap_flags  = tl_v2_cap_flags,
        .first.lane_type        = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags    = UCT_IFACE_FLAG_PUT_ZCOPY,
        .middle.tl_v2_cap_flags = tl_v2_cap_flags,
        .middle.lane_type       = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs         = UCP_PROTO_COMMON_OFFSET_INVALID,
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

static void
ucp_proto_put_offload_zcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_put_offload_zcopy_probe_param(
            init_params, 30, UCP_PROTO_COMMON_INIT_FLAG_FAILOVER, 0);
}

ucp_proto_t ucp_put_offload_zcopy_proto = {
    .name     = "put/offload/zcopy",
    .desc     = UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .dt_mask  = UCP_DT_MASK_CONTIG_IOV,
    .probe    = ucp_proto_put_offload_zcopy_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_offload_zcopy_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};

static void
ucp_proto_put_sgl_offload_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_put_offload_zcopy_probe_param(
            init_params, 30, 0, UCT_IFACE_FLAG_V2_PUT_SGL_ZCOPY);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_sgl_offload_post(ucp_request_t *req,
                               const ucp_proto_multi_lane_priv_t *lpriv,
                               void *const *buffers, const size_t *lengths,
                               uct_mem_h const *memhs,
                               const uint64_t *remote_addrs,
                               uct_rkey_t const *rkeys, size_t count)
{
    ucp_ep_t *ep    = req->send.ep;
    uct_ep_h uct_ep = ucp_ep_get_lane(ep, lpriv->super.lane);
    ucs_status_t status;

    status = uct_ep_put_sgl_zcopy(uct_ep, buffers, lengths, memhs, remote_addrs,
                                  rkeys, NULL, NULL, count,
                                  &req->send.state.uct_comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucp_proto_put_offload_update_remote_flush(ep, lpriv->flush_sys_dev_mask,
                                                  rkeys[0], uct_ep,
                                                  remote_addrs[0]);
    }

    return status;
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_put_sgl_elem_fits(size_t length, size_t max_frag_length)
{
    return (length > 0) && (length <= max_frag_length);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_sgl_offload_send_frag(ucp_request_t *req,
                                    const ucp_proto_multi_lane_priv_t *lpriv,
                                    size_t max_frag_length,
                                    ucp_datatype_iter_t *next_iter)
{
    ucp_ep_t *ep                 = req->send.ep;
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_md_index_t md_index      = ucp_ep_md_index(ep, lpriv->super.lane);
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

    uct_memh = (sgl_memhs != NULL) ? sgl_memhs[elem_index]->uct[md_index] :
                                     UCT_MEM_HANDLE_NULL;
    uct_rkey = ucp_rkey_get_tl_rkey(req->send.rma.sgl.rkeys[elem_index],
                                    lpriv->super.rkey_index);

    return ucp_proto_put_sgl_offload_post(req, lpriv, &buffer, &length,
                                          &uct_memh, &remote_addr, &uct_rkey,
                                          1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_sgl_offload_send_func(ucp_request_t *req,
                                    const ucp_proto_multi_lane_priv_t *lpriv,
                                    ucp_datatype_iter_t *next_iter,
                                    ucp_lane_index_t *lane_shift)
{
    ucp_ep_t *ep                 = req->send.ep;
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_md_index_t md_index      = ucp_ep_md_index(ep, lpriv->super.lane);
    ucp_rsc_index_t rkey_index   = lpriv->super.rkey_index;
    size_t max_frag_length       = lpriv->max_frag;
    ucp_mem_h *sgl_memhs         = dt_iter->type.sgl.memhs;
    ucp_rkey_h const *sgl_rkeys  = req->send.rma.sgl.rkeys;
    void *const *buffers         = dt_iter->type.sgl.buffers;
    const size_t *lengths        = dt_iter->type.sgl.lengths;
    const uint64_t *remote_addrs = req->send.rma.sgl.remote_addrs;
    size_t start_index           = dt_iter->offset;
    size_t max_elem_count        = ucs_min(lpriv->max_sgl_zcopy_count,
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
                     !ucp_proto_put_sgl_elem_fits(lengths[start_index],
                                                  max_frag_length))) {
        return ucp_proto_put_sgl_offload_send_frag(req, lpriv, max_frag_length,
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
        if (!ucp_proto_put_sgl_elem_fits(lengths[idx], max_frag_length)) {
            break;
        }

        uct_memhs[elem_count] = (sgl_memhs != NULL) ?
                                sgl_memhs[idx]->uct[md_index] :
                                UCT_MEM_HANDLE_NULL;
        uct_rkeys[elem_count] = ucp_rkey_get_tl_rkey(sgl_rkeys[idx],
                                                     rkey_index);
    }

    next_iter->offset               = start_index + elem_count;
    next_iter->type.sgl.frag_offset = 0;

    status = ucp_proto_put_sgl_offload_post(req, lpriv, &buffers[start_index],
                                            &lengths[start_index], uct_memhs,
                                            &remote_addrs[start_index],
                                            uct_rkeys, elem_count);

    ucs_free_on_stack(uct_memhs, uct_memhs_size);
    ucs_free_on_stack(uct_rkeys, uct_rkeys_size);

    return status;
}

static ucs_status_t
ucp_proto_put_sgl_offload_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCS_BIT(UCP_DATATYPE_SGL),
            ucp_proto_put_sgl_offload_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

ucp_proto_t ucp_put_sgl_offload_proto = {
    .name     = "put/sgl/offload",
    .desc     = "sgl",
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_SGL),
    .probe    = ucp_proto_put_sgl_offload_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_sgl_offload_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};

static void
ucp_proto_put_sgl_offload_sw_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_put_offload_zcopy_probe_param(init_params, 20, 0, 0);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_sgl_offload_sw_send_func(ucp_request_t *req,
                                       const ucp_proto_multi_lane_priv_t *lpriv,
                                       ucp_datatype_iter_t *next_iter,
                                       ucp_lane_index_t *lane_shift)
{
    ucp_ep_t *ep                 = req->send.ep;
    ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;
    ucp_lane_index_t lane        = lpriv->super.lane;
    uct_ep_h uct_ep              = ucp_ep_get_lane(ep, lane);
    ucp_md_index_t md_index      = ucp_ep_md_index(ep, lane);
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
    ucs_status_t status;

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

    tl_rkey    = ucp_rkey_get_tl_rkey(sgl_rkeys[elem_index], rkey_index);
    iov.buffer = buffer;
    iov.length = length;
    iov.memh   = (sgl_memhs != NULL) ? sgl_memhs[elem_index]->uct[md_index] :
                                       UCT_MEM_HANDLE_NULL;
    iov.stride = 0;
    iov.count  = 1;

    status = uct_ep_put_zcopy(uct_ep, &iov, 1, remote_addr, tl_rkey,
                              &req->send.state.uct_comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        ucp_proto_put_offload_update_remote_flush(ep, lpriv->flush_sys_dev_mask,
                                                  tl_rkey, uct_ep, remote_addr);
    }

    return status;
}

static ucs_status_t
ucp_proto_put_sgl_offload_sw_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    /* coverity[tainted_data_downcast] */
    return ucp_proto_multi_zcopy_progress(
            req, req->send.proto_config->priv, ucp_proto_multi_rma_init_func,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCS_BIT(UCP_DATATYPE_SGL),
            ucp_proto_put_sgl_offload_sw_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion);
}

ucp_proto_t ucp_put_sgl_offload_sw_proto = {
    .name     = "put/sgl/offload_sw",
    .desc     = "sgl " UCP_PROTO_RMA_EMULATION_DESC,
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_SGL),
    .probe    = ucp_proto_put_sgl_offload_sw_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_sgl_offload_sw_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};
