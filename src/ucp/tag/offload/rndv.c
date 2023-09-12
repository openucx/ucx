/**
 * Copyright (C) 2023, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/proto/proto_single.inl>
#include <ucp/rndv/proto_rndv.h>
#include <ucp/rndv/proto_rndv.inl>
#include <ucp/tag/offload.h>
#include <ucp/tag/tag_rndv.h>


/* Init HW rendezvous as a single protocol which does not require operation on
 * remote side (which is the case when tag is matched - everyting is done by the
 * HW/FW).
 */
static ucs_status_t
ucp_tag_rndv_offload_proto_init(const ucp_proto_init_params_t *init_params)
{
    ucp_worker_h worker                   = init_params->worker;
    ucp_context_h context                 = worker->context;
    ucp_proto_single_init_params_t params = {
       .super.super         = *init_params,
       .super.latency       = 0,
       .super.overhead      = 40e-9,
       .super.cfg_thresh    = context->config.ext.rndv_thresh,
       .super.cfg_priority  = 60,
       .super.min_length    = ucp_ep_tag_offload_min_rndv_thresh(
                                  context, init_params->ep_config_key),
       .super.max_length    = SIZE_MAX,
       .super.min_iov       = 0,
       .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
       .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                           cap.tag.rndv.max_zcopy),
       .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t,
                                           cap.tag.rndv.max_iov),
       .super.hdr_size      = 0,
       .super.send_op       = UCT_EP_OP_RNDV_ZCOPY,
       .super.memtype_op    = UCT_EP_OP_LAST,
       .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                              UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                              UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
       .super.exclude_map   = 0,
       .lane_type           = UCP_LANE_TYPE_TAG,
       .tl_cap_flags        = UCT_IFACE_FLAG_TAG_RNDV_ZCOPY
    };

    if (!ucp_tag_rndv_check_op_id(init_params) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static void
ucp_tag_rndv_offload_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_send_request_id_release(req);
    ucp_proto_request_zcopy_complete(req, uct_comp->status);
}

static ucs_status_t
ucp_tag_rndv_offload_request_init(ucp_request_t *req, ucp_md_map_t md_map,
                                  uct_completion_callback_t comp_func,
                                  unsigned uct_reg_flags, unsigned dt_mask)
{
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    status = ucp_ep_resolve_remote_id(req->send.ep, spriv->super.lane);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assertv(dt_mask == UCS_BIT(UCP_DATATYPE_CONTIG), "dt_mask=0x%x",
                dt_mask);

    status = ucp_proto_request_zcopy_init(req, md_map, comp_func, uct_reg_flags,
                                          dt_mask);
    if (status != UCS_OK) {
        return status;
    }

    ucp_send_request_id_alloc(req);

    return UCS_OK;
}

static ucs_status_t
ucp_tag_rndv_offload_send_func(ucp_request_t *req,
                               const ucp_proto_single_priv_t *spriv,
                               uct_iov_t *iov)
{
    ucp_tag_offload_unexp_rndv_hdr_t rndv_hdr = {
        .ep_id    = ucp_send_request_get_ep_remote_id(req),
        .req_id   = ucp_send_request_get_id(req),
        .md_index = spriv->super.md_index
    };
    void *rndv_op;

    rndv_op = uct_ep_tag_rndv_zcopy(ucp_ep_get_fast_lane(req->send.ep,
                                                         spriv->super.lane),
                                    req->send.msg_proto.tag,
                                    &rndv_hdr, sizeof(rndv_hdr), iov, 1, 0,
                                    &req->send.state.uct_comp);

    if (ucs_unlikely(UCS_PTR_IS_ERR(rndv_op))) {
        return UCS_PTR_STATUS(rndv_op);
    }

    req->flags                   |= UCP_REQUEST_FLAG_OFFLOADED;
    req->send.tag_offload.rndv_op = rndv_op;

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_rndv_offload_proto_progress, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_RMA | UCT_MD_MEM_FLAG_HIDE_ERRORS,
            ucp_tag_rndv_offload_send_func, NULL,
            ucp_tag_rndv_offload_completion, ucp_tag_rndv_offload_request_init);
}

ucp_proto_t ucp_tag_rndv_offload_proto = {
    .name     = "tag/rndv/offload",
    .desc     = "rendezvous tag offload",
    .flags    = 0,
    .init     = ucp_tag_rndv_offload_proto_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_tag_rndv_offload_proto_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_request_zcopy_id_reset
};

/* SW emulation of rndv offload protocol. This protocol sends regular RTS as
 * offloaded eager message and this RTS is always handled by SW on the receiver.
 * It can be used when real rndv offload is not supported (e.g. for non-contig
 * or very large messages) or when it is enabled by protocol selection.
 */
static ucs_status_t
ucp_tag_rndv_offload_sw_proto_init(const ucp_proto_init_params_t *init_params)
{
    ucp_worker_h worker                      = init_params->worker;
    ucp_context_h context                    = worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 40e-9,
        .super.cfg_thresh    = context->config.ext.rndv_thresh,
        .super.cfg_priority  = 60,
        .super.min_length    = ucp_ep_tag_offload_min_rndv_thresh(
                                   context, init_params->ep_config_key),
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        /* SW rendezvous request over tag offload is implemented in UCT via tag
         * eager short API. Currently, UCT iface capabilities do not contain any
         * fields related to SW rendezvous request, so check for tag eager short
         * caps instead.
         * TODO: use proper rndv request caps when UCT API is updated.
         */
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.tag.eager.max_short),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_EAGER_SHORT,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .super.exclude_map   = 0,
        .remote_op_id        = UCP_OP_ID_RNDV_RECV,
        .unpack_time         = UCS_LINEAR_FUNC_ZERO,
        .perf_bias           = context->config.ext.rndv_perf_diff / 100.0,
        .mem_info.type       = init_params->select_param->mem_type,
        .mem_info.sys_dev    = init_params->select_param->sys_dev,
        .ctrl_msg_name       = UCP_PROTO_RNDV_RTS_NAME,
        .md_map              = 0
    };

    if (!ucp_tag_rndv_check_op_id(init_params) ||
        !ucp_ep_config_key_has_tag_lane(init_params->ep_config_key)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_ctrl_init(&params,
                                    init_params->ep_config_key->tag_lane);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_tag_rndv_offload_sw_proto_progress, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h ep        = req->send.ep;
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    ucp_rndv_rts_hdr_t *rts_hdr;
    ucs_status_t status;
    size_t rts_hdr_size;

    status = UCS_PROFILE_CALL(ucp_proto_rndv_rts_request_init, req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    /* Send regular RTS, remote side will complete the protocol in SW even if
     * this message is matched by the HW. */
    rpriv        = req->send.proto_config->priv;
    rts_hdr      = ucs_alloca(sizeof(*rts_hdr) + rpriv->packed_rkey_size);
    rts_hdr_size = ucp_tag_rndv_proto_rts_pack(rts_hdr, req);
    status       = uct_ep_tag_rndv_request(ep->uct_eps[rpriv->lane],
                                           req->send.msg_proto.tag, rts_hdr,
                                           rts_hdr_size, 0);

    return ucp_proto_single_status_handle(req, 0, NULL, rpriv->lane, status);
}

ucp_proto_t ucp_tag_rndv_offload_sw_proto = {
    .name     = "tag/rndv/offload_sw",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_tag_rndv_offload_sw_proto_init,
    .query    = ucp_proto_rndv_rts_query,
    .progress = {ucp_tag_rndv_offload_sw_proto_progress},
    .abort    = ucp_proto_rndv_rts_abort,
    .reset    = ucp_proto_rndv_rts_reset
};
