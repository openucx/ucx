/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/tag/offload.h>
#include <ucp/tag/proto_eager.inl>
#include <ucp/proto/proto_single.inl>

#define UCP_PROTO_EAGER_OFFLOAD_DESC "eager offloaded"


static ucs_status_t
ucp_proto_eager_tag_offload_short_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    status = uct_ep_tag_eager_short(ep->uct_eps[spriv->super.lane],
                                    req->send.msg_proto.tag,
                                    req->send.state.dt_iter.type.contig.buffer,
                                    req->send.state.dt_iter.length);
    if (status == UCS_ERR_NO_RESOURCE) {
        req->send.lane = spriv->super.lane; /* for pending add */
        return status;
    }

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter,
                              UCS_BIT(UCP_DATATYPE_CONTIG));

    ucs_assert(status != UCS_INPROGRESS);
    ucp_request_complete_send(req, status);

    return UCS_OK;
}

static ucs_status_t ucp_proto_eager_tag_offload_short_init(
        const ucp_proto_init_params_t *init_params)
{
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    ucp_proto_single_init_params_t params        = {
        .super.super         = *init_params,
        .super.latency       = -150e-9, /* no extra mem access to fetch data */
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.tag.eager.max_short),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_tag_t),
        .super.send_op       = UCT_EP_OP_EAGER_SHORT,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY,
        .lane_type           = UCP_LANE_TYPE_TAG,
        .tl_cap_flags        = UCT_IFACE_FLAG_TAG_EAGER_SHORT
    };

    if (!ucp_proto_eager_check_op_id(init_params, UCP_OP_ID_TAG_SEND, 1) ||
        !ucp_proto_is_short_supported(select_param)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

ucp_proto_t ucp_eager_tag_offload_short_proto = {
    .name     = "egr/offload/short",
    .desc     = UCP_PROTO_EAGER_OFFLOAD_DESC " " UCP_PROTO_SHORT_DESC,
    .flags    = UCP_PROTO_FLAG_TAG_SHORT,
    .init     = ucp_proto_eager_tag_offload_short_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_tag_offload_short_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};

static size_t ucp_eager_tag_offload_pack(void *dest, void *arg)
{
    ucp_request_t *req = arg;
    ucp_datatype_iter_t next_iter;

    ucs_assert(req->send.state.dt_iter.offset == 0);

    return ucp_datatype_iter_next_pack(&req->send.state.dt_iter,
                                       req->send.ep->worker, SIZE_MAX,
                                       &next_iter, dest);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_tag_offload_bcopy_common(ucp_request_t *req,
                                         const ucp_proto_single_priv_t *spriv,
                                         uint64_t imm_data)
{
    ssize_t packed_len;

    packed_len = uct_ep_tag_eager_bcopy(req->send.ep->uct_eps[spriv->super.lane],
                                        req->send.msg_proto.tag, imm_data,
                                        ucp_eager_tag_offload_pack, req, 0);

    return ucs_likely(packed_len >= 0) ? UCS_OK : packed_len;
}

static ucs_status_t ucp_proto_eager_tag_offload_bcopy_init_common(
        const ucp_proto_init_params_t *init_params, ucp_proto_id_t op_id)
{
    ucp_context_t *context                = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 5e-9,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.tag.eager.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_tag_t),
        .super.send_op       = UCT_EP_OP_EAGER_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY,
        .lane_type           = UCP_LANE_TYPE_TAG,
        .tl_cap_flags        = UCT_IFACE_FLAG_TAG_EAGER_BCOPY
    };

    /* offload proto can not be used if no tag offload lane configured */
    if (!ucp_proto_eager_check_op_id(init_params, op_id, 1)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}


static ucs_status_t
ucp_proto_eager_tag_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    status = ucp_proto_eager_tag_offload_bcopy_common(req, spriv, 0ul);

    return ucp_proto_single_status_handle(
            req, 0, ucp_proto_request_bcopy_complete_success, spriv->super.lane,
            status);
}

static ucs_status_t ucp_proto_eager_tag_offload_bcopy_init(
        const ucp_proto_init_params_t *init_params)
{
    return ucp_proto_eager_tag_offload_bcopy_init_common(init_params,
                                                         UCP_OP_ID_TAG_SEND);
}

ucp_proto_t ucp_tag_offload_eager_bcopy_single_proto = {
    .name     = "egr/offload/bcopy",
    .desc     = UCP_PROTO_EAGER_OFFLOAD_DESC " " UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_tag_offload_bcopy_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_tag_offload_bcopy_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_sync_tag_offload_bcopy_posted(ucp_request_t *req)
{
    ucp_tag_offload_sync_posted(req);
    return ucp_proto_eager_sync_bcopy_send_completed(req);
}

static ucs_status_t
ucp_proto_eager_sync_tag_offload_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    status = ucp_proto_eager_tag_offload_bcopy_common(
            req, spriv, ucp_send_request_get_ep_remote_id(req));
    return ucp_proto_single_status_handle(
            req, 0, ucp_proto_eager_sync_tag_offload_bcopy_posted,
            spriv->super.lane, status);
}

static ucs_status_t ucp_proto_eager_sync_tag_offload_bcopy_init(
        const ucp_proto_init_params_t *init_params)
{
    return ucp_proto_eager_tag_offload_bcopy_init_common(
            init_params, UCP_OP_ID_TAG_SEND_SYNC);
}

ucp_proto_t ucp_eager_sync_bcopy_single_proto = {
    .name     = "egrsnc/offload/bcopy",
    .desc     = UCP_PROTO_EAGER_OFFLOAD_DESC " " UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_sync_tag_offload_bcopy_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_sync_tag_offload_bcopy_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};

static ucs_status_t ucp_proto_eager_tag_offload_zcopy_init_common(
        const ucp_proto_init_params_t *init_params, ucp_proto_id_t op_id)
{
    ucp_context_t *context                = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 30,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.tag.eager.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t,
                                            cap.tag.eager.max_iov),
        .super.hdr_size      = sizeof(ucp_tag_t),
        .super.send_op       = UCT_EP_OP_EAGER_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_TAG,
        .tl_cap_flags        = UCT_IFACE_FLAG_TAG_EAGER_ZCOPY
    };

    /* offload proto can not be used if no tag offload lane configured */
    if (!ucp_proto_eager_check_op_id(init_params, op_id, 1) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_single_init(&params);
}

static ucs_status_t ucp_proto_eager_tag_offload_zcopy_init(
        const ucp_proto_init_params_t *init_params)
{
    return ucp_proto_eager_tag_offload_zcopy_init_common(init_params,
                                                         UCP_OP_ID_TAG_SEND);
}

static ucs_status_t
ucp_proto_tag_offload_zcopy_send_func(ucp_request_t *req,
                                      const ucp_proto_single_priv_t *spriv,
                                      uct_iov_t *iov)
{
    return uct_ep_tag_eager_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                                  req->send.msg_proto.tag, 0ul, iov, 1, 0,
                                  &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_eager_tag_offload_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_LOCAL_READ,
            ucp_proto_tag_offload_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion, ucp_proto_request_zcopy_init);
}

ucp_proto_t ucp_tag_offload_eager_zcopy_single_proto = {
    .name     = "egr/offload/zcopy",
    .desc     = UCP_PROTO_EAGER_OFFLOAD_DESC " " UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_tag_offload_zcopy_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_tag_offload_zcopy_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};

static ucs_status_t ucp_proto_eager_sync_tag_offload_zcopy_init(
        const ucp_proto_init_params_t *init_params)
{
    return ucp_proto_eager_tag_offload_zcopy_init_common(
            init_params, UCP_OP_ID_TAG_SEND_SYNC);
}

static ucs_status_t
ucp_proto_tag_offload_zcopy_sync_send_func(ucp_request_t *req,
                                           const ucp_proto_single_priv_t *spriv,
                                           uct_iov_t *iov)
{
    return uct_ep_tag_eager_zcopy(req->send.ep->uct_eps[spriv->super.lane],
                                  req->send.msg_proto.tag,
                                  ucp_send_request_get_ep_remote_id(req), iov,
                                  1, 0, &req->send.state.uct_comp);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_eager_sync_tag_offload_zcopy_send_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucp_proto_request_zcopy_cleanup(req, UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_proto_eager_sync_send_completed_common(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_eager_sync_tag_offload_zcopy_posted(ucp_request_t *req)
{
    ucp_tag_offload_sync_posted(req);
    return ucp_request_invoke_uct_completion_success(req);
}

static ucs_status_t
ucp_proto_eager_sync_tag_offload_zcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_LOCAL_READ,
            ucp_proto_tag_offload_zcopy_sync_send_func,
            ucp_proto_eager_sync_tag_offload_zcopy_posted,
            ucp_proto_eager_sync_tag_offload_zcopy_send_completion,
            ucp_proto_request_zcopy_init);
}

ucp_proto_t ucp_eager_sync_zcopy_single_proto = {
    .name     = "egrsnc/offload/zcopy",
    .desc     = UCP_PROTO_EAGER_OFFLOAD_DESC " " UCP_PROTO_ZCOPY_DESC,
    .flags    = 0,
    .init     = ucp_proto_eager_sync_tag_offload_zcopy_init,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_sync_tag_offload_zcopy_progress},
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};
