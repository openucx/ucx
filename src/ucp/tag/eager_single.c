/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020-2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "eager.h"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/sys/string.h>

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_single.inl>
#include <ucp/proto/proto_common.inl>


static ucs_status_t ucp_eager_short_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucs_status_t status;

    status = uct_ep_am_short(ucp_ep_get_fast_lane(req->send.ep,
                                                  spriv->super.lane),
                             UCP_AM_ID_EAGER_ONLY, req->send.msg_proto.tag,
                             req->send.state.dt_iter.type.contig.buffer,
                             req->send.state.dt_iter.length);
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req->send.lane = spriv->super.lane; /* for pending add */
        return status;
    }

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                              UCS_BIT(UCP_DATATYPE_CONTIG));

    ucs_assert(status != UCS_INPROGRESS);
    ucp_request_complete_send(req, status);
    return UCS_OK;
}

static void
ucp_proto_eager_short_probe(const ucp_proto_init_params_t *init_params)
{
    const ucp_proto_select_param_t *select_param = init_params->select_param;
    ucp_proto_single_init_params_t params        = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_short),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_SHORT,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_SHORT
    };

    /* AM based proto can not be used if tag offload lane configured */
    if (!ucp_tag_eager_check_op_id(init_params, UCP_OP_ID_TAG_SEND, 0) ||
        !ucp_proto_is_short_supported(select_param)) {
        return;
    }

    ucp_proto_single_probe(&params);
}

ucp_proto_t ucp_eager_short_proto = {
    .name     = "egr/short",
    .desc     = "eager " UCP_PROTO_SHORT_DESC,
    .flags    = UCP_PROTO_FLAG_AM_SHORT,
    .probe    = ucp_proto_eager_short_probe,
    .query    = ucp_proto_single_query,
    .progress = {ucp_eager_short_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static size_t ucp_eager_single_pack(void *dest, void *arg)
{
    ucp_eager_hdr_t *hdr = dest;
    ucp_request_t *req   = arg;
    ucp_datatype_iter_t next_iter;
    size_t packed_size;

    ucs_assert(req->send.state.dt_iter.offset == 0);
    hdr->super.tag = req->send.msg_proto.tag;
    packed_size    = ucp_datatype_iter_next_pack(&req->send.state.dt_iter,
                                                 req->send.ep->worker,
                                                 SIZE_MAX, &next_iter, hdr + 1);
    return sizeof(*hdr) + packed_size;
}

static ucs_status_t ucp_eager_bcopy_single_progress(uct_pending_req_t *self)
{
    ucp_request_t                   *req = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_EAGER_ONLY, spriv->super.lane, ucp_eager_single_pack,
            req, SIZE_MAX, ucp_proto_request_bcopy_complete_success, 1);
}

static void
ucp_proto_eager_bcopy_single_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                = init_params->worker->context;
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_single,
        .super.cfg_thresh    = context->config.ext.bcopy_thresh,
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_BCOPY
    };

    /* AM based proto can not be used if tag offload lane configured */
    if (!ucp_tag_eager_check_op_id(init_params, UCP_OP_ID_TAG_SEND, 0)) {
        return;
    }

    ucp_proto_single_probe(&params);
}

ucp_proto_t ucp_eager_bcopy_single_proto = {
    .name     = "egr/single/bcopy",
    .desc     = UCP_PROTO_EAGER_BCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_proto_eager_bcopy_single_probe,
    .query    = ucp_proto_single_query,
    .progress = {ucp_eager_bcopy_single_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};

static void
ucp_proto_eager_zcopy_single_probe(const ucp_proto_init_params_t *init_params)
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
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.am.max_iov),
        .super.hdr_size      = sizeof(ucp_tag_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY   |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG  |
                               UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        .lane_type           = UCP_LANE_TYPE_AM,
        .tl_cap_flags        = UCT_IFACE_FLAG_AM_ZCOPY
    };

    /* AM based proto can not be used if tag offload lane configured */
    if (!ucp_tag_eager_check_op_id(init_params, UCP_OP_ID_TAG_SEND, 0) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return;
    }

    ucp_proto_single_probe(&params);
}

static ucs_status_t
ucp_proto_eager_zcopy_send_func(ucp_request_t *req,
                                const ucp_proto_single_priv_t *spriv,
                                uct_iov_t *iov)
{
    ucp_eager_hdr_t hdr = {
        .super.tag = req->send.msg_proto.tag
    };

    return uct_ep_am_zcopy(ucp_ep_get_fast_lane(req->send.ep,
                                                spriv->super.lane),
                           UCP_AM_ID_EAGER_ONLY, &hdr, sizeof(hdr), iov, 1, 0,
                           &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_eager_zcopy_single_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_zcopy_single_progress(
            req, UCT_MD_MEM_ACCESS_LOCAL_READ, ucp_proto_eager_zcopy_send_func,
            ucp_request_invoke_uct_completion_success,
            ucp_proto_request_zcopy_completion, ucp_proto_request_zcopy_init);
}

ucp_proto_t ucp_eager_zcopy_single_proto = {
    .name     = "egr/single/zcopy",
    .desc     = UCP_PROTO_EAGER_ZCOPY_DESC,
    .flags    = 0,
    .probe    = ucp_proto_eager_zcopy_single_probe,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_eager_zcopy_single_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_request_zcopy_reset
};
