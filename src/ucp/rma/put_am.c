/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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


static size_t ucp_proto_put_am_bcopy_pack(void *dest, void *arg)
{
    ucp_proto_multi_pack_ctx_t *pack_ctx = arg;
    ucp_request_t                   *req = pack_ctx->req;
    ucp_put_hdr_t                  *puth = dest;

    puth->address  = req->send.rma.remote_addr +
                     req->send.state.dt_iter.offset;
    puth->ep_id    = ucp_send_request_get_ep_remote_id(req);
    puth->mem_type = req->send.rma.rkey->mem_type;

    return sizeof(*puth) + ucp_proto_multi_data_pack(pack_ctx, puth + 1);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_am_bcopy_send_func(ucp_request_t *req,
                                 const ucp_proto_multi_lane_priv_t *lpriv,
                                 ucp_datatype_iter_t *next_iter,
                                 ucp_lane_index_t *lane_shift)
{
    ucp_proto_multi_pack_ctx_t pack_ctx = {
        .req         = req,
        .max_payload = ucp_proto_multi_max_payload(req, lpriv,
                                                   sizeof(ucp_put_hdr_t)),
        .next_iter   = next_iter
    };

    return ucp_rma_sw_do_am_bcopy(req, UCP_AM_ID_PUT, lpriv->super.lane,
                                  ucp_proto_put_am_bcopy_pack, &pack_ctx, NULL);
}

static ucs_status_t ucp_proto_put_am_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                  = ucs_container_of(self, ucp_request_t,
                                                           send.uct);
    /* coverity[tainted_data_downcast] */
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_ep_resolve_remote_id(req->send.ep,
                                          mpriv->lanes[0].super.lane);
        if (status != UCS_OK) {
            return status;
        }

        ucp_proto_multi_request_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, mpriv,
                                    ucp_proto_put_am_bcopy_send_func,
                                    ucp_proto_request_bcopy_complete_success,
                                    UCP_DT_MASK_CONTIG_IOV);
}

static ucs_status_t
ucp_proto_put_am_bcopy_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 40e-9,
        .super.cfg_thresh    = ucp_proto_sw_rma_cfg_thresh(
                                   context, context->config.ext.bcopy_thresh),
        .super.cfg_priority  = 20,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_put_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_SHORT,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_CAP_SEG_SIZE |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .max_lanes           = 1,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_AM_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_AM,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_AM_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_AM,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID
    };

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_multi_init(&params, init_params->priv,
                                init_params->priv_size);
}

ucp_proto_t ucp_put_am_bcopy_proto = {
    .name     = "put/am/bcopy",
    .desc     = UCP_PROTO_RMA_EMULATION_DESC,
    .flags    = 0,
    .init     = ucp_proto_put_am_bcopy_init,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_am_bcopy_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};
