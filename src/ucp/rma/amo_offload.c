/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.inl"

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_single.inl>


static void ucp_proto_amo_completed(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucp_request_complete_send(req, self->status);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_amo_progress(uct_pending_req_t *self, int op_id, size_t op_size)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucp_ep_t *ep                         = req->send.ep;
    uint64_t value                       = req->send.amo.value;
    uint64_t remote_addr                 = req->send.amo.remote_addr;
    uct_atomic_op_t op                   = req->send.amo.uct_op;
    uint64_t *result64                   = req->send.buffer;
    uint32_t *result32                   = req->send.buffer;
    ucs_status_t status;
    uct_rkey_t tl_rkey;

    req->send.lane = spriv->super.lane;
    tl_rkey        = ucp_rma_request_get_tl_rkey(req, spriv->super.rkey_index);

    if ((op_id != UCP_OP_ID_AMO_POST) &&
        !(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        req->send.state.uct_comp.count  = 1;
        req->send.state.uct_comp.status = UCS_OK;
        req->send.state.uct_comp.func   = ucp_proto_amo_completed;
        req->flags                     |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    if (op_size == sizeof(uint64_t)) {
        if (op_id == UCP_OP_ID_AMO_POST) {
            status = UCS_PROFILE_CALL(uct_ep_atomic64_post,
                                      ep->uct_eps[req->send.lane], op, value,
                                      remote_addr, tl_rkey);
        } else if (op_id == UCP_OP_ID_AMO_FETCH) {
            status = UCS_PROFILE_CALL(uct_ep_atomic64_fetch,
                                      ep->uct_eps[req->send.lane], op, value,
                                      result64, remote_addr, tl_rkey,
                                      &req->send.state.uct_comp);
        } else {
            ucs_assert(op_id == UCP_OP_ID_AMO_CSWAP);
            status = UCS_PROFILE_CALL(uct_ep_atomic_cswap64,
                                      ep->uct_eps[req->send.lane], value,
                                      *result64, remote_addr, tl_rkey, result64,
                                      &req->send.state.uct_comp);
        }
    } else {
        ucs_assert(op_size == sizeof(uint32_t));
        if (op_id == UCP_OP_ID_AMO_POST) {
            status = UCS_PROFILE_CALL(uct_ep_atomic32_post,
                                      ep->uct_eps[req->send.lane], op, value,
                                      remote_addr, tl_rkey);
        } else if (op_id == UCP_OP_ID_AMO_FETCH) {
            status = UCS_PROFILE_CALL(uct_ep_atomic32_fetch,
                                      ep->uct_eps[req->send.lane], op, value,
                                      result32, remote_addr, tl_rkey,
                                      &req->send.state.uct_comp);
        } else {
            ucs_assert(op_id == UCP_OP_ID_AMO_CSWAP);
            status = UCS_PROFILE_CALL(uct_ep_atomic_cswap32,
                                      ep->uct_eps[req->send.lane], value,
                                      *result32, remote_addr, tl_rkey, result32,
                                      &req->send.state.uct_comp);
        }
    }

    if (status == UCS_INPROGRESS) {
        return UCS_OK;
    }

    /* Complete for UCS_OK and unexpected errors */
    if (status != UCS_ERR_NO_RESOURCE) {
        ucp_request_complete_send(req, status);
    }

    return status;
}

static ucs_status_t
ucp_proto_amo_init(const ucp_proto_init_params_t *init_params, int op_id,
                   size_t length)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = 0,
        .super.cfg_priority  = 20,
        .super.min_length    = length,
        .super.max_length    = length,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_MAX_FRAG,
        .lane_type           = UCP_LANE_TYPE_AMO,
        .tl_cap_flags        = 0
    };

    if (op_id != UCP_OP_ID_AMO_POST) {
        params.super.flags |= UCP_PROTO_COMMON_INIT_FLAG_RESPONSE;
    }

    return ucp_proto_single_init(&params);
}

#define UCP_PROTO_AMO_REGISTER(_id, _size, _op_id)                         \
static ucs_status_t                                                        \
ucp_amo_progress##_id##_size(uct_pending_req_t *self) {                    \
    return ucp_proto_amo_progress(self, (_op_id), (_size)/8);              \
}                                                                          \
static ucs_status_t                                                        \
ucp_amo_init##_id##_size(const ucp_proto_init_params_t *init_params) {     \
    UCP_RMA_PROTO_INIT_CHECK(init_params, (_op_id));                       \
    return ucp_proto_amo_init(init_params, (_op_id), (_size)/8);           \
}                                                                          \
static ucp_proto_t ucp_amo_proto##_id##_size = {                           \
    .name       = "amo" #_size "/" #_id "/offload",                        \
    .init       = ucp_amo_init##_id##_size,                                \
    .progress   = ucp_amo_progress##_id##_size,                            \
    .config_str = ucp_proto_single_config_str,                             \
};                                                                         \
UCP_PROTO_REGISTER(&ucp_amo_proto##_id##_size)

UCP_PROTO_AMO_REGISTER(post,  32, UCP_OP_ID_AMO_POST);
UCP_PROTO_AMO_REGISTER(fetch, 32, UCP_OP_ID_AMO_FETCH);
UCP_PROTO_AMO_REGISTER(cswap, 32, UCP_OP_ID_AMO_CSWAP);

UCP_PROTO_AMO_REGISTER(post,  64, UCP_OP_ID_AMO_POST);
UCP_PROTO_AMO_REGISTER(fetch, 64, UCP_OP_ID_AMO_FETCH);
UCP_PROTO_AMO_REGISTER(cswap, 64, UCP_OP_ID_AMO_CSWAP);
