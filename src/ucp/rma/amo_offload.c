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


typedef void (*ucp_amo_pack_arg_callback_t)(ucp_request_t *, size_t);

static void ucp_proto_amo_completed(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucp_request_complete_send(req, self->status);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_amo_arg_pack(ucp_request_t *req, size_t op_size)
{
    memcpy(&req->send.amo.value, req->send.state.dt_iter.type.contig.buffer,
           op_size);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_amo_mtype_arg_pack(ucp_request_t *req, size_t op_size)
{
    ucp_datatype_iter_t next_iter;

    ucp_datatype_iter_next_pack(&req->send.state.dt_iter,
                                req->send.ep->worker, SIZE_MAX,
                                &next_iter, &req->send.amo.value);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_amo_progress(uct_pending_req_t *self, ucp_operation_id_t op_id,
                       size_t op_size, ucp_amo_pack_arg_callback_t pack_arg)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucp_ep_t *ep                         = req->send.ep;
    uint64_t remote_addr                 = req->send.amo.remote_addr;
    uct_atomic_op_t op                   = req->send.amo.uct_op;
    uint64_t *result64                   = req->send.buffer;
    uint32_t *result32                   = req->send.buffer;
    uint64_t value;
    ucs_status_t status;
    uct_rkey_t tl_rkey;

    req->send.lane = spriv->super.lane;
    tl_rkey        = ucp_rkey_get_tl_rkey(req->send.rma.rkey,
                                          spriv->super.rkey_index);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        pack_arg(req, op_size);

        if (op_id != UCP_OP_ID_AMO_POST) {
            ucp_proto_completion_init(&req->send.state.uct_comp,
                                      ucp_proto_amo_completed);
        }
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    value = req->send.amo.value;

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

    if (status == UCS_OK) {
        /* fast path is OK */
        ucp_request_complete_send(req, status);
    } else if (status == UCS_INPROGRESS) {
        ucs_assert(op_id != UCP_OP_ID_AMO_POST);
    } else if (status == UCS_ERR_NO_RESOURCE) {
        /* keep on pending queue */
        return UCS_ERR_NO_RESOURCE;
    } else {
        ucp_proto_request_abort(req, status);
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_amo_progress_offload(uct_pending_req_t *self,
                               ucp_operation_id_t op_id, size_t op_size)
{
    return ucp_proto_amo_progress(self, op_id, op_size,
                                  ucp_proto_amo_arg_pack);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_amo_progress_mtype(uct_pending_req_t *self,
                             ucp_operation_id_t op_id, size_t op_size)
{
    return ucp_proto_amo_progress(self, op_id, op_size,
                                  ucp_proto_amo_mtype_arg_pack);
}

static ucs_status_t
ucp_proto_amo_init(const ucp_proto_init_params_t *init_params,
                   ucp_operation_id_t op_id, size_t length,
                   uct_ep_operation_t memtype_op)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = 0,
        .super.cfg_priority  = 20,
        .super.min_length    = length,
        .super.max_length    = length,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = (op_id == UCP_OP_ID_AMO_POST) ?
                               UCT_EP_OP_ATOMIC_POST : UCT_EP_OP_ATOMIC_FETCH,
        .super.memtype_op    = memtype_op,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_AMO,
        .tl_cap_flags        = 0
    };

    UCP_RMA_PROTO_INIT_CHECK(init_params, op_id);

    if (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (op_id != UCP_OP_ID_AMO_POST) {
        params.super.flags |= UCP_PROTO_COMMON_INIT_FLAG_RESPONSE;
    }

    return ucp_proto_single_init(&params);
}

#define UCP_PROTO_AMO_REGISTER(_id, _op_id, _bits, _memtype_op, _sub_id) \
    static ucs_status_t ucp_amo_progress_##_id(uct_pending_req_t *self) \
    { \
        return ucp_proto_amo_progress_##_sub_id(self, (_op_id), (_bits) / 8); \
    } \
    \
    static ucs_status_t ucp_amo_init_##_id( \
            const ucp_proto_init_params_t *init_params) \
    { \
        return ucp_proto_amo_init(init_params, (_op_id), (_bits) / 8, \
                                  (_memtype_op)); \
    } \
    \
    ucp_proto_t ucp_amo_proto_##_id = { \
        .name     = "amo" #_bits "/" #_id "/" #_sub_id, \
        .desc     = #_sub_id, \
        .init     = ucp_amo_init_##_id, \
        .query    = ucp_proto_single_query, \
        .progress = {ucp_amo_progress_##_id} \
    };

#define UCP_PROTO_AMO_REGISTER_MTYPE(_id, _op_id, _bits) \
    UCP_PROTO_AMO_REGISTER(_id,         _op_id, _bits, UCT_EP_OP_LAST,      offload) \
    UCP_PROTO_AMO_REGISTER(_id##_mtype, _op_id, _bits, UCT_EP_OP_GET_SHORT, mtype)

#define UCP_PROTO_AMO_REGISTER_BITS(_id, _op_id) \
    UCP_PROTO_AMO_REGISTER_MTYPE(_id##32, _op_id, 32) \
    UCP_PROTO_AMO_REGISTER_MTYPE(_id##64, _op_id, 64)

UCP_PROTO_AMO_REGISTER_BITS(post,  UCP_OP_ID_AMO_POST)
UCP_PROTO_AMO_REGISTER_BITS(fetch, UCP_OP_ID_AMO_FETCH)
UCP_PROTO_AMO_REGISTER_BITS(cswap, UCP_OP_ID_AMO_CSWAP)
