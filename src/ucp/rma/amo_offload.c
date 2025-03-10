/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
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
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.inl>


static UCS_F_ALWAYS_INLINE void
ucp_amo_memtype_unpack_reply_buffer(ucp_request_t *req)
{
    ucp_dt_contig_unpack(req->send.ep->worker, req->send.amo.reply_buffer,
                         &req->send.amo.result, req->send.state.dt_iter.length,
                         ucp_amo_request_reply_mem_type(req),
                         req->send.state.dt_iter.length);
}

static void ucp_proto_amo_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucp_request_complete_send(req, self->status);
}

static void ucp_proto_amo_completion_mtype(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    ucp_amo_memtype_unpack_reply_buffer(req);
    ucp_request_complete_send(req, self->status);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_amo_progress(uct_pending_req_t *self, ucp_operation_id_t op_id,
                       size_t op_size, int is_memtype)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    ucp_ep_t *ep                         = req->send.ep;
    uint64_t remote_addr                 = req->send.amo.remote_addr;
    uct_atomic_op_t op                   = req->send.amo.uct_op;
    uct_completion_callback_t comp_cb;
    ucs_memory_type_t mem_type;
    uct_ep_h uct_ep;
    uint64_t value;
    ucs_status_t status;
    uct_rkey_t tl_rkey;
    void *result;

    req->send.lane = spriv->super.lane;
    uct_ep         = ucp_ep_get_fast_lane(ep, req->send.lane);
    tl_rkey        = ucp_rkey_get_tl_rkey(req->send.amo.rkey,
                                          spriv->super.rkey_index);

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        if (!(req->flags & UCP_REQUEST_FLAG_PROTO_AMO_PACKED)) {
            mem_type = is_memtype ? req->send.state.dt_iter.mem_info.type :
                                    UCS_MEMORY_TYPE_HOST;
            ucp_dt_contig_pack(req->send.ep->worker, &req->send.amo.value,
                               req->send.state.dt_iter.type.contig.buffer,
                               op_size, mem_type, op_size);
            req->flags |= UCP_REQUEST_FLAG_PROTO_AMO_PACKED;
        }

        if (op_id != UCP_OP_ID_AMO_POST) {
            comp_cb = is_memtype ? ucp_proto_amo_completion_mtype :
                                   ucp_proto_amo_completion;
            ucp_proto_completion_init(&req->send.state.uct_comp, comp_cb);
        }

        if (op_id == UCP_OP_ID_AMO_CSWAP) {
            ucp_dt_contig_pack(ep->worker, &req->send.amo.result,
                               req->send.amo.reply_buffer, op_size,
                               ucp_amo_request_reply_mem_type(req), op_size);
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    value  = req->send.amo.value;
    result = is_memtype ? &req->send.amo.result : req->send.amo.reply_buffer;

    if (op_size == sizeof(uint64_t)) {
        if (op_id == UCP_OP_ID_AMO_POST) {
            status = UCS_PROFILE_CALL(uct_ep_atomic64_post, uct_ep, op, value,
                                      remote_addr, tl_rkey);
        } else if (op_id == UCP_OP_ID_AMO_FETCH) {
            status = UCS_PROFILE_CALL(uct_ep_atomic64_fetch, uct_ep, op, value,
                                      result, remote_addr, tl_rkey,
                                      &req->send.state.uct_comp);
        } else {
            ucs_assert(op_id == UCP_OP_ID_AMO_CSWAP);
            status = UCS_PROFILE_CALL(uct_ep_atomic_cswap64, uct_ep, value,
                                      *(uint64_t*)result, remote_addr, tl_rkey,
                                      result, &req->send.state.uct_comp);
        }
    } else {
        ucs_assert(op_size == sizeof(uint32_t));
        if (op_id == UCP_OP_ID_AMO_POST) {
            status = UCS_PROFILE_CALL(uct_ep_atomic32_post, uct_ep, op, value,
                                      remote_addr, tl_rkey);
        } else if (op_id == UCP_OP_ID_AMO_FETCH) {
            status = UCS_PROFILE_CALL(uct_ep_atomic32_fetch, uct_ep, op, value,
                                      result, remote_addr, tl_rkey,
                                      &req->send.state.uct_comp);
        } else {
            ucs_assert(op_id == UCP_OP_ID_AMO_CSWAP);
            status = UCS_PROFILE_CALL(uct_ep_atomic_cswap32, uct_ep, value,
                                      *(uint32_t*)result, remote_addr, tl_rkey,
                                      result, &req->send.state.uct_comp);
        }
    }

    if (status == UCS_OK) {
        /* fast path is OK */
        if ((op_id != UCP_OP_ID_AMO_POST) && is_memtype) {
            ucp_amo_memtype_unpack_reply_buffer(req);
        }
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

static void ucp_proto_amo_probe(const ucp_proto_init_params_t *init_params,
                                ucp_operation_id_t op_id, size_t length,
                                int is_memtype)
{
    ucp_worker_h worker              = init_params->worker;
    ucs_memory_type_t reply_mem_type =
            init_params->select_param->op.reply.mem_type;
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
        .super.memtype_op    = is_memtype ? UCT_EP_OP_GET_SHORT : UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = UCP_LANE_TYPE_AMO,
        .tl_cap_flags        = 0
    };

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_init_check_op(init_params, UCS_BIT(op_id))) {
        return;
    }

    if (op_id != UCP_OP_ID_AMO_POST) {
        params.super.flags |= UCP_PROTO_COMMON_INIT_FLAG_RESPONSE;
        if (!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(reply_mem_type) &&
            (!is_memtype || (worker->mem_type_ep[reply_mem_type] == NULL))) {
            /* Check if reply buffer memory type is supported */
            return;
        }
    }

    ucp_proto_single_probe(&params);
}

static void ucp_proto_amo_query(const ucp_proto_query_params_t *params,
                                ucp_proto_query_attr_t *attr, const char *name,
                                int is_memtype)
{
    UCS_STRING_BUFFER_FIXED(config_strb, attr->config, sizeof(attr->config));
    UCS_STRING_BUFFER_FIXED(desc_strb, attr->desc, sizeof(attr->desc));
    const ucp_proto_single_priv_t *spriv = params->priv;
    ucs_memory_type_t send_mem_type      = params->select_param->mem_type;
    ucs_memory_type_t reply_mem_type;

    if (is_memtype && !UCP_MEM_IS_ACCESSIBLE_FROM_CPU(send_mem_type)) {
        ucs_string_buffer_appendf(&desc_strb, "copy from %s, ",
                                  ucs_memory_type_names[send_mem_type]);
    }

    ucs_string_buffer_appendf(&desc_strb, "atomic %s", name);
    ucs_string_buffer_rbrk(&desc_strb, "/");

    if (is_memtype &&
        (ucp_proto_select_op_id(params->select_param) != UCP_OP_ID_AMO_POST)) {
        reply_mem_type = params->select_param->op.reply.mem_type;
        if (!UCP_MEM_IS_ACCESSIBLE_FROM_CPU(reply_mem_type)) {
            ucs_string_buffer_appendf(&desc_strb, ", copy to %s, ",
                                      ucs_memory_type_names[reply_mem_type]);
        }
    }

    attr->max_msg_length = SIZE_MAX;
    attr->is_estimation  = 0;
    attr->lane_map       = UCS_BIT(spriv->super.lane);
    ucp_proto_common_lane_priv_str(params, &spriv->super, 1, 1, &config_strb);
}

/*
 * "amoNN/[post|fetch|cswap]"       - send and reply buffers must be host memory
 * "amoNN/[post|fetch|cswap]/mtype" - any supported memory type, use pack/unpack
 *
 * @param _bits       32/64
 * @param _id         post/fetch/cswap[_mtype]
 * @param _name       post/fetch/swap[/mtype]
 * @param _op_id      UCP_OP_ID_AMO_POST/UCP_OP_ID_AMO_FETCH/UCP_OP_ID_AMO_CSWAP
 * @param _memtype_op
 */
#define UCP_PROTO_AMO_REGISTER(_bits, _id, _name, _op_id, _is_memtype) \
    \
    static ucs_status_t ucp_proto_amo##_bits##_id##_progress( \
            uct_pending_req_t *self) \
    { \
        return ucp_proto_amo_progress(self, _op_id, sizeof(uint##_bits##_t), \
                                      _is_memtype); \
    } \
    \
    static void ucp_proto_amo##_bits##_##_id##_probe( \
            const ucp_proto_init_params_t *init_params) \
    { \
        ucp_proto_amo_probe(init_params, _op_id, sizeof(uint##_bits##_t), \
                            _is_memtype); \
    } \
    \
    static void ucp_proto_amo##_bits##_##_id##_query( \
            const ucp_proto_query_params_t *params, \
            ucp_proto_query_attr_t *attr) \
    { \
        return ucp_proto_amo_query(params, attr, _name, _is_memtype); \
    } \
    \
    ucp_proto_t ucp_amo##_bits##_##_id##_proto = { \
        .name     = "amo" #_bits "/" _name, \
        .desc     = NULL, \
        .probe    = ucp_proto_amo##_bits##_##_id##_probe, \
        .query    = ucp_proto_amo##_bits##_##_id##_query, \
        .progress = {ucp_proto_amo##_bits##_id##_progress}, \
        .abort    = ucp_proto_abort_fatal_not_implemented, \
        .reset    = ucp_proto_request_bcopy_reset \
    };

#define UCP_PROTO_AMO_REGISTER_MTYPE(_bits, _id, _op_id) \
    UCP_PROTO_AMO_REGISTER(_bits, _id, #_id, _op_id, 0) \
    UCP_PROTO_AMO_REGISTER(_bits, _id##_mtype, #_id "/mtype", _op_id, 1)

#define UCP_PROTO_AMO_REGISTER_BITS(_id, _op_id) \
    UCP_PROTO_AMO_REGISTER_MTYPE(32, _id, _op_id) \
    UCP_PROTO_AMO_REGISTER_MTYPE(64, _id, _op_id)

UCP_PROTO_AMO_REGISTER_BITS(post,  UCP_OP_ID_AMO_POST)
UCP_PROTO_AMO_REGISTER_BITS(fetch, UCP_OP_ID_AMO_FETCH)
UCP_PROTO_AMO_REGISTER_BITS(cswap, UCP_OP_ID_AMO_CSWAP)
