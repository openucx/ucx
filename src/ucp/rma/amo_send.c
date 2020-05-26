/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <ucs/sys/stubs.h>

#include <inttypes.h>


#define UCP_AMO_CHECK_PARAM(_context, _remote_addr, _size, _opcode, \
                            _last_opcode, _action) \
    { \
        if (ENABLE_PARAMS_CHECK && \
            ucs_unlikely(((_remote_addr) % (_size)) != 0)) { \
            ucs_error("atomic variable must be naturally aligned " \
                      "(remote address 0x%"PRIx64", size %zu)", (_remote_addr), \
                      (_size)); \
            _action; \
        } \
        \
        if (ENABLE_PARAMS_CHECK && \
            ucs_unlikely(((_size) != 4) && (_size != 8))) { \
            ucs_error("invalid atomic operation size: %zu", (_size)); \
            _action; \
        } \
        \
        UCP_CONTEXT_CHECK_FEATURE_FLAGS((_context), ((_size) == 4) ? \
                                        UCP_FEATURE_AMO32 : UCP_FEATURE_AMO64, \
                                        _action); \
        \
        if (ENABLE_PARAMS_CHECK && \
            (ucs_unlikely((_opcode) >= (_last_opcode)))) { \
            ucs_error("invalid atomic opcode %d ", _opcode); \
            _action; \
        } \
    }


#define UCP_AMO_CHECK_PARAM_NBX(_context, _remote_addr, _size, _count, \
                                _opcode, _last_opcode, _action) \
    { \
        if (ENABLE_PARAMS_CHECK) { \
            if ((_count) != 1) { \
                ucs_error("unsupported number of elements: %zu", (_count)); \
                _action; \
            } \
        } \
        \
        UCP_AMO_CHECK_PARAM(_context, _remote_addr, _size, _opcode, \
                            _last_opcode, _action); \
    }


static uct_atomic_op_t ucp_uct_op_table[] = {
    [UCP_ATOMIC_POST_OP_ADD]    = UCT_ATOMIC_OP_ADD,
    [UCP_ATOMIC_POST_OP_AND]    = UCT_ATOMIC_OP_AND,
    [UCP_ATOMIC_POST_OP_OR]     = UCT_ATOMIC_OP_OR,
    [UCP_ATOMIC_POST_OP_XOR]    = UCT_ATOMIC_OP_XOR
};

static uct_atomic_op_t ucp_uct_atomic_op_table[] = {
    [UCP_ATOMIC_OP_ADD]         = UCT_ATOMIC_OP_ADD,
    [UCP_ATOMIC_OP_AND]         = UCT_ATOMIC_OP_AND,
    [UCP_ATOMIC_OP_OR]          = UCT_ATOMIC_OP_OR,
    [UCP_ATOMIC_OP_XOR]         = UCT_ATOMIC_OP_XOR,
    [UCP_ATOMIC_OP_SWAP]        = UCT_ATOMIC_OP_SWAP,
    [UCP_ATOMIC_OP_CSWAP]       = UCT_ATOMIC_OP_CSWAP
};


static void ucp_amo_completed_single(uct_completion_t *self,
                                     ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucp_trace_req(req, "invoking completion");
    ucp_request_complete_send(req, status);
}

static UCS_F_ALWAYS_INLINE void
ucp_amo_init_common(ucp_request_t *req, ucp_ep_h ep, uct_atomic_op_t op,
                    uint64_t remote_addr, ucp_rkey_h rkey, uint64_t value,
                    size_t size)
{
    req->flags                = 0;
    req->send.ep              = ep;
    req->send.length          = size;
    req->send.amo.uct_op      = op;
    req->send.amo.remote_addr = remote_addr;
    req->send.amo.rkey        = rkey;
    req->send.amo.value       = value;
#if UCS_ENABLE_ASSERT
    req->send.lane            = UCP_NULL_LANE;
#endif
}

static UCS_F_ALWAYS_INLINE void
ucp_amo_init_fetch(ucp_request_t *req, ucp_ep_h ep, void *buffer,
                   uct_atomic_op_t op, size_t op_size, uint64_t remote_addr,
                   ucp_rkey_h rkey, uint64_t value, const ucp_amo_proto_t *proto)
{
    ucp_amo_init_common(req, ep, op, remote_addr, rkey, value, op_size);
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_amo_completed_single;
    req->send.uct.func              = proto->progress_fetch;
    req->send.buffer                = buffer;
}

static UCS_F_ALWAYS_INLINE
void ucp_amo_init_post(ucp_request_t *req, ucp_ep_h ep, uct_atomic_op_t op,
                       size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey,
                       uint64_t value, const ucp_amo_proto_t *proto)
{
    ucp_amo_init_common(req, ep, op, remote_addr, rkey, value, op_size);
    req->send.uct.func = proto->progress_post;
}

ucs_status_ptr_t ucp_atomic_fetch_nb(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                                     uint64_t value, void *result, size_t op_size,
                                     uint64_t remote_addr, ucp_rkey_h rkey,
                                     ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FIELD_REPLY_BUFFER,
        .datatype     = ucp_dt_make_contig(op_size),
        .cb.send      = (ucp_send_nbx_callback_t)cb,
        .reply_buffer = result
    };

    /* Note: opcode transition from ucp_atomic_fetch_op_t to ucp_atomic_op_t */
    return ucp_atomic_op_nbx(ep, (ucp_atomic_op_t)opcode, &value, 1,
                             remote_addr, rkey, &param);
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_atomic_op_nbx,
                 (ep, opcode, buffer, count, remote_addr, rkey, param),
                 ucp_ep_h ep, ucp_atomic_op_t opcode, const void *buffer,
                 size_t count, uint64_t remote_addr, ucp_rkey_h rkey,
                 const ucp_request_param_t *param)
{
    ucs_status_ptr_t status_p;
    ucs_status_t status;
    ucp_request_t *req;
    uint64_t value;
    size_t op_size;

    if (ucs_unlikely(!(param->op_attr_mask & UCP_OP_ATTR_FIELD_DATATYPE))) {
        ucs_error("missing atomic operation datatype");
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    if (param->datatype == ucp_dt_make_contig(8)) {
        value   = *(uint64_t*)buffer;
        op_size = sizeof(uint64_t);
    } else if (param->datatype == ucp_dt_make_contig(4)) {
        value   = *(uint32_t*)buffer;
        op_size = sizeof(uint32_t);
    } else {
        ucs_error("invalid atomic operation datatype: %zu", param->datatype);
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCP_AMO_CHECK_PARAM_NBX(ep->worker->context, remote_addr, op_size,
                            count, opcode, UCP_ATOMIC_OP_LAST,
                            return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("atomic_op_nbx opcode %d buffer %p result %p "
                  "datatype %zu remote_addr %"PRIx64" rkey %p to %s cb %p",
                  opcode, buffer,
                  (param->op_attr_mask & UCP_OP_ATTR_FIELD_REPLY_BUFFER) ?
                  param->reply_buffer : NULL, param->datatype,
                  remote_addr, rkey, ucp_ep_peer_name(ep),
                  (param->op_attr_mask & UCP_OP_ATTR_FIELD_CALLBACK) ?
                  param->cb.send : NULL);

    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        status_p = UCS_STATUS_PTR(status);
        goto out;
    }

    req = ucp_request_get_param(ep->worker, param,
                                {status_p = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                 goto out;});

    if (param->op_attr_mask & UCP_OP_ATTR_FIELD_REPLY_BUFFER) {
        ucp_amo_init_fetch(req, ep, param->reply_buffer,
                           ucp_uct_atomic_op_table[opcode], op_size,
                           remote_addr, rkey, value, rkey->cache.amo_proto);
        status_p = ucp_rma_send_request(req, param);
    } else {
        ucp_amo_init_post(req, ep, ucp_uct_atomic_op_table[opcode], op_size,
                          remote_addr, rkey, value, rkey->cache.amo_proto);

        status_p = ucp_rma_send_request(req, param);
        if (UCS_PTR_IS_PTR(status_p)) {
            ucp_request_release(status_p);
        }
        status_p = UCS_STATUS_PTR(UCS_OK);
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status_p;
}

ucs_status_t ucp_atomic_post(ucp_ep_h ep, ucp_atomic_post_op_t opcode, uint64_t value,
                             size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_ptr_t status_p;
    ucs_status_t status;
    ucp_request_t *req;

    UCP_AMO_CHECK_PARAM(ep->worker->context, remote_addr, op_size, opcode,
                        UCP_ATOMIC_POST_OP_LAST, return UCS_ERR_INVALID_PARAM);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("atomic_post opcode %d value %"PRIu64" size %zu "
                  "remote_addr %"PRIx64" rkey %p to %s",
                  opcode, value, op_size, remote_addr, rkey,
                  ucp_ep_peer_name(ep));

    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        goto out;
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    ucp_amo_init_post(req, ep, ucp_uct_op_table[opcode], op_size, remote_addr,
                      rkey, value, rkey->cache.amo_proto);

    status_p = ucp_rma_send_request_cb(req, (ucp_send_callback_t)ucs_empty_function);
    if (UCS_PTR_IS_PTR(status_p)) {
        ucp_request_release(status_p);
        status = UCS_OK;
    } else {
        status = UCS_PTR_STATUS(status_p);
    }

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status;
}

static inline ucs_status_t
ucp_atomic_fetch_b(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode, uint64_t value,
                   void *result, size_t size, uint64_t remote_addr,
                   ucp_rkey_h rkey, const char *op_name)
{
    void *request;

    request = ucp_atomic_fetch_nb(ep, opcode, value, result, size, remote_addr,
                                  rkey, (ucp_send_callback_t)ucs_empty_function);
    return ucp_rma_wait(ep->worker, request, op_name);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_add32, (ep, add, remote_addr, rkey),
                 ucp_ep_h ep, uint32_t add, uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_atomic_post(ep, UCP_ATOMIC_POST_OP_ADD, add, sizeof(add),
                           remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_add64, (ep, add, remote_addr, rkey),
                 ucp_ep_h ep, uint64_t add, uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_atomic_post(ep, UCP_ATOMIC_POST_OP_ADD, add, sizeof(add),
                           remote_addr, rkey);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_fadd32, (ep, add, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t add, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    return ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_FADD, add, result,
                              sizeof(add), remote_addr, rkey, "atomic_fadd32");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_fadd64, (ep, add, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t add, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    return ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_FADD, add, result,
                              sizeof(add), remote_addr, rkey, "atomic_fadd64");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap32, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    return ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_SWAP, swap, result,
                              sizeof(swap), remote_addr, rkey, "atomic_swap32");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap64, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    return ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_SWAP, swap, result,
                              sizeof(swap), remote_addr, rkey, "atomic_swap64");
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_atomic_cswap_b(ucp_ep_h ep, uint64_t compare, uint64_t swap, size_t size,
                   uint64_t remote_addr, ucp_rkey_h rkey, void *result,
                   const char *op_name)
{
    char tmp[sizeof(swap)]; /* sufficient storage for maximal operand size */
    ucs_status_t status;

    memcpy(tmp, &swap, size);
    status = ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_CSWAP, compare, &tmp,
                                size, remote_addr, rkey, op_name);
    if (status == UCS_OK) {
        memcpy(result, tmp, size);
    }
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap32,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t compare, uint32_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint32_t *result)
{
    return ucp_atomic_cswap_b(ep, compare, swap, sizeof(swap), remote_addr,
                              rkey, result, "atomic_cswap32");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap64,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t compare, uint64_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint64_t *result)
{
    return ucp_atomic_cswap_b(ep, compare, swap, sizeof(swap), remote_addr,
                              rkey, result, "atomic_cswap64");
}
