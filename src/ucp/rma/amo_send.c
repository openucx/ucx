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


static uct_atomic_op_t ucp_uct_op_table[] = {
    [UCP_ATOMIC_POST_OP_ADD]    = UCT_ATOMIC_OP_ADD,
    [UCP_ATOMIC_POST_OP_AND]    = UCT_ATOMIC_OP_AND,
    [UCP_ATOMIC_POST_OP_OR]     = UCT_ATOMIC_OP_OR,
    [UCP_ATOMIC_POST_OP_XOR]    = UCT_ATOMIC_OP_XOR
};

static uct_atomic_op_t ucp_uct_fop_table[] = {
    [UCP_ATOMIC_FETCH_OP_FADD]  = UCT_ATOMIC_OP_ADD,
    [UCP_ATOMIC_FETCH_OP_FAND]  = UCT_ATOMIC_OP_AND,
    [UCP_ATOMIC_FETCH_OP_FOR]   = UCT_ATOMIC_OP_OR,
    [UCP_ATOMIC_FETCH_OP_FXOR]  = UCT_ATOMIC_OP_XOR,
    [UCP_ATOMIC_FETCH_OP_SWAP]  = UCT_ATOMIC_OP_SWAP,
    [UCP_ATOMIC_FETCH_OP_CSWAP] = UCT_ATOMIC_OP_CSWAP,
};


static void ucp_amo_completed_single(uct_completion_t *self,
                                     ucs_status_t status)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);
    ucs_trace("invoking completion on AMO request %p", req);
    ucp_request_complete_send(req, status);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_amo_init_common(ucp_request_t *req, ucp_ep_h ep, uct_atomic_op_t op,
                    uint64_t remote_addr, ucp_rkey_h rkey, uint64_t value,
                    size_t size, const ucp_atomic_loopback_ctx_t *loopback_ctx)
{
    req->flags                = 0;
    req->send.ep              = ep;
    req->send.length          = size;
    req->send.amo.uct_op      = op;
    req->send.amo.remote_addr = remote_addr;
    req->send.amo.rkey        = rkey;
    req->send.amo.value       = value;

    if (ucs_unlikely(loopback_ctx != NULL)) {
        req->send.amo.looback_ctx = ucs_malloc(sizeof(*loopback_ctx),
                                               "amo loopback context");
        if (req->send.amo.looback_ctx == NULL) {
            ucs_error("failed to allocate amo loopback context");
            return UCS_ERR_NO_MEMORY;
        }

        *req->send.amo.looback_ctx = *loopback_ctx;
    }

#if UCS_ENABLE_ASSERT
    req->send.lane            = UCP_NULL_LANE;
#endif
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_amo_init_fetch(ucp_request_t *req, ucp_ep_h ep, void *buffer,
                   uct_atomic_op_t op, size_t op_size, uint64_t remote_addr,
                   ucp_rkey_h rkey, uint64_t value,
                   const ucp_amo_proto_t *proto,
                   const ucp_atomic_loopback_ctx_t *loopback_ctx)
{
    ucs_status_t status = ucp_amo_init_common(req, ep, op, remote_addr, rkey,
                                              value, op_size, loopback_ctx);
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_amo_completed_single;
    req->send.uct.func              = proto->progress_fetch;
    if (ucs_likely(loopback_ctx == NULL)) {
        req->send.buffer            = buffer;
    } else {
        ucs_assert(buffer == NULL);
        req->send.buffer            = &req->send.amo.looback_ctx->reply_data;
    }

    return status;
}

static UCS_F_ALWAYS_INLINE
ucs_status_t ucp_amo_init_post(ucp_request_t *req, ucp_ep_h ep,
                               uct_atomic_op_t op, size_t op_size,
                               uint64_t remote_addr, ucp_rkey_h rkey,
                               uint64_t value, const ucp_amo_proto_t *proto,
                               const ucp_atomic_loopback_ctx_t *loopback_ctx)
{
    ucs_status_t status = ucp_amo_init_common(req, ep, op, remote_addr, rkey,
                                              value, op_size, loopback_ctx);

    req->send.uct.func = proto->progress_post;
    return status;
}

ucs_status_ptr_t
ucp_atomic_fetch_internal(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                          uint64_t value, void *result, size_t op_size,
                          uint64_t remote_addr, ucp_rkey_h rkey,
                          const ucp_atomic_loopback_ctx_t *loopback_ctx,
                          ucp_send_callback_t cb)
{
    ucs_status_t status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    ucp_request_t *req;

    if (status != UCS_OK) {
        return UCS_STATUS_PTR(UCS_ERR_UNREACHABLE);
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    if (ucs_unlikely(result == NULL)) {
        ucs_assert(cb == ucp_amo_sw_loopback_completion_cb);
        status = ucp_amo_init_fetch(req, ep, NULL, ucp_uct_fop_table[opcode],
                                    op_size, remote_addr, rkey, value,
                                    rkey->cache.amo_proto, loopback_ctx);
    } else {
        status = ucp_amo_init_fetch(req, ep, result, ucp_uct_fop_table[opcode],
                                    op_size, remote_addr, rkey, value,
                                    rkey->cache.amo_proto, loopback_ctx);
    }

    if (ucs_unlikely(status != UCS_OK)) {
        ucp_request_put(req);
        return UCS_STATUS_PTR(status);
    }

    return ucp_rma_send_request_cb(req, cb);
}

ucs_status_ptr_t ucp_atomic_fetch_nb(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                                     uint64_t value, void *result, size_t op_size,
                                     uint64_t remote_addr, ucp_rkey_h rkey,
                                     ucp_send_callback_t cb)
{
    ucs_status_ptr_t status_p;

    UCP_AMO_CHECK_PARAM(ep->worker->context, remote_addr, op_size, opcode,
                        UCP_ATOMIC_FETCH_OP_LAST,
                        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("atomic_fetch_nb opcode %d value %"PRIu64" buffer %p size %zu"
                  " remote_addr %"PRIx64" rkey %p to %s cb %p",
                  opcode, value, result, op_size, remote_addr, rkey,
                  ucp_ep_peer_name(ep), cb);

    status_p = ucp_atomic_fetch_internal(ep, opcode, value, result, op_size,
                                         remote_addr, rkey, NULL, cb);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    return status_p;
}

ucs_status_t ucp_atomic_post_internal(ucp_ep_h ep, ucp_atomic_post_op_t opcode,
                                      uint64_t value, size_t op_size,
                                      uint64_t remote_addr, ucp_rkey_h rkey,
                                      const ucp_atomic_loopback_ctx_t *loopback_ctx,
                                      ucp_send_callback_t completion_cb)
{
    ucs_status_t status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    ucs_status_ptr_t status_p;
    ucp_request_t *req;

    if (status != UCS_OK) {
        return status;
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        return UCS_ERR_NO_MEMORY;
    }

    status = ucp_amo_init_post(req, ep, ucp_uct_op_table[opcode], op_size,
                               remote_addr, rkey, value, rkey->cache.amo_proto,
                               loopback_ctx);
    if (ucs_unlikely(status != UCS_OK)) {
        ucp_request_put(req);
        return status;
    }

    status_p = ucp_rma_send_request_cb(req, completion_cb);
    if (UCS_PTR_IS_PTR(status_p)) {
        ucp_request_release(status_p);
        return UCS_OK;
    }

    return UCS_PTR_STATUS(status_p);
}


ucs_status_t ucp_atomic_post(ucp_ep_h ep, ucp_atomic_post_op_t opcode, uint64_t value,
                             size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_t status;

    UCP_AMO_CHECK_PARAM(ep->worker->context, remote_addr, op_size, opcode,
                        UCP_ATOMIC_POST_OP_LAST, return UCS_ERR_INVALID_PARAM);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("atomic_post opcode %d value %"PRIu64" size %zu "
                  "remote_addr %"PRIx64" rkey %p to %s",
                  opcode, value, op_size, remote_addr, rkey,
                  ucp_ep_peer_name(ep));
    status = ucp_atomic_post_internal(ep, opcode, value, op_size, remote_addr,
                                      rkey, NULL,
                                      (ucp_send_callback_t)ucs_empty_function);
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
