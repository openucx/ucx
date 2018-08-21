/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rma.h"
#include "rma.inl"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <inttypes.h>


#define UCP_AMO_CHECK_PARAM(_remote_addr, _size, _opcode, _last_opcode, \
                            _err_status) \
    { \
        if (ENABLE_PARAMS_CHECK && \
            ucs_unlikely(((_remote_addr) % (_size)) != 0)) { \
            ucs_error("atomic variable must be naturally aligned " \
                      "(remote address 0x%"PRIx64", size %zu)", (_remote_addr), \
                      (_size)); \
            return _err_status; \
        } \
        \
        if (ENABLE_PARAMS_CHECK && \
            (ucs_unlikely((_opcode) >= (_last_opcode)))) { \
            ucs_error("invalid atomic opcode %d ", _opcode); \
            return _err_status; \
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
#if ENABLE_ASSERT
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
    ucs_status_ptr_t status_p;
    ucs_status_t status;
    ucp_request_t *req;

    UCP_AMO_CHECK_PARAM(remote_addr, op_size, opcode, UCP_ATOMIC_FETCH_OP_LAST,
                        UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        status_p = UCS_STATUS_PTR(UCS_ERR_UNREACHABLE);
        goto out;
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        status_p = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ucp_amo_init_fetch(req, ep, result, ucp_uct_fop_table[opcode], op_size,
                       remote_addr, rkey, value, rkey->cache.amo_proto);

    status_p = ucp_rma_send_request_cb(req, cb);

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
    uct_atomic_op_t op;

    UCP_AMO_CHECK_PARAM(remote_addr, op_size, opcode, UCP_ATOMIC_POST_OP_LAST,
                        UCS_ERR_INVALID_PARAM);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    op     = ucp_uct_op_table[opcode];
    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        goto out;
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    ucp_amo_init_post(req, ep, op, op_size, remote_addr, rkey, value,
                      rkey->cache.amo_proto);

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
                                  rkey, (void*)ucs_empty_function);
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
