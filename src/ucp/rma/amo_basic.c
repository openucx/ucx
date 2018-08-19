/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rma.inl"

#include <ucs/profile/profile.h>


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

static inline
ucs_status_t ucp_amo_check_send_status(ucp_request_t *req, ucs_status_t status)
{
    if (status == UCS_INPROGRESS) {
        return UCS_OK;
    }
    /* Complete for UCS_OK and unexpected errors */
    if (status != UCS_ERR_NO_RESOURCE) {
        ucp_request_complete_send(req, status);
    }
    return status;
}

static ucs_status_t ucp_amo_basic_progress_post(uct_pending_req_t *self)
{
    ucp_request_t *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rkey_h rkey      = req->send.amo.rkey;
    ucp_ep_t *ep         = req->send.ep;
    uint64_t value       = req->send.amo.value;
    uint64_t remote_addr = req->send.amo.remote_addr;
    uct_atomic_op_t op   = req->send.amo.uct_op;
    ucs_status_t status;

    req->send.lane = rkey->cache.amo_lane;
    if (req->send.length == sizeof(uint64_t)) {
        status = UCS_PROFILE_CALL(uct_ep_atomic64_post,
                                  ep->uct_eps[req->send.lane], op, value,
                                  remote_addr, rkey->cache.amo_rkey);
    } else {
        ucs_assert(req->send.length == sizeof(uint32_t));
        status = UCS_PROFILE_CALL(uct_ep_atomic32_post,
                                  ep->uct_eps[req->send.lane], op, value,
                                  remote_addr, rkey->cache.amo_rkey);
    }

    return ucp_amo_check_send_status(req, status);
}

static ucs_status_t ucp_amo_basic_progress_fetch(uct_pending_req_t *self)
{
    ucp_request_t *req    = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rkey_h rkey       = req->send.amo.rkey;
    ucp_ep_t *ep          = req->send.ep;
    uint64_t value        = req->send.amo.value;
    uint64_t *result      = (void*)req->send.buffer;
    uint64_t remote_addr  = req->send.amo.remote_addr;
    uct_atomic_op_t op    = req->send.amo.uct_op;
    ucs_status_t status;

    req->send.lane = rkey->cache.amo_lane;
    if (req->send.length == sizeof(uint64_t)) {
        if (op != UCT_ATOMIC_OP_CSWAP) {
            status = uct_ep_atomic64_fetch(ep->uct_eps[req->send.lane],
                                           op, value, result,
                                           remote_addr,
                                           rkey->cache.amo_rkey,
                                           &req->send.state.uct_comp);
        } else {
            status = uct_ep_atomic_cswap64(ep->uct_eps[req->send.lane],
                                           value, *result,
                                           remote_addr, rkey->cache.amo_rkey, result,
                                           &req->send.state.uct_comp);
        }
    } else {
        ucs_assert(req->send.length == sizeof(uint32_t));
        if (op != UCT_ATOMIC_OP_CSWAP) {
            status = uct_ep_atomic32_fetch(ep->uct_eps[req->send.lane],
                                           op, value, (uint32_t*)result,
                                           remote_addr,
                                           rkey->cache.amo_rkey,
                                           &req->send.state.uct_comp);
        } else {
            status = uct_ep_atomic_cswap32(ep->uct_eps[req->send.lane],
                                           value, *result, remote_addr,
                                           rkey->cache.amo_rkey, (uint32_t*)result,
                                           &req->send.state.uct_comp);
        }
    }

    return ucp_amo_check_send_status(req, status);
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
                   ucp_rkey_h rkey, uint64_t value)
{
    ucp_amo_init_common(req, ep, op, remote_addr, rkey, value, op_size);
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_amo_completed_single;
    req->send.uct.func              = ucp_amo_basic_progress_fetch;
    req->send.buffer                = buffer;
}

static UCS_F_ALWAYS_INLINE
void ucp_amo_init_post(ucp_request_t *req, ucp_ep_h ep, uct_atomic_op_t op,
                       size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey,
                       uint64_t value)
{
    ucp_amo_init_common(req, ep, op, remote_addr, rkey, value, op_size);
    req->send.uct.func              = ucp_amo_basic_progress_post;
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
                       remote_addr, rkey, value);

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

    ucp_amo_init_post(req, ep, op, op_size, remote_addr, rkey, value);

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
    return ucp_rma_wait(ep->worker,
                        ucp_atomic_fetch_nb(ep, UCP_ATOMIC_FETCH_OP_FADD, add,
                                            result, sizeof(add), remote_addr,
                                            rkey, (void*)ucs_empty_function),
                        "atomic_fadd64");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap32, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint32_t *result)
{
    return ucp_rma_wait(ep->worker,
                        ucp_atomic_fetch_nb(ep, UCP_ATOMIC_FETCH_OP_SWAP, swap,
                                            result, sizeof(swap), remote_addr,
                                            rkey, (void*)ucs_empty_function),
                        "atomic_swap32");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_swap64, (ep, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t swap, uint64_t remote_addr, ucp_rkey_h rkey,
                 uint64_t *result)
{
    return ucp_rma_wait(ep->worker,
                        ucp_atomic_fetch_nb(ep, UCP_ATOMIC_FETCH_OP_SWAP, swap,
                                            result, sizeof(swap), remote_addr,
                                            rkey, (void*)ucs_empty_function),
                        "atomic_swap64");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap32,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint32_t compare, uint32_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint32_t *result)
{
    ucs_status_t status;
    uint32_t tmp = swap;

    status = ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_CSWAP, compare, &tmp,
                                sizeof(swap), remote_addr, rkey, "atomic_cswap32");
    if (status == UCS_OK) {
        *result = tmp;
    }
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_atomic_cswap64,
                 (ep, compare, swap, remote_addr, rkey, result),
                 ucp_ep_h ep, uint64_t compare, uint64_t swap,
                 uint64_t remote_addr, ucp_rkey_h rkey, uint64_t *result)
{
    ucs_status_t status;
    uint64_t tmp = swap;

    status = ucp_atomic_fetch_b(ep, UCP_ATOMIC_FETCH_OP_CSWAP, compare, &tmp,
                                sizeof(swap), remote_addr, rkey, "atomic_cswap64");
    if (status == UCS_OK) {
        *result = tmp;
    }
    return status;
}
