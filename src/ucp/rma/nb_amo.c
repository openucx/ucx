/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "amo.inl"
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/sys/preprocessor.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/log.h>
#include <inttypes.h>

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

static ucs_status_t
_UCP_PROGRESS_AMO_NAME(uct_ep_atomic_post)(uct_pending_req_t *_self)
{
    ucp_request_t *req   = ucs_container_of(_self, ucp_request_t, send.uct);
    ucp_rkey_h rkey      = req->send.amo.rkey;
    ucp_ep_t *ep         = req->send.ep;
    uint64_t value       = req->send.amo.value;
    uint64_t remote_addr = req->send.amo.remote_addr;
    uct_atomic_op_t op   = req->send.amo.uct_op;
    ucs_status_t status;

    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        return UCS_ERR_UNREACHABLE;
    }

    req->send.lane = rkey->cache.amo_lane;
    if (req->send.amo.size == sizeof(uint64_t)) {
        status = UCS_PROFILE_CALL(uct_ep_atomic64_post,
                                  ep->uct_eps[req->send.lane], op, value,
                                  remote_addr, rkey->cache.amo_rkey);
    } else {
        ucs_assert(req->send.amo.size == sizeof(uint32_t));
        status = UCS_PROFILE_CALL(uct_ep_atomic32_post,
                                  ep->uct_eps[req->send.lane], op, value,
                                  remote_addr, rkey->cache.amo_rkey);
    }

    return ucp_amo_check_send_status(req, status);
}

static ucs_status_t
_UCP_PROGRESS_AMO_NAME(uct_ep_atomic_fetch)(uct_pending_req_t *_self)
{
    ucp_request_t *req    = ucs_container_of(_self, ucp_request_t, send.uct);
    ucp_rkey_h rkey       = req->send.amo.rkey;
    ucp_ep_t *ep          = req->send.ep;
    uint64_t value        = req->send.amo.value;
    uint64_t *result      = req->send.amo.result;
    uint64_t remote_addr  = req->send.amo.remote_addr;
    uct_atomic_op_t op    = req->send.amo.uct_op;
    ucs_status_t status;

    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        return UCS_ERR_UNREACHABLE;
    }

    req->send.lane = rkey->cache.amo_lane;
    if (req->send.amo.size == sizeof(uint64_t)) {
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
        ucs_assert(req->send.amo.size == sizeof(uint32_t));
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

static inline void init_amo_common(ucp_request_t *req, ucp_ep_h ep, uct_atomic_op_t op,
                                   uint64_t remote_addr, ucp_rkey_h rkey,
                                   uint64_t value, size_t size)
{
    req->flags                = 0;
    req->send.ep              = ep;
    req->send.amo.uct_op      = op;
    req->send.amo.remote_addr = remote_addr;
    req->send.amo.rkey        = rkey;
    req->send.amo.value       = value;
    req->send.amo.size        = size;
#if ENABLE_ASSERT
    req->send.lane            = UCP_NULL_LANE;
#endif
}

static inline void init_amo_fetch(ucp_request_t *req, ucp_ep_h ep, void *buffer,
                                  uct_atomic_op_t op, size_t op_size,
                                  uint64_t remote_addr, ucp_rkey_h rkey, uint64_t value)
{
    init_amo_common(req, ep, op, remote_addr, rkey, value, op_size);
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_amo_completed_single;
    req->send.amo.result            = buffer;
    req->send.uct.func              = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_fetch);
}

static inline void init_amo_post(ucp_request_t *req, ucp_ep_h ep,
                                 uct_atomic_op_t op,
                                 size_t op_size, uint64_t remote_addr,
                                 ucp_rkey_h rkey, uint64_t value)
{
    init_amo_common(req, ep, op, remote_addr, rkey, value, op_size);
    req->send.uct.func = _UCP_PROGRESS_AMO_NAME(uct_ep_atomic_post);
}

ucs_status_ptr_t ucp_atomic_fetch_nb(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                                     uint64_t value, void *result, size_t op_size,
                                     uint64_t remote_addr, ucp_rkey_h rkey,
                                     ucp_send_callback_t cb)
{
    ucp_request_t *req;
    ucs_status_ptr_t status_p;

    UCP_RMA_CHECK_ATOMIC_PTR(remote_addr, op_size);
    if (ENABLE_PARAMS_CHECK && ucs_unlikely((opcode) >= UCP_ATOMIC_FETCH_OP_LAST)) {
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);
    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    init_amo_fetch(req, ep, result, ucp_uct_fop_table[opcode], op_size,
                   remote_addr, rkey, value);

    status_p = ucp_amo_send_request(req, cb);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return status_p;
}

ucs_status_t ucp_atomic_post(ucp_ep_h ep, ucp_atomic_post_op_t opcode, uint64_t value,
                             size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey)    
{
    ucs_status_ptr_t status_p;
    ucs_status_t status;
    ucp_request_t *req;
    uct_atomic_op_t op;

    if (ENABLE_PARAMS_CHECK && ucs_unlikely(opcode >= UCP_ATOMIC_POST_OP_LAST)) {
        return UCS_ERR_INVALID_PARAM;
    }

    op     = ucp_uct_op_table[opcode];
    status = ucp_rma_check_atomic(remote_addr, op_size);
    if (status != UCS_OK) {
        return status;
    }
    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    status = UCP_RKEY_RESOLVE(rkey, ep, amo);
    if (status != UCS_OK) {
        goto out;
    }

    if (op_size == sizeof(uint32_t)) {
        status = UCS_PROFILE_CALL(uct_ep_atomic32_post, ep->uct_eps[rkey->cache.amo_lane], op,
                                  (uint32_t)value, remote_addr, rkey->cache.amo_rkey);
    } else if (op_size == sizeof(uint64_t)) {
        status = UCS_PROFILE_CALL(uct_ep_atomic64_post, ep->uct_eps[rkey->cache.amo_lane], op,
                                  (uint64_t)value, remote_addr, rkey->cache.amo_rkey);
    } else {
        status =  UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req = ucp_request_get(ep->worker);
        if (ucs_unlikely(NULL == req)) {
            status = UCS_ERR_NO_MEMORY;
            goto out;
        }

        init_amo_post(req, ep, op, op_size, remote_addr, rkey, value);

        status_p = ucp_amo_send_request(req, (ucp_send_callback_t)ucs_empty_function);
        if (UCS_PTR_IS_PTR(status_p)) {
            ucp_request_release(status_p);
            status = UCS_OK;
        } else {
            status = UCS_PTR_STATUS(status_p);
        }
    }
out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return status;
}
