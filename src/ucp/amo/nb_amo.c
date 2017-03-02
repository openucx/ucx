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
#include <ucs/debug/profile.h>
#include <ucs/debug/log.h>
#include <inttypes.h>

ucs_status_ptr_t ucp_atomic_fetch_nb(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                                     uint64_t value, void *result, size_t op_size,
                                     uint64_t remote_addr, ucp_rkey_h rkey,
                                     ucp_send_callback_t cb)
{
    ucp_request_t *req;
    ucs_status_ptr_t status;
    UCP_RMA_CHECK_ATOMIC_PTR(remote_addr, op_size);
    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);
    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(NULL == req)) {
        UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }
    init_amo_req(req, ep, result, opcode, op_size, remote_addr, rkey, value);
    status = ucp_amo_send_request(req, cb);
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return status;
}

ucs_status_t ucp_atomic_post(ucp_ep_h ep, ucp_atomic_post_op_t opcode, uint64_t value,
                             size_t op_size, uint64_t remote_addr, ucp_rkey_h rkey)    
{
    ucs_status_ptr_t status_p;
    ucs_status_t status;
    ucp_request_t *req;

    if (ucs_unlikely(opcode != UCP_ATOMIC_POST_OP_ADD)) {
        return UCS_ERR_INVALID_PARAM;
    }
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
        status = UCS_PROFILE_CALL(uct_ep_atomic_add32, ep->uct_eps[rkey->cache.amo_lane],
                                  (uint32_t)value, remote_addr, rkey->cache.amo_rkey);
    } else if (op_size == sizeof(uint64_t)) {
        status = UCS_PROFILE_CALL(uct_ep_atomic_add64, ep->uct_eps[rkey->cache.amo_lane],
                                  (uint64_t)value, remote_addr, rkey->cache.amo_rkey);
    }
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        req = ucp_request_get(ep->worker);
        if (ucs_unlikely(NULL == req)) {
            UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
            return UCS_ERR_NO_MEMORY;
        }
        init_amo_post(req, ep, opcode, op_size, remote_addr, rkey, value);
        status_p = ucp_amo_send_request(req, (ucp_send_callback_t)ucs_empty_function);
        if (UCS_PTR_IS_PTR(status_p)) {
            ucp_request_release(status_p);
            status = UCS_INPROGRESS;
        } else {
            status = UCS_PTR_STATUS(status_p);
        }
    }
out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return status;
}
