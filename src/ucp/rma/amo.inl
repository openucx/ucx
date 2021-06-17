/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_AMO_INL_
#define UCP_AMO_INL_

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_types.h>
#include <ucp/core/ucp_request.inl>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_amo_check_send_status(ucp_request_t *req, ucs_status_t status)
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

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_amo_progress_post(ucp_request_t *req, uct_rkey_t tl_rkey)
{
    ucp_ep_t *ep         = req->send.ep;
    uint64_t value       = req->send.amo.value;
    uint64_t remote_addr = req->send.amo.remote_addr;
    uct_atomic_op_t op   = req->send.amo.uct_op;
    ucs_status_t status;

    if (req->send.length == sizeof(uint64_t)) {
        status = UCS_PROFILE_CALL(uct_ep_atomic64_post,
                                  ep->uct_eps[req->send.lane], op, value,
                                  remote_addr, tl_rkey);
    } else {
        ucs_assert(req->send.length == sizeof(uint32_t));
        status = UCS_PROFILE_CALL(uct_ep_atomic32_post,
                                  ep->uct_eps[req->send.lane], op, value,
                                  remote_addr, tl_rkey);
    }

    return ucp_amo_check_send_status(req, status);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_amo_progress_fetch(ucp_request_t *req, uct_rkey_t tl_rkey)
{
    ucp_ep_t *ep         = req->send.ep;
    uint64_t value       = req->send.amo.value;
    uint64_t *result     = req->send.buffer;
    uint64_t remote_addr = req->send.amo.remote_addr;
    uct_atomic_op_t op   = req->send.amo.uct_op;
    ucs_status_t status;

    if (req->send.length == sizeof(uint64_t)) {
        if (op != UCT_ATOMIC_OP_CSWAP) {
            status = uct_ep_atomic64_fetch(ep->uct_eps[req->send.lane],
                                           op, value, result,
                                           remote_addr, tl_rkey,
                                           &req->send.state.uct_comp);
        } else {
            status = uct_ep_atomic_cswap64(ep->uct_eps[req->send.lane],
                                           value, *result,
                                           remote_addr, tl_rkey, result,
                                           &req->send.state.uct_comp);
        }
    } else {
        ucs_assert(req->send.length == sizeof(uint32_t));
        if (op != UCT_ATOMIC_OP_CSWAP) {
            status = uct_ep_atomic32_fetch(ep->uct_eps[req->send.lane],
                                           op, value, (uint32_t*)result,
                                           remote_addr, tl_rkey,
                                           &req->send.state.uct_comp);
        } else {
            status = uct_ep_atomic_cswap32(ep->uct_eps[req->send.lane],
                                           value, *(uint32_t*)result, remote_addr,
                                           tl_rkey, (uint32_t*)result,
                                           &req->send.state.uct_comp);
        }
    }

    return ucp_amo_check_send_status(req, status);
}

#endif
