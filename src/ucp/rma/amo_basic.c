/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rma.h"
#include "rma.inl"

#include <ucs/profile/profile.h>


static UCS_F_ALWAYS_INLINE
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
    uint64_t *result      = req->send.buffer;
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

ucp_amo_proto_t ucp_amo_basic_proto = {
    .name           = "basic_amo",
    .progress_fetch = ucp_amo_basic_progress_fetch,
    .progress_post  = ucp_amo_basic_progress_post
};
