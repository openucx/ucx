/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_AMO_INL_
#define UCP_AMO_INL_

#include <inttypes.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_ep.inl>
#include <ucs/sys/preprocessor.h>
#include <ucs/debug/log.h>
#include <ucs/profile/profile.h>

static inline
ucs_status_t ucp_amo_check_send_status(ucp_request_t *req, ucs_status_t status);

#define _UCP_PROGRESS_AMO_NAME(_function)           \
    UCS_PP_TOKENPASTE(ucp_amo_progress_, _function)

#define UCP_RMA_CHECK_ATOMIC_PTR(_addr, _op_size) \
    do { \
        ucs_status_t status = ucp_rma_check_atomic(_addr, _op_size); \
        \
        if (status != UCS_OK) { \
            return UCS_STATUS_PTR(status); \
        } \
    } while(0)

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

static inline ucs_status_t ucp_rma_check_atomic(uint64_t remote_addr, size_t size)
{
    if (ENABLE_PARAMS_CHECK && ((remote_addr % size) != 0)) {
        ucs_debug("Error: Atomic variable must be naturally aligned "
                  "(got address 0x%"PRIx64", atomic size %zu)", (remote_addr),
                  (size));
        return UCS_ERR_INVALID_PARAM;
    }
    return UCS_OK;
}

static inline ucs_status_ptr_t 
ucp_amo_send_request(ucp_request_t *req, ucp_send_callback_t cb)
{
    ucs_status_t status = ucp_request_send(req);

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucs_mpool_put(req);
        return UCS_STATUS_PTR(status);
    }
    ucs_trace_req("returning amo request %p, status %s", req,
                  ucs_status_string(status));
    ucp_request_set_callback(req, send.cb, cb);
    return req + 1;
}
#endif
