/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_context.h"
#include "ucp_worker.h"
#include "ucp_request.inl"

#include <ucp/tag/match.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>


int ucp_request_is_completed(void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    return !!(req->flags & UCP_REQUEST_FLAG_COMPLETED);
}

void ucp_request_release(void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    ucs_trace_data("release request %p (%p) flags: 0x%x", req, req + 1, req->flags);

    if ((req->flags |= UCP_REQUEST_FLAG_RELEASED) & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_data("put %p to mpool", req);
        ucs_mpool_put_inline(req);
    }
}

void ucp_request_cancel(ucp_worker_h worker, void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        return;
    }

    if (req->flags & UCP_REQUEST_FLAG_EXPECTED) {
        ucp_tag_cancel_expected(worker->context, req);
        ucp_request_complete_recv(req, UCS_ERR_CANCELED, NULL);
    }
}

static void ucp_worker_request_init_proxy(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_context_h context = worker->context;
    ucp_request_t *req = obj;

    if (context->config.request.init != NULL) {
        context->config.request.init(req + 1);
    }
}

static void ucp_worker_request_fini_proxy(ucs_mpool_t *mp, void *obj)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_context_h context = worker->context;
    ucp_request_t *req = obj;

    if (context->config.request.cleanup != NULL) {
        context->config.request.cleanup(req + 1);
    }
}

ucs_mpool_ops_t ucp_request_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucp_worker_request_init_proxy,
    .obj_cleanup   = ucp_worker_request_fini_proxy
};

ucs_status_t ucp_request_release_pending_send(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_complete_send(req, UCS_ERR_CANCELED);
    return UCS_OK;
}

/*
 * @return Whether completed.
 *         *req_status if filled with the completion status if completed.
 */
static ucs_status_t ucp_request_try_send(ucp_request_t *req, ucs_status_t *req_status)
{
    char UCS_V_UNUSED func_name[128];
    ucs_status_t status;
    uct_ep_h uct_ep;

    status = req->send.uct.func(&req->send.uct);
    if (status == UCS_OK) {
        /* Completed the operation */
        *req_status = UCS_OK;
        return 1;
    } else if (status == UCS_INPROGRESS) {
        /* Not completed, but made progress */
        return 0;
    } else if (status != UCS_ERR_NO_RESOURCE) {
        /* Unexpected error */
        *req_status = status;
        return 1;
    }

    ucs_assert(status == UCS_ERR_NO_RESOURCE);
    ucs_assertv(req->send.lane != UCP_NULL_LANE, "%s() did not set req->send.lane",
                ucs_debug_get_symbol_name(req->send.uct.func, func_name,
                                          sizeof(func_name)));

    /* No send resources, try to add to pending queue */
    uct_ep = req->send.ep->uct_eps[req->send.lane];
    status = uct_ep_pending_add(uct_ep, &req->send.uct);
    if (status == UCS_OK) {
        ucs_trace_data("ep %p: added pending uct request %p to lane[%d]=%p",
                       req->send.ep, req, req->send.lane, uct_ep);
        *req_status = UCS_INPROGRESS;
        return 1;
    } else if (status == UCS_ERR_BUSY) {
        /* Could not add, try to send again */
        return 0;
    } else {
        /* Unexpected error while adding to pending */
        ucs_assert(status != UCS_INPROGRESS);
        *req_status = status;
        return 1;
    }
}

ucs_status_t ucp_request_start_send(ucp_request_t *req)
{
    ucs_status_t status = UCS_ERR_NOT_IMPLEMENTED;
    while (!ucp_request_try_send(req, &status));
    return status;
}

