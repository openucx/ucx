/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "match.h"
#include "eager.h"
#include "rndv.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/dt/dt_generic.h>
#include <ucs/arch/arch.h>
#include <ucs/datastruct/mpool.inl>
#include <string.h>


static ucs_status_t
ucp_tag_send_start_req(ucp_ep_h ep, const void *buffer, size_t count,
                       ucp_datatype_t datatype, ucp_tag_t tag,
                       ucp_request_t *req)
{
    size_t rndv_thresh = ep->worker->context->config.rndv_thresh;
    ucp_dt_generic_t *dt_gen;
    void *state;

    req->status            = UCS_INPROGRESS;
    req->send.ep           = ep;
    req->send.buffer       = buffer;
    req->send.count        = count;
    req->send.datatype     = datatype;
    req->send.state.offset = 0;
    req->send.tag          = tag;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        /* TODO check for zero-copy */
        req->send.length = ucp_contig_dt_length(datatype, count);
        if (req->send.length <= rndv_thresh) {
            req->send.uct.func = ucp_tag_progress_eager_contig;
            return UCS_OK;
        }
        break;

    case UCP_DATATYPE_GENERIC:
        dt_gen = ucp_dt_generic(datatype);
        state = dt_gen->ops->start_pack(dt_gen->context, buffer, count);

        req->send.state.dt.generic.state = state;
        req->send.length = dt_gen->ops->packed_size(state);
        if (req->send.length <= rndv_thresh) {
            req->send.uct.func = ucp_tag_progress_eager_generic;
            return UCS_OK;
        }
        break;

    default:
        return UCS_ERR_INVALID_PARAM;
    }

    return ucp_tag_send_start_rndv(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_send_try(ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_tag_t tag)
{
    size_t length;

    if ((datatype & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_CONTIG) {
        length = ucp_contig_dt_length(datatype, count);
        if (length <= ep->config.max_short_egr) {
            return ucp_tag_send_eager_short(ep, tag, buffer, length);
        }
    }

    return UCS_ERR_NO_RESOURCE; /* Fallback to slower progress */
}

static void ucp_tag_send_blocking_completion(void *request, ucs_status_t status)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    req->status = status;
}

ucs_status_t ucp_tag_send(ucp_ep_h ep, const void *buffer, size_t count,
                          uintptr_t datatype, ucp_tag_t tag)
{
    ucs_status_t status;
    ucp_request_t req;

    ucs_trace_req("send buffer %p count %zu tag %"PRIx64, buffer, count, tag);

    status = ucp_tag_send_try(ep, buffer, count, datatype, tag);
    if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
        return UCS_OK;
    }

    req.flags    = UCP_REQUEST_FLAG_BLOCKING;
    req.cb.send  = ucp_tag_send_blocking_completion;

    status = ucp_tag_send_start_req(ep, buffer, count, datatype, tag, &req);
    if (status != UCS_OK) {
        return status;
    }

    do {
        ucp_worker_progress(ep->worker);
        req.send.uct.func(&req.send.uct);
        /* coverity[loop_condition] */
    } while (!(req.flags & UCP_REQUEST_FLAG_COMPLETED));
    return req.status;
}

ucs_status_ptr_t ucp_tag_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                 uintptr_t datatype, ucp_tag_t tag,
                                 ucp_send_callback_t cb)
{
    ucs_status_t status;
    ucp_request_t *req;

    ucs_trace_req("send_nb buffer %p count %zu tag %"PRIx64, buffer, count, tag);

    status = ucp_tag_send_try(ep, buffer, count, datatype, tag);
    if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
        return UCS_STATUS_PTR(status); /* UCS_OK also goes here */
    }

    req = ucs_mpool_get(&ep->worker->req_mp);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    VALGRIND_MAKE_MEM_DEFINED(req + 1, ep->worker->context->config.request.size);

    req->flags   = 0;
    req->cb.send = cb;

    status = ucp_tag_send_start_req(ep, buffer, count, datatype, tag, req);
    if (status != UCS_OK) {
        return UCS_STATUS_PTR(status); /* UCS_OK also goes here */
    }

    if (!(req->flags & UCP_REQUEST_FLAG_COMPLETED)) {
        ucp_ep_add_pending(ep, ep->uct_ep, req);
        ucp_worker_progress(ep->worker);
    }

    ucs_trace_req("send_nb returning request %p", req);
    return req + 1;
}

