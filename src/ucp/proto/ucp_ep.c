/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <string.h>


static void ucp_ep_progress_pending(ucs_callback_t *self)
{
    ucp_ep_h ep = ucs_container_of(self, ucp_ep_t, notify);
    ucp_worker_h worker = ep->worker;
    ucp_ep_pending_op_t *op;
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_trace_func("ep=%p", ep);

    UCS_ASYNC_BLOCK(&worker->async);

    while (!ucs_queue_is_empty(&ep->pending_q)) {
        op = ucs_queue_head_elem_non_empty(&ep->pending_q, ucp_ep_pending_op_t,
                                           queue);
        status = op->progress(ep, op, &uct_ep);
        if (status == UCS_ERR_NO_RESOURCE) {
            /* We could not progress the operation. Request another notification
             * from the transport, and keep the endpoint in pending state.
             */
            uct_ep_req_notify(uct_ep, &ep->notify);
            goto out;
        }

        ucs_queue_pull_non_empty(&ep->pending_q);
        ucs_free(op);
    }

    ep->state &= ~UCP_EP_STATE_PENDING;

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
}

ucs_status_t ucp_ep_create(ucp_worker_h worker, ucp_address_t *address,
                           ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_h ep;

    ep = ucs_calloc(1, sizeof(*ep), "ucp ep");
    if (ep == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    ep->worker               = worker;
    ep->config.max_short_tag = SIZE_MAX;
    ep->config.max_short_put = SIZE_MAX;
    ep->config.max_bcopy_put = SIZE_MAX;
    ep->config.max_bcopy_get = SIZE_MAX;
    ep->notify.func          = ucp_ep_progress_pending;
    ep->state                = 0;
    ucs_queue_head_init(&ep->pending_q);

    status = ucp_ep_wireup_start(ep, address);
    if (status != UCS_OK) {
        goto err_free;
    }

    *ep_p = ep;
    return UCS_OK;

err_free:
    ucs_free(ep);
err:
    return status;
}

static void ucs_ep_purge_pending(ucp_ep_h ep)
{
    ucp_ep_pending_op_t *op;

    ucs_queue_for_each_extract(op, &ep->pending_q, queue, 1) {
        ucs_free(op); /* TODO release callback */
    }
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    ucp_ep_wireup_stop(ep);
    ucs_ep_purge_pending(ep);
    while (uct_ep_flush(ep->uct.ep) != UCS_OK) {
        ucp_worker_progress(ep->worker);
    }
    uct_ep_destroy(ep->uct.ep);
    ucs_free(ep);
}

