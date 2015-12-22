/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_ep.h"
#include "ucp_worker.h"

#include <ucp/wireup/wireup.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <string.h>


ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *peer_name, const char *message, ucp_ep_h *ep_p)
{
    ucp_ep_h ep;

    ep = ucs_calloc(1, sizeof(*ep), "ucp ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate ep");
        return UCS_ERR_NO_MEMORY;
    }

    ep->worker               = worker;
    ep->uct_ep               = NULL;
    ep->config.max_short_egr = SIZE_MAX;
    ep->config.max_bcopy_egr = SIZE_MAX;
    ep->config.max_short_put = SIZE_MAX;
    ep->config.max_bcopy_put = SIZE_MAX;
    ep->config.max_bcopy_get = SIZE_MAX;
    ep->dest_uuid            = dest_uuid;
    ep->rsc_index            = -1;
    ep->dst_pd_index         = -1;
    ep->state                = 0;
    sglib_hashed_ucp_ep_t_add(worker->ep_hash, ep);
#if ENABLE_DEBUG_DATA
    ucs_snprintf_zero(ep->peer_name, UCP_PEER_NAME_MAX, "%s", peer_name);
#endif

    ucs_debug("created ep %p to %s 0x%"PRIx64"->0x%"PRIx64" %s", ep,
              ucp_ep_peer_name(ep), worker->uuid, ep->dest_uuid, message);
    *ep_p = ep;
    return UCS_OK;
}

static ucs_status_t ucp_pending_req_release(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, uct);

    ucp_request_complete(req, req->cb.send, UCS_ERR_CANCELED)
    return UCS_OK;
}

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep)
{
    ucs_assert_always(uct_ep != NULL);
    uct_ep_pending_purge(uct_ep, ucp_pending_req_release);
    uct_ep_destroy(uct_ep);
}

void ucp_ep_add_pending(uct_ep_h uct_ep, ucp_request_t *req)
{
    ucs_status_t status;

    ucs_trace_data("add pending request %p uct %p to uct_ep %p", req,
                   &req->uct, uct_ep);

    for (;;) {
        status = uct_ep_pending_add(uct_ep, &req->uct);
        if (status != UCS_ERR_BUSY) {
            ucs_assert(status == UCS_OK);
            return; /* Added to pending */
        }

        /* Forced progress */
        status = req->uct.func(&req->uct);
        if (status == UCS_OK) {
            return; /* Completed the operation */
        }
    }
}

ucs_status_t ucp_ep_create(ucp_worker_h worker, ucp_address_t *address,
                           ucp_ep_h *ep_p)
{
    uint64_t dest_uuid = ucp_address_uuid(address);
    char peer_name[UCP_PEER_NAME_MAX];
    ucs_status_t status;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    ep = ucp_worker_ep_find(worker, dest_uuid);
    if (ep != NULL) {
        ucs_debug("returning existing ep %p which is already connected to %"PRIx64,
                  ep, ep->dest_uuid);
        goto out;
    }

    ucp_address_peer_name(address, peer_name);
    status = ucp_ep_new(worker, dest_uuid, peer_name, " from api call", &ep);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucp_wireup_start(ep, address);
    if (status != UCS_OK) {
        goto err_free;
    }

out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    *ep_p = ep;
    return UCS_OK;

err_free:
    ucs_free(ep);
err:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    ucs_debug("destroy ep %p", ep);
    ucp_wireup_stop(ep);
    if (ep->state & UCP_EP_STATE_READY_TO_SEND) {
        ucp_ep_destroy_uct_ep_safe(ep, ep->uct_ep);
    } else {
        ucs_assert(ep->uct_ep == NULL);
    }
    ucs_free(ep);
}

