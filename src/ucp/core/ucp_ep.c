/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
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
    ep->dest_uuid            = dest_uuid;
    ep->rsc_index            = worker->context->num_tls;
    ep->dst_pd_index         = UCP_NULL_RESOURCE;
    ep->state                = 0;
    sglib_hashed_ucp_ep_t_add(worker->ep_hash, ep);
#if ENABLE_DEBUG_DATA
    ucs_snprintf_zero(ep->peer_name, UCP_WORKER_NAME_MAX, "%s", peer_name);
#endif

    ucs_debug("created ep %p to %s 0x%"PRIx64"->0x%"PRIx64" %s", ep,
              ucp_ep_peer_name(ep), worker->uuid, ep->dest_uuid, message);
    *ep_p = ep;
    return UCS_OK;
}

void ucp_ep_delete(ucp_ep_h ep)
{
    sglib_hashed_ucp_ep_t_delete(ep->worker->ep_hash, ep);
    ucs_free(ep);
}

static ucs_status_t ucp_pending_req_release(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    ucp_request_complete(req, req->cb.send, UCS_ERR_CANCELED)
    return UCS_OK;
}

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep)
{
    ucs_assert_always(uct_ep != NULL);
    uct_ep_pending_purge(uct_ep, ucp_pending_req_release);
    uct_ep_destroy(uct_ep);
}

ucs_status_t ucp_ep_add_pending_uct(ucp_ep_h ep, uct_ep_h uct_ep,
                                    uct_pending_req_t *req)
{
    ucs_status_t status;

    ucs_assertv(req->func != NULL, "req=%p", req);

    status = uct_ep_pending_add(uct_ep, req);
    if (status != UCS_ERR_BUSY) {
        ucs_assert(status == UCS_OK);
        ucs_trace_data("added pending uct request %p to uct_ep %p", req, uct_ep);
        return UCS_OK; /* Added to pending */
    }

    /* Forced progress */
    status = req->func(req);
    if (status == UCS_OK) {
        return UCS_OK; /* Completed the operation */
    }

    return UCS_ERR_NO_PROGRESS;
}

void ucp_ep_add_pending(ucp_ep_h ep, uct_ep_h uct_ep, ucp_request_t *req,
                        int progress)
{
    ucs_status_t status;

    req->send.ep = ep;
    status = ucp_ep_add_pending_uct(ep, uct_ep, &req->send.uct);
    while (status != UCS_OK) {
        if (progress) {
            ucp_worker_progress(ep->worker);
        }
        status = ucp_ep_add_pending_uct(ep, uct_ep, &req->send.uct);
    }
}

ucs_status_t ucp_ep_create(ucp_worker_h worker, const ucp_address_t *address,
                           ucp_ep_h *ep_p)
{
    char peer_name[UCP_WORKER_NAME_MAX];
    ucs_status_t status;
    uint64_t dest_uuid;
    unsigned address_count;
    ucp_address_entry_t *address_list;
    ucp_ep_h ep;

    UCS_ASYNC_BLOCK(&worker->async);

    status = ucp_address_unpack(address, &dest_uuid, peer_name, sizeof(peer_name),
                                &address_count, &address_list);
    if (status != UCS_OK) {
        ucs_error("failed to unpack remote address: %s", ucs_status_string(status));
        goto out;
    }

    ep = ucp_worker_ep_find(worker, dest_uuid);
    if (ep != NULL) {
        /* TODO handle a case where the existing endpoint is incomplete */
        ucs_assert(ep->dst_pd_index != UCP_NULL_RESOURCE);
        ucs_debug("returning existing ep %p which is already connected to %"PRIx64,
                  ep, ep->dest_uuid);
        *ep_p = ep;
        status = UCS_OK;
        goto out_free_address;
    }

    status = ucp_ep_new(worker, dest_uuid, peer_name, " from api call", &ep);
    if (status != UCS_OK) {
        goto out_free_address;
    }

    status = ucp_wireup_start(ep, address_list, address_count);
    if (status != UCS_OK) {
        goto out_delete_ep;
    }

    *ep_p = ep;
    goto out_free_address;

out_delete_ep:
    ucp_ep_delete(ep);
out_free_address:
    ucs_free(address_list);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    ucs_debug("destroy ep %p", ep);

    sglib_hashed_ucp_ep_t_delete(ep->worker->ep_hash, ep);
    ucp_wireup_stop(ep);
    if (ep->state & UCP_EP_STATE_READY_TO_SEND) {
        ucp_ep_destroy_uct_ep_safe(ep, ep->uct_ep);
    } else {
        ucs_assert(ep->uct_ep == NULL);
    }
    ucs_free(ep);
}

void ucp_ep_send_reply(ucp_request_t *req, int progress)
{
    ucp_ep_h ep = req->send.ep;
    ucp_ep_add_pending(ep, ep->uct_ep, req, progress);
}
