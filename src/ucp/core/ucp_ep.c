/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_ep.h"
#include "ucp_request.h"
#include "ucp_worker.h"

#include <ucp/wireup/stub_ep.h>
#include <ucp/wireup/wireup.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <string.h>


static ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                               const char *peer_name, const char *message,
                               ucp_ep_h *ep_p)
{
    ucp_ep_h ep;

    ep = ucs_calloc(1, sizeof(*ep), "ucp ep");
    if (ep == NULL) {
        ucs_error("Failed to allocate ep");
        return UCS_ERR_NO_MEMORY;
    }

    ep->worker               = worker;
    ep->dest_uuid            = dest_uuid;
    ep->rma_dst_pdi          = UCP_NULL_RESOURCE;
    ep->amo_dst_pdi          = UCP_NULL_RESOURCE;
    ep->cfg_index            = 0;
    ep->flags                = 0;
#if ENABLE_DEBUG_DATA
    ucs_snprintf_zero(ep->peer_name, UCP_WORKER_NAME_MAX, "%s", peer_name);
#endif
    sglib_hashed_ucp_ep_t_add(worker->ep_hash, ep);

    *ep_p                    = ep;
    ucs_debug("created ep %p to %s 0x%"PRIx64"->0x%"PRIx64" %s", ep, peer_name,
              worker->uuid, ep->dest_uuid, message);
    return UCS_OK;
}

static void ucp_ep_delete(ucp_ep_h ep)
{
    sglib_hashed_ucp_ep_t_delete(ep->worker->ep_hash, ep);
    ucs_free(ep);
}

ucs_status_t ucp_ep_create_connected(ucp_worker_h worker, uint64_t dest_uuid,
                                     const char *peer_name, unsigned address_count,
                                     const ucp_address_entry_t *address_list,
                                     const char *message, ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_h ep = NULL;

    status = ucp_ep_new(worker, dest_uuid, peer_name, message, &ep);
    if (status != UCS_OK) {
        goto err;
    }

    /* initialize transport endpoints */
    status = ucp_ep_init_trasports(ep, address_count, address_list);
    if (status != UCS_OK) {
        goto err_delete;
    }

    *ep_p = ep;
    return UCS_OK;

err_delete:
    ucp_ep_delete(ep);
err:
    return status;
}

ucs_status_t ucp_ep_create_stub(ucp_worker_h worker, uint64_t dest_uuid,
                                const char *message, ucp_ep_h *ep_p)
{
    ucs_status_t status;
    ucp_ep_op_t optype;
    ucp_ep_h ep = NULL;

    status = ucp_ep_new(worker, dest_uuid, "??", message, &ep);
    if (status != UCS_OK) {
        goto err;
    }

    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        status = ucp_stub_ep_create(ep, optype, 0, NULL, &ep->uct_eps[optype]);
        if (status != UCS_OK) {
            goto err_destroy_uct_eps;
        }
    }

    *ep_p = ep;
    return UCS_OK;

err_destroy_uct_eps:
    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (ep->uct_eps[optype] != NULL) {
            uct_ep_destroy(ep->uct_eps[optype]);
        }
    }
    ucp_ep_delete(ep);
err:
    return status;
}

ucs_status_t ucp_ep_pending_req_release(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    ucp_request_complete(req, req->cb.send, UCS_ERR_CANCELED)
    return UCS_OK;
}

ucs_status_t ucp_ep_add_pending_uct(ucp_ep_h ep, uct_ep_h uct_ep,
                                    uct_pending_req_t *req)
{
    ucs_status_t status;

    ucs_assertv(req->func != NULL, "req=%p", req);

    status = uct_ep_pending_add(uct_ep, req);
    if (status != UCS_ERR_BUSY) {
        ucs_assert(status == UCS_OK);
        ucs_trace_data("ep %p: added pending uct request %p to uct_ep %p", ep,
                       req, uct_ep);
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
        ucs_debug("returning existing ep %p which is already connected to %"PRIx64,
                  ep, ep->dest_uuid);
        *ep_p = ep;
        status = UCS_OK;
        goto out_free_address;
    }

    status = ucp_ep_create_connected(worker, dest_uuid, peer_name, address_count,
                                     address_list, " from api call", &ep);
    if (status != UCS_OK) {
        goto out_free_address;
    }

    /* send initial wireup message */
    if (!(ep->flags & UCP_EP_FLAG_LOCAL_CONNECTED)) {
        status = ucp_wireup_send_request(ep);
        if (status != UCS_OK) {
            goto err_destroy_ep;
        }
    }

    *ep_p = ep;
    goto out_free_address;

err_destroy_ep:
    ucp_ep_destroy(ep);
out_free_address:
    ucs_free(address_list);
out:
    UCS_ASYNC_UNBLOCK(&worker->async);
    return status;
}

static void ucp_ep_destory_uct_eps(ucp_ep_h ep)
{
    uct_ep_h uct_ep;
    int optype;

    for (optype = 0; optype < UCP_EP_OP_LAST; ++optype) {
        if (!ucp_ep_is_op_primary(ep, optype)) {
            continue;
        }

        uct_ep = ep->uct_eps[optype];
        uct_ep_pending_purge(uct_ep, ucp_ep_pending_req_release);
        ucs_debug("destroy ep %p op %d uct_ep %p", ep, optype, uct_ep);
        uct_ep_destroy(uct_ep);
    }
}

void ucp_ep_destroy(ucp_ep_h ep)
{
    ucp_worker_h worker = ep->worker;

    ucs_debug("destroy ep %p", ep);

    UCS_ASYNC_BLOCK(&worker->async);
    sglib_hashed_ucp_ep_t_delete(worker->ep_hash, ep);
    ucp_ep_destory_uct_eps(ep);
    UCS_ASYNC_UNBLOCK(&worker->async);

    ucs_free(ep);
}

void ucp_ep_send_reply(ucp_request_t *req, ucp_ep_op_t optype, int progress)
{
    ucp_ep_h ep = req->send.ep;
    ucp_ep_add_pending(ep, ep->uct_eps[optype], req, progress);
}

int ucp_ep_is_op_primary(ucp_ep_h ep, ucp_ep_op_t optype)
{
    ucp_ep_config_t *config = ucp_ep_config(ep);
    return (config->rscs[optype] != UCP_NULL_RESOURCE) && /* exists */
           (config->dups[optype] == UCP_EP_OP_LAST);      /* not a duplicate */
}
