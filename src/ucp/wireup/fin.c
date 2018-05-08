/**
* Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "fin.h"

#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_ep.inl>

static size_t ucp_fin_msg_pack(void *dest, void *arg)
{
    memcpy(dest, arg, sizeof(ucp_fin_msg_t));
    return sizeof(ucp_fin_msg_t);
}

static ucs_status_t ucp_fin_msg_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_h      ep   = req->send.ep;
    ssize_t       len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    if (ucp_ep_ext_gen(ep)->dest_ep_ptr == 0) {
        req->send.fin.ep_id  = ep->worker->uuid;
        req->send.fin.is_ptr = 0;
    } else {
        req->send.fin.ep_id  = ucp_ep_ext_gen(ep)->dest_ep_ptr;
        req->send.fin.is_ptr = 1;
    }

    len = uct_ep_am_bcopy(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_FIN,
                          ucp_fin_msg_pack, &req->send.fin, 0);
    if (len > 0) {
        ucs_trace_req("putting request %p FIN of ep %p with flags %d",
                      req, ep, ep->flags);
        ucp_request_put(req);
        return UCS_OK;
    }
    return len;
}

static void ucp_ep_fin_cb(ucp_request_t *req)
{
    ucp_ep_h ep = req->send.ep;

    ucs_assertv((req->status == UCS_OK) ||
                (req->status == UCS_ERR_ENDPOINT_TIMEOUT ||
                (req->status == UCS_ERR_CANCELED)),
                "ep %p FIN msg is failed with status %s", ep,
                ucs_status_string(req->status));

    ucs_debug("ep %p, flags %d: completed FIN msg", ep, ep->flags);
    ucs_trace_req("putting request %p FIN flush of ep %p with flags %d",
                  req, ep, ep->flags);
    ucp_request_complete_send(req, req->status);
}

ucs_status_t ucp_fin_msg_send(ucp_ep_h ep)
{
    ucp_request_t *fin_req;
    ucp_request_t *flush_req;

    if (ep->flags & UCP_EP_FLAG_FIN_REQ_QUEUED) {
        return UCS_OK;
    }

    fin_req = ucp_request_get(ep->worker);
    if (fin_req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucs_debug("ep %p, flags %d: sending FIN msg", ep, ep->flags);

    fin_req->flags         = 0;
    fin_req->send.ep       = ep;
    fin_req->send.uct.func = ucp_fin_msg_progress;
    fin_req->send.buffer   = &fin_req->send.fin;
    fin_req->send.datatype = ucp_dt_make_contig(sizeof(ucp_fin_msg_t));
    ucs_trace_req("getting FIN request %p of ep %p with flags %d",
                  fin_req, ep, ep->flags);
    ucp_request_send_state_init(fin_req, fin_req->send.datatype, 1);
    ep->flags |= UCP_EP_FLAG_FIN_REQ_QUEUED;
    ucp_request_send(fin_req);

    ucs_debug("ep %p, flags: %d: flushing FIN msg", ep, ep->flags);

    flush_req = ucp_ep_flush_internal(ep, UCT_FLUSH_FLAG_LOCAL, NULL, 0,
                                      ucp_ep_fin_cb);
    if (UCS_PTR_IS_ERR(flush_req)) {
        return UCS_PTR_STATUS(flush_req);
    }

    ucp_request_free(flush_req);
    return UCS_OK;
}

static ucs_status_t
ucp_fin_msg_handler(void *arg, void *data, size_t length, unsigned flags)
{
    ucp_worker_h  worker = arg;
    ucp_fin_msg_t *msg   = data;
    ucp_ep_h      ep;

    ucs_assert(msg->ep_id != 0);
    if (msg->is_ptr) {
        ep = ucp_worker_get_ep_by_ptr(worker, msg->ep_id);
    } else {
        ep = ucp_worker_get_ep_by_uuid(worker, msg->ep_id);
        if (ep == NULL) {
            ucs_debug("got FIN msg for nonexisting ep, uuid: 0x%"PRIx64,
                      msg->ep_id);
            return UCS_OK;
        }
    }
    ep->flags |= UCP_EP_FLAG_FIN_MSG_RECVD;
    ucs_debug("ep %p, flags: %d: got FIN msg", ep, ep->flags);
    if (ucs_test_all_flags(ep->flags, UCP_EP_MASK_FIN_DONE)) {
        ucp_ep_disconnected(ep, 1);
        return UCS_OK;
    }

    ucp_fin_msg_send(ep);

    /* Notify user about disconnect */
    if (!(ep->flags & UCP_EP_FLAG_HIDDEN)) {
        if (ucp_ep_ext_gen(ep)->err_cb) {
            ucp_ep_ext_gen(ep)->err_cb(ucp_ep_ext_gen(ep)->user_data, ep,
                                       UCS_ERR_REMOTE_DISCONNECT);
        }
    }

    return UCS_OK;
}

static void
ucp_fin_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type, uint8_t id,
                 const void *data, size_t length, char *buffer, size_t max)
{
    const ucp_fin_msg_t *msg = data;
    char *p, *end;

    p   = buffer;
    end = buffer + max;
    snprintf(p, end - p, "FIN [%s 0x%"PRIx64"]",
             msg->is_ptr ? "ep_ptr" : "uuid", msg->ep_id);
}

UCP_DEFINE_AM(-1, UCP_AM_ID_FIN, ucp_fin_msg_handler,
              ucp_fin_msg_dump, UCT_CB_FLAG_SYNC);
