/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "rma.h"
#include "rma.inl"

#include <ucs/profile/profile.h>
#include <ucp/core/ucp_request.inl>


static size_t ucp_rma_sw_put_pack_cb(void *dest, void *arg)
{
    ucp_request_t *req  = arg;
    ucp_ep_t *ep        = req->send.ep;
    ucp_put_hdr_t *puth = dest;
    size_t length;

    puth->address = req->send.rma.remote_addr;
    puth->ep_ptr  = ucp_ep_dest_ep_ptr(ep);

    ucs_assert(puth->ep_ptr != 0);

    length = ucs_min(req->send.length,
                     ucp_ep_config(ep)->am.max_bcopy - sizeof(*puth));
    memcpy(puth + 1, req->send.buffer, length);

    return sizeof(*puth) + length;
}

static ucs_status_t ucp_rma_sw_progress_put(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;
    ucs_status_t status;

    ucs_assert(req->send.lane == ucp_ep_get_am_lane(ep));

    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_PUT,
                                 ucp_rma_sw_put_pack_cb, req, 0);
    if (packed_len > 0) {
        status = UCS_OK;
        ucp_ep_rma_remote_request_sent(ep);
    } else {
        status = (ucs_status_t)packed_len;
    }

    return ucp_rma_request_advance(req, packed_len - sizeof(ucp_put_hdr_t),
                                   status);
}

static size_t ucp_rma_sw_get_req_pack_cb(void *dest, void *arg)
{
    ucp_request_t *req         = arg;
    ucp_get_req_hdr_t *getreqh = dest;

    getreqh->address      = req->send.rma.remote_addr;
    getreqh->length       = req->send.length;
    getreqh->req.ep_ptr   = ucp_ep_dest_ep_ptr(req->send.ep);
    getreqh->req.reqptr  = (uintptr_t)req;
    ucs_assert(getreqh->req.ep_ptr != 0);

    return sizeof(*getreqh);
}

static ucs_status_t ucp_rma_sw_progress_get(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ucs_status_t status;
    ssize_t packed_len;

    ucs_assert(req->send.lane == ucp_ep_get_am_lane(ep));

    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_GET_REQ,
                                 ucp_rma_sw_get_req_pack_cb, req, 0);
    if (packed_len < 0) {
        status = (ucs_status_t)packed_len;
        if (status != UCS_ERR_NO_RESOURCE) {
            ucp_request_complete_send(req, status);
        }
        return status;
    }

    /* get request packet sent, complete the request object when all data arrives */
    ucs_assert(packed_len == sizeof(ucp_get_req_hdr_t));
    ucp_ep_rma_remote_request_sent(ep);
    return UCS_OK;
}

ucp_rma_proto_t ucp_rma_sw_proto = {
    .name         = "sw_rma",
    .progress_put = ucp_rma_sw_progress_put,
    .progress_get = ucp_rma_sw_progress_get
};

static size_t ucp_rma_sw_pack_rma_ack(void *dest, void *arg)
{
    ucp_cmpl_hdr_t *hdr = dest;
    ucp_request_t *req = arg;

    hdr->ep_ptr = ucp_ep_dest_ep_ptr(req->send.ep);
    return sizeof(*hdr);
}

static ucs_status_t ucp_progress_rma_cmpl(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len;

    req->send.lane = ucp_ep_get_am_lane(ep);

    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_CMPL,
                                 ucp_rma_sw_pack_rma_ack, req, 0);
    if (packed_len < 0) {
        return (ucs_status_t)packed_len;
    }

    ucs_assert(packed_len == sizeof(ucp_cmpl_hdr_t));
    ucp_request_put(req);
    return UCS_OK;
}

void ucp_rma_sw_send_cmpl(ucp_ep_h ep)
{
    ucp_request_t *req;

    req = ucp_request_get(ep->worker);
    ucs_assert(req != NULL);

    req->send.ep       = ep;
    req->send.uct.func = ucp_progress_rma_cmpl;
    ucp_request_send(req);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_put_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_put_hdr_t *puth = data;
    ucp_worker_h worker = arg;

    memcpy((void*)puth->address, puth + 1, length - sizeof(*puth));
    ucp_rma_sw_send_cmpl(ucp_worker_get_ep_by_ptr(worker, puth->ep_ptr));
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rma_cmpl_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_cmpl_hdr_t *putackh = data;
    ucp_worker_h worker     = arg;
    ucp_ep_h ep             = ucp_worker_get_ep_by_ptr(worker, putackh->ep_ptr);

    ucp_ep_rma_remote_request_completed(ep);
    return UCS_OK;
}

static size_t ucp_rma_sw_pack_get_reply(void *dest, void *arg)
{
    ucp_rma_rep_hdr_t *hdr = dest;
    ucp_request_t *req    = arg;
    size_t length;

    length   = ucs_min(req->send.length,
                       ucp_ep_config(req->send.ep)->am.max_bcopy - sizeof(*hdr));
    hdr->req = req->send.get_reply.req;
    memcpy(hdr + 1, req->send.buffer, length);

    return sizeof(*hdr) + length;
}

static ucs_status_t ucp_progress_get_reply(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep       = req->send.ep;
    ssize_t packed_len, payload_len;

    req->send.lane = ucp_ep_get_am_lane(ep);
    packed_len = uct_ep_am_bcopy(ep->uct_eps[req->send.lane], UCP_AM_ID_GET_REP,
                                 ucp_rma_sw_pack_get_reply, req, 0);
    if (packed_len < 0) {
        return (ucs_status_t)packed_len;
    }

    payload_len = packed_len - sizeof(ucp_rma_rep_hdr_t);
    ucs_assert(payload_len >= 0);

    req->send.buffer += payload_len;
    req->send.length -= payload_len;

    if (req->send.length == 0) {
        ucp_request_put(req);
        return UCS_OK;
    } else {
        return UCS_INPROGRESS;
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_get_req_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_get_req_hdr_t *getreqh = data;
    ucp_worker_h worker        = arg;
    ucp_ep_h ep                = ucp_worker_get_ep_by_ptr(worker,
                                                          getreqh->req.ep_ptr);
    ucp_request_t *req;

    req = ucp_request_get(worker);
    ucs_assert(req != NULL);

    req->send.ep            = ep;
    req->send.buffer        = (void*)getreqh->address;
    req->send.length        = getreqh->length;
    req->send.get_reply.req = getreqh->req.reqptr;
    req->send.uct.func      = ucp_progress_get_reply;

    ucp_request_send(req);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_get_rep_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_rma_rep_hdr_t *getreph = data;
    size_t frag_length         = length - sizeof(*getreph);
    ucp_request_t *req         = (ucp_request_t*)getreph->req;
    ucp_ep_h ep                = req->send.ep;

    memcpy(req->send.buffer, getreph + 1, frag_length);

    /* complete get request on last fragment of the reply */
    if (ucp_rma_request_advance(req, frag_length, UCS_OK) == UCS_OK) {
        ucp_ep_rma_remote_request_completed(ep);
    }

    return UCS_OK;
}

static void ucp_rma_sw_dump_packet(ucp_worker_h worker, uct_am_trace_type_t type,
                                   uint8_t id, const void *data, size_t length,
                                   char *buffer, size_t max)
{
    const ucp_get_req_hdr_t *geth;
    const ucp_rma_rep_hdr_t *reph;
    const ucp_cmpl_hdr_t *cmplh;
    const ucp_put_hdr_t *puth;
    size_t header_len;
    char *p;

    switch (id) {
    case UCP_AM_ID_PUT:
        puth = data;
        snprintf(buffer, max, "PUT [addr 0x%lx ep_ptr 0x%lx]", puth->address,
                 puth->ep_ptr);
        header_len = sizeof(*puth);
        break;
    case UCP_AM_ID_GET_REQ:
        geth = data;
        snprintf(buffer, max, "GET_REQ [addr 0x%lx len %zu reqptr 0x%lx ep 0x%lx]",
                 geth->address, geth->length, geth->req.reqptr, geth->req.ep_ptr);
        return;
    case UCP_AM_ID_GET_REP:
        reph = data;
        snprintf(buffer, max, "GET_REP [reqptr 0x%lx]", reph->req);
        header_len = sizeof(*reph);
        break;
    case UCP_AM_ID_CMPL:
        cmplh = data;
        snprintf(buffer, max, "CMPL [ep_ptr 0x%lx]", cmplh->ep_ptr);
        return;
    default:
        return;
    }

    p = buffer + strlen(buffer);
    ucp_dump_payload(worker->context, p, buffer + max - p, data + header_len,
                     length - header_len);
}

UCP_DEFINE_AM(UCP_FEATURE_RMA, UCP_AM_ID_PUT, ucp_put_handler,
              ucp_rma_sw_dump_packet, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_RMA, UCP_AM_ID_GET_REQ, ucp_get_req_handler,
              ucp_rma_sw_dump_packet, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_RMA, UCP_AM_ID_GET_REP, ucp_get_rep_handler,
              ucp_rma_sw_dump_packet, UCT_CB_FLAG_SYNC);
UCP_DEFINE_AM(UCP_FEATURE_RMA|UCP_FEATURE_AMO, UCP_AM_ID_CMPL,
              ucp_rma_cmpl_handler, ucp_rma_sw_dump_packet, UCT_CB_FLAG_SYNC);

UCP_DEFINE_AM_PROXY(UCP_AM_ID_PUT);
UCP_DEFINE_AM_PROXY(UCP_AM_ID_GET_REQ);
