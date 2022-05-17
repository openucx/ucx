/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucs/profile/profile.h>
#include <ucp/core/ucp_request.inl>

#include <ucp/proto/proto_common.inl>


static size_t ucp_rma_sw_put_pack_cb(void *dest, void *arg)
{
    ucp_request_t *req  = arg;
    ucp_ep_t *ep        = req->send.ep;
    ucp_put_hdr_t *puth = dest;
    size_t length;

    puth->address  = req->send.rma.remote_addr;
    puth->ep_id    = ucp_ep_remote_id(ep);
    puth->mem_type = UCS_MEMORY_TYPE_HOST;

    ucs_assert(puth->ep_id != UCS_PTR_MAP_KEY_INVALID);

    length = ucs_min(req->send.length,
                     ucp_ep_config(ep)->am.max_bcopy - sizeof(*puth));
    memcpy(puth + 1, req->send.buffer, length);

    return sizeof(*puth) + length;
}

static ucs_status_t ucp_rma_sw_progress_put(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ssize_t packed_len = 0;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(req->send.ep);
    status         = ucp_rma_sw_do_am_bcopy(req, UCP_AM_ID_PUT, req->send.lane,
                                            ucp_rma_sw_put_pack_cb, req,
                                            &packed_len);
    return ucp_rma_request_advance(req, packed_len - sizeof(ucp_put_hdr_t),
                                   status, UCS_PTR_MAP_KEY_INVALID);
}

static size_t ucp_rma_sw_get_req_pack_cb(void *dest, void *arg)
{
    ucp_request_t *req         = arg;
    ucp_get_req_hdr_t *getreqh = dest;

    getreqh->address    = req->send.rma.remote_addr;
    getreqh->length     = req->send.length;
    getreqh->req.ep_id  = ucp_send_request_get_ep_remote_id(req);
    getreqh->mem_type   = req->send.rma.rkey->mem_type;
    getreqh->req.req_id = ucp_send_request_get_id(req);
    ucs_assert(getreqh->req.ep_id != UCS_PTR_MAP_KEY_INVALID);

    return sizeof(*getreqh);
}

static ucs_status_t ucp_rma_sw_progress_get(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ssize_t packed_len = 0;
    ucs_status_t status;

    req->send.lane = ucp_ep_get_am_lane(req->send.ep);
    ucp_send_request_id_alloc(req);

    status = ucp_rma_sw_do_am_bcopy(req, UCP_AM_ID_GET_REQ, req->send.lane,
                                    ucp_rma_sw_get_req_pack_cb, req,
                                    &packed_len);
    if (status != UCS_OK) {
        ucp_send_request_id_release(req);
        if (ucs_unlikely(status != UCS_ERR_NO_RESOURCE)) {
            /* completed with error */
            ucp_request_complete_send(req, status);
            return UCS_OK;
        }
    }

    /* If completed with UCS_OK, it means that get request packet sent,
     * complete the request object when all data arrives */
    ucs_assert((status != UCS_OK) || (packed_len == sizeof(ucp_get_req_hdr_t)));
    return status;
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

    hdr->ep_id = ucp_send_request_get_ep_remote_id(req);
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
    if (req == NULL) {
        ucs_error("failed to allocate put completion");
        return;
    }

    ucp_request_send_state_init(req, ucp_dt_make_contig(1),
                                sizeof(ucp_cmpl_hdr_t));

    req->flags         = 0;
    req->send.ep       = ep;
    req->send.uct.func = ucp_progress_rma_cmpl;
    ucp_request_send(req);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_put_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_put_hdr_t *puth = data;
    ucp_worker_h worker = arg;
    ucp_ep_h ep;

    /* allow getting closed EP to be used for sending a completion to enable flush
     * on a peer
     */
    UCP_WORKER_GET_EP_BY_ID(&ep, worker, puth->ep_id, return UCS_OK,
                            "SW PUT request");
    ucp_dt_contig_unpack(worker, (void*)puth->address, puth + 1,
                         length - sizeof(*puth), puth->mem_type);
    ucp_rma_sw_send_cmpl(ep);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rma_cmpl_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_cmpl_hdr_t *putackh = data;
    ucp_worker_h worker     = arg;
    ucp_ep_h ep;

    /* allow getting closed EP to be used for handling a completion to enable flush
     * on a peer
     */
    UCP_WORKER_GET_EP_BY_ID(&ep, worker, putackh->ep_id, return UCS_OK,
                            "SW RMA completion");
    ucp_ep_rma_remote_request_completed(ep);
    return UCS_OK;
}

static size_t ucp_rma_sw_pack_get_reply(void *dest, void *arg)
{
    ucp_request_data_hdr_t *hdr = dest;
    ucp_request_t *req          = arg;
    size_t length;

    length      = ucs_min(req->send.length,
                          ucp_ep_config(req->send.ep)->am.max_bcopy -
                          sizeof(*hdr));
    hdr->req_id = req->send.get_reply.remote_req_id;
    hdr->offset = req->send.state.dt_iter.offset;
    ucp_dt_contig_pack(req->send.ep->worker, hdr + 1,
                       (char*)req->send.buffer + hdr->offset, length,
                       req->send.mem_type);

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

    payload_len = packed_len - sizeof(ucp_request_data_hdr_t);
    ucs_assert(payload_len >= 0);

    req->send.length               -= payload_len;
    req->send.state.dt_iter.offset += payload_len;

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
    ucp_ep_h ep;
    ucp_request_t *req;

    /* allow getting closed EP to be used for sending a GET operation data to enable
     * flush on a peer
     */
    UCP_WORKER_GET_EP_BY_ID(&ep, worker, getreqh->req.ep_id, return UCS_OK,
                            "SW GET request");
    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_error("failed to allocate get reply");
        return UCS_OK;
    }

    ucp_request_send_state_init(req, ucp_dt_make_contig(1), length);

    req->flags                        = 0;
    req->send.ep                      = ep;
    req->send.buffer                  = (void*)getreqh->address;
    req->send.length                  = getreqh->length;
    req->send.get_reply.remote_req_id = getreqh->req.req_id;
    req->send.state.dt_iter.offset    = 0;
    req->send.uct.func                = ucp_progress_get_reply;
    if (ep->worker->context->config.ext.proto_enable) {
        req->send.mem_type = getreqh->mem_type;
    } else {
        req->send.mem_type = UCS_MEMORY_TYPE_HOST;
    }

    ucp_request_send(req);
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_get_rep_handler, (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucp_worker_h worker             = arg;
    ucp_request_data_hdr_t *getreph = data;
    size_t frag_length              = length - sizeof(*getreph);
    ucp_request_t *req;
    ucp_ep_h ep;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, getreph->req_id, 0, return UCS_OK,
                               "GET reply data %p", getreph);
    ep = req->send.ep;
    if (ep->worker->context->config.ext.proto_enable) {
        ucp_datatype_iter_unpack(&req->send.state.dt_iter, worker, frag_length,
                                 getreph->offset, getreph + 1);
        req->send.state.completed_size += frag_length;
        if (req->send.state.completed_size == req->send.length) {
            ucp_send_request_id_release(req);
            ucp_proto_request_bcopy_complete_success(req);
            ucp_ep_rma_remote_request_completed(ep);
        }
    } else {
        memcpy(req->send.buffer, getreph + 1, frag_length);

        /* complete get request on last fragment of the reply */
        if (ucp_rma_request_advance(req, frag_length, UCS_OK,
                                    getreph->req_id) == UCS_OK) {
            ucp_ep_rma_remote_request_completed(ep);
        }
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
        snprintf(buffer, max, "PUT [addr 0x%"PRIx64" ep_id 0x%"PRIx64" %s]",
                 puth->address, puth->ep_id,
                 ucs_memory_type_names[puth->mem_type]);
        header_len = sizeof(*puth);
        break;
    case UCP_AM_ID_GET_REQ:
        geth = data;
        snprintf(buffer, max, "GET_REQ [addr 0x%"PRIx64" len %"PRIu64
                 " req_id 0x%"PRIx64" ep_id 0x%"PRIx64" %s]", geth->address,
                 geth->length, geth->req.req_id, geth->req.ep_id,
                 ucs_memory_type_names[geth->mem_type]);
        return;
    case UCP_AM_ID_GET_REP:
        reph = data;
        snprintf(buffer, max, "GET_REP [req_id 0x%"PRIx64"]", reph->req_id);
        header_len = sizeof(*reph);
        break;
    case UCP_AM_ID_CMPL:
        cmplh = data;
        snprintf(buffer, max, "CMPL [ep_id 0x%"PRIx64"]", cmplh->ep_id);
        return;
    default:
        return;
    }

    p = buffer + strlen(buffer);
    ucp_dump_payload(worker->context, p, buffer + max - p,
                     UCS_PTR_BYTE_OFFSET(data, header_len),
                     length - header_len);
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_RMA, UCP_AM_ID_PUT, ucp_put_handler,
                         ucp_rma_sw_dump_packet, 0);
UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_RMA, UCP_AM_ID_GET_REQ, ucp_get_req_handler,
                         ucp_rma_sw_dump_packet, 0);
UCP_DEFINE_AM(UCP_FEATURE_RMA, UCP_AM_ID_GET_REP, ucp_get_rep_handler,
              ucp_rma_sw_dump_packet, 0);
UCP_DEFINE_AM(UCP_FEATURE_RMA|UCP_FEATURE_AMO, UCP_AM_ID_CMPL,
              ucp_rma_cmpl_handler, ucp_rma_sw_dump_packet, 0);
