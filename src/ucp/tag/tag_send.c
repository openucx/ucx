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
#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/instrument.h>
#include <string.h>

static ucs_status_t ucp_tag_req_start_contig(ucp_request_t *req, size_t count,
                                             ssize_t max_short, size_t zcopy_thresh,
                                             size_t rndv_thresh,
                                             const ucp_proto_t *proto)
{
    ucp_ep_config_t *config = ucp_ep_config(req->send.ep);
    size_t only_hdr_size = proto->only_hdr_size;
    ucs_status_t status;
    size_t max_zcopy;
    ssize_t length;

    length           = ucp_contig_dt_length(req->send.datatype, count);
    req->send.length = length;

    if (length <= max_short) {
        /* short */
        req->send.uct.func = proto->contig_short;
    } else if (length >= rndv_thresh) {
        /* rendezvous */
        status = ucp_tag_send_start_rndv(req);
        if (status != UCS_OK) {
            return status;
        }
    } else if (length < zcopy_thresh) {
        /* bcopy */
        if (req->send.length <= config->max_am_bcopy - only_hdr_size) {
            req->send.uct.func = proto->bcopy_single;
        } else {
            req->send.uct.func = proto->bcopy_multi;
        }
    } else {
        /* eager zcopy */
        status = ucp_request_send_buffer_reg(req, ucp_ep_get_am_lane(req->send.ep));
        if (status != UCS_OK) {
            return status;
        }

        req->send.uct_comp.func = proto->contig_zcopy_completion;

        max_zcopy = config->max_am_zcopy;
        if (req->send.length <= max_zcopy - only_hdr_size) {
            req->send.uct_comp.count = 1;
            req->send.uct.func = proto->contig_zcopy_single;
        } else {
            /* calculate number of zcopy fragments */
            req->send.uct_comp.count = 1 +
                    (length + proto->first_hdr_size - proto->mid_hdr_size - 1) /
                    (max_zcopy - proto->mid_hdr_size);
            req->send.uct.func = proto->contig_zcopy_multi;
        }
    }
    return UCS_OK;
}

static ucs_status_t ucp_tag_req_start_iov(ucp_request_t *req, size_t count,
                                          ssize_t max_short, size_t zcopy_thresh,
                                          size_t rndv_thresh,
                                          const ucp_proto_t *proto)
{
    ucp_ep_config_t *config = ucp_ep_config(req->send.ep);
    size_t only_hdr_size = proto->only_hdr_size;

    req->send.length = ucp_dt_iov_length(req->send.buffer, count);
    req->send.state.dt.iov.iovcnt_offset = 0;
    req->send.state.dt.iov.iov_offset    = 0;

    /* bcopy */
    if (req->send.length <= config->max_am_bcopy - only_hdr_size) {
        req->send.uct.func = proto->bcopy_single;
    } else {
        req->send.uct.func = proto->bcopy_multi;
    }
    return UCS_OK;
}

static void ucp_tag_req_start_generic(ucp_request_t *req, size_t count,
                                      size_t rndv_thresh,
                                      const ucp_proto_t *proto)
{
    ucp_ep_config_t *config = ucp_ep_config(req->send.ep);
    ucp_dt_generic_t *dt_gen;
    size_t length;
    void *state;

    dt_gen = ucp_dt_generic(req->send.datatype);
    state = dt_gen->ops.start_pack(dt_gen->context, req->send.buffer, count);

    req->send.state.dt.generic.state = state;
    req->send.length = length = dt_gen->ops.packed_size(state);

    if (length <= config->max_am_bcopy - proto->only_hdr_size) {
        req->send.uct.func = proto->bcopy_single;
    } else {
        req->send.uct.func = proto->bcopy_multi;
    }
}

static inline ucs_status_ptr_t
ucp_tag_send_req(ucp_request_t *req, size_t count, ssize_t max_short,
                 size_t zcopy_thresh, size_t rndv_thresh, ucp_send_callback_t cb,
                 const ucp_proto_t *proto)
{
    ucs_status_t status;

    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        status = ucp_tag_req_start_contig(req, count, max_short, zcopy_thresh,
                                          rndv_thresh, proto);
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
        break;

    case UCP_DATATYPE_IOV:
        status = ucp_tag_req_start_iov(req, count, max_short, zcopy_thresh,
                                       rndv_thresh, proto);
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
        break;

    case UCP_DATATYPE_GENERIC:
        ucp_tag_req_start_generic(req, count, rndv_thresh, proto);
        break;

    default:
        ucs_error("Invalid data type");
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_start_send(req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucs_mpool_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucs_trace_req("returning send request %p", req);
    req->send.cb = cb;
    return req + 1;
}

/* Will be called if request is completed internally before returned to user */
static void ucp_tag_stub_send_completion(void *request, ucs_status_t status)
{
    ucs_assertv(status == UCS_OK, "status=%s", ucs_status_string(status));
}

static void ucp_tag_send_req_init(ucp_request_t* req, ucp_ep_h ep,
                                  const void* buffer, uintptr_t datatype,
                                  ucp_tag_t tag)
{
    req->flags             = 0;
    req->send.ep           = ep;
    req->send.buffer       = buffer;
    req->send.datatype     = datatype;
    req->send.cb           = ucp_tag_stub_send_completion;
    req->send.tag          = tag;
    req->send.state.offset = 0;
#if ENABLE_ASSERT
    req->send.lane         = UCP_NULL_LANE;
#endif
}

ucs_status_ptr_t ucp_tag_send_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                 uintptr_t datatype, ucp_tag_t tag,
                                 ucp_send_callback_t cb)
{
    ucs_status_t status;
    ucp_request_t *req;
    size_t length;

    ucs_trace_req("send_nb buffer %p count %zu tag %"PRIx64" to %s cb %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), cb);

    if (ucs_likely((datatype & UCP_DATATYPE_CLASS_MASK) == UCP_DATATYPE_CONTIG)) {
        length = ucp_contig_dt_length(datatype, count);
        UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_UCP_TX,
                              "ucp_tag_send_nb (eager - start)",
                              buffer, length);
        if (ucs_likely(length <= ucp_ep_config(ep)->max_eager_short)) {
            status = ucp_tag_send_eager_short(ep, tag, buffer, length);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_UCP_TX,
                                      "ucp_tag_send_nb (eager - finish)",
                                      buffer, length);
                return UCS_STATUS_PTR(status); /* UCS_OK also goes here */
            }
        }
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_UCP_TX, "ucp_tag_send_nb", req,
                          ucp_dt_length(datatype, count, buffer, &req->send.state));

    ucp_tag_send_req_init(req, ep, buffer, datatype, tag);

    return ucp_tag_send_req(req, count,
                            ucp_ep_config(ep)->max_eager_short,
                            ucp_ep_config(ep)->zcopy_thresh,
                            ucp_ep_config(ep)->rndv_thresh,
                            cb, &ucp_tag_eager_proto);
}

ucs_status_ptr_t ucp_tag_send_sync_nb(ucp_ep_h ep, const void *buffer, size_t count,
                                      ucp_datatype_t datatype, ucp_tag_t tag,
                                      ucp_send_callback_t cb)
{
    ucp_request_t *req;

    ucs_trace_req("send_sync_nb buffer %p count %zu tag %"PRIx64" to %s cb %p",
                  buffer, count, tag, ucp_ep_peer_name(ep), cb);

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    UCS_INSTRUMENT_RECORD(UCS_INSTRUMENT_TYPE_UCP_TX, "ucp_tag_send_sync_nb", req,
                          ucp_dt_length(datatype, count, buffer, &req->send.state));

    /* Remote side needs to send reply, so have it connect to us */
    ucp_ep_connect_remote(ep);

    ucp_tag_send_req_init(req, ep, buffer, datatype, tag);

    return ucp_tag_send_req(req, count,
                            -1, /* disable short method */
                            ucp_ep_config(ep)->sync_zcopy_thresh,
                            ucp_ep_config(ep)->sync_rndv_thresh,
                            cb, &ucp_tag_eager_sync_proto);
}

void ucp_tag_eager_sync_send_ack(ucp_worker_h worker, uint64_t sender_uuid,
                                 uintptr_t remote_request)
{
    ucp_request_t *req;

    ucs_trace_req("send_sync_ack sender_uuid %"PRIx64" remote_request 0x%lx",
                  sender_uuid, remote_request);

    req = ucp_worker_allocate_reply(worker, sender_uuid);
    req->send.uct.func             = ucp_proto_progress_am_bcopy_single;
    req->send.proto.am_id          = UCP_AM_ID_EAGER_SYNC_ACK;
    req->send.proto.remote_request = remote_request;
    req->send.proto.status         = UCS_OK;
    ucp_request_start_send(req);
}

/**
 * Common function to copy data with specified offset from particular datatype
 * to the descriptor
 */
size_t ucp_tag_pack_dt_copy(void *dest, const void *src, ucp_frag_state_t *state,
                                size_t length, ucp_datatype_t datatype)
{
    ucp_dt_generic_t *dt;
    size_t result_len = 0;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        memcpy(dest, src + state->offset, length);
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
        ucp_dt_iov_gather(dest, src, length, &state->dt.iov.iov_offset,
                          &state->dt.iov.iovcnt_offset);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt = ucp_dt_generic(datatype);
        result_len = dt->ops.pack(state->dt.generic.state, state->offset, dest,
                                  length);
        break;

    default:
        ucs_error("Invalid data type");
    }

    state->offset += result_len;
    return result_len;
}

