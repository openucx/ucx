/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto.h>
#include <ucp/dt/dt.h>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_send_am_short(ucp_ep_t *ep, const void *buffer, size_t length)
{
    UCS_STATIC_ASSERT(sizeof(ep->dest_uuid) == sizeof(uint64_t));

    return uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_STREAM_DATA,
                           ep->worker->uuid, buffer, length);
}

static void ucp_stream_send_req_init(ucp_request_t* req, ucp_ep_h ep,
                                     const void* buffer, uintptr_t datatype,
                                     uint16_t flags)
{
    req->flags             = flags;
    req->send.ep           = ep;
    req->send.buffer       = buffer;
    req->send.datatype     = datatype;
    req->send.reg_rsc      = UCP_NULL_RESOURCE;
    req->send.lane         = ep->am_lane;

    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.tag, sizeof(req->send.tag));
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.uct_comp, sizeof(req->send.uct_comp));
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.state.offset,
                                sizeof(req->send.state.offset));
}

static ucs_status_t ucp_stream_req_start(ucp_request_t *req, size_t count,
                                         ssize_t max_short,
                                         size_t *zcopy_thresh_arr,
                                         size_t rndv_rma_thresh,
                                         size_t rndv_am_thresh,
                                         const ucp_proto_t *proto)
{
    size_t zcopy_thresh = 0;
    size_t length;

    ucs_assertv_always(UCP_DT_IS_CONTIG(req->send.datatype),
                       "Only UCP_DATATYPE_CONTIG type is implemented");

    length                  = ucp_contig_dt_length(req->send.datatype, count);
    req->send.length        = length;
    req->send.uct_comp.func = NULL;

    ucs_trace_req("select request(%p) progress algorithm datatype=%lx buffer=%p "
                  " length=%zu max_short=%zd rndv_rma_thresh=%zu rndv_am_thresh=%zu "
                  "zcopy_thresh=%zu",
                  req, req->send.datatype, req->send.buffer, length, max_short,
                  rndv_rma_thresh, rndv_am_thresh, zcopy_thresh);

    if ((ssize_t)length <= max_short) {
        /* short */
        req->send.uct.func = proto->contig_short;
        UCS_PROFILE_REQUEST_EVENT(req, "start_contig_short", req->send.length);
    } else {
        ucs_error("Not implemented");
        return UCS_ERR_NOT_IMPLEMENTED;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_send_req(ucp_request_t *req, size_t count, ssize_t max_short,
                    size_t *zcopy_thresh, size_t rndv_rma_thresh, size_t rndv_am_thresh,
                    ucp_send_callback_t cb, const ucp_proto_t *proto)
{
    ucs_status_t status;

    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        status = ucp_stream_req_start(req, count, max_short, zcopy_thresh,
                                      rndv_rma_thresh, rndv_am_thresh, proto);
        if (status != UCS_OK) {
            return UCS_STATUS_PTR(status);
        }
        break;
    case UCP_DATATYPE_IOV:
    case UCP_DATATYPE_GENERIC:
        ucs_error("Not implemented datatype");
        return UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
    default:
        ucs_error("Invalid data type");
        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
    }

    ucp_request_send_stat(req);

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_start_send(req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucp_request_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucp_request_set_callback(req, send.cb, cb)
    ucs_trace_req("returning send request %p", req);
    return req + 1;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_send_nb,
                 (ep, buffer, count, datatype, cb, flags),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 uintptr_t datatype, ucp_send_callback_t cb, unsigned flags)
{
    ucp_request_t    *req;
    size_t           length;
    ucs_status_t     status;
    ucs_status_ptr_t ret = UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);

    UCP_THREAD_CS_ENTER_CONDITIONAL(&ep->worker->mt_lock);

    ucs_trace_req("send_nb buffer %p count %zu to %s cb %p flags %u",
                  buffer, count, ucp_ep_peer_name(ep), cb, flags);

    if (ucs_unlikely(flags != 0)) {
        goto out;
    }

    if (ucs_likely(UCP_DT_IS_CONTIG(datatype))) {
        length = ucp_contig_dt_length(datatype, count);
        if (ucs_likely((ssize_t)length <= ucp_ep_config(ep)->am.max_short)) {
            status = UCS_PROFILE_CALL(ucp_stream_send_am_short, ep, buffer,
                                      length);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                UCP_EP_STAT_TAG_OP(ep, EAGER);
                ret = UCS_STATUS_PTR(status); /* UCS_OK also goes here */
                goto out;
            }
        }
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely(req == NULL)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        goto out;
    }

    ucp_stream_send_req_init(req, ep, buffer, datatype, flags);

    ret = ucp_stream_send_req(req, count,
                              ucp_ep_config(ep)->am.max_short,
                              ucp_ep_config(ep)->am.zcopy_thresh,
                              SIZE_MAX, /* NOTE: disable, not implemented */
                              SIZE_MAX, /* NOTE: disable, not implemented */
                              cb, ucp_ep_config(ep)->stream.proto);

out:
    UCP_THREAD_CS_EXIT_CONDITIONAL(&ep->worker->mt_lock);
    return ret;
}

static ucs_status_t ucp_stream_contig_am_short(uct_pending_req_t *self)
{
    ucp_request_t  *req   = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t   status = ucp_stream_send_am_short(req->send.ep,
                                                     req->send.buffer,
                                                     req->send.length);
    if (ucs_likely(status == UCS_OK)) {
        ucp_request_complete_send(req, UCS_OK);
    }
    return status;
}

const ucp_proto_t ucp_stream_am_proto = {
    .contig_short            = ucp_stream_contig_am_short,
    .bcopy_single            = NULL,
    .bcopy_multi             = NULL,
    .zcopy_single            = NULL,
    .zcopy_multi             = NULL,
    .zcopy_completion        = NULL,
    .only_hdr_size           = 0, /* sizeof(ucp_stream_am_hdr_t) */
    .first_hdr_size          = 0, /* sizeof(ucp_stream_am_hdr_t) */
    .mid_hdr_size            = 0  /* sizeof(ucp_stream_am_hdr_t) */
};
