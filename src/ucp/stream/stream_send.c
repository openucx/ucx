/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_am.inl>
#include <ucp/stream/stream.h>
#include <ucp/dt/dt.h>
#include <ucp/dt/dt.inl>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_send_am_short(ucp_ep_t *ep, const void *buffer, size_t length)
{
    UCS_STATIC_ASSERT(sizeof(ep->worker->uuid) == sizeof(uint64_t));

    return uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_STREAM_DATA,
                           ucp_ep_dest_ep_ptr(ep), buffer, length);
}

static void ucp_stream_send_req_init(ucp_request_t* req, ucp_ep_h ep,
                                     const void* buffer, uintptr_t datatype,
                                     size_t count, uint16_t flags)
{
    req->flags             = flags;
    req->send.ep           = ep;
    req->send.buffer       = (void*)buffer;
    req->send.datatype     = datatype;
    req->send.lane         = ep->am_lane;
    ucp_request_send_state_init(req, datatype, count);
    req->send.length       = ucp_dt_length(req->send.datatype, count,
                                           req->send.buffer,
                                           &req->send.state.dt);
    ucp_memory_type_detect_mds(ep->worker->context, (void *)buffer,
                               req->send.length, &req->send.mem_type);
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.tag, sizeof(req->send.tag));
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_send_req(ucp_request_t *req, size_t count,
                    const ucp_ep_msg_config_t* msg_config,
                    ucp_send_callback_t cb, const ucp_proto_t *proto)
{
    size_t zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config,
                                                        count, SIZE_MAX);
    ssize_t max_short   = ucp_proto_get_short_max(req, msg_config);

    ucs_status_t status = ucp_request_send_start(req, max_short, zcopy_thresh,
                                                 SIZE_MAX, count, msg_config,
                                                 proto);
    if (status != UCS_OK) {
        return UCS_STATUS_PTR(status);
    }

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    status = ucp_request_send(req);
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
    ucs_status_ptr_t ret;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    ucs_trace_req("stream_send_nb buffer %p count %zu to %s cb %p flags %u",
                  buffer, count, ucp_ep_peer_name(ep), cb, flags);

    if (ucs_unlikely(flags != 0)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
        goto out;
    }

    status = ucp_ep_resolve_dest_ep_ptr(ep, ep->am_lane);
    if (status != UCS_OK) {
        ret = UCS_STATUS_PTR(status);
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

    ucp_stream_send_req_init(req, ep, buffer, datatype, count, flags);

    ret = ucp_stream_send_req(req, count, &ucp_ep_config(ep)->am, cb,
                              ucp_ep_config(ep)->stream.proto);

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
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

static size_t ucp_stream_pack_am_single_dt(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr = dest;
    ucp_request_t       *req = arg;
    size_t              length;

    hdr->ep_ptr = ucp_request_get_dest_ep_ptr(req);

    ucs_assert(req->send.state.dt.offset == 0);

    length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                         req->send.mem_type, hdr + 1, req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static ucs_status_t ucp_stream_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status;

    status = ucp_do_am_bcopy_single(self, UCP_AM_ID_STREAM_DATA,
                                    ucp_stream_pack_am_single_dt);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    }
    return status;
}

static size_t ucp_stream_pack_am_first_dt(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr = dest;
    ucp_request_t       *req = arg;
    size_t              length;

    hdr->ep_ptr = ucp_request_get_dest_ep_ptr(req);
    length      = ucp_ep_config(req->send.ep)->am.max_bcopy - sizeof(*hdr);

    ucs_assert(req->send.state.dt.offset == 0);
    ucs_assert(req->send.length > length);
    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                                      req->send.mem_type, hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

static size_t ucp_stream_pack_am_middle_dt(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr = dest;
    ucp_request_t       *req = arg;
    size_t              length;

    hdr->ep_ptr = ucp_request_get_dest_ep_ptr(req);
    length      = ucs_min(ucp_ep_config(req->send.ep)->am.max_bcopy - sizeof(*hdr),
                          req->send.length - req->send.state.dt.offset);
    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                                      req->send.mem_type, hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

static ucs_status_t ucp_stream_bcopy_multi(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_multi(self,
                                                UCP_AM_ID_STREAM_DATA,
                                                UCP_AM_ID_STREAM_DATA,
                                                sizeof(ucp_stream_am_hdr_t),
                                                ucp_stream_pack_am_first_dt,
                                                ucp_stream_pack_am_middle_dt, 0);
    if (status == UCS_OK) {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
        ucp_request_send_generic_dt_finish(req);
        ucp_request_complete_send(req, UCS_OK);
    } else if (status == UCP_STATUS_PENDING_SWITCH) {
        status = UCS_OK;
    }
    return status;
}

static ucs_status_t ucp_stream_eager_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t       *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_stream_am_hdr_t hdr;

    hdr.ep_ptr = ucp_request_get_dest_ep_ptr(req);
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_STREAM_DATA, &hdr,
                                  sizeof(hdr), ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_stream_eager_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t       *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_stream_am_hdr_t hdr;

    hdr.ep_ptr = ucp_request_get_dest_ep_ptr(req);
    return ucp_do_am_zcopy_multi(self,
                                 UCP_AM_ID_STREAM_DATA,
                                 UCP_AM_ID_STREAM_DATA,
                                 &hdr, sizeof(hdr), &hdr, sizeof(hdr),
                                 ucp_proto_am_zcopy_req_complete, 0);
}

const ucp_proto_t ucp_stream_am_proto = {
    .contig_short            = ucp_stream_contig_am_short,
    .bcopy_single            = ucp_stream_bcopy_single,
    .bcopy_multi             = ucp_stream_bcopy_multi,
    .zcopy_single            = ucp_stream_eager_zcopy_single,
    .zcopy_multi             = ucp_stream_eager_zcopy_multi,
    .zcopy_completion        = ucp_proto_am_zcopy_completion,
    .only_hdr_size           = sizeof(ucp_stream_am_hdr_t),
    .first_hdr_size          = sizeof(ucp_stream_am_hdr_t),
    .mid_hdr_size            = sizeof(ucp_stream_am_hdr_t)
};
