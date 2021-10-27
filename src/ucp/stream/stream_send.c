/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/proto/proto_am.inl>
#include <ucp/stream/stream.h>
#include <ucp/dt/dt.h>
#include <ucp/dt/dt.inl>


#define UCP_STREAM_SEND_CHECK_STATUS(_ep, _status, _ret, _done) \
    if (ucs_likely((_status) != UCS_ERR_NO_RESOURCE)) { \
        _ret = UCS_STATUS_PTR(_status); /* UCS_OK also goes here */ \
        _done; \
    }

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_send_am_short(ucp_ep_t *ep, const void *buffer, size_t length)
{
    UCS_STATIC_ASSERT(sizeof(ep->worker->uuid) == sizeof(uint64_t));

    return uct_ep_am_short(ucp_ep_get_am_uct_ep(ep), UCP_AM_ID_STREAM_DATA,
                           ucp_ep_remote_id(ep), buffer, length);
}

static void ucp_stream_send_req_init(ucp_request_t *req, ucp_ep_h ep,
                                     const void *buffer, uintptr_t datatype,
                                     size_t count, uint32_t flags,
                                     const ucp_request_param_t *param)
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
    req->send.mem_type     = ucp_request_get_memory_type(ep->worker->context,
                                                         (void*)buffer,
                                                         req->send.length, param);
    VALGRIND_MAKE_MEM_UNDEFINED(&req->send.msg_proto.tag,
                                sizeof(req->send.msg_proto.tag));
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_stream_send_req(ucp_request_t *req, size_t count,
                    const ucp_ep_msg_config_t* msg_config,
                    const ucp_request_param_t *param,
                    const ucp_request_send_proto_t *proto)
{
    size_t zcopy_thresh = ucp_proto_get_zcopy_threshold(req, msg_config,
                                                        count, SIZE_MAX);
    ssize_t max_short   = ucp_proto_get_short_max(req, msg_config);

    ucs_status_t status = ucp_request_send_start(req, max_short, zcopy_thresh,
                                                 SIZE_MAX, count, 0,
                                                 req->send.length, msg_config,
                                                 proto, param);
    if (status != UCS_OK) {
        return UCS_STATUS_PTR(status);
    }

    /*
     * Start the request.
     * If it is completed immediately, release the request and return the status.
     * Otherwise, return the request.
     */
    ucp_request_send(req);
    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        /* Coverity wrongly resolves completion callback function to
         * 'ucp_cm_client_connect_progress' */
        /* coverity[offset_free] */
        ucp_request_imm_cmpl_param(param, req, send);
    }

    ucp_request_set_send_callback_param(param, req, send);
    ucs_trace_req("returning send request %p", req);
    return req + 1;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_send_nb,
                 (ep, buffer, count, datatype, cb, flags),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 ucp_datatype_t datatype, ucp_send_callback_t cb,
                 unsigned flags)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE |
                        UCP_OP_ATTR_FIELD_CALLBACK |
                        UCP_OP_ATTR_FIELD_FLAGS,
        .cb.send      = (ucp_send_nbx_callback_t)cb,
        .flags        = flags,
        .datatype     = datatype
    };

    return ucp_stream_send_nbx(ep, buffer, count, &param);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_stream_send_nbx_am_short(ucp_ep_t *ep, const void *buffer, size_t length)
{
    if (ucs_likely((ssize_t)length <= ucp_ep_config(ep)->am.max_short)) {
        return UCS_PROFILE_CALL(ucp_stream_send_am_short, ep, buffer, length);
    }

    return UCS_ERR_NO_RESOURCE;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_stream_send_nbx,
                 (ep, buffer, count, param),
                 ucp_ep_h ep, const void *buffer, size_t count,
                 const ucp_request_param_t *param)
{
    ucp_datatype_t datatype;
    ucp_request_t *req;
    size_t length;
    ucs_status_t status;
    ucs_status_ptr_t ret;
    uint32_t attr_mask;
    uint32_t flags;

    UCP_CONTEXT_CHECK_FEATURE_FLAGS(ep->worker->context, UCP_FEATURE_STREAM,
                                    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM));
    UCP_REQUEST_CHECK_PARAM(param);

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);

    flags = ucp_request_param_flags(param);

    ucs_trace_req("stream_send_nbx buffer %p count %zu to %s cb %p flags %u",
                  buffer, count, ucp_ep_peer_name(ep),
                  param->op_attr_mask & UCP_OP_ATTR_FIELD_CALLBACK ?
                  param->cb.send : NULL, flags);

    if (ucs_unlikely(flags != 0)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NOT_IMPLEMENTED);
        goto out;
    }

    status = ucp_ep_resolve_remote_id(ep, ep->am_lane);
    if (status != UCS_OK) {
        ret = UCS_STATUS_PTR(status);
        goto out;
    }

    if (ucs_memtype_cache_is_empty()) {
        attr_mask = param->op_attr_mask &
                    (UCP_OP_ATTR_FIELD_DATATYPE | UCP_OP_ATTR_FLAG_NO_IMM_CMPL);
        if (ucs_likely(attr_mask == 0)) {
            status = ucp_stream_send_nbx_am_short(ep, buffer, count);
            UCP_STREAM_SEND_CHECK_STATUS(ep, status, ret, goto out);
            datatype = ucp_dt_make_contig(1);
        } else if (attr_mask == UCP_OP_ATTR_FIELD_DATATYPE) {
            datatype = param->datatype;
            if (UCP_DT_IS_CONTIG(datatype)) {
                length = ucp_contig_dt_length(datatype, count);
                status = ucp_stream_send_nbx_am_short(ep, buffer, length);
                UCP_STREAM_SEND_CHECK_STATUS(ep, status, ret, goto out);
            }
        } else {
            datatype = ucp_dt_make_contig(1);
        }
    } else {
        datatype = ucp_request_param_datatype(param);
    }

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        ret = UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
        goto out;
    }

    req = ucp_request_get_param(ep->worker, param,
                                {
                                    ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                    goto out;
                                });

    ucp_stream_send_req_init(req, ep, buffer, datatype, count, flags, param);

    ret = ucp_stream_send_req(req, count, &ucp_ep_config(ep)->am, param,
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
    return ucp_am_short_handle_status_from_pending(req, status);
}

static size_t ucp_stream_pack_am_single_dt(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr = dest;
    ucp_request_t       *req = arg;
    size_t              length;

    hdr->ep_id = ucp_send_request_get_ep_remote_id(req);

    ucs_assert(req->send.state.dt.offset == 0);

    length = ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                         req->send.mem_type, hdr + 1, req->send.buffer,
                         &req->send.state.dt, req->send.length);
    ucs_assert(length == req->send.length);
    return sizeof(*hdr) + length;
}

static ucs_status_t ucp_stream_bcopy_single(uct_pending_req_t *self)
{
    ucs_status_t status = ucp_do_am_bcopy_single(self, UCP_AM_ID_STREAM_DATA,
                                                 ucp_stream_pack_am_single_dt);

    return ucp_am_bcopy_handle_status_from_pending(self, 0, 0, status);
}

static size_t ucp_stream_pack_am_first_dt(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr = dest;
    ucp_request_t       *req = arg;
    size_t              length;

    hdr->ep_id = ucp_send_request_get_ep_remote_id(req);
    length     = ucs_min(ucp_ep_config(req->send.ep)->am.max_bcopy - sizeof(*hdr),
                         req->send.length);

    ucs_assert(req->send.state.dt.offset == 0);
    return sizeof(*hdr) + ucp_dt_pack(req->send.ep->worker, req->send.datatype,
                                      req->send.mem_type, hdr + 1, req->send.buffer,
                                      &req->send.state.dt, length);
}

static size_t ucp_stream_pack_am_middle_dt(void *dest, void *arg)
{
    ucp_stream_am_hdr_t *hdr = dest;
    ucp_request_t       *req = arg;
    size_t              length;

    hdr->ep_id = ucp_send_request_get_ep_remote_id(req);
    length     = ucs_min(ucp_ep_config(req->send.ep)->am.max_bcopy - sizeof(*hdr),
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
                                                ucp_stream_pack_am_first_dt,
                                                ucp_stream_pack_am_middle_dt, 0);

    return ucp_am_bcopy_handle_status_from_pending(self, 1, 0, status);
}

static ucs_status_t ucp_stream_eager_zcopy_single(uct_pending_req_t *self)
{
    ucp_request_t       *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_stream_am_hdr_t hdr;

    hdr.ep_id = ucp_send_request_get_ep_remote_id(req);
    return ucp_do_am_zcopy_single(self, UCP_AM_ID_STREAM_DATA, &hdr,
                                  sizeof(hdr), NULL, 0ul,
                                  ucp_proto_am_zcopy_req_complete);
}

static ucs_status_t ucp_stream_eager_zcopy_multi(uct_pending_req_t *self)
{
    ucp_request_t       *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_stream_am_hdr_t hdr;

    hdr.ep_id = ucp_send_request_get_ep_remote_id(req);
    return ucp_do_am_zcopy_multi(self, UCP_AM_ID_STREAM_DATA,
                                 UCP_AM_ID_STREAM_DATA, &hdr, sizeof(hdr), &hdr,
                                 sizeof(hdr), NULL, 0ul, 0ul,
                                 ucp_proto_am_zcopy_req_complete, 0);
}

const ucp_request_send_proto_t ucp_stream_am_proto = {
    .contig_short            = ucp_stream_contig_am_short,
    .bcopy_single            = ucp_stream_bcopy_single,
    .bcopy_multi             = ucp_stream_bcopy_multi,
    .zcopy_single            = ucp_stream_eager_zcopy_single,
    .zcopy_multi             = ucp_stream_eager_zcopy_multi,
    .zcopy_completion        = ucp_proto_am_zcopy_completion,
    .only_hdr_size           = sizeof(ucp_stream_am_hdr_t)
};
