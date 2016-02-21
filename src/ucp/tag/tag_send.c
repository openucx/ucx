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
#include <ucp/dt/dt_generic.h>
#include <ucs/datastruct/mpool.inl>
#include <string.h>


static ucs_status_t ucp_tag_req_start_contig(ucp_request_t *req, size_t count,
                                             size_t max_short, size_t zcopy_thresh,
                                             size_t rndv_thresh,
                                             const ucp_proto_t *proto)
{
    ucp_ep_config_t *config = ucp_ep_config(req->send.ep);
    size_t only_hdr_size = proto->only_hdr_size;
    ucs_status_t status;
    size_t max_zcopy;
    size_t length;

    length           = ucp_contig_dt_length(req->send.datatype, count);
    req->send.length = length;

    if (length <= max_short) {
        /* short */
        req->send.uct.func = proto->contig_short;
    } else if (length >= rndv_thresh) {
        /* rendezvous */
        ucp_tag_send_start_rndv(req);
    } else if (length < zcopy_thresh) {
        /* bcopy */
        if (req->send.length <= config->max_am_bcopy - only_hdr_size) {
            req->send.uct.func = proto->contig_bcopy_single;
        } else {
            req->send.uct.func = proto->contig_bcopy_multi;
        }
    } else {
        /* zcopy */
        status = ucp_request_send_buffer_reg(req);
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

static void ucp_tag_req_start_generic(ucp_request_t *req, size_t count,
                                      size_t rndv_thresh,
                                      const ucp_proto_t *progress)
{
    ucp_ep_config_t *config = ucp_ep_config(req->send.ep);
    ucp_dt_generic_t *dt_gen;
    size_t length;
    void *state;

    dt_gen = ucp_dt_generic(req->send.datatype);
    state = dt_gen->ops.start_pack(dt_gen->context, req->send.buffer, count);

    req->send.state.dt.generic.state = state;
    req->send.length = length = dt_gen->ops.packed_size(state);

    if (length >= rndv_thresh) {
        ucp_tag_send_start_rndv(req);
    } else if (length <= config->max_am_bcopy - progress->only_hdr_size) {
        req->send.uct.func = progress->generic_single;
    } else {
        req->send.uct.func = progress->generic_multi;
    }
}

static inline ucs_status_ptr_t
ucp_tag_send_req(ucp_request_t *req, size_t count, size_t max_short,
                 size_t zcopy_thresh, size_t rndv_thresh, const ucp_proto_t *proto)
{
    ucs_status_t status;
    ucp_ep_h ep = req->send.ep;

    switch (req->send.datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        status = ucp_tag_req_start_contig(req, count, max_short, zcopy_thresh,
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

    ucp_ep_add_pending(ep, ep->uct_ep, req, 1);
    ucp_worker_progress(ep->worker);
    ucs_trace_req("returning send request %p", req);
    return req + 1;
}

void ucp_send_req_init(ucp_request_t* req, ucp_ep_h ep, const void* buffer,
                       uintptr_t datatype, ucp_tag_t tag, ucp_send_callback_t cb)
{
    VALGRIND_MAKE_MEM_DEFINED(req + 1, ep->worker->context->config.request.size);
    req->flags             = 0;
    req->cb.send           = cb;
    req->send.ep           = ep;
    req->send.buffer       = buffer;
    req->send.datatype     = datatype;
    req->send.state.offset = 0;
    req->send.tag          = tag;
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
        if (ucs_likely(length <= ucp_ep_config(ep)->max_eager_short)) {
            status = ucp_tag_send_eager_short(ep, tag, buffer, length);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                return UCS_STATUS_PTR(status); /* UCS_OK also goes here */
            }
        }
    }

    req = ucs_mpool_get_inline(&ep->worker->req_mp);
    if (req == NULL) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
    }

    ucp_send_req_init(req, ep, buffer, datatype, tag, cb);

    return ucp_tag_send_req(req, count,
                            ucp_ep_config(ep)->max_eager_short,
                            ucp_ep_config(ep)->zcopy_thresh,
                            ucp_ep_config(ep)->rndv_thresh,
                            &ucp_tag_eager_proto);
}
