/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_context.h"
#include "ucp_worker.h"
#include "ucp_request.inl"

#include <ucp/proto/proto.h>

#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>


int ucp_request_is_completed(void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    return !!(req->flags & UCP_REQUEST_FLAG_COMPLETED);
}

ucs_status_t ucp_request_check_status(void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_assert(req->status != UCS_INPROGRESS);
        return req->status;
    }
    return UCS_INPROGRESS;
}

ucs_status_t ucp_tag_recv_request_test(void *request, ucp_tag_recv_info_t *info)
{
    ucp_request_t *req   = (ucp_request_t*)request - 1;
    ucs_status_t  status = ucp_request_check_status(request);

    if (status != UCS_INPROGRESS) {
        ucs_assert(req->flags & UCP_REQUEST_FLAG_RECV);
        *info = req->recv.tag.info;
    }

    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_release_common(void *request, uint8_t cb_flag, const char *debug_name)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    ucp_worker_h UCS_V_UNUSED worker = ucs_container_of(ucs_mpool_obj_owner(req),
                                                        ucp_worker_t, req_mp);
    uint16_t flags;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    flags = req->flags;
    ucs_trace_req("%s request %p (%p) "UCP_REQUEST_FLAGS_FMT, debug_name,
                  req, req + 1, UCP_REQUEST_FLAGS_ARG(flags));

    ucs_assert(!(flags & UCP_REQUEST_DEBUG_FLAG_EXTERNAL));
    ucs_assert(!(flags & UCP_REQUEST_FLAG_RELEASED));

    if (ucs_likely(flags & UCP_REQUEST_FLAG_COMPLETED)) {
        ucp_request_put(req);
    } else {
        req->flags = (flags | UCP_REQUEST_FLAG_RELEASED) & ~cb_flag;
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
}

UCS_PROFILE_FUNC_VOID(ucp_request_release, (request), void *request)
{
    /* mark request as released */
    ucp_request_release_common(request, 0, "release");
}

UCS_PROFILE_FUNC_VOID(ucp_request_free, (request), void *request)
{
    /* mark request as released and disable the callback */
    ucp_request_release_common(request, UCP_REQUEST_FLAG_CALLBACK, "free");
}

UCS_PROFILE_FUNC_VOID(ucp_request_cancel, (worker, request),
                      ucp_worker_h worker, void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        return;
    }

    if (req->flags & UCP_REQUEST_FLAG_EXPECTED) {
        UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);
        UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->context->mt_lock);

        ucp_tag_exp_remove(&worker->context->tm, req);
        /* If tag posted to the transport need to wait its completion */
        if (!(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
            ucp_request_complete_tag_recv(req, UCS_ERR_CANCELED);
        }

        UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->context->mt_lock);
        UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);
    }
}

static void ucp_worker_request_init_proxy(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_context_h context = worker->context;
    ucp_request_t *req = obj;

    if (context->config.request.init != NULL) {
        context->config.request.init(req + 1);
    }
}

static void ucp_worker_request_fini_proxy(ucs_mpool_t *mp, void *obj)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_context_h context = worker->context;
    ucp_request_t *req = obj;

    if (context->config.request.cleanup != NULL) {
        context->config.request.cleanup(req + 1);
    }
}

ucs_mpool_ops_t ucp_request_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucp_worker_request_init_proxy,
    .obj_cleanup   = ucp_worker_request_fini_proxy
};

int ucp_request_pending_add(ucp_request_t *req, ucs_status_t *req_status)
{
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_assertv(req->send.lane != UCP_NULL_LANE, "%s() did not set req->send.lane",
                ucs_debug_get_symbol_name(req->send.uct.func));

    uct_ep = req->send.ep->uct_eps[req->send.lane];
    status = uct_ep_pending_add(uct_ep, &req->send.uct);
    if (status == UCS_OK) {
        ucs_trace_data("ep %p: added pending uct request %p to lane[%d]=%p",
                       req->send.ep, req, req->send.lane, uct_ep);
        *req_status = UCS_INPROGRESS;
        return 1;
    } else if (status == UCS_ERR_BUSY) {
        /* Could not add, try to send again */
        return 0;
    }
    /* Unexpected error while adding to pending */
    ucs_assert(status != UCS_INPROGRESS);
    *req_status = status;
    return 1;
}

static UCS_F_ALWAYS_INLINE
void ucp_iov_buffer_memh_dereg(uct_md_h uct_md, uct_mem_h *memh,
                               size_t count)
{
    size_t it;

    for (it = 0; it < count; ++it) {
        if (memh[it] != UCT_MEM_HANDLE_NULL) {
            uct_md_mem_dereg(uct_md, memh[it]);
        }
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_request_memory_reg,
                 (context, rsc_index, buffer, length, datatype, state),
                 ucp_context_t *context, ucp_rsc_index_t rsc_index, void *buffer,
                 size_t length, ucp_datatype_t datatype, ucp_dt_state_t *state)
{
    ucp_rsc_index_t mdi = context->tl_rscs[rsc_index].md_index;
    uct_md_h uct_md     = context->tl_mds[mdi].md;
    uct_md_attr_t *uct_md_attr;
    size_t iov_it, iovcnt;
    const ucp_dt_iov_t *iov;
    uct_mem_h *memh;
    ucs_status_t status;

    status = UCS_OK;
    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        status = uct_md_mem_reg(uct_md, buffer, length, UCT_MD_MEM_ACCESS_RMA,
                                &state->dt.contig.memh);
        break;
    case UCP_DATATYPE_IOV:
        iovcnt = state->dt.iov.iovcnt;
        iov    = buffer;
        memh   = ucs_malloc(sizeof(*memh) * iovcnt, "IOV memh");
        if (NULL == memh) {
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }
        for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
            if (iov[iov_it].length) {
                status = uct_md_mem_reg(uct_md, iov[iov_it].buffer,
                                        iov[iov_it].length,
                                        UCT_MD_MEM_ACCESS_RMA, &memh[iov_it]);
                if (status != UCS_OK) {
                    /* unregister previously registered memory */
                    ucp_iov_buffer_memh_dereg(uct_md, memh, iov_it);
                    ucs_free(memh);
                    goto err;
                }
            } else {
                memh[iov_it] = UCT_MEM_HANDLE_NULL; /* Indicator for zero length */
            }
        }
        state->dt.iov.memh = memh;
        break;
    default:
        status = UCS_ERR_INVALID_PARAM;
        ucs_error("Invalid data type %lx", datatype);
    }

err:
    if (status != UCS_OK) {
        uct_md_attr = &context->tl_mds[mdi].attr;
        ucs_error("failed to register user buffer [datatype=%lx address=%p "
                  "len=%zu pd=\"%s\"]: %s", datatype, buffer, length,
                  uct_md_attr->component_name, ucs_status_string(status));
    }
    return status;
}

UCS_PROFILE_FUNC_VOID(ucp_request_memory_dereg,
                      (context, rsc_index, datatype, state),
                      ucp_context_t *context, ucp_rsc_index_t rsc_index,
                      ucp_datatype_t datatype, ucp_dt_state_t *state)
{
    ucp_rsc_index_t mdi = context->tl_rscs[rsc_index].md_index;
    uct_md_h uct_md     = context->tl_mds[mdi].md;
    uct_mem_h *memh;
    size_t iov_it;

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (state->dt.contig.memh != UCT_MEM_HANDLE_NULL) {
            uct_md_mem_dereg(uct_md, state->dt.contig.memh);
        }
        break;
    case UCP_DATATYPE_IOV:
        memh = state->dt.iov.memh;
        for (iov_it = 0; iov_it < state->dt.iov.iovcnt; ++iov_it) {
            if (memh[iov_it] != UCT_MEM_HANDLE_NULL) { /* skip zero memh */
                uct_md_mem_dereg(uct_md, memh[iov_it]);
            }
        }
        ucs_free(state->dt.iov.memh);
        break;
    default:
        ucs_error("Invalid data type");
    }
}

ucs_status_t ucp_request_send_buffer_reg(ucp_request_t *req,
                                         ucp_lane_index_t lane)
{
    ucp_context_t *context    = req->send.ep->worker->context;
    req->send.reg_rsc         = ucp_ep_get_rsc_index(req->send.ep, lane);
    ucs_assert(req->send.reg_rsc != UCP_NULL_RESOURCE);

    return ucp_request_memory_reg(context, req->send.reg_rsc,
                                  (void*)req->send.buffer, req->send.length,
                                  req->send.datatype, &req->send.state.dt);
}

void ucp_request_send_buffer_dereg(ucp_request_t *req, ucp_lane_index_t lane)
{
    ucp_context_t *context    = req->send.ep->worker->context;
    ucs_assert(req->send.reg_rsc != UCP_NULL_RESOURCE);
    ucp_request_memory_dereg(context, req->send.reg_rsc, req->send.datatype,
                             &req->send.state.dt);
    req->send.reg_rsc = UCP_NULL_RESOURCE;
}

/* NOTE: deprecated */
ucs_status_t ucp_request_test(void *request, ucp_tag_recv_info_t *info)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        if (req->flags & UCP_REQUEST_FLAG_RECV) {
            *info = req->recv.tag.info;
        }
        ucs_assert(req->status != UCS_INPROGRESS);
        return req->status;
    }
    return UCS_INPROGRESS;
}

ucs_status_t
ucp_request_send_start(ucp_request_t *req, ssize_t max_short,
                       size_t zcopy_thresh, size_t seg_size,
                       size_t zcopy_max, const ucp_proto_t *proto)
{
    size_t       length = req->send.length;
    ucs_status_t status;

    if ((ssize_t)length <= max_short) {
        /* short */
        req->send.uct.func = proto->contig_short;
        UCS_PROFILE_REQUEST_EVENT(req, "start_contig_short", req->send.length);
        return UCS_OK;
    } else if (length < zcopy_thresh) {
        /* bcopy */
        ucp_request_send_state_reset(req, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        if (length < seg_size) {
            req->send.uct.func   = proto->bcopy_single;
            UCS_PROFILE_REQUEST_EVENT(req, "start_bcopy_single", req->send.length);
        } else {
            req->send.uct.func   = proto->bcopy_multi;
            UCS_PROFILE_REQUEST_EVENT(req, "start_bcopy_multi", req->send.length);
        }
        return UCS_OK;
    } else if (length < zcopy_max) {
        /* zcopy */
        ucp_request_send_state_reset(req, proto->zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_ZCOPY_AM);
        status = ucp_request_send_buffer_reg(req, req->send.lane);
        if (status != UCS_OK) {
            return status;
        }

        if (length < seg_size) {
            req->send.uct.func   = proto->zcopy_single;
            UCS_PROFILE_REQUEST_EVENT(req, "start_zcopy_single", req->send.length);
        } else {
            req->send.uct.func   = proto->zcopy_multi;
            UCS_PROFILE_REQUEST_EVENT(req, "start_zcopy_multi", req->send.length);
        }
        return UCS_OK;
    }

    return UCS_ERR_NO_PROGRESS;
}
