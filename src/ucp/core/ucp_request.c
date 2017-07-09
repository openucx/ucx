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

ucs_status_t ucp_request_test(void *request, ucp_tag_recv_info_t *info)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        if (req->flags & UCP_REQUEST_FLAG_RECV) {
            *info = req->recv.info;
        }
        ucs_assert(req->status != UCS_INPROGRESS);
        return req->status;
    }
    return UCS_INPROGRESS;
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
            ucp_request_complete_recv(req, UCS_ERR_CANCELED);
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

void ucp_request_release_pending_send(uct_pending_req_t *self, void *arg)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_complete_send(req, UCS_ERR_CANCELED);
}

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
void ucp_multiple_memh_dereg(uct_md_h uct_md, uct_mem_h *memh, size_t count)
{
    size_t it;

    for (it = 0; it < count; ++it) {
        if (memh[it] != UCT_MEM_HANDLE_NULL) {
            uct_md_mem_dereg(uct_md, memh[it]);
        }
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_request_memory_reg,
                 (context, rsc_index, buffer, length, datatype, state, ep),
                 ucp_context_t *context, ucp_rsc_index_t rsc_index, void *buffer,
                 size_t length, ucp_datatype_t datatype, ucp_dt_state_t *state,
                 uct_ep_h ep)
{
    ucp_rsc_index_t mdi = context->tl_rscs[rsc_index].md_index;
    uct_md_h uct_md     = context->tl_mds[mdi].md;
    ucp_dt_reusable_t *reusable;
    uct_md_attr_t *uct_md_attr;
    size_t iov_it, iovcnt;
    const ucp_dt_iov_t *iov;
    uct_mem_h *memh;
    ucs_status_t status;
    size_t extent;

    status = UCS_OK;
    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        status = uct_md_mem_reg(uct_md, buffer, length, 0, &state->dt.contig.memh);
        break;

    case UCP_DATATYPE_STRIDE:
        extent = ucp_dt_extent(datatype, state->dt.stride.count, NULL, NULL);
        status = uct_md_mem_reg(uct_md, buffer, extent, 0, &state->dt.stride.memh);
        break;

    case UCP_DATATYPE_STRIDE_R:
        /* If this is not the first time - just update the pointers and GO */
        reusable = UCP_DT_GET_REUSABLE(datatype);
        if (reusable->stride_memh != UCT_MEM_HANDLE_NULL) {
            if (ucs_unlikely(reusable->nc_status != UCS_OK)) {
                return status;
            }

            uct_md_attr = &context->tl_mds[mdi].attr;
            if (uct_md_attr->cap.flags & UCT_MD_FLAG_REG_NC) {
                status = ucp_dt_reusable_update(ep, buffer, length, datatype, state);
                state->dt.stride.contig_memh = reusable->nc_memh;
            }
            state->dt.stride.memh = reusable->stride_memh;
            break;
        }

        /* Map the entire extent of buffers potentially sent */
        extent = ucp_dt_extent(datatype, state->dt.stride.count, NULL, NULL);
        status = uct_md_mem_reg(uct_md, buffer, extent, 0, &state->dt.stride.memh);

        /* If non-contiguous bind is not supported - use the existing mapping */
        uct_md_attr = &context->tl_mds[mdi].attr;
        if (!(uct_md_attr->cap.flags & UCT_MD_FLAG_REG_NC)) {
            break;
        }

        /* make sure the call to uct_md_mem_reg() succeeded */
        if (status != UCS_OK) {
            break;
        }

        status = ucp_dt_reusable_create(ep, buffer, length, datatype, state);
        state->dt.stride.contig_memh = reusable->nc_memh;
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
            size_t iov_len = ucp_dt_extent(iov->dt, iov->count, NULL, NULL);
            if (iov_len) {
                status = uct_md_mem_reg(uct_md, iov->buffer, iov_len, 0,
                                        &memh[iov_it]);
                if (status != UCS_OK) {
                    /* unregister previously registered memory */
                    ucp_multiple_memh_dereg(uct_md, memh, iov_it);
                    ucs_free(memh);
                    goto err;
                }
            } else {
                memh[iov_it] = UCT_MEM_HANDLE_NULL; /* Indicator for zero length */
            }
        }
        state->dt.iov.memh = memh;
        break;

    case UCP_DATATYPE_IOV_R:
        /* If this is not the first time - just update the pointers and GO */
        reusable = UCP_DT_GET_REUSABLE(datatype);
        if (reusable->stride_memh != UCT_MEM_HANDLE_NULL) {
            if (ucs_unlikely(reusable->nc_status != UCS_OK)) {
                return status;
            }

            uct_md_attr = &context->tl_mds[mdi].attr;
            if (uct_md_attr->cap.flags & UCT_MD_FLAG_REG_NC) {
                status = ucp_dt_reusable_update(ep, buffer, length, datatype, state);
                state->dt.iov.contig_memh = reusable->nc_memh;
            }
            state->dt.iov.memh = reusable->stride_memh;
            break;
        }

        /* Map the entire extent of buffers potentially sent */
        iovcnt = state->dt.iov.iovcnt;
        iov    = buffer;
        memh   = ucs_malloc(sizeof(*memh) * iovcnt, "IOV memh reusable");
        if (NULL == memh) {
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }
        for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
            size_t iov_len = ucp_dt_extent(iov->dt, iov->count, NULL, NULL);
            if (iov_len) {
                status = uct_md_mem_reg(uct_md, iov->buffer, iov_len, 0,
                                        &memh[iov_it]);
                if (status != UCS_OK) {
                    /* unregister previously registered memory */
                    ucp_multiple_memh_dereg(uct_md, memh, iov_it);
                    ucs_free(memh);
                    goto err;
                }
            } else {
                memh[iov_it] = UCT_MEM_HANDLE_NULL; /* Indicator for zero length */
            }
        }
        state->dt.iov.memh = memh;

        /* If non-contiguous bind is not supported - use the existing mapping */
        uct_md_attr = &context->tl_mds[mdi].attr;
        if (!(uct_md_attr->cap.flags & UCT_MD_FLAG_REG_NC)) {
            break;
        }

        status = ucp_dt_reusable_create(ep, buffer, length, datatype, state);
        state->dt.iov.contig_memh = reusable->nc_memh;
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

    case UCP_DATATYPE_STRIDE:
        if (state->dt.stride.memh != UCT_MEM_HANDLE_NULL) {
            uct_md_mem_dereg(uct_md, state->dt.stride.memh);
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

    case UCP_DATATYPE_IOV_R:
    case UCP_DATATYPE_STRIDE_R:
        break;

    default:
        ucs_error("Invalid data type");
    }
}

ucs_status_t ucp_request_send_buffer_reg(ucp_request_t *req,
                                         ucp_lane_index_t lane)
{
    uct_ep_h ep               = req->send.ep->uct_eps[lane];
    ucp_context_t *context    = req->send.ep->worker->context;
    req->send.reg_rsc         = ucp_ep_get_rsc_index(req->send.ep, lane);
    ucs_assert(req->send.reg_rsc != UCP_NULL_RESOURCE);

    return ucp_request_memory_reg(context, req->send.reg_rsc,
                                  (void*)req->send.buffer, req->send.length,
                                  req->send.datatype, &req->send.state, ep);
}

void ucp_request_send_buffer_dereg(ucp_request_t *req, ucp_lane_index_t lane)
{
    ucp_context_t *context    = req->send.ep->worker->context;
    ucs_assert(req->send.reg_rsc != UCP_NULL_RESOURCE);
    ucp_request_memory_dereg(context, req->send.reg_rsc, req->send.datatype,
                             &req->send.state);
    req->send.reg_rsc = UCP_NULL_RESOURCE;
}
