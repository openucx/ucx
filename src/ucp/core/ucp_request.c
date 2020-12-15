/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_context.h"
#include "ucp_worker.h"
#include "ucp_request.inl"

#include <ucp/proto/proto_am.h>

#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>


const ucp_request_param_t ucp_request_null_param = { .op_attr_mask = 0 };


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
        ucs_assert(req->flags & UCP_REQUEST_FLAG_RECV_TAG);
        *info = req->recv.tag.info;
    }

    return status;
}

ucs_status_t ucp_stream_recv_request_test(void *request, size_t *length_p)
{
    ucp_request_t *req   = (ucp_request_t*)request - 1;
    ucs_status_t  status = ucp_request_check_status(request);

    if (status != UCS_INPROGRESS) {
        ucs_assert(req->flags & UCP_REQUEST_FLAG_STREAM_RECV);
        *length_p = req->recv.stream.length;
    }

    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_request_release_common(void *request, uint8_t cb_flag, const char *debug_name)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    ucp_worker_h UCS_V_UNUSED worker = ucs_container_of(ucs_mpool_obj_owner(req),
                                                        ucp_worker_t, req_mp);
    uint32_t flags;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

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

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
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

UCS_PROFILE_FUNC(void*, ucp_request_alloc,
                 (worker),
                 ucp_worker_h worker)
{
    return NULL;
}

UCS_PROFILE_FUNC_VOID(ucp_request_cancel, (worker, request),
                      ucp_worker_h worker, void *request)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    int removed;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        return;
    }

    if (req->flags & UCP_REQUEST_FLAG_EXPECTED) {
        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

        removed = ucp_tag_exp_remove(&worker->tm, req);
        /* If tag posted to the transport need to wait its completion */
        if (removed && !(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
            ucp_request_complete_tag_recv(req, UCS_ERR_CANCELED);
        }

        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
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

ucs_mpool_ops_t ucp_rndv_get_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

int ucp_request_pending_add(ucp_request_t *req, unsigned pending_flags)
{
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_assertv(req->send.lane != UCP_NULL_LANE, "%s() did not set req->send.lane",
                ucs_debug_get_symbol_name(req->send.uct.func));

    uct_ep = req->send.ep->uct_eps[req->send.lane];
    status = uct_ep_pending_add(uct_ep, &req->send.uct, pending_flags);
    if (status == UCS_OK) {
        ucs_trace_data("ep %p: added pending uct request %p to lane[%d]=%p",
                       req->send.ep, req, req->send.lane, uct_ep);
        req->send.pending_lane = req->send.lane;
        return 1;
    } else if (status == UCS_ERR_BUSY) {
        /* Could not add, try to send again */
        return 0;
    }

    /* Unexpected error while adding to pending */
    ucs_fatal("invalid return status from uct_ep_pending_add(): %s",
              ucs_status_string(status));
}

static void ucp_request_dt_dereg(ucp_context_t *context, ucp_dt_reg_t *dt_reg,
                                 size_t count, ucp_request_t *req_dbg)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        ucp_trace_req(req_dbg, "mem dereg buffer %ld/%ld md_map 0x%"PRIx64,
                      i, count, dt_reg[i].md_map);
        ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL, UCS_MEMORY_TYPE_HOST, NULL,
                          dt_reg[i].memh, &dt_reg[i].md_map);
        ucs_assert(dt_reg[i].md_map == 0);
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_request_memory_reg,
                 (context, md_map, buffer, length, datatype, state, mem_type, req_dbg, uct_flags),
                 ucp_context_t *context, ucp_md_map_t md_map, void *buffer,
                 size_t length, ucp_datatype_t datatype, ucp_dt_state_t *state,
                 ucs_memory_type_t mem_type, ucp_request_t *req_dbg, unsigned uct_flags)
{
    size_t iov_it, iovcnt;
    const ucp_dt_iov_t *iov;
    ucp_dt_reg_t *dt_reg;
    ucs_status_t status;
    int flags;
    int level;

    ucs_trace_func("context=%p md_map=0x%"PRIx64" buffer=%p length=%zu "
                   "datatype=0x%"PRIx64" state=%p", context, md_map, buffer,
                   length, datatype, state);

    status = UCS_OK;
    flags  = UCT_MD_MEM_ACCESS_RMA | uct_flags;
    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        ucs_assert(ucs_popcount(md_map) <= UCP_MAX_OP_MDS);
        status = ucp_mem_rereg_mds(context, md_map, buffer, length, flags,
                                   NULL, mem_type, NULL, state->dt.contig.memh,
                                   &state->dt.contig.md_map);
        ucp_trace_req(req_dbg, "mem reg md_map 0x%"PRIx64"/0x%"PRIx64,
                      state->dt.contig.md_map, md_map);
        break;
    case UCP_DATATYPE_IOV:
        iovcnt = state->dt.iov.iovcnt;
        iov    = buffer;
        dt_reg = ((state->dt.iov.dt_reg == NULL) ?
                  ucs_calloc(iovcnt, sizeof(*dt_reg), "iov_dt_reg") :
                  state->dt.iov.dt_reg);
        if (NULL == dt_reg) {
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }
        for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
            if (iov[iov_it].length) {
                status = ucp_mem_rereg_mds(context, md_map, iov[iov_it].buffer,
                                           iov[iov_it].length, flags, NULL,
                                           mem_type, NULL, dt_reg[iov_it].memh,
                                           &dt_reg[iov_it].md_map);
                if (status != UCS_OK) {
                    /* unregister previously registered memory */
                    ucp_request_dt_dereg(context, dt_reg, iov_it, req_dbg);
                    ucs_free(dt_reg);
                    goto err;
                }
                ucp_trace_req(req_dbg,
                              "mem reg iov %ld/%ld md_map 0x%"PRIx64"/0x%"PRIx64,
                              iov_it, iovcnt, dt_reg[iov_it].md_map, md_map);
            }
        }
        state->dt.iov.dt_reg = dt_reg;
        break;
    default:
        status = UCS_ERR_INVALID_PARAM;
        ucs_error("Invalid data type 0x%"PRIx64, datatype);
    }

err:
    if (status != UCS_OK) {
        level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ?
                UCS_LOG_LEVEL_DEBUG : UCS_LOG_LEVEL_ERROR;
        ucs_log(level,
                "failed to register user buffer datatype 0x%"PRIx64
                " address %p len %zu: %s", datatype, buffer, length,
                ucs_status_string(status));
    }
    return status;
}

UCS_PROFILE_FUNC_VOID(ucp_request_memory_dereg, (context, datatype, state, req_dbg),
                      ucp_context_t *context, ucp_datatype_t datatype,
                      ucp_dt_state_t *state, ucp_request_t *req_dbg)
{
    ucs_trace_func("context=%p datatype=0x%"PRIx64" state=%p", context,
                   datatype, state);

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        ucp_request_dt_dereg(context, &state->dt.contig, 1, req_dbg);
        break;
    case UCP_DATATYPE_IOV:
        if (state->dt.iov.dt_reg != NULL) {
            ucp_request_dt_dereg(context, state->dt.iov.dt_reg,
                                 state->dt.iov.iovcnt, req_dbg);
            ucs_free(state->dt.iov.dt_reg);
            state->dt.iov.dt_reg = NULL;
        }
        break;
    default:
        break;
    }
}

/* NOTE: deprecated */
ucs_status_t ucp_request_test(void *request, ucp_tag_recv_info_t *info)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        if (req->flags & UCP_REQUEST_FLAG_RECV_TAG) {
            *info = req->recv.tag.info;
        }
        ucs_assert(req->status != UCS_INPROGRESS);
        return req->status;
    }
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE
void ucp_request_init_multi_proto(ucp_request_t *req,
                                  uct_pending_callback_t multi_func,
                                  const char *multi_func_str)
{
    req->send.uct.func = multi_func;

    if (req->flags & (UCP_REQUEST_FLAG_SEND_TAG |
                      UCP_REQUEST_FLAG_SEND_AM)) {
        req->send.msg_proto.message_id = req->send.ep->worker->am_message_id++;
        req->send.am_bw_index          = 0;
    }

    req->send.pending_lane = UCP_NULL_LANE;
    UCS_PROFILE_REQUEST_EVENT(req, multi_func_str, req->send.length);
}

static UCS_F_ALWAYS_INLINE void ucp_send_request_init_id(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_SYNC) {
        ucs_assert(req->flags & UCP_REQUEST_FLAG_SEND_TAG);
        ucp_send_request_set_id(req);
    }
}

ucs_status_t
ucp_request_send_start(ucp_request_t *req, ssize_t max_short,
                       size_t zcopy_thresh, size_t zcopy_max,
                       size_t dt_count, size_t priv_iov_count,
                       size_t length, const ucp_ep_msg_config_t* msg_config,
                       const ucp_request_send_proto_t *proto)
{
    ucs_status_t status;
    int          multi;

    req->status = UCS_INPROGRESS;

    if ((ssize_t)length <= max_short) {
        /* short */
        req->send.uct.func = proto->contig_short;
        UCS_PROFILE_REQUEST_EVENT(req, "start_contig_short", req->send.length);
        return UCS_OK;
    } else if (length < zcopy_thresh) {
        /* bcopy */
        ucp_request_send_state_reset(req, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);
        ucs_assert(msg_config->max_bcopy >= proto->only_hdr_size);
        if (length <= (msg_config->max_bcopy - proto->only_hdr_size)) {
            req->send.uct.func = proto->bcopy_single;
            UCS_PROFILE_REQUEST_EVENT(req, "start_bcopy_single", req->send.length);
        } else {
            ucp_request_init_multi_proto(req, proto->bcopy_multi,
                                         "start_bcopy_multi");
        }

        ucp_send_request_init_id(req);
        return UCS_OK;
    } else if (length < zcopy_max) {
        /* zcopy */
        ucp_request_send_state_reset(req, proto->zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_ZCOPY_AM);
        status = ucp_request_send_buffer_reg_lane(req, req->send.lane, 0);
        if (status != UCS_OK) {
            return status;
        }

        if (ucs_unlikely(length > msg_config->max_zcopy - proto->only_hdr_size)) {
            multi = 1;
        } else if (ucs_unlikely(UCP_DT_IS_IOV(req->send.datatype))) {
            if (dt_count <= (msg_config->max_iov - priv_iov_count)) {
                multi = 0;
            } else {
                multi = ucp_dt_iov_count_nonempty(req->send.buffer, dt_count) >
                        (msg_config->max_iov - priv_iov_count);
            }
        } else {
            multi = 0;
        }

        if (multi) {
            ucp_request_init_multi_proto(req, proto->zcopy_multi,
                                         "start_zcopy_multi");
        } else {
            req->send.uct.func = proto->zcopy_single;
            UCS_PROFILE_REQUEST_EVENT(req, "start_zcopy_single", req->send.length);
        }

        ucp_send_request_init_id(req);
        return UCS_OK;
    }

    return UCS_ERR_NO_PROGRESS;
}

void ucp_request_send_state_ff(ucp_request_t *req, ucs_status_t status)
{
    /*
     * FIXME should not fast-forward requests owned by UCT
     */
    ucp_trace_req(req, "fast-forward with status %s", ucs_status_string(status));

    if (req->send.state.uct_comp.func == ucp_ep_flush_completion) {
        ucp_ep_flush_request_ff(req, status);
    } else if (req->send.state.uct_comp.func) {
        req->send.state.dt.offset      = req->send.length;
        req->send.state.uct_comp.count = 0;
        uct_completion_update_status(&req->send.state.uct_comp, status);
        req->send.state.uct_comp.func(&req->send.state.uct_comp);
    } else {
        ucp_request_complete_send(req, status);
    }
}

ucs_status_t ucp_request_recv_msg_truncated(ucp_request_t *req, size_t length,
                                            size_t offset)
{
    ucp_dt_generic_t *dt_gen;

    ucs_debug("message truncated: recv_length %zu offset %zu buffer_size %zu",
              length, offset, req->recv.length);

    if (UCP_DT_IS_GENERIC(req->recv.datatype)) {
        dt_gen = ucp_dt_to_generic(req->recv.datatype);
        UCS_PROFILE_NAMED_CALL_VOID("dt_finish", dt_gen->ops.finish,
                                    req->recv.state.dt.generic.state);
    }

    return UCS_ERR_MESSAGE_TRUNCATED;
}
