/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <uct/api/v2/uct_v2.h>
#include "ucp_context.h"
#include "ucp_worker.h"
#include "ucp_request.inl"

#include <ucp/proto/proto_am.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/tag/tag_rndv.h>

#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/debug_int.h>
#include <ucs/debug/log.h>


const ucp_request_param_t ucp_request_null_param = { .op_attr_mask = 0 };

static ucs_memory_type_t ucp_request_get_mem_type(ucp_request_t *req)
{
    if (req->flags & (UCP_REQUEST_FLAG_SEND_AM | UCP_REQUEST_FLAG_SEND_TAG)) {
        if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
            return req->send.state.dt_iter.mem_info.type;
        } else {
            return req->send.mem_type;
        }
    } else if (req->flags &
               (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG)) {
        return req->recv.mem_type;
    } else {
        return UCS_MEMORY_TYPE_UNKNOWN;
    }
}

static void
ucp_request_str(ucp_request_t *req, ucs_string_buffer_t *strb, int recurse)
{
    const char *progress_func_name;
    const char *comp_func_name;
    ucp_ep_config_t *config;
    ucp_ep_h ep;

    ucs_string_buffer_appendf(strb, "flags:0x%x ", req->flags);

    if (req->flags & UCP_REQUEST_FLAG_PROTO_SEND) {
        ucp_proto_config_info_str(req->send.ep->worker, req->send.proto_config,
                                  req->send.state.dt_iter.length, strb);
        return;
    }

    if (req->flags & (UCP_REQUEST_FLAG_SEND_AM | UCP_REQUEST_FLAG_SEND_TAG)) {
        ucs_string_buffer_appendf(strb, "send length %zu ", req->send.length);

        progress_func_name = ucs_debug_get_symbol_name(req->send.uct.func);
        ucs_string_buffer_appendf(strb, "%s() ", progress_func_name);

        if (req->flags & UCP_REQUEST_FLAG_CALLBACK) {
            comp_func_name = ucs_debug_get_symbol_name(req->send.cb);
            ucs_string_buffer_appendf(strb, "comp:%s()", comp_func_name);
        }

        if (recurse) {
            ep     = req->send.ep;
            config = ucp_ep_config(ep);
            ucp_ep_config_lane_info_str(ep->worker, &config->key, NULL,
                                        req->send.lane, UCP_NULL_RESOURCE,
                                        strb);
        }
    } else if (req->flags &
               (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG)) {
#if ENABLE_DEBUG_DATA
        if (req->recv.proto_rndv_config != NULL) {
            /* Print the send protocol of the rendezvous request */
            ucp_proto_config_info_str(req->recv.worker,
                                      req->recv.proto_rndv_config,
                                      req->recv.length, strb);
            return;
        }
#endif
        ucs_string_buffer_appendf(strb, "recv length %zu ", req->recv.length);
    } else {
        ucs_string_buffer_appendf(strb, "<no debug info>");
        return;
    }

    ucs_string_buffer_appendf(
            strb, "%s memory",
            ucs_memory_type_names[ucp_request_get_mem_type(req)]);
}

ucs_status_t ucp_request_query(void *request, ucp_request_attr_t *attr)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;
    ucs_string_buffer_t strb;

    if (req->flags & UCP_REQUEST_FLAG_RELEASED) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (attr->field_mask & UCP_REQUEST_ATTR_FIELD_INFO_STRING) {
        if (!(attr->field_mask & UCP_REQUEST_ATTR_FIELD_INFO_STRING_SIZE)) {
            return UCS_ERR_INVALID_PARAM;
        }

        ucs_string_buffer_init_fixed(&strb, attr->debug_string,
                                     attr->debug_string_size);
        ucp_request_str(req, &strb, 1);
    }

    if (attr->field_mask & UCP_REQUEST_ATTR_FIELD_STATUS) {
        attr->status = ucp_request_check_status(request);
    }

    if (attr->field_mask & UCP_REQUEST_ATTR_FIELD_MEM_TYPE) {
        attr->mem_type = ucp_request_get_mem_type(req);
    }

    return UCS_OK;
}

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

    if (req->flags & UCP_REQUEST_FLAG_RECV_TAG) {
        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

        removed = ucp_tag_exp_remove(&worker->tm, req);
        /* If tag posted to the transport need to wait its completion */
        if (removed && !(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
            ucp_request_complete_tag_recv(req, UCS_ERR_CANCELED);
        }

        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    }
}

static void
ucp_worker_request_init_proxy(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucp_worker_h worker   = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_context_h context = worker->context;
    ucp_request_t *req    = obj;

    ucp_request_id_reset(req);

    if (context->config.request.init != NULL) {
        context->config.request.init(req + 1);
    }
}

static void ucp_worker_request_fini_proxy(ucs_mpool_t *mp, void *obj)
{
    ucp_worker_h worker   = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_context_h context = worker->context;
    ucp_request_t *req    = obj;

    ucp_request_id_check(req, ==, UCS_PTR_MAP_KEY_INVALID);

    if (context->config.request.cleanup != NULL) {
        context->config.request.cleanup(req + 1);
    }
}

static void
ucp_request_mpool_obj_str(ucs_mpool_t *mp, void *obj, ucs_string_buffer_t *strb)
{
    ucp_request_t *req = obj;

    ucp_request_str(req, strb, 0);
}

ucs_mpool_ops_t ucp_request_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucp_worker_request_init_proxy,
    .obj_cleanup   = ucp_worker_request_fini_proxy,
    .obj_str       = ucp_request_mpool_obj_str
};

ucs_mpool_ops_t ucp_rndv_get_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

int ucp_request_pending_add(ucp_request_t *req)
{
    ucs_status_t status;
    uct_ep_h uct_ep;

    ucs_assertv(req->send.lane != UCP_NULL_LANE, "%s() did not set req->send.lane",
                ucs_debug_get_symbol_name(req->send.uct.func));

    uct_ep = req->send.ep->uct_eps[req->send.lane];
    status = uct_ep_pending_add(uct_ep, &req->send.uct, 0);
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

static unsigned ucp_request_dt_invalidate_progress(void *arg)
{
    ucp_request_t *req = arg;

    ucp_request_complete_send(req, req->status);
    return 1;
}

static void ucp_request_mem_invalidate_completion(uct_completion_t *comp)
{
    ucp_request_t *req         = ucs_container_of(comp, ucp_request_t,
                                                  send.state.uct_comp);
    uct_worker_cb_id_t prog_id = UCS_CALLBACKQ_ID_NULL;

    uct_worker_progress_register_safe(req->send.invalidate.worker->uct,
                                      ucp_request_dt_invalidate_progress,
                                      req, UCS_CALLBACKQ_FLAG_ONESHOT,
                                      &prog_id);
}

static ucp_md_map_t ucp_request_get_invalidation_map(ucp_request_t *req)
{
    ucp_ep_h ep              = req->send.ep;
    ucp_ep_config_key_t *key = &ucp_ep_config(ep)->key;
    ucp_lane_index_t lane;
    ucp_lane_index_t i;
    ucp_md_map_t inv_map;

    for (i = 0, inv_map = 0;
         (key->rma_bw_lanes[i] != UCP_NULL_LANE) && (i < UCP_MAX_LANES); i++) {
        lane = key->rma_bw_lanes[i];

        if (!ucp_ep_is_lane_p2p(ep, lane)) {
            ucs_assert(ucp_ep_get_iface_attr(ep, lane)->cap.flags &
                       UCT_IFACE_FLAG_GET_ZCOPY);
            ucs_assert(ucp_ep_md_attr(ep, lane)->cap.flags &
                       UCT_MD_FLAG_INVALIDATE);
            inv_map |= UCS_BIT(ucp_ep_md_index(ep, lane));
        }
    }

    return inv_map & req->send.state.dt.dt.contig.md_map;
}

void ucp_request_dt_invalidate(ucp_request_t *req, ucs_status_t status)
{
    uct_md_mem_dereg_params_t params = {
        .field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH |
                      UCT_MD_MEM_DEREG_FIELD_FLAGS |
                      UCT_MD_MEM_DEREG_FIELD_COMPLETION,
        .flags      = UCT_MD_MEM_DEREG_FLAG_INVALIDATE,
        .comp       = &req->send.state.uct_comp
    };
    ucp_worker_h worker   = req->send.ep->worker;
    ucp_context_h context = worker->context;
    uct_mem_h *uct_memh   = req->send.state.dt.dt.contig.memh;
    ucp_md_map_t invalidate_map;
    unsigned md_index;
    unsigned memh_index;

    ucs_assert(status != UCS_OK);
    ucs_assert(ucp_ep_config(req->send.ep)->key.err_mode !=
               UCP_ERR_HANDLING_MODE_NONE);
    ucs_assert(UCP_DT_IS_CONTIG(req->send.datatype));

    invalidate_map                  = ucp_request_get_invalidation_map(req);
    req->send.ep                    = NULL;
    req->send.state.uct_comp.count  = 1;
    req->send.state.uct_comp.func   = ucp_request_mem_invalidate_completion;
    req->send.state.uct_comp.status = UCS_OK;
    req->send.invalidate.worker     = worker;
    req->status                     = status;

    ucp_trace_req(req, "mem dereg buffer md_map 0x%"PRIx64, invalidate_map);
    /* dereg all lanes except for 'invalidate_map' */
    ucp_mem_rereg_mds(context, invalidate_map, NULL, 0, 0, NULL,
                      UCS_MEMORY_TYPE_HOST, NULL,
                      req->send.state.dt.dt.contig.memh,
                      &req->send.state.dt.dt.contig.md_map);
    ucp_trace_req(req, "mem invalidate buffer md_map 0x%"PRIx64,
                  req->send.state.dt.dt.contig.md_map);

    memh_index = 0;
    ucs_log_indent(1);
    ucs_for_each_bit(md_index, req->send.state.dt.dt.contig.md_map) {
        ucs_trace_req("invalidating memh[%d]=%p from md[%d]", memh_index,
                      uct_memh[memh_index], md_index);
        req->send.state.uct_comp.count++;
        params.memh = uct_memh[memh_index];
        status      = uct_md_mem_dereg_v2(context->tl_mds[md_index].md,
                                          &params);
        if (status != UCS_OK) {
            ucs_warn("failed to dereg from md[%d]=%s: %s", md_index,
                     context->tl_mds[md_index].rsc.md_name,
                     ucs_status_string(status));
            req->send.state.uct_comp.count--;
        }
        memh_index++;
    }

    ucs_log_indent(-1);
    ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
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
                 (context, md_map, buffer, length, datatype, state, mem_type,
                  req, uct_flags),
                 ucp_context_t *context, ucp_md_map_t md_map, void *buffer,
                 size_t length, ucp_datatype_t datatype, ucp_dt_state_t *state,
                 ucs_memory_type_t mem_type, ucp_request_t *req,
                 unsigned uct_flags)
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

    if (req->flags & UCP_REQUEST_FLAG_USER_MEMH) {
        ucs_assert(UCP_DT_IS_CONTIG(datatype));

        /* All memory domains that we need were provided by user memh */
        if (ucs_likely(ucs_test_all_flags(state->dt.contig.md_map, md_map))) {
            ucp_trace_req(req, "memh already registered");
            return UCS_OK;
        }

        /* We can't mix user-provided memh with internal registrations, since
         * would need to track which ones to release.
         * Forget about what user provided and register what we need.
         */
        ucp_trace_req(req, "mds 0x%" PRIx64 " not registered - drop user memh",
                      md_map & ~state->dt.contig.md_map);
        req->flags             &= ~UCP_REQUEST_FLAG_USER_MEMH;
        state->dt.contig.md_map = 0;
    }

    status = UCS_OK;
    flags  = UCT_MD_MEM_ACCESS_RMA | uct_flags;
    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        ucs_assert(ucs_popcount(md_map) <= UCP_MAX_OP_MDS);
        status = ucp_mem_rereg_mds(context, md_map, buffer, length, flags,
                                   NULL, mem_type, NULL, state->dt.contig.memh,
                                   &state->dt.contig.md_map);
        ucp_trace_req(req, "mem reg md_map 0x%" PRIx64 "/0x%" PRIx64,
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
                    ucp_request_dt_dereg(context, dt_reg, iov_it, req);
                    ucs_free(dt_reg);
                    goto err;
                }
                ucp_trace_req(req,
                              "mem reg iov %ld/%ld md_map 0x%" PRIx64
                              "/0x%" PRIx64,
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

UCS_PROFILE_FUNC_VOID(ucp_request_memory_dereg, (context, datatype, state, req),
                      ucp_context_t *context, ucp_datatype_t datatype,
                      ucp_dt_state_t *state, ucp_request_t *req)
{
    ucs_trace_func("context=%p datatype=0x%"PRIx64" state=%p", context,
                   datatype, state);

    if (req->flags & UCP_REQUEST_FLAG_USER_MEMH) {
        return;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        ucp_request_dt_dereg(context, &state->dt.contig, 1, req);
        break;
    case UCP_DATATYPE_IOV:
        if (state->dt.iov.dt_reg != NULL) {
            ucp_request_dt_dereg(context, state->dt.iov.dt_reg,
                                 state->dt.iov.iovcnt, req);
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

ucs_status_t ucp_request_send_start(ucp_request_t *req, ssize_t max_short,
                                    size_t zcopy_thresh, size_t zcopy_max,
                                    size_t dt_count, size_t priv_iov_count,
                                    size_t length,
                                    const ucp_ep_msg_config_t *msg_config,
                                    const ucp_request_send_proto_t *proto,
                                    const ucp_request_param_t *param)
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

        return UCS_OK;
    } else if (length < zcopy_max) {
        /* zcopy */
        ucp_request_send_state_reset(req, proto->zcopy_completion,
                                     UCP_REQUEST_SEND_PROTO_ZCOPY_AM);
        status = ucp_send_request_set_user_memh(
                req, ucp_ep_config(req->send.ep)->am_bw_prereg_md_map, param);
        if (status != UCS_OK) {
            return status;
        }

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

        return UCS_OK;
    }

    return UCS_ERR_NO_PROGRESS;
}

void ucp_request_send_state_ff(ucp_request_t *req, ucs_status_t status)
{
    ucp_trace_req(req, "fast-forward with status %s",
                  ucs_status_string(status));

    ucs_assertv(UCS_STATUS_IS_ERR(status), "status=%s",
                ucs_status_string(status));

    /* Set REMOTE_COMPLETED flag to make sure that TAG/Sync operations will be
     * fully completed here */
    req->flags |= UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED;
    ucp_send_request_id_release(req);

    if (req->send.uct.func == ucp_proto_progress_am_single) {
        req->send.proto.comp_cb(req);
    } else if (req->send.uct.func == ucp_wireup_msg_progress) {
        /* Sending EP_REMOVED/EP_CHECK/ACK WIREUP_MSGs could be scheduled on
         * UCT endpoint which is not a WIREUP_EP. Other WIREUP MSGs should not
         * be returned from 'uct_ep_pending_purge()', since they are released
         * by WIREUP endpoint's purge function
         */
        ucs_assertv((req->send.wireup.type == UCP_WIREUP_MSG_EP_REMOVED) ||
                    (req->send.wireup.type == UCP_WIREUP_MSG_EP_CHECK) ||
                    (req->send.wireup.type == UCP_WIREUP_MSG_ACK),
                    "req %p ep %p: got %s message", req, req->send.ep,
                    ucp_wireup_msg_str(req->send.wireup.type));
        ucs_free(req->send.buffer);
        ucp_request_mem_free(req);
    } else if (req->send.state.uct_comp.func == ucp_ep_flush_completion) {
        ucp_ep_flush_request_ff(req, status);
    } else if (req->send.uct.func == ucp_worker_discard_uct_ep_pending_cb) {
        /* Discard operations with flush(LOCAL) could be started (e.g. closing
         * unneeded UCT EPs from intersection procedure), convert them to
         * flush(CANCEL) to avoid flushing failed UCT EPs
         */
        req->send.discard_uct_ep.ep_flush_flags |= UCT_FLUSH_FLAG_CANCEL;
        ucp_worker_discard_uct_ep_progress(req);
    } else if (req->send.state.uct_comp.func != NULL) {
        /* Fast-forward the sending state to complete the operation when last
         * network completion callback is called
         */
        req->send.state.dt.offset = req->send.length;
        uct_completion_update_status(&req->send.state.uct_comp, status);

        /* If nothing is in-flight, call completion callback to ensure cleanup
         * of zero-copy resources
         */
        ucp_send_request_invoke_uct_completion(req);
    } else if ((req->send.uct.func == ucp_proto_progress_rndv_rtr) ||
               (req->send.uct.func == ucp_proto_progress_am_rndv_rts) ||
               (req->send.uct.func == ucp_proto_progress_tag_rndv_rts)) {
        /* Canceling control message which asks for remote side to reply is
         * equivalent to reply not being received */
        ucp_ep_req_purge(req->send.ep, req, status, 1);
    } else {
        ucp_request_send_buffer_dereg(req);
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

void ucp_request_purge_enqueue_cb(uct_pending_req_t *self, void *arg)
{
    ucp_request_t *req      = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_queue_head_t *queue = arg;

    ucs_trace_req("ep %p: extracted request %p from pending queue",
                  req->send.ep, req);
    ucs_queue_push(queue, (ucs_queue_elem_t*)&req->send.uct.priv);
}

ucs_status_t ucp_request_progress_wrapper(uct_pending_req_t *self)
{
    ucp_request_t *req       = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_t *proto = req->send.proto_config->proto;
    uct_pending_callback_t progress_cb;
    ucs_status_t status;

    progress_cb = proto->progress[req->send.proto_stage];
    ucp_trace_req(req,
                  "progress %s {%s} ep_cfg[%d] rkey_cfg[%d] offset %zu/%zu",
                  proto->name, ucs_debug_get_symbol_name(progress_cb),
                  req->send.proto_config->ep_cfg_index,
                  req->send.proto_config->rkey_cfg_index,
                  req->send.state.dt_iter.offset,
                  req->send.state.dt_iter.length);

    ucs_log_indent(1);
    status = progress_cb(self);
    if (UCS_STATUS_IS_ERR(status)) {
        ucp_trace_req(req, "progress protocol %s returned: %s lane %d",
                      proto->name, ucs_status_string(status), req->send.lane);
    } else {
        ucp_trace_req(req, "progress protocol %s returned: %s", proto->name,
                      ucs_status_string(status));
    }
    ucs_log_indent(-1);
    return status;
}
