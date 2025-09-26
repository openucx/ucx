/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_context.h"
#include "ucp_worker.h"
#include "ucp_request.inl"
#include "ucp_mm.inl"

#include <ucp/proto/proto_am.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/tag/tag_rndv.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/debug/debug_int.h>
#include <ucs/debug/log.h>


const ucp_request_param_t ucp_request_null_param = { .op_attr_mask = 0 };

static const char *ucp_request_flag_names[] = {
    [ucs_ilog2(UCP_REQUEST_FLAG_COMPLETED)]             = "cpml",
    [ucs_ilog2(UCP_REQUEST_FLAG_RELEASED)]              = "rls",
    [ucs_ilog2(UCP_REQUEST_FLAG_PROTO_SEND)]            = "proto",
    [ucs_ilog2(UCP_REQUEST_FLAG_SYNC_LOCAL_COMPLETED)]  = "loc_cmpl",
    [ucs_ilog2(UCP_REQUEST_FLAG_SYNC_REMOTE_COMPLETED)] = "rm_cmpl",
    [ucs_ilog2(UCP_REQUEST_FLAG_CALLBACK)]              = "cb",
    [ucs_ilog2(UCP_REQUEST_FLAG_PROTO_INITIALIZED)]     = "init",
    [ucs_ilog2(UCP_REQUEST_FLAG_SYNC)]                  = "sync",
    [ucs_ilog2(UCP_REQUEST_FLAG_PROTO_AMO_PACKED)]      = "amo_pack",
    [ucs_ilog2(UCP_REQUEST_FLAG_OFFLOADED)]             = "offld",
    [ucs_ilog2(UCP_REQUEST_FLAG_BLOCK_OFFLOAD)]         = "blk_offld",
    [ucs_ilog2(UCP_REQUEST_FLAG_STREAM_RECV_WAITALL)]   = "strm_r_wtall",
    [ucs_ilog2(UCP_REQUEST_FLAG_SEND_AM)]               = "snd_am",
    [ucs_ilog2(UCP_REQUEST_FLAG_SEND_TAG)]              = "snd_tag",
    [ucs_ilog2(UCP_REQUEST_FLAG_RNDV_FRAG)]             = "rndv_fr",
    [ucs_ilog2(UCP_REQUEST_FLAG_RECV_AM)]               = "rcv_am",
    [ucs_ilog2(UCP_REQUEST_FLAG_RECV_TAG)]              = "rcv_tag",
    [ucs_ilog2(UCP_REQUEST_FLAG_RKEY_INUSE)]            = "rk_use",
    [ucs_ilog2(UCP_REQUEST_FLAG_USER_HEADER_COPIED)]    = "hdr_copy",

#if UCS_ENABLE_ASSERT
    [ucs_ilog2(UCP_REQUEST_FLAG_STREAM_RECV)]           = "strm_rcv",
    [ucs_ilog2(UCP_REQUEST_DEBUG_FLAG_EXTERNAL)]        = "extrn",
    [ucs_ilog2(UCP_REQUEST_FLAG_SUPER_VALID)]           = "spr_vld",
#endif
};

static ucs_memory_type_t ucp_request_get_mem_type(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_SEND) {
        return req->send.state.dt_iter.mem_info.type;
    } else if (req->flags & (UCP_REQUEST_FLAG_SEND_AM | UCP_REQUEST_FLAG_SEND_TAG)) {
        return req->send.mem_type;
    } else if (req->flags &
               (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG)) {
        return req->recv.dt_iter.mem_info.type;
    } else {
        return UCS_MEMORY_TYPE_UNKNOWN;
    }
}

static void
ucp_request_str(ucp_request_t *req, ucp_worker_h worker,
                ucs_string_buffer_t *strb, int recurse)
{
    const char *progress_func_name;
    const char *comp_func_name;
    ucp_ep_config_t *config;
    ucp_ep_h ep;

    ucs_string_buffer_appendf(strb, "{");
    ucs_string_buffer_append_flags(strb, req->flags, ucp_request_flag_names);
    ucs_string_buffer_appendf(strb, "} ");

    if (req->flags & UCP_REQUEST_FLAG_PROTO_SEND) {
        ucp_proto_config_info_str(worker, req->send.proto_config,
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
            ucp_ep_config_lane_info_str(worker, &config->key, NULL,
                                        req->send.lane, UCP_NULL_RESOURCE,
                                        strb);
        }
    } else if (req->flags &
               (UCP_REQUEST_FLAG_RECV_AM | UCP_REQUEST_FLAG_RECV_TAG)) {
#if ENABLE_DEBUG_DATA
        if (req->recv.proto_rndv_config != NULL) {
            /* Print the send protocol of the rendezvous request */
            ucp_proto_config_info_str(worker, req->recv.proto_rndv_config,
                                      req->recv.dt_iter.length, strb);
            return;
        }
#endif
        ucs_string_buffer_appendf(strb, "recv length %zu ",
                                  req->recv.dt_iter.length);
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
    ucp_request_t *req  = (ucp_request_t*)request - 1;
    ucp_worker_h worker = ucs_container_of(ucs_mpool_obj_owner(req),
                                           ucp_worker_t, req_mp);

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
        ucp_request_str(req, worker, &strb, 1);
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
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, req_mp);
    ucp_request_t *req  = obj;

    ucp_request_str(req, worker, strb, 0);
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

    uct_ep = ucp_ep_get_lane(req->send.ep, req->send.lane);
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

static unsigned ucp_request_memh_invalidate_progress(void *arg)
{
    ucp_request_t *req = arg;

    ucp_request_complete_send(req, req->status);
    return 1;
}

static void ucp_request_mem_invalidate_completion(void *arg)
{
    ucp_request_t *req  = arg;
    ucp_worker_h worker = req->send.invalidate.worker;

    ucs_callbackq_add_oneshot(&worker->uct->progress_q, worker,
                              ucp_request_memh_invalidate_progress, req);
}

static void
ucp_request_dt_dereg(ucp_mem_h *memhs, size_t count, ucp_request_t *req_dbg)
{
    size_t i;

    for (i = 0; i < count; ++i) {
        ucp_trace_req(req_dbg, "mem dereg buffer %ld/%ld md_map 0x%" PRIx64, i,
                      count, memhs[i]->md_map);
        ucp_memh_put(memhs[i]);
    }
}

static ucp_md_map_t ucp_request_get_invalidation_map(ucp_ep_h ep)
{
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
            ucs_assert(ucp_ep_md_attr(ep, lane)->flags &
                       UCT_MD_FLAG_INVALIDATE_RMA);
            inv_map |= UCS_BIT(ucp_ep_md_index(ep, lane));
        }
    }

    return inv_map;
}

int ucp_request_memh_invalidate(ucp_request_t *req, ucs_status_t status)
{
    ucp_ep_h ep                      = req->send.ep;
    ucp_err_handling_mode_t err_mode = ucp_ep_config(ep)->key.err_mode;
    ucp_worker_h worker              = ep->worker;
    ucp_context_h context            = worker->context;
    ucp_mem_h *memh_p;
    ucp_md_map_t invalidate_map;

    if ((err_mode != UCP_ERR_HANDLING_MODE_PEER) ||
        !(req->flags & UCP_REQUEST_FLAG_RKEY_INUSE)) {
        return 0;
    }

    /* Get the contig memh from the request basing on the proto version */
    if (context->config.ext.proto_enable) {
        ucs_assertv(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG,
                    "dt_class=%s",
                    ucp_datatype_class_names[req->send.state.dt_iter.dt_class]);
        memh_p = &req->send.state.dt_iter.type.contig.memh;
    } else {
        ucs_assertv(UCP_DT_IS_CONTIG(req->send.datatype), "datatype=0x%" PRIx64,
                    req->send.datatype);
        memh_p = &req->send.state.dt.dt.contig.memh;
    }

    if ((*memh_p == NULL) || ucp_memh_is_user_memh(*memh_p)) {
        return 0;
    }

    ucs_assert(status != UCS_OK);

    req->send.invalidate.worker = worker;
    req->status                 = status;

    invalidate_map = ucp_request_get_invalidation_map(ep);
    ucp_trace_req(req, "mem invalidate buffer md_map 0x%" PRIx64 "/0x%" PRIx64,
                  invalidate_map, (*memh_p)->md_map);
    ucp_memh_invalidate(context, *memh_p, ucp_request_mem_invalidate_completion,
                        req, invalidate_map);

    ucp_memh_put(*memh_p);
    *memh_p = NULL;
    return 1;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_request_memory_reg,
                 (context, md_map, buffer, length, datatype, state, mem_type,
                  req, uct_flags),
                 ucp_context_t *context, ucp_md_map_t md_map, void *buffer,
                 size_t length, ucp_datatype_t datatype, ucp_dt_state_t *state,
                 ucs_memory_type_t mem_type, ucp_request_t *req,
                 unsigned uct_flags)
{
    ucp_md_map_t reg_md_map = md_map & context->reg_md_map[mem_type];
    int flags               = UCT_MD_MEM_ACCESS_RMA | uct_flags;
    ucs_status_t status     = UCS_OK;
    size_t iov_it, iovcnt;
    const ucp_dt_iov_t *iov;
    ucp_mem_h *memhs;
    int level;

    ucs_trace_func("context=%p md_map=0x%"PRIx64" buffer=%p length=%zu "
                   "datatype=0x%"PRIx64" state=%p", context, md_map, buffer,
                   length, datatype, state);

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        status = ucp_memh_get_or_update(context, buffer, length, mem_type,
                                        reg_md_map, flags,
                                        &state->dt.contig.memh);
        if (status != UCS_OK) {
            goto err;
        }
        ucp_trace_req(req, "mem reg md_map 0x%" PRIx64 "/0x%" PRIx64,
                      state->dt.contig.memh->md_map, reg_md_map);
        break;

    case UCP_DATATYPE_IOV:
        ucs_assert(!(flags & UCT_MD_MEM_FLAG_HIDE_ERRORS));
        iovcnt = state->dt.iov.iovcnt;
        iov    = buffer;
        if (ucs_unlikely(state->dt.iov.memhs != NULL)) {
            memhs = state->dt.iov.memhs;
        } else {
            memhs = ucs_calloc(iovcnt, sizeof(*memhs), "iov_memhs");
            if (NULL == memhs) {
                status = UCS_ERR_NO_MEMORY;
                goto err;
            }
        }

        for (iov_it = 0; iov_it < iovcnt; ++iov_it) {
            status = ucp_memh_get_or_update(context, iov[iov_it].buffer,
                                            iov[iov_it].length, mem_type,
                                            reg_md_map, flags, &memhs[iov_it]);
            if (status != UCS_OK) {
                /* unregister previously registered memory */
                /* coverity[check_after_deref] */
                if (state->dt.iov.memhs == NULL) {
                    ucp_request_dt_dereg(memhs, iov_it, req);
                } else {
                    ucp_request_dt_dereg(memhs, iovcnt, req);
                    state->dt.iov.memhs = NULL;
                }
                ucs_free(memhs);
                goto err;
            }

            ucp_trace_req(req,
                          "mem reg iov %ld/%ld md_map 0x%" PRIx64 "/0x%" PRIx64,
                          iov_it, iovcnt, memhs[iov_it]->md_map, reg_md_map);
        }
        state->dt.iov.memhs = memhs;
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

UCS_PROFILE_FUNC_VOID(ucp_request_memory_dereg, (datatype, state, req),
                      ucp_datatype_t datatype, ucp_dt_state_t *state,
                      ucp_request_t *req)
{
    ucs_trace_func("datatype=0x%" PRIx64 " state=%p", datatype, state);

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if (state->dt.contig.memh != NULL) {
            ucp_request_dt_dereg(&state->dt.contig.memh, 1, req);
            state->dt.contig.memh = NULL;
        }
        break;
    case UCP_DATATYPE_IOV:
        if (state->dt.iov.memhs != NULL) {
            ucp_request_dt_dereg(state->dt.iov.memhs, state->dt.iov.iovcnt,
                                 req);
            ucs_free(state->dt.iov.memhs);
            state->dt.iov.memhs = NULL;
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

        status = ucp_request_send_reg_lane(req, req->send.lane);
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
    ucs_debug("message truncated: recv_length %zu offset %zu buffer_size %zu",
              length, offset, req->recv.dt_iter.length);

    ucp_datatype_iter_cleanup(&req->recv.dt_iter, 0, UCP_DT_MASK_ALL);
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

    ucp_worker_track_ep_usage(req);

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
