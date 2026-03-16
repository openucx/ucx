/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCP_RNDV_MTYPE_INL_
#define UCP_RNDV_MTYPE_INL_

#include "proto_rndv.inl"
#include "rndv.h"

#include <ucp/core/ucp_worker.h>


static ucp_ep_h ucp_proto_rndv_mtype_ep(ucp_worker_t *worker,
                                        ucs_memory_type_t frag_mem_type,
                                        ucs_memory_type_t buf_mem_type)
{
    if (worker->mem_type_ep[buf_mem_type] != NULL) {
        return worker->mem_type_ep[buf_mem_type];
    }

    return worker->mem_type_ep[frag_mem_type];
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_mtype_init(const ucp_proto_init_params_t *init_params,
                          ucs_memory_type_t frag_mem_type,
                          ucp_md_map_t *mdesc_md_map_p, size_t *frag_size_p)
{
    ucp_worker_h worker        = init_params->worker;
    ucp_context_h context      = worker->context;
    ucs_memory_type_t mem_type = init_params->select_param->mem_type;

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (ucp_proto_rndv_mtype_ep(worker, frag_mem_type, mem_type) == NULL) ||
        !init_params->worker->context->config.ext.memtype_copy_enable ||
        !ucp_proto_init_check_op(init_params, UCP_PROTO_RNDV_OP_ID_MASK)) {
        return UCS_ERR_UNSUPPORTED;
    }

    *mdesc_md_map_p = context->reg_md_map[frag_mem_type];
    *frag_size_p    = context->config.ext.rndv_frag_size[frag_mem_type];

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_mtype_request_init(ucp_request_t *req,
                                  ucs_memory_type_t frag_mem_type,
                                  ucs_sys_device_t frag_sys_dev,
                                  const size_t max_frags,
                                  ucs_queue_head_t *pending_q)
{
    ucp_worker_h worker = req->send.ep->worker;

    /* Check throttling limit. If no resource at the moment, queue the
     * request in RTR pending queue and return NO_RESOURCE. */
    if (worker->rndv_mtype_fc.active_frags >= max_frags) {
        ucs_trace_req("mtype_fc: fragments throttle limit reached (%zu/%zu)",
                      worker->rndv_mtype_fc.active_frags, max_frags);
        UCS_STATS_UPDATE_COUNTER(worker->stats,
                                 UCP_WORKER_STAT_RNDV_MTYPE_FC_THROTTLED, 1);
        ucs_queue_push(pending_q, &req->send.rndv.ppln.queue_elem);
        return UCS_ERR_NO_RESOURCE;
    }

    req->send.rndv.mdesc = ucp_rndv_mpool_get(worker, frag_mem_type,
                                              frag_sys_dev);
    if (req->send.rndv.mdesc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_proto_rndv_mtype_get_memh(ucp_request_t *req, ucp_md_index_t md_index)
{
    ucp_mem_desc_t *mdesc = req->send.rndv.mdesc;

    if (md_index == UCP_NULL_RESOURCE) {
        return UCT_MEM_HANDLE_NULL;
    }

    ucs_assertv(UCS_BIT(md_index) & mdesc->memh->md_map,
                "md_index=%d md_map=0x%" PRIx64, md_index,
                mdesc->memh->md_map);
    return mdesc->memh->uct[md_index];
}

static ucp_ep_h ucp_proto_rndv_req_mtype_ep(ucp_request_t *req)
{
    ucp_ep_h mem_type_ep;

    mem_type_ep = ucp_proto_rndv_mtype_ep(req->send.ep->worker,
                                          req->send.rndv.mdesc->memh->mem_type,
                                          req->send.state.dt_iter.mem_info.type);
    ucs_assert(mem_type_ep != NULL);

    return mem_type_ep;
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_proto_rndv_mtype_get_req_memh(ucp_request_t *req)
{
    ucp_ep_h mtype_ep          = ucp_proto_rndv_req_mtype_ep(req);
    ucp_lane_index_t lane      = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    ucp_md_index_t md_index    = ucp_ep_md_index(mtype_ep, lane);

    return ucp_proto_rndv_mtype_get_memh(req, md_index);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_iov_init(ucp_request_t *req, void *buffer,
                              size_t length, size_t offset, uct_mem_h memh,
                              uct_iov_t *iov)
{
    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);

    iov->length = length;
    iov->buffer = UCS_PTR_BYTE_OFFSET(buffer, offset);
    iov->memh   = memh;
    iov->stride = 0;
    iov->count  = 1;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_next_iov(ucp_request_t *req,
                              const ucp_proto_rndv_bulk_priv_t *rpriv,
                              const ucp_proto_multi_lane_priv_t *lpriv,
                              ucp_datatype_iter_t *next_iter, uct_iov_t *iov)
{
    size_t max_payload = ucp_proto_rndv_bulk_max_payload(req, rpriv, lpriv);
    size_t length      = ucp_datatype_iter_next(&req->send.state.dt_iter,
                                                max_payload, next_iter);
    uct_mem_h memh     = ucp_proto_rndv_mtype_get_memh(req,
                                                       lpriv->super.md_index);

    ucp_proto_rndv_mtype_iov_init(req, req->send.rndv.mdesc->ptr, length,
                                  req->send.state.dt_iter.offset, memh, iov);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_mtype_copy(
        ucp_request_t *req, void *buffer, uct_mem_h memh,
        ucs_memory_type_t frag_mem_type, uct_ep_put_zcopy_func_t copy_func,
        uct_completion_callback_t comp_func, const char *mode)
{
    ucp_ep_h mtype_ep     = ucp_proto_rndv_mtype_ep(
                              req->send.ep->worker, frag_mem_type,
                              req->send.state.dt_iter.mem_info.type);
    ucp_lane_index_t lane = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    ucp_context_t UCS_V_UNUSED *context = req->send.ep->worker->context;
    ucs_status_t status;
    uct_iov_t iov;

    ucp_trace_req(req, "buffer %p copy-%s %p %s-%s using memtype-ep %p lane[%d]",
                  buffer, mode, req->send.state.dt_iter.type.contig.buffer,
                  ucs_memory_type_names[req->send.state.dt_iter.mem_info.type],
                  ucs_memory_type_names[frag_mem_type],
                  mtype_ep, lane);

    ucp_proto_completion_init(&req->send.state.uct_comp, comp_func);

    /* Set up IOV pointing to the mdesc */
    ucp_proto_rndv_mtype_iov_init(req, buffer, req->send.state.dt_iter.length,
                                  0, memh, &iov);

    /* Copy from mdesc to user buffer */
    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);
    status = copy_func(ucp_ep_get_lane(mtype_ep, lane), &iov, 1,
                       (uintptr_t)req->send.state.dt_iter.type.contig.buffer,
                       UCT_INVALID_RKEY, &req->send.state.uct_comp);
    ucp_trace_req(req, "buffer %p copy returned %s", buffer,
                  ucs_status_string(status));
    ucs_assert(status != UCS_ERR_NO_RESOURCE);

    if (status != UCS_INPROGRESS) {
        ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
    }

    return status;
}

/* Reschedule callback for throttled mtype requests */
static UCS_F_ALWAYS_INLINE
unsigned ucp_proto_rndv_mtype_fc_reschedule_cb(void *arg)
{
    ucp_request_t *req = arg;
    ucp_request_send(req);
    return 1;
}

/*
 * Staging-buffer flow-control throttle limit per operation type.
 *
 * To prevent deadlock under memory pressure, each operation type is capped at a
 * different share of the total fragment budget (fc_max_frags).  Operations that
 * free more resources on completion receive a larger cap, so they can always
 * make progress and unblock others:
 *
 *   PUT (level 0, 100%)            – frees local staging buffer AND remote RTR buffer.
 *   GET (level 1, 100%-tier_step)  – frees local staging buffer only.
 *   RTR (level 2, 100%-2*tier_step) – triggers a remote PUT allocation, adding pressure.
 *
 * The step between tiers is configurable via UCX_RNDV_MTYPE_FC_TIER_STEP
 * (default 10%).  The exact fractions are not performance-sensitive; only the
 * strict ordering PUT > GET > RTR matters.
 * The same ordering is used when dequeueing pending requests, see
 * ucp_proto_rndv_mtype_fc_decrement().
 */
static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_mtype_fc_limit(size_t fc_max, unsigned level,
                              unsigned tier_step)
{
    return fc_max - level * (fc_max * tier_step / 100);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_mtype_fc_put_limit(size_t fc_max, unsigned tier_step)
{
    return ucp_proto_rndv_mtype_fc_limit(fc_max, 0, tier_step);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_mtype_fc_get_limit(size_t fc_max, unsigned tier_step)
{
    return ucp_proto_rndv_mtype_fc_limit(fc_max, 1, tier_step);
}

static UCS_F_ALWAYS_INLINE size_t
ucp_proto_rndv_mtype_fc_rtr_limit(size_t fc_max, unsigned tier_step)
{
    return ucp_proto_rndv_mtype_fc_limit(fc_max, 2, tier_step);
}

/**
 * Decrement active_frags counter and reschedule pending request.
 * Dequeue priority: PUT > GET > RTR (same ordering as the throttle limits
 * defined by ucp_proto_rndv_mtype_fc_limit()).
 *
 * Priority rationale:
 * PUT - Remote is blocked waiting for our data. Scheduling PUT unblocks remote
 *       as well.
 * GET - Self-contained fetch operation. Completes without causing remote
 *       allocations, but scheduling it doesn't unblock another buffer.
 * RTR - Scheduling RTR triggers a remote PUT allocation, increasing total
 *       memory pressure.
 */
static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_fc_decrement(ucp_request_t *req)
{
    ucp_worker_h worker    = req->send.ep->worker;
    ucs_queue_elem_t *elem = NULL;
    ucp_request_t *pending_req;

    ucs_assert(worker->rndv_mtype_fc.active_frags > 0);
    worker->rndv_mtype_fc.active_frags--;

    /* Dequeue with priority: PUT > GET > RTR */
    if (!ucs_queue_is_empty(&worker->rndv_mtype_fc.put_pending_q)) {
        elem = ucs_queue_pull(&worker->rndv_mtype_fc.put_pending_q);
    } else if (!ucs_queue_is_empty(&worker->rndv_mtype_fc.get_pending_q)) {
        elem = ucs_queue_pull(&worker->rndv_mtype_fc.get_pending_q);
    } else if (!ucs_queue_is_empty(&worker->rndv_mtype_fc.rtr_pending_q)) {
        elem = ucs_queue_pull(&worker->rndv_mtype_fc.rtr_pending_q);
    }

    if (elem == NULL) {
        return;
    }

    pending_req = ucs_container_of(elem, ucp_request_t,
                                   send.rndv.ppln.queue_elem);
    ucs_callbackq_add_oneshot(&worker->uct->progress_q, pending_req,
                              ucp_proto_rndv_mtype_fc_reschedule_cb,
                              pending_req);
}

/**
 * Release the staging buffer and decrement the FC active fragments counter.
 * This pairs with ucp_proto_rndv_mtype_request_init() which allocates the
 * mdesc and increments the counter.
 */
static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_mdesc_release(ucp_request_t *req)
{
    ucs_mpool_put_inline(req->send.rndv.mdesc);
    ucp_proto_rndv_mtype_fc_decrement(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_mdesc_mtype_copy(ucp_request_t *req,
                                uct_ep_put_zcopy_func_t copy_func,
                                uct_completion_callback_t comp_func,
                                const char *mode)
{
    ucp_mem_desc_t *mdesc = req->send.rndv.mdesc;

    return ucp_proto_rndv_mtype_copy(
                 req, mdesc->ptr, ucp_proto_rndv_mtype_get_req_memh(req),
                 mdesc->memh->mem_type, copy_func, comp_func, mode);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_query_desc(const ucp_proto_query_params_t *params,
                                ucs_memory_type_t frag_mem_type,
                                ucp_proto_query_attr_t *attr,
                                const char *xfer_desc)
{
    UCS_STRING_BUFFER_FIXED(strb, attr->desc, sizeof(attr->desc));
    ucp_context_h context      = params->worker->context;
    ucs_memory_type_t mem_type = params->select_param->mem_type;
    ucp_ep_h mtype_ep;
    ucp_lane_index_t lane;
    ucp_rsc_index_t rsc_index;
    const char *tl_name;

    /* Make coverity happy */
    ucs_assertv(frag_mem_type < UCS_MEMORY_TYPE_UNKNOWN, "frag_mem_type = %u",
                frag_mem_type);

    mtype_ep  = ucp_proto_rndv_mtype_ep(params->worker, frag_mem_type,
                                        mem_type);
    ucs_assert(mtype_ep != NULL);

    lane      = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    rsc_index = ucp_ep_get_rsc_index(mtype_ep, lane);
    tl_name   = context->tl_rscs[rsc_index].tl_rsc.tl_name;

    if (ucp_proto_select_op_id(params->select_param) == UCP_OP_ID_RNDV_SEND) {
        ucs_string_buffer_appendf(&strb, "%s, ", tl_name);
    }

    ucs_string_buffer_appendf(&strb, "%s", xfer_desc);

    if (ucp_proto_select_op_id(params->select_param) == UCP_OP_ID_RNDV_RECV) {
        ucs_string_buffer_appendf(&strb, ", %s", tl_name);
    }

    ucs_string_buffer_appendf(&strb, ", frag %s",
                              ucs_memory_type_names[frag_mem_type]);
}

#endif
