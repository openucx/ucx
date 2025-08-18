/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucp/dt/dt_contig.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/stubs.h>

#include <ucp/core/ucp_rkey.inl>
#include <ucp/proto/proto_common.inl>
#include <ucp/api/cuda/ucp_def.h>


#define UCP_RMA_CHECK_BUFFER(_buffer, _action) \
    do { \
        if (ENABLE_PARAMS_CHECK && ucs_unlikely((_buffer) == NULL)) { \
            _action; \
        } \
    } while (0)


#define UCP_RMA_CHECK_ZERO_LENGTH(_length, _action) \
    do { \
        if ((_length) == 0) { \
            _action; \
        } \
    } while (0)


#define UCP_RMA_CHECK(_context, _buffer, _length) \
    do { \
        UCP_CONTEXT_CHECK_FEATURE_FLAGS(_context, UCP_FEATURE_RMA, \
                                        return UCS_ERR_INVALID_PARAM); \
        UCP_RMA_CHECK_ZERO_LENGTH(_length, return UCS_OK); \
        UCP_RMA_CHECK_BUFFER(_buffer, return UCS_ERR_INVALID_PARAM); \
    } while (0)


#define UCP_RMA_CHECK_PTR(_context, _buffer, _length) \
    do { \
        UCP_CONTEXT_CHECK_FEATURE_FLAGS(_context, UCP_FEATURE_RMA, \
                                        return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM)); \
        UCP_RMA_CHECK_ZERO_LENGTH(_length, return NULL); \
        UCP_RMA_CHECK_BUFFER(_buffer, \
                             return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM)); \
    } while (0)


/* request can be released if
 *  - all fragments were sent (length == 0) (bcopy & zcopy mix)
 *  - all zcopy fragments are done (uct_comp.count == 0)
 *  - and request was allocated from the mpool
 *    (checked in ucp_request_complete_send)
 *
 * Request can be released either immediately or in the completion callback.
 * We must check req length in the completion callback to avoid the following
 * scenario:
 *  partial_send;no_resos;progress;
 *  send_completed;cb called;req free(ooops);
 *  next_partial_send; (oops req already freed)
 */
ucs_status_t ucp_rma_request_advance(ucp_request_t *req, ssize_t frag_length,
                                     ucs_status_t status,
                                     ucs_ptr_map_key_t req_id)
{
    ucs_assert(status != UCS_ERR_NOT_IMPLEMENTED);

    ucp_request_send_state_advance(req, NULL, UCP_REQUEST_SEND_PROTO_RMA,
                                   status);

    if (ucs_unlikely(UCS_STATUS_IS_ERR(status))) {
        if (status == UCS_ERR_NO_RESOURCE) {
            return UCS_ERR_NO_RESOURCE;
        }

        return UCS_OK;
    }

    ucs_assert(frag_length >= 0);
    ucs_assert(req->send.length >= frag_length);
    req->send.length -= frag_length;
    if (req->send.length == 0) {
        /* bcopy is the fast path */
        ucp_send_request_invoke_uct_completion(req);
        return UCS_OK;
    }
    req->send.buffer           = UCS_PTR_BYTE_OFFSET(req->send.buffer, frag_length);
    req->send.rma.remote_addr += frag_length;
    return UCS_INPROGRESS;
}

static void ucp_rma_request_bcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    if (ucs_likely(req->send.length == req->send.state.dt.offset)) {
        ucp_send_request_id_release(req);
        ucp_request_complete_send(req, self->status);
    }
}

static void ucp_rma_request_zcopy_completion(uct_completion_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t,
                                          send.state.uct_comp);

    if (ucs_likely(req->send.length == req->send.state.dt.offset)) {
        ucp_send_request_id_release(req);
        ucp_request_send_buffer_dereg(req);
        ucp_request_complete_send(req, self->status);
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_rma_request_init(ucp_request_t *req, ucp_ep_h ep, const void *buffer,
                     size_t length, uint64_t remote_addr, ucp_rkey_h rkey,
                     uct_pending_callback_t cb, size_t zcopy_thresh,
                     const ucp_request_param_t *param)
{
    ucp_context_h context = ep->worker->context;
    ucs_status_t status;

    req->flags                = 0;
    req->send.ep              = ep;
    req->send.buffer          = (void*)buffer;
    req->send.datatype        = ucp_dt_make_contig(1);
    req->send.mem_type        = ucp_request_get_memory_type(
                                    context, buffer, length,
                                    ucp_dt_make_contig(1), length, param);
    req->send.length          = length;
    req->send.rma.remote_addr = remote_addr;
    req->send.rma.rkey        = rkey;
    req->send.uct.func        = cb;
    req->send.lane            = rkey->cache.rma_lane;
    ucp_request_send_state_init(req, ucp_dt_make_contig(1), length);
    ucp_request_send_state_reset(req,
                                 (length < zcopy_thresh) ?
                                 ucp_rma_request_bcopy_completion :
                                 ucp_rma_request_zcopy_completion,
                                 UCP_REQUEST_SEND_PROTO_RMA);
#if UCS_ENABLE_ASSERT
    req->send.cb              = NULL;
#endif
    if (length < zcopy_thresh) {
        return UCS_OK;
    }

    status = ucp_send_request_set_user_memh(req,
                                            ucp_ep_config(ep)->key.rma_md_map,
                                            param);
    if (status != UCS_OK) {
        return status;
    }

    return ucp_request_send_reg_lane(req, req->send.lane);
}

static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_rma_nonblocking(ucp_ep_h ep, const void *buffer, size_t length,
                    uint64_t remote_addr, ucp_rkey_h rkey,
                    uct_pending_callback_t progress_cb, size_t zcopy_thresh,
                    const ucp_request_param_t *param)
{
    ucs_status_t status;
    ucp_request_t *req;

    req = ucp_request_get_param(ep->worker, param,
                                {return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);});

    status = ucp_rma_request_init(req, ep, buffer, length, remote_addr, rkey,
                                  progress_cb, zcopy_thresh, param);
    if (ucs_unlikely(status != UCS_OK)) {
        return UCS_STATUS_PTR(status);
    }

    return ucp_rma_send_request(req, param);
}

ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_ptr_t status_ptr;

    status_ptr = ucp_put_nbx(ep, buffer, length, remote_addr, rkey,
                             &ucp_request_null_param);
    if (UCS_PTR_IS_PTR(status_ptr)) {
        ucp_request_free(status_ptr);
        return UCS_INPROGRESS;
    }

    /* coverity[overflow] */
    return UCS_PTR_STATUS(status_ptr);
}

ucs_status_ptr_t ucp_put_nb(ucp_ep_h ep, const void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb
    };

    return ucp_put_nbx(ep, buffer, length, remote_addr, rkey, &param);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_put_send_short(ucp_ep_h ep, const void *buffer, size_t length,
                   uint64_t remote_addr, ucp_rkey_h rkey,
                   const ucp_request_param_t *param)
{
    const ucp_rkey_config_t *rkey_config;
    uct_rkey_t tl_rkey;
    ucs_status_t status;

    if (ucs_unlikely(param->op_attr_mask & (UCP_OP_ATTR_FIELD_DATATYPE |
                                            UCP_OP_ATTR_FLAG_NO_IMM_CMPL))) {
        return UCS_ERR_NO_RESOURCE;
    }

    rkey_config = ucp_rkey_config(ep->worker, rkey);
    if (ucs_unlikely(!ucp_proto_select_is_short(ep, &rkey_config->put_short,
                                                length))) {
        return UCS_ERR_NO_RESOURCE;
    }

    tl_rkey = ucp_rkey_get_tl_rkey(rkey, rkey_config->put_short.rkey_index);

    if (ucs_unlikely(ucp_ep_rma_is_fence_required(ep))) {
        /* TODO: check support for fence in fast path short */
        return UCS_ERR_NO_RESOURCE;
    }

    status = UCS_PROFILE_CALL(uct_ep_put_short,
                              ucp_ep_get_fast_lane(ep,
                                                   rkey_config->put_short.lane),
                              buffer, length, remote_addr, tl_rkey);
    if (status == UCS_OK) {
        ep->ext->unflushed_lanes |= UCS_BIT(rkey_config->put_short.lane);
    }

    return status;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucp_ep_rma_batch_prepare,
                 (ep, iov, iovcnt, signal_va, signal_rkey, param), ucp_ep_h ep,
                 const ucp_rma_iov_t *iov, size_t iovcnt, uint64_t signal_va,
                 ucp_rkey_h signal_rkey, ucp_ep_prepare_batch_param_t *param)
{
    ucp_request_t *req;

    if (!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
        return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
    }

    req = ucp_request_get(ep->worker);
    if (ucs_unlikely((req == NULL))) {
        return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
    }

    req->send.batch.iovcnt = 0;
    req->send.batch.iov    = NULL;
    if (iovcnt != 0) {
        if (iov == NULL) {
            goto invalid_param;
        }

        req->send.batch.iovcnt = iovcnt;
        req->send.batch.iov = ucs_malloc(iovcnt * sizeof(*req->send.batch.iov),
                                         "request send batch");
        if (req->send.batch.iov == NULL) {
            ucp_request_put(req);
            return UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
        }

        memcpy(req->send.batch.iov, iov, sizeof(*iov) * iovcnt);
    } else if (signal_va == 0) {
        goto invalid_param;
    }

    req->send.batch.signal_va   = 0;
    req->send.batch.signal_rkey = NULL;
    if (signal_va != 0) {
        if (signal_rkey == NULL) {
            goto invalid_param;
        }

        req->send.batch.signal_va   = signal_va;
        req->send.batch.signal_rkey = signal_rkey;
    }

    req->flags         = UCP_REQUEST_FLAG_COMPLETED | UCP_REQUEST_FLAG_BATCH;
    req->send.batch.ep = ep;
    req->send.batch.exported = 0;

    return req + 1;

invalid_param:
    ucp_request_put(req);
    return UCS_STATUS_PTR(UCS_ERR_INVALID_PARAM);
}

static ucs_status_t ucp_ep_rma_batch_populate(const ucp_request_t *req,
                                              ucp_md_index_t md_index,
                                              ucp_md_index_t remote_md_index,
                                              uct_rma_iov_t *iov,
                                              uint64_t *sig_va,
                                              uct_rkey_t *sig_rkey)
{
    ucp_rma_iov_t *rma_iov = req->send.batch.iov;
    ucp_mem_h ucp_memh;
    uct_mem_h memh;
    int i;
    uint8_t rkey_index;
    uct_rkey_t uct_rkey;

    /* Prepare signal area */
    if (req->send.batch.signal_va != 0) {
        rkey_index = ucs_bitmap2idx(req->send.batch.signal_rkey->md_map,
                                    remote_md_index);
        uct_rkey   = ucp_rkey_get_tl_rkey(req->send.batch.signal_rkey,
                                          rkey_index);
        if (uct_rkey == UCT_INVALID_RKEY) {
            return UCS_ERR_INVALID_PARAM;
        }

        *sig_va   = req->send.batch.signal_va;
        *sig_rkey = uct_rkey;
    } else {
        *sig_va   = 0;
        *sig_rkey = 0;
    }

    for (i = 0; i < req->send.batch.iovcnt; i++) {
        /* Local registration */
        ucp_memh = rma_iov[i].memh;

        if (!(ucp_memh->md_map & UCS_BIT(md_index))) {
            return UCS_ERR_INVALID_PARAM;
        }

        memh = ucp_memh->uct[md_index];
        if (memh == UCT_MEM_HANDLE_NULL) {
            return UCS_ERR_INVALID_PARAM;
        }

        /* Remote registration */
        rkey_index = ucs_bitmap2idx(rma_iov[i].rkey->md_map, remote_md_index);
        uct_rkey   = ucp_rkey_get_tl_rkey(rma_iov[i].rkey, rkey_index);
        if (uct_rkey == UCT_INVALID_RKEY) {
            return UCS_ERR_INVALID_PARAM;
        }

        iov[i].local_va  = rma_iov[i].local_va;
        iov[i].remote_va = rma_iov[i].remote_va;
        iov[i].length    = rma_iov[i].length;
        iov[i].rkey      = uct_rkey;
        iov[i].memh      = memh;

        ucs_trace("batch populate: i=%d va=%p rva=%lx length=%zu rkey=%lx "
                  "memh=%p ucp_memh->md_map=%lx",
                  i, iov[i].local_va, iov[i].remote_va, iov[i].length,
                  iov[i].rkey, iov[i].memh, ucp_memh->md_map);
    }

    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_rma_batch_release,
                 (request, batch), void *request, ucp_batch_h batch)
{
    ucp_request_t *req = (ucp_request_t *)request - 1;
    ucp_ep_h ep        = req->send.batch.ep;
    uct_ep_h uct_ep;
    ucp_batch_t host_batch;

    if (!(req->flags & UCP_REQUEST_FLAG_BATCH) ||
        (req->send.batch.exported < 1)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if (batch == NULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    ucp_mem_type_pack(ep->worker, &host_batch, batch, sizeof(host_batch),
                      UCS_MEMORY_TYPE_CUDA);

    req->send.batch.exported--;
    uct_ep = host_batch.host.uct_ep;
    uct_ep_batch_release(uct_ep, host_batch.uct_batch);

    uct_mem_free(&host_batch.host.mem);
    return UCS_OK;
}

static ucs_status_t ucp_ep_rma_batch_create(ucp_worker_h worker,
                                            uct_ep_h uct_ep,
                                            uct_batch_h uct_batch,
                                            ucp_batch_h *batch,
                                            const uct_allocated_memory_t *mem)
{
    ucp_batch_t host_batch;
    ucs_status_t status;

    host_batch.host.uct_ep = uct_ep;
    host_batch.host.mem    = *mem;
    host_batch.uct_batch   = uct_batch;

    status = uct_ep_export_dev(uct_ep, &host_batch.exported_uct_ep);
    if (status != UCS_OK) {
        ucs_error("failed to export uct_ep");
        uct_mem_free(&host_batch.host.mem);
        return status;
    }

    if (!host_batch.exported_uct_ep) {
        ucs_error("exported_uct_ep is NULL");
        uct_mem_free(&host_batch.host.mem);
        return UCS_ERR_NO_RESOURCE;
    }

    ucp_mem_type_unpack(worker, mem->address, &host_batch, mem->length,
                        UCS_MEMORY_TYPE_CUDA);

    *batch = mem->address;
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_rma_batch_export, (request, batch),
                 void *request, ucp_batch_h *batch)
{
    ucs_status_t status   = UCS_ERR_NO_RESOURCE;
    ucp_request_t *req    = (ucp_request_t*)request - 1;
    ucp_ep_h ep           = req->send.batch.ep;
    ucp_context_h context = ep->worker->context;
    uct_rma_iov_t *iov    = NULL;
    int found_lane        = 0;
    ucp_md_map_t remote_md_map;
    ucp_rkey_h rkey;
    ucp_lane_index_t lane;
    struct ucp_ep_config *ep_config;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_md_index_t lane_local_md_index, lane_remote_md_index;
    uct_ep_h uct_ep;
    uct_batch_h uct_batch;
    uint64_t sig_va;
    uct_rkey_t sig_rkey;
    ucs_memory_info_t mem_info;
    ucp_rkey_config_t *rkey_config;
    uct_allocated_memory_t mem;
    ucp_rsc_index_t lane_rsc_index;
    ucs_sys_device_t lane_local_sys_dev, local_sys_dev, remote_sys_dev;
    ucs_sys_device_t lane_remote_sys_dev;
    uct_iface_attr_t *iface_attr;
    ucp_tl_resource_desc_t *lane_tl_rsc;

    if (!(req->flags & UCP_REQUEST_FLAG_BATCH)) {
        ucs_error("request is not a batch");
        return UCS_ERR_INVALID_PARAM;
    }

    /*
     * 0. User sets cuda context to desired device before doing export to GPU
     * 1. Allocate ucp batch with UCS_SYS_DEVICE_ID_UNKNOWN sys_dev
     * 2. Detect allocated sys_dev
     * 3. Remote MD index filter
     * 4. Local GPU lanes filter
     * 5. Remote GPU lanes filter
     * 6. Batch support filter
     * 7. Populate uct batch
     * 8. Create ucp batch
     * 9. Export ucp batch
     */

    /* Step 1: Allocate ucp_batch with UCS_SYS_DEVICE_ID_UNKNOWN sys_dev */
    status = ucp_mem_do_alloc(context, NULL, sizeof(**batch),
                              UCT_MD_MEM_ACCESS_LOCAL_READ |
                              UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                              UCS_MEMORY_TYPE_CUDA, UCS_SYS_DEVICE_ID_UNKNOWN,
                              "ucp_batch_t", &mem);
    if (status != UCS_OK) {
        ucs_error("failed to allocate ucp_batch");
        return status;
    }

    /* Step 2: Detect allocated sys_dev */
    ucp_memory_detect_internal(context, mem.address, mem.length, &mem_info);
    if (mem_info.sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_error("detected unknown sys_dev");
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    local_sys_dev = mem_info.sys_dev;
    ucs_trace("detected local_sys_dev %u", local_sys_dev);

    if (req->send.batch.signal_va != 0 && req->send.batch.signal_rkey != NULL) {
        rkey = req->send.batch.signal_rkey;
        if (rkey->mem_type != UCS_MEMORY_TYPE_CUDA) {
            ucs_error("signal rkey is not CUDA");
            status = UCS_ERR_INVALID_PARAM;
            goto err;
        }
    } else if (req->send.batch.iovcnt > 0) {
        rkey = req->send.batch.iov[0].rkey;
    } else {
        ucs_error("ep %p should have iovcnt > 0 or signal_va != 0 and "
                  "signal_rkey != NULL", ep);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    rkey_cfg_index = rkey->cfg_index;
    remote_md_map  = rkey->md_map;

    for (int i = 0; i < req->send.batch.iovcnt; i++) {
        rkey = req->send.batch.iov[i].rkey;
        if ((rkey->mem_type != UCS_MEMORY_TYPE_CUDA) ||
            (rkey->cfg_index != rkey_cfg_index)) {
            ucs_error("invalid rkey, mem_type %u, cfg_index %u",
                      rkey->mem_type, rkey->cfg_index);
            status = UCS_ERR_INVALID_PARAM;
            goto err;
        }

        remote_md_map &= rkey->md_map;
        if (remote_md_map == 0) {
            ucs_error("remote_md_map is 0");
            status = UCS_ERR_INVALID_PARAM;
            goto err;
        }
    }

    ucs_trace_req("ep %p iovcnt %zu remote_md_map %lx search batch lane", ep,
                  req->send.batch.iovcnt, remote_md_map);

    if (req->send.batch.iovcnt != 0) {
        iov = ucs_malloc(req->send.batch.iovcnt * sizeof(*iov),
                         "req batch iov");
        if (iov == NULL) {
            ucs_error("ep %p failed to allocate iov", ep);
            status = UCS_ERR_NO_MEMORY;
            goto err;
        }
    }

    ep_config      = ucp_ep_config(ep);
    rkey_config    = ucp_rkey_config(ep->worker, rkey);
    remote_sys_dev = rkey_config->key.sys_dev;

    if (remote_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_error("ep %p remote_sys_dev is unknown iovcnt %zu", ep,
                  req->send.batch.iovcnt);
        status = UCS_ERR_UNSUPPORTED;
        goto err;
    }

    for (lane = 0; lane < ep_config->key.num_lanes; ++lane) {
        uct_ep               = ucp_ep_get_lane(ep, lane);
        lane_local_md_index  = ep_config->md_index[lane];
        lane_remote_md_index = ep_config->key.lanes[lane].dst_md_index;
        lane_rsc_index       = ep_config->key.lanes[lane].rsc_index;
        lane_remote_sys_dev  = ep_config->key.lanes[lane].dst_sys_dev;
        lane_tl_rsc          = &context->tl_rscs[lane_rsc_index];
        lane_local_sys_dev   = lane_tl_rsc->tl_rsc.sys_device;

        /* Step 3: Remote MD index filter */
        if (!(remote_md_map & UCS_BIT(lane_remote_md_index))) {
            ucs_trace("ep %p skipping lane[%u]: lane_remote_md_index %u not in "
                      "remote_md_map %lx " UCT_TL_RESOURCE_DESC_FMT,
                      ep, lane, lane_remote_md_index, remote_md_map,
                      UCT_TL_RESOURCE_DESC_ARG(&lane_tl_rsc->tl_rsc));
            continue;
        }

        /* Step 4: Local GPU lanes filter */
        if (local_sys_dev != lane_local_sys_dev) {
            ucs_trace("ep %p skipping lane[%u]: local_sys_dev %u != "
                      "lane_local_sys_dev %u " UCT_TL_RESOURCE_DESC_FMT,
                      ep, lane, local_sys_dev, lane_local_sys_dev,
                      UCT_TL_RESOURCE_DESC_ARG(&lane_tl_rsc->tl_rsc));
            continue;
        }

        /* Step 5: Remote GPU lanes filter */
        if (remote_sys_dev != lane_remote_sys_dev) {
            ucs_trace("ep %p skipping lane[%u]: remote_sys_dev %u != "
                      "lane_remote_sys_dev %u",
                      ep, lane, remote_sys_dev, lane_remote_sys_dev);
            continue;
        }

        /* Step 6: Batch support filter */
        iface_attr = ucp_worker_iface_get_attr(ep->worker, lane_rsc_index);
        if (!(iface_attr->cap.flags & UCT_IFACE_FLAG_PUT_BATCH)) {
            ucs_trace("ep %p skipping lane[%u]: doesn't support batch "
                      UCT_TL_RESOURCE_DESC_FMT,
                      ep, lane, UCT_TL_RESOURCE_DESC_ARG(&lane_tl_rsc->tl_rsc));
            continue;
        }

        /* Step 7: Populate uct batch */
        status = ucp_ep_rma_batch_populate(req, lane_local_md_index,
                                           lane_remote_md_index, iov, &sig_va,
                                           &sig_rkey);
        if (status != UCS_OK) {
            ucs_trace("ep %p could not populate uct batch for lane[%u] "
                      UCT_TL_RESOURCE_DESC_FMT,
                      ep, lane, UCT_TL_RESOURCE_DESC_ARG(&lane_tl_rsc->tl_rsc));
            continue;
        }

        /* Step 8: Create ucp batch */
        status = uct_ep_batch_prepare(uct_ep, iov, req->send.batch.iovcnt,
                                      sig_va, sig_rkey, &uct_batch);
        if (status == UCS_OK) {
            ucs_trace("ep %p selected lane[%u] " UCT_TL_RESOURCE_DESC_FMT
                      "md[%u] sys_dev %u -> md[%u] sys_dev %u",
                      ep, lane, UCT_TL_RESOURCE_DESC_ARG(&lane_tl_rsc->tl_rsc),
                      lane_local_md_index, lane_local_sys_dev,
                      lane_remote_md_index, lane_remote_sys_dev);

            /* Step 9: Export ucp batch */
            status = ucp_ep_rma_batch_create(ep->worker, uct_ep, uct_batch,
                                             batch, &mem);
            if (status == UCS_OK) {
                req->send.batch.exported++;
                found_lane = 1;
                break;
            } else {
                ucs_trace("ep %p failed to create batch for lane[%u] "
                          UCT_TL_RESOURCE_DESC_FMT,
                          ep, lane,
                          UCT_TL_RESOURCE_DESC_ARG(&lane_tl_rsc->tl_rsc));
                uct_ep_batch_release(uct_ep, uct_batch);
            }
        }
    }

    if (!found_lane) {
        ucs_error("failed to find a lane");
        status = UCS_ERR_NO_RESOURCE;
    }

    ucs_free(iov);
    return status;

err:
    uct_mem_free(&mem);
    return status;
}

ucs_status_ptr_t ucp_put_nbx(ucp_ep_h ep, const void *buffer, size_t count,
                             uint64_t remote_addr, ucp_rkey_h rkey,
                             const ucp_request_param_t *param)
{
    ucp_worker_h worker     = ep->worker;
    size_t contig_length    = 0;
    ucp_datatype_t datatype = ucp_dt_make_contig(1);
    ucp_ep_rma_config_t *rma_config;
    ucs_status_ptr_t ret;
    ucs_status_t status;
    ucp_request_t *req;

    UCP_REQUEST_CHECK_PARAM(param);
    UCP_RMA_CHECK_PTR(worker->context, buffer, count);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    ucs_trace_req("put_nbx buffer %p count %zu remote_addr %" PRIx64
                  " rkey %p to %s cb %p",
                  buffer, count, remote_addr, rkey, ucp_ep_peer_name(ep),
                  ucp_request_param_send_callback(param));

    if (worker->context->config.ext.proto_enable) {
        status = ucp_put_send_short(ep, buffer, count, remote_addr, rkey, param);
        if (ucs_likely(status != UCS_ERR_NO_RESOURCE) ||
            ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
            ret = UCS_STATUS_PTR(status);
            goto out_unlock;
        }

        req = ucp_request_get_param(worker, param,
                                    {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                    goto out_unlock;});
        req->send.rma.rkey        = rkey;
        req->send.rma.remote_addr = remote_addr;

        if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FIELD_DATATYPE)) {
            datatype = param->datatype;
            if (UCP_DT_IS_CONTIG(datatype)) {
                contig_length = ucp_contig_dt_length(datatype, count);
            }
        } else {
            contig_length = count;
        }

        ret = ucp_proto_request_send_op(
                ep, &ucp_rkey_config(worker, rkey)->proto_select,
                rkey->cfg_index, req, ucp_ep_rma_get_fence_flag(ep),
                UCP_OP_ID_PUT, buffer, count, datatype, contig_length, param, 0,
                0);
    } else {
        status = UCP_RKEY_RESOLVE(rkey, ep, rma);
        if (status != UCS_OK) {
            ret = UCS_STATUS_PTR(status);
            goto out_unlock;
        }

        /* Fast path for a single short message */
        if (ucs_likely(!(param->op_attr_mask & UCP_OP_ATTR_FLAG_NO_IMM_CMPL) &&
                       ((ssize_t)count <= rkey->cache.max_put_short))) {
            status = UCS_PROFILE_CALL(
                    uct_ep_put_short,
                    ucp_ep_get_fast_lane(ep, rkey->cache.rma_lane), buffer,
                    count, remote_addr, rkey->cache.rma_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                ret = UCS_STATUS_PTR(status);
                goto out_unlock;
            }
        }

        if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
            ret = UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
            goto out_unlock;
        }

        rma_config = &ucp_ep_config(ep)->rma[rkey->cache.rma_lane];
        ret = ucp_rma_nonblocking(ep, buffer, count, remote_addr, rkey,
                                  UCP_RKEY_RMA_PROTO(rkey->cache.rma_proto_index)->progress_put,
                                  rma_config->put_zcopy_thresh, param);
    }

out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_ptr_t status_ptr;

    status_ptr = ucp_get_nbx(ep, buffer, length, remote_addr, rkey,
                             &ucp_request_null_param);
    if (UCS_PTR_IS_PTR(status_ptr)) {
        ucp_request_free(status_ptr);
        return UCS_INPROGRESS;
    }

    /* coverity[overflow] */
    return UCS_PTR_STATUS(status_ptr);
}

ucs_status_ptr_t ucp_get_nb(ucp_ep_h ep, void *buffer, size_t length,
                            uint64_t remote_addr, ucp_rkey_h rkey,
                            ucp_send_callback_t cb)
{
    ucp_request_param_t param = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = (ucp_send_nbx_callback_t)cb
    };

    return ucp_get_nbx(ep, buffer, length, remote_addr, rkey, &param);
}

ucs_status_ptr_t ucp_get_nbx(ucp_ep_h ep, void *buffer, size_t count,
                             uint64_t remote_addr, ucp_rkey_h rkey,
                             const ucp_request_param_t *param)
{
    ucp_worker_h worker  = ep->worker;
    size_t contig_length = 0;
    ucp_ep_rma_config_t *rma_config;
    ucs_status_ptr_t ret;
    ucs_status_t status;
    ucp_request_t *req;
    uintptr_t datatype;

    if (ucs_unlikely(param->op_attr_mask & UCP_OP_ATTR_FLAG_FORCE_IMM_CMPL)) {
        return UCS_STATUS_PTR(UCS_ERR_NO_RESOURCE);
    }

    UCP_REQUEST_CHECK_PARAM(param);
    UCP_RMA_CHECK_PTR(worker->context, buffer, count);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    ucs_trace_req("get_nbx buffer %p count %zu remote_addr %" PRIx64
                  " rkey %p from %s cb %p",
                  buffer, count, remote_addr, rkey, ucp_ep_peer_name(ep),
                  ucp_request_param_send_callback(param));

    if (worker->context->config.ext.proto_enable) {
        datatype = ucp_request_param_datatype(param);
        req = ucp_request_get_param(worker, param,
                                    {ret = UCS_STATUS_PTR(UCS_ERR_NO_MEMORY);
                                    goto out_unlock;});

        req->send.rma.rkey             = rkey;
        req->send.rma.remote_addr      = remote_addr;
        req->send.state.completed_size = 0;
        if (UCP_DT_IS_CONTIG(datatype)) {
            contig_length = ucp_contig_dt_length(datatype, count);
        }

        ret = ucp_proto_request_send_op(
                ep, &ucp_rkey_config(worker, rkey)->proto_select,
                rkey->cfg_index, req, ucp_ep_rma_get_fence_flag(ep),
                UCP_OP_ID_GET, buffer, count, datatype, contig_length, param, 0,
                0);
    } else {
        status = UCP_RKEY_RESOLVE(rkey, ep, rma);
        if (status != UCS_OK) {
            ret = UCS_STATUS_PTR(status);
            goto out_unlock;
        }

        rma_config = &ucp_ep_config(ep)->rma[rkey->cache.rma_lane];
        ret        = ucp_rma_nonblocking(ep, buffer, count, remote_addr, rkey,
                                         UCP_RKEY_RMA_PROTO(rkey->cache.rma_proto_index)->progress_get,
                                         rma_config->get_zcopy_thresh, param);
    }

out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_put, (ep, buffer, length, remote_addr, rkey),
                 ucp_ep_h ep, const void *buffer, size_t length,
                 uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_rma_wait(ep->worker,
                        ucp_put_nb(ep, buffer, length, remote_addr, rkey,
                                   (ucp_send_callback_t)ucs_empty_function),
                        "put");
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_get, (ep, buffer, length, remote_addr, rkey),
                 ucp_ep_h ep, void *buffer, size_t length,
                 uint64_t remote_addr, ucp_rkey_h rkey)
{
    return ucp_rma_wait(ep->worker,
                        ucp_get_nb(ep, buffer, length, remote_addr, rkey,
                                   (ucp_send_callback_t)ucs_empty_function),
                        "get");
}
