/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_ipc_ep.h"
#include "cuda_ipc_iface_address.h"
#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"
#include "cuda_ipc.inl"

#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <uct/api/device/uct_device_types.h>
#include <uct/api/uct_def.h>
#include <uct/base/uct_iov.inl>
#include <uct/base/uct_log.h>
#include <uct/cuda/base/cuda_iface.h>

#define UCT_CUDA_IPC_PUT 0
#define UCT_CUDA_IPC_GET 1


static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_ep_t, const uct_ep_params_t *params)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_cuda_ipc_iface_t);
    size_t iface_addr_length;

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    iface_addr_length          = UCT_EP_PARAM_VALUE(params, iface_addr_length,
                                                    IFACE_ADDR_LENGTH, 0);
    self->remote_iface_address = uct_cuda_ipc_iface_address_unpack(
            params->iface_addr, iface_addr_length);
    self->device_ep            = NULL;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_ep_t)
{
    uct_cuda_ipc_destroy_cache_by_iface_address(&self->remote_iface_address);
    if (self->device_ep != NULL) {
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuMemFree((CUdeviceptr)self->device_ep));
    }
}

UCS_CLASS_DEFINE(uct_cuda_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_ep_t, uct_ep_t);

#define uct_cuda_ipc_trace_data(_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_addr), (_rkey))

int uct_cuda_ipc_ep_is_connected(const uct_ep_h tl_ep,
                                 const uct_ep_is_connected_params_t *params)
{
    const uct_cuda_ipc_ep_t *ep = ucs_derived_of(tl_ep, uct_cuda_ipc_ep_t);

    if (!uct_base_ep_is_connected(tl_ep, params)) {
        return 0;
    }

    /* TODO: Enhance the check to account for PID namespace and device address.
             Comparing PIDs alone can yield false positives: PIDs are unique
             only within a PID namespace, so unrelated peers (e.g., on another
             node or in another namespace on the same host) may report the same
             PID number. */
    return ep->remote_iface_address.pid == *(const pid_t*)params->iface_addr;
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_cuda_ipc_ctx_rsc_get(
        uct_cuda_ipc_iface_t *iface, uct_cuda_ipc_ctx_rsc_t **ctx_rsc_p)
{
    unsigned long long ctx_id;
    ucs_status_t status;
    CUresult result;
    uct_cuda_ctx_rsc_t *ctx_rsc;

    result = uct_cuda_ctx_get_id(NULL, &ctx_id);
    if (ucs_unlikely(result != CUDA_SUCCESS)) {
        UCT_CUDADRV_LOG(cuCtxGetId, UCS_LOG_LEVEL_ERROR, result);
        return UCS_ERR_IO_ERROR;
    }

    status = uct_cuda_base_ctx_rsc_get(&iface->super, ctx_id, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *ctx_rsc_p = ucs_derived_of(ctx_rsc, uct_cuda_ipc_ctx_rsc_t);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_get_stream_and_event(uct_cuda_ipc_iface_t *iface,
                                  unsigned stream_id,
                                  uct_cuda_queue_desc_t **q_desc_p,
                                  CUstream **stream_p,
                                  uct_cuda_ipc_event_desc_t **event_p)
{
    uct_cuda_ipc_ctx_rsc_t *ctx_rsc;
    ucs_status_t status;

    status = uct_cuda_ipc_ctx_rsc_get(iface, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *q_desc_p = &ctx_rsc->queue_desc[stream_id % iface->config.max_streams];
    *stream_p = &(*q_desc_p)->stream;

    status = uct_cuda_base_init_stream(*stream_p);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    *event_p = uct_cuda_base_event_desc_mpool_get(&ctx_rsc->super.event_mp);
    if (ucs_unlikely(*event_p == NULL)) {
        ucs_error("Failed to allocate cuda_ipc event object");
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_event_record_and_enqueue(uct_cuda_ipc_iface_t *iface,
                                      uct_cuda_queue_desc_t *q_desc,
                                      uct_cuda_ipc_event_desc_t *event,
                                      CUstream stream)
{
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuEventRecord(event->super.event,
                                                    stream));
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    if (ucs_queue_is_empty(&q_desc->event_queue)) {
        ucs_queue_push(&iface->super.active_queue, &q_desc->queue);
    }

    ucs_queue_push(&q_desc->event_queue, &event->super.queue);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_post_cuda_async_copy(uct_ep_h tl_ep, uint64_t remote_addr,
                                  const uct_iov_t *iov, uct_rkey_t rkey,
                                  uct_completion_t *comp, int direction)
{
    uct_cuda_ipc_iface_t *iface       = ucs_derived_of(tl_ep->iface,
                                                       uct_cuda_ipc_iface_t);
    uct_cuda_ipc_unpacked_rkey_t *key = (uct_cuda_ipc_unpacked_rkey_t *)rkey;
    CUdevice cuda_device;
    int is_ctx_pushed;
    void *mapped_rem_addr;
    const void *mapped_addr;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    uct_cuda_queue_desc_t *q_desc;
    ucs_status_t status;
    CUdeviceptr dst, src;
    CUstream *stream;

    if (ucs_unlikely(0 == iov[0].length)) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)iov[0].buffer,
                                             &cuda_device, &is_ctx_pushed);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    status = uct_cuda_ipc_get_remote_address(&key->super, remote_addr,
                                             cuda_device, &mapped_rem_addr,
                                             &mapped_addr);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    status = uct_cuda_ipc_get_stream_and_event(iface, key->stream_id,
                                               &q_desc, &stream,
                                               &cuda_ipc_event);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    dst = (CUdeviceptr)
        ((direction == UCT_CUDA_IPC_PUT) ? mapped_rem_addr : iov[0].buffer);
    src = (CUdeviceptr)
        ((direction == UCT_CUDA_IPC_PUT) ? iov[0].buffer : mapped_rem_addr);

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemcpyDtoDAsync(dst, src, iov[0].length,
                                                        *stream));
    if (UCS_OK != status) {
        ucs_mpool_put(cuda_ipc_event);
        goto out;
    }

    status = uct_cuda_ipc_event_record_and_enqueue(iface, q_desc,
                                                   cuda_ipc_event, *stream);
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_mpool_put(cuda_ipc_event);
        goto out;
    }

    cuda_ipc_event->super.comp  = comp;
    cuda_ipc_event->mapped_addr = mapped_addr;
    cuda_ipc_event->d_bptr      = (uintptr_t)key->super.super.d_bptr;
    cuda_ipc_event->pid         = key->super.super.pid;
    cuda_ipc_event->pid_ns      = key->super.pid_ns;
    cuda_ipc_event->cuda_device = cuda_device;
#if CUDA_VERSION >= 13000
    cuda_ipc_event->sgl_mapping = NULL;
#endif
    ucs_trace("cuMemcpyDtoDAsync issued :%p dst:%p, src:%p  len:%ld",
             cuda_ipc_event, (void *) dst, (void *) src, iov[0].length);
    status = UCS_INPROGRESS;

out:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_ipc_post_cuda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_CUDA_IPC_GET);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_ep_put_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_ipc_post_cuda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_CUDA_IPC_PUT);
    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_ipc_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                                uct_iov_total_length(iov, iovcnt));
    return status;
}

#if CUDA_VERSION >= 13000
UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_ep_put_sgl_zcopy,
                 (tl_ep, buffers, lengths, memhs, remote_addrs, rkeys, counts,
                  strides, count, comp),
                 uct_ep_h tl_ep, void * const *buffers,
                 const size_t *lengths, uct_mem_h const *memhs UCS_V_UNUSED,
                 const uint64_t *remote_addrs, uct_rkey_t const *rkeys,
                 const size_t *counts, const size_t *strides,
                 size_t count, uct_completion_t *comp)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                 uct_cuda_ipc_iface_t);
    uct_cuda_ipc_unpacked_rkey_t *key;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    uct_cuda_ipc_sgl_mapping_t *mapping;
    uct_cuda_queue_desc_t *q_desc;
    CUmemcpyAttributes attr;
    void *mapped_rem_addr;
    const void *mapped_addr;
    CUdeviceptr *cuda_dsts;
    size_t attrs_idx;
    size_t i, total_length;
    CUdevice cuda_device;
    CUstream *stream;
    ucs_status_t status;
    int is_ctx_pushed;

    /* TODO: add strided elements support */
    if (ucs_unlikely((counts != NULL) || (strides != NULL))) {
        ucs_error("cuda_ipc put_sgl_zcopy does not support strided elements");
        return UCS_ERR_UNSUPPORTED;
    }

    total_length = 0;
    for (i = 0; i < count; i++) {
        total_length += lengths[i];
    }

    if (ucs_unlikely(total_length == 0)) {
        ucs_trace_data("Zero length put_sgl_zcopy: skip it");
        return UCS_OK;
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)buffers[0],
                                             &cuda_device, &is_ctx_pushed);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    mapping = ucs_malloc(sizeof(*mapping) +
                         count * (sizeof(*mapping->entries) +
                                  sizeof(CUdeviceptr)),
                         "cuda_ipc_sgl_mapping");
    if (ucs_unlikely(mapping == NULL)) {
        ucs_error("Failed to allocate cuda_ipc sgl mapping");
        status = UCS_ERR_NO_MEMORY;
        goto out_ctx;
    }

    mapping->count   = 0;
    mapping->entries = (uct_cuda_ipc_sgl_entry_t *)(mapping + 1);
    cuda_dsts        = (CUdeviceptr *)(mapping->entries + count);

    memset(&attr, 0, sizeof(attr));
    attr.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attrs_idx           = 0;

    for (i = 0; i < count; i++) {
        key = (uct_cuda_ipc_unpacked_rkey_t *)rkeys[i];

        status = uct_cuda_ipc_get_remote_address(&key->super, remote_addrs[i],
                                                 cuda_device, &mapped_rem_addr,
                                                 &mapped_addr);
        if (ucs_unlikely(status != UCS_OK)) {
            goto out_unmap;
        }

        mapping->entries[i].pid         = key->super.super.pid;
        mapping->entries[i].pid_ns      = key->super.pid_ns;
        mapping->entries[i].d_bptr      = (uintptr_t)key->super.super.d_bptr;
        mapping->entries[i].mapped_addr = mapped_addr;
        cuda_dsts[i]                    = (CUdeviceptr)mapped_rem_addr;
        mapping->count++;
    }

    key    = (uct_cuda_ipc_unpacked_rkey_t *)rkeys[0];
    status = uct_cuda_ipc_get_stream_and_event(iface, key->stream_id,
                                               &q_desc, &stream,
                                               &cuda_ipc_event);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out_unmap;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyBatchAsync(cuda_dsts, (CUdeviceptr *)buffers,
                               (size_t *)lengths, count, &attr, &attrs_idx, 1,
                               *stream));
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_mpool_put(cuda_ipc_event);
        goto out_unmap;
    }

    status = uct_cuda_ipc_event_record_and_enqueue(iface, q_desc,
                                                   cuda_ipc_event, *stream);
    if (ucs_unlikely(status != UCS_OK)) {
        ucs_mpool_put(cuda_ipc_event);
        goto out_unmap;
    }

    cuda_ipc_event->super.comp  = comp;
    cuda_ipc_event->sgl_mapping = mapping;
    cuda_ipc_event->mapped_addr = NULL;
    cuda_ipc_event->d_bptr      = 0;
    cuda_ipc_event->pid         = 0;
    cuda_ipc_event->cuda_device = cuda_device;

    ucs_trace("cuMemcpyBatchAsync issued: count=%zu total_length=%zu",
              count, total_length);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      total_length);

    status = UCS_INPROGRESS;
    goto out_ctx;

out_unmap:
    uct_cuda_ipc_sgl_mapping_destroy(
            mapping, cuda_device, uct_cuda_ipc_component.enable_remote_cache);

out_ctx:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
    return status;
}
#endif /* CUDA_VERSION >= 13000 */

ucs_status_t uct_cuda_ipc_ep_get_device_ep(uct_ep_h tl_ep,
                                           uct_device_ep_h *device_ep_p)
{
    uct_cuda_ipc_ep_t *ep = ucs_derived_of(tl_ep, uct_cuda_ipc_ep_t);
    uct_device_ep_t device_ep;
    ucs_status_t status;

    if (ep->device_ep != NULL) {
        goto out;
    }

    device_ep.uct_tl_id = UCT_DEVICE_TL_CUDA_IPC;
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemAlloc((CUdeviceptr *)&ep->device_ep, sizeof(uct_device_ep_t)));
    if (status != UCS_OK) {
        goto err;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyHtoD((CUdeviceptr)ep->device_ep, &device_ep, sizeof(uct_device_ep_t)));
    if (status != UCS_OK) {
        goto err_free_mem;
    }

out:
    *device_ep_p = ep->device_ep;
    return UCS_OK;
err_free_mem:
    cuMemFree((CUdeviceptr)ep->device_ep);
    ep->device_ep = NULL;
err:
    return status;
}
