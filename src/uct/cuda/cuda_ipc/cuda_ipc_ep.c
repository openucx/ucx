/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

 #ifdef HAVE_CONFIG_H
 #  include "config.h"
 #endif

#include <uct/cuda/base/cuda_iface.h>
#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>

#include "cuda_ipc_ep.h"
#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"
#include "cuda_ipc.inl"

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>

#define UCT_CUDA_IPC_PUT 0
#define UCT_CUDA_IPC_GET 1


static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_ep_t, const uct_ep_params_t *params)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(params->iface,
                                                 uct_cuda_ipc_iface_t);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

    self->remote_pid = *(const pid_t*)params->iface_addr;
    self->device_ep  = NULL;
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_ep_t)
{
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

    return ep->remote_pid == *(pid_t*)params->iface_addr;
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_cuda_ipc_ctx_rsc_get(
        uct_cuda_ipc_iface_t *iface, uct_cuda_ipc_ctx_rsc_t **ctx_rsc_p)
{
    unsigned long long ctx_id;
    ucs_status_t status;
    CUresult result;
    uct_cuda_ctx_rsc_t *ctx_rsc;

    result = uct_cuda_base_ctx_get_id(NULL, &ctx_id);
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
    void *mapped_addr;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    uct_cuda_ipc_ctx_rsc_t *ctx_rsc;
    uct_cuda_queue_desc_t *q_desc;
    ucs_status_t status;
    CUdeviceptr dst, src;
    CUcontext UCS_V_UNUSED cuda_context;
    CUstream *stream;
    size_t offset;

    if (ucs_unlikely(0 == iov[0].length)) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)iov[0].buffer,
                                             &cuda_device, &is_ctx_pushed);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    status = uct_cuda_ipc_map_memhandle(&key->super, cuda_device, &mapped_addr,
                                        UCS_LOG_LEVEL_ERROR);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    status = uct_cuda_ipc_ctx_rsc_get(iface, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    offset          = (uintptr_t)remote_addr - (uintptr_t)key->super.d_bptr;
    mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
    ucs_assert(offset <= key->super.b_len);

    /* round-robin */
    q_desc = &ctx_rsc->queue_desc[key->stream_id % iface->config.max_streams];
    stream = &q_desc->stream;
    status = uct_cuda_base_init_stream(stream);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    if (ucs_unlikely(stream == NULL)) {
        ucs_error("stream=%d for dev_num=%d not available", key->stream_id,
                  cuda_device);
        status = UCS_ERR_IO_ERROR;
        goto out;
    }

    cuda_ipc_event = ucs_mpool_get(&ctx_rsc->super.event_mp);
    if (ucs_unlikely(cuda_ipc_event == NULL)) {
        ucs_error("Failed to allocate cuda_ipc event object");
        status = UCS_ERR_NO_MEMORY;
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

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuEventRecord(cuda_ipc_event->super.event,
                                                    *stream));
    if (UCS_OK != status) {
        ucs_mpool_put(cuda_ipc_event);
        goto out;
    }

    if (ucs_queue_is_empty(&q_desc->event_queue)) {
        ucs_queue_push(&iface->super.active_queue, &q_desc->queue);
    }

    ucs_queue_push(&q_desc->event_queue, &cuda_ipc_event->super.queue);
    cuda_ipc_event->super.comp  = comp;
    cuda_ipc_event->mapped_addr = mapped_addr;
    cuda_ipc_event->d_bptr      = (uintptr_t)key->super.d_bptr;
    cuda_ipc_event->pid         = key->super.pid;
    cuda_ipc_event->cuda_device = cuda_device;
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
