/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#include "cuda_ipc_ep.h"
#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>

#define UCT_CUDA_IPC_PUT 0
#define UCT_CUDA_IPC_GET 1

static UCS_CLASS_INIT_FUNC(uct_cuda_ipc_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_ipc_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_ep_t)
{
}

UCS_CLASS_DEFINE(uct_cuda_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_ep_t, uct_ep_t);

#define uct_cuda_ipc_trace_data(_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_addr), (_rkey))

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_get_mapped_addr(uct_cuda_ipc_ep_t *ep, uct_cuda_ipc_key_t *key,
                             int cu_device, uint64_t remote_addr,
                             void **mapped_rem_addr, void *buffer)
{
    int offset, same_ctx = 0;
    void *mapped_addr;
    ucs_status_t status;
    CUcontext local_ptr_ctx;
    CUcontext remote_ptr_ctx;
    CUpointer_attribute attrib;

    if (key->dev_num == (int) cu_device) {
        attrib = CU_POINTER_ATTRIBUTE_CONTEXT;
        status = UCT_CUDADRV_FUNC(cuPointerGetAttribute((void *) &remote_ptr_ctx,
                                                        attrib,
                                                        (CUdeviceptr) remote_addr));
        if (UCS_OK != status) {
            return status;
        }

        status = UCT_CUDADRV_FUNC(cuPointerGetAttribute((void *) &local_ptr_ctx,
                                                        attrib,
                                                        (CUdeviceptr) buffer));
        if (UCS_OK != status) {
            return status;
        }

        same_ctx = (local_ptr_ctx == remote_ptr_ctx) ? 1 : 0;
    }

    if (same_ctx) {
        *mapped_rem_addr = (void *) remote_addr;
    } else {
        status =
            UCT_CUDADRV_FUNC(cuIpcOpenMemHandle((CUdeviceptr *) &mapped_addr,
                                                key->ph,
                                                CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS));
        if (UCS_OK != status) {
            return status;
        }

        offset = (uintptr_t) remote_addr - (uintptr_t) key->d_rem_bptr;
        if (offset > key->b_rem_len) {
            ucs_fatal("Access memory outside memory range attempt\n");
            return UCS_ERR_IO_ERROR;
        }

        *mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
    }
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_post_cuda_async_copy(uct_ep_h tl_ep, uint64_t remote_addr,
                                  const uct_iov_t *iov, uct_rkey_t rkey,
                                  uct_completion_t *comp, int direction)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_ipc_iface_t);
    uct_cuda_ipc_ep_t *ep       = ucs_derived_of(tl_ep, uct_cuda_ipc_ep_t);
    uct_cuda_ipc_key_t *key     = (uct_cuda_ipc_key_t *) rkey;
    void *mapped_rem_addr       = NULL;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    ucs_queue_head_t *outstanding_queue;
    ucs_status_t status;
    CUdeviceptr dst, src;
    CUdevice cu_device;
    CUstream stream;

    if (0 == iov[0].length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    GET_CUDA_DEVICE(cu_device);

    status = uct_cuda_ipc_get_mapped_addr(ep, key, cu_device, remote_addr,
                                          &mapped_rem_addr, iov[0].buffer);
    if (UCS_OK != status) {
        return status;
    }

    if (0 == iface->streams_initialized) {
        status = uct_cuda_ipc_iface_init_streams(iface);
        if (UCS_OK != status) {
            return status;
        }
    }

    stream            = iface->stream_d2d[key->dev_num];
    outstanding_queue = &iface->outstanding_d2d_event_q;
    cuda_ipc_event    = ucs_mpool_get(&iface->event_desc);

    if (ucs_unlikely(cuda_ipc_event == NULL)) {
        ucs_error("Failed to allocate cuda_ipc event object");
        return UCS_ERR_NO_MEMORY;
    }

    dst = (CUdeviceptr)
        ((direction == UCT_CUDA_IPC_PUT) ? mapped_rem_addr : iov[0].buffer);
    src = (CUdeviceptr)
        ((direction == UCT_CUDA_IPC_PUT) ? iov[0].buffer : mapped_rem_addr);

    status = UCT_CUDADRV_FUNC(cuMemcpyDtoDAsync(dst, src, iov[0].length, stream));
    if (UCS_OK != status) {
        return status;
    }

    status = UCT_CUDADRV_FUNC(cuEventRecord(cuda_ipc_event->event, stream));
    if (UCS_OK != status) {
        return status;
    }

    ucs_queue_push(outstanding_queue, &cuda_ipc_event->queue);
    cuda_ipc_event->comp = comp;
    ucs_trace("cuMemcpyDtoDAsync issued :%p dst:%p, src:%p  len:%ld",
             cuda_ipc_event, (void *) dst, (void *) src, iov[0].length);
    return UCS_INPROGRESS;
}

ucs_status_t uct_cuda_ipc_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_ipc_post_cuda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_CUDA_IPC_GET);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_ipc_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                            uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_cuda_ipc_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_ipc_post_cuda_async_copy(tl_ep, remote_addr, iov,
                                               rkey, comp, UCT_CUDA_IPC_PUT);
    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_ipc_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                                uct_iov_total_length(iov, iovcnt));
    return status;
}
