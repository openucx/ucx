/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_ep.h"
#include "cuda_ipc_iface.h"
#include "cuda_ipc_md.h"

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
    uct_cuda_ipc_iface_t *iface         = ucs_derived_of(params->iface,
                                                         uct_cuda_ipc_iface_t);
    uct_cuda_base_sys_dev_map_t *remote = (uct_cuda_base_sys_dev_map_t*)
                                          params->iface_addr;
    uct_cuda_base_sys_dev_map_t *hash;
    khiter_t khiter;
    int khret;
    int i;

    ucs_recursive_spin_lock(&iface->rem_iface_addr_lock);

    khiter = kh_put(cuda_ipc_rem_iface_addr, &iface->rem_iface_addr_hash,
                    remote->pid, &khret);
    if ((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
        (khret == UCS_KH_PUT_BUCKET_CLEAR)) {
        hash        = &kh_val(&iface->rem_iface_addr_hash, khiter);
        hash->count = remote->count;
        hash->pid   = remote->pid;

        for (i = 0; i < remote->count; i++) {
            hash->sys_dev[i] = remote->sys_dev[i];
            hash->bus_id[i]  = remote->bus_id[i];
            ucs_trace("peer pid %ld sys_dev %u bus_id %u",
                      (long)hash->pid, (unsigned)hash->sys_dev[i],
                      (unsigned)hash->bus_id[i]);
        }
    } else if (khret != UCS_KH_PUT_KEY_PRESENT) {
        ucs_error("unable to use cuda_ipc remote_iface_addr hash");
    }
    ucs_recursive_spin_unlock(&iface->rem_iface_addr_lock);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    self->remote_pid = remote->pid;

    return uct_ep_keepalive_init(&self->keepalive, self->remote_pid);
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ipc_ep_t)
{
}

UCS_CLASS_DEFINE(uct_cuda_ipc_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ipc_ep_t, uct_ep_t);

#define uct_cuda_ipc_trace_data(_addr, _rkey, _fmt, ...)     \
    ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_addr), (_rkey))

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_post_cuda_async_copy(uct_ep_h tl_ep, uint64_t remote_addr,
                                  const uct_iov_t *iov, uct_rkey_t rkey,
                                  uct_completion_t *comp, int direction)
{
    uct_cuda_ipc_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_ipc_iface_t);
    uct_cuda_ipc_key_t *key     = (uct_cuda_ipc_key_t *) rkey;
    void *mapped_rem_addr;
    void *mapped_addr;
    uct_cuda_ipc_event_desc_t *cuda_ipc_event;
    ucs_queue_head_t *outstanding_queue;
    ucs_status_t status;
    CUdeviceptr dst, src;
    CUstream stream;
    size_t offset;

    if (0 == iov[0].length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }

    status = uct_cuda_ipc_map_memhandle(key, &mapped_addr);
    if (status != UCS_OK) {
        return UCS_ERR_IO_ERROR;
    }

    /* ensure context is set before creating events/streams */
    if (iface->cuda_context == NULL) {
        UCT_CUDA_FUNC_LOG_ERR(cuCtxGetCurrent(&iface->cuda_context));
        if (iface->cuda_context == NULL) {
            ucs_error("attempt to perform cuda memcpy without active context");
            return UCS_ERR_IO_ERROR;
        }
    }

    offset          = (uintptr_t)remote_addr - (uintptr_t)key->d_bptr;
    mapped_rem_addr = (void *) ((uintptr_t) mapped_addr + offset);
    ucs_assert(offset <= key->b_len);

    if (!iface->streams_initialized) {
        status = uct_cuda_ipc_iface_init_streams(iface);
        if (UCS_OK != status) {
            return status;
        }
    }

    key->dev_num %= iface->config.max_streams; /* round-robin */

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

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemcpyDtoDAsync(dst, src, iov[0].length,
                                                        stream));
    if (UCS_OK != status) {
        ucs_mpool_put(cuda_ipc_event);
        return status;
    }

    iface->stream_refcount[key->dev_num]++;
    cuda_ipc_event->stream_id = key->dev_num;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuEventRecord(cuda_ipc_event->event,
                                                    stream));
    if (UCS_OK != status) {
        ucs_mpool_put(cuda_ipc_event);
        return status;
    }

    ucs_queue_push(outstanding_queue, &cuda_ipc_event->queue);
    cuda_ipc_event->comp        = comp;
    cuda_ipc_event->mapped_addr = mapped_addr;
    cuda_ipc_event->d_bptr      = (uintptr_t)key->d_bptr;
    cuda_ipc_event->pid         = key->pid;
    ucs_trace("cuMemcpyDtoDAsync issued :%p dst:%p, src:%p  len:%ld",
             cuda_ipc_event, (void *) dst, (void *) src, iov[0].length);
    return UCS_INPROGRESS;
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

ucs_status_t uct_cuda_ipc_ep_check(const uct_ep_h tl_ep, unsigned flags,
                                   uct_completion_t *comp)
{
    uct_cuda_ipc_ep_t *ep = ucs_derived_of(tl_ep, uct_cuda_ipc_ep_t);

    return uct_ep_keepalive_check(tl_ep, &ep->keepalive, ep->remote_pid, flags,
                                  comp);
}
