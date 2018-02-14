/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cuda_copy_ep.h"
#include "cuda_copy_iface.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_cuda_copy_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_cuda_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_copy_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_copy_ep_t, uct_ep_t);

#define uct_cuda_copy_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_post_cuda_async_copy(uct_ep_h tl_ep, void *dst, void *src, size_t length,
                                   int direction, cudaStream_t stream,
                                   ucs_queue_head_t *outstanding_queue,
                                   uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    uct_cuda_copy_event_desc_t *cuda_event;
    ucs_status_t status;

    if (!length) {
        return UCS_OK;
    }

    cuda_event = ucs_mpool_get(&iface->cuda_event_desc);
    if (ucs_unlikely(cuda_event == NULL)) {
        ucs_error("Failed to allocate cuda event object");
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_CUDA_FUNC(cudaMemcpyAsync(dst, src, length, direction, stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_CUDA_FUNC(cudaEventRecord(cuda_event->event, stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }
    ucs_queue_push(outstanding_queue, &cuda_event->queue);
    cuda_event->comp = comp;

    ucs_trace("cuda async issued :%p dst:%p, src:%p  len:%ld",
             cuda_event, dst, src, length);
    return UCS_INPROGRESS;
}

ucs_status_t uct_cuda_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    ucs_status_t status;

    if (iface->stream_d2h == 0) {
        status = UCT_CUDA_FUNC(cudaStreamCreateWithFlags(&iface->stream_d2h,
                               cudaStreamNonBlocking));
        if (UCS_OK != status) {
            return UCS_ERR_IO_ERROR;
        }
    }

    status = uct_cuda_copy_post_cuda_async_copy(tl_ep, iov[0].buffer, (void *)remote_addr,
                                                iov[0].length, cudaMemcpyDeviceToHost,
                                                iface->stream_d2h,
                                                &iface->outstanding_d2h_cuda_event_q, comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;
}

ucs_status_t uct_cuda_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp)
{

    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    ucs_status_t status;

    if (iface->stream_h2d == 0) {
        status = UCT_CUDA_FUNC(cudaStreamCreateWithFlags(&iface->stream_h2d,
                               cudaStreamNonBlocking));
        if (UCS_OK != status) {
            return UCS_ERR_IO_ERROR;
        }
    }

    status = uct_cuda_copy_post_cuda_async_copy(tl_ep, (void *)remote_addr,  iov[0].buffer,
                                                iov[0].length, cudaMemcpyHostToDevice,
                                                iface->stream_h2d,
                                                &iface->outstanding_h2d_cuda_event_q, comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;

}
