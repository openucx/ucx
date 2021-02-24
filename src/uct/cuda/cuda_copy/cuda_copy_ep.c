/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_copy_ep.h"
#include "cuda_copy_iface.h"
#include <uct/cuda/base/cuda_md.h>

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_cuda_copy_ep_t, const uct_ep_params_t *params)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(params->iface,
                                                  uct_cuda_copy_iface_t);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super);

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_copy_ep_t)
{
}

UCS_CLASS_DEFINE(uct_cuda_copy_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_copy_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_copy_ep_t, uct_ep_t);

#define uct_cuda_copy_trace_data(_remote_addr, _rkey, _fmt, ...) \
     ucs_trace_data(_fmt " to %"PRIx64"(%+ld)", ## __VA_ARGS__, (_remote_addr), \
                    (_rkey))


static UCS_F_ALWAYS_INLINE cudaStream_t
*uct_cuda_copy_get_stream(uct_cuda_copy_iface_t *iface,
                         ucs_memory_type_t src_type,
                         ucs_memory_type_t dst_type)
{
    cudaStream_t *stream;
    ucs_status_t status;

    stream = &iface->stream[src_type][dst_type];
    if (*stream == 0) {
        status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamCreateWithFlags(stream,
                                                                 cudaStreamNonBlocking));
        if (status != UCS_OK ) {
            stream = NULL;
        }
    }

    return stream;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_post_cuda_async_copy(uct_ep_h tl_ep, void *dst, void *src, size_t length,
                                   uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    uct_cuda_copy_event_desc_t *cuda_event;
    ucs_status_t status;
    cudaStream_t *stream;
    ucs_queue_head_t *event_q;
    ucs_memory_type_t src_type;
    ucs_memory_type_t dst_type;

    if (!length) {
        return UCS_OK;
    }

    status = uct_cuda_base_detect_memory_type(NULL, src, 0, &src_type);
    if (UCS_OK != status) {
        src_type = UCS_MEMORY_TYPE_HOST;
    }

    status = uct_cuda_base_detect_memory_type(NULL, dst, 0, &dst_type);
    if (UCS_OK != status) {
        dst_type = UCS_MEMORY_TYPE_HOST;
    }

    cuda_event = ucs_mpool_get(&iface->cuda_event_desc);
    if (ucs_unlikely(cuda_event == NULL)) {
        ucs_error("Failed to allocate cuda event object");
        return UCS_ERR_NO_MEMORY;
    }

    stream = uct_cuda_copy_get_stream(iface, src_type, dst_type);
    if (ucs_unlikely(stream == NULL)) {
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(dst, src, length, cudaMemcpyDefault,
                                                   *stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_CUDA_FUNC_LOG_ERR(cudaEventRecord(cuda_event->event, *stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    event_q = &iface->outstanding_event_q[src_type][dst_type];
    ucs_queue_push(event_q, &cuda_event->queue);
    cuda_event->comp = comp;

    ucs_trace("cuda async issued :%p dst:%p, src:%p  len:%ld",
             cuda_event, dst, src, length);
    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;

    status = uct_cuda_copy_post_cuda_async_copy(tl_ep, iov[0].buffer, (void *)remote_addr,
                                                iov[0].length, comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        VALGRIND_MAKE_MEM_DEFINED(iov[0].buffer, iov[0].length);
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_copy_trace_data(remote_addr, rkey, "GET_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_put_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{

    ucs_status_t status;

    status = uct_cuda_copy_post_cuda_async_copy(tl_ep, (void *)remote_addr,  iov[0].buffer,
                                                iov[0].length, comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_copy_trace_data(remote_addr, rkey, "PUT_ZCOPY [length %zu]",
                             uct_iov_total_length(iov, iovcnt));
    return status;

}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_put_short,
                 (tl_ep, buffer, length, remote_addr, rkey),
                 uct_ep_h tl_ep, const void *buffer, unsigned length,
                 uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    ucs_memory_type_t src_type;
    ucs_memory_type_t dst_type;
    cudaStream_t *stream;
    ucs_status_t status;

    status = uct_cuda_base_detect_memory_type(NULL, buffer, 0, &src_type);
    if (UCS_OK != status) {
        src_type = UCS_MEMORY_TYPE_HOST;
    }

    status = uct_cuda_base_detect_memory_type(NULL, (void*)remote_addr, 0, &dst_type);
    if (UCS_OK != status) {
        dst_type = UCS_MEMORY_TYPE_HOST;
    }

    stream = uct_cuda_copy_get_stream(iface, src_type, dst_type);
    if (ucs_unlikely(stream == NULL)) {
        return UCS_ERR_IO_ERROR;
    }

    UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync((void*)remote_addr, buffer, length,
                                          cudaMemcpyDefault, *stream));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamSynchronize(*stream));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, SHORT, length);
    ucs_trace_data("PUT_SHORT size %d from %p to %p",
                   length, buffer, (void *)remote_addr);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_get_short,
                 (tl_ep, buffer, length, remote_addr, rkey),
                 uct_ep_h tl_ep, void *buffer, unsigned length,
                 uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    ucs_memory_type_t src_type;
    ucs_memory_type_t dst_type;
    cudaStream_t *stream;
    ucs_status_t status;

    status = uct_cuda_base_detect_memory_type(NULL, (void*)remote_addr, 0, &src_type);
    if (UCS_OK != status) {
        src_type = UCS_MEMORY_TYPE_HOST;
    }

    status = uct_cuda_base_detect_memory_type(NULL, buffer, 0, &dst_type);
    if (UCS_OK != status) {
        dst_type = UCS_MEMORY_TYPE_HOST;
    }

    stream = uct_cuda_copy_get_stream(iface, src_type, dst_type);
    if (ucs_unlikely(stream == NULL)) {
        return UCS_ERR_IO_ERROR;
    }

    UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(buffer, (void*)remote_addr, length,
                                          cudaMemcpyDefault, *stream));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamSynchronize(*stream));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}

