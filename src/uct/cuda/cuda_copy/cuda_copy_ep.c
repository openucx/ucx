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

/* It is relatively inexpensive to query memory attributes compared to actual
 * cudaMemcpy cost. Query time is between 0.05 to 0.2 us on x86 but memcpy cost
 * is generally > 7 us */
#define UCT_CUDA_COPY_GET_STREAM_EVENT_INDEX(_src_index, _dst_index, _src, _dst) { \
    ucs_memory_type_t __src_type; \
    ucs_memory_type_t __dst_type; \
    ucs_status_t __status; \
    __status = uct_cuda_base_detect_memory_type(NULL, _src, 0, &__src_type); \
    if (UCS_OK != __status) { \
        __src_type = UCS_MEMORY_TYPE_HOST; \
    } \
    __status = uct_cuda_base_detect_memory_type(NULL, _dst, 0, &__dst_type); \
    if (UCS_OK != __status) { \
        __dst_type = UCS_MEMORY_TYPE_HOST; \
    } \
    uct_cuda_copy_get_mem_type_index(__src_type, _src_index); \
    uct_cuda_copy_get_mem_type_index(__dst_type, _dst_index); \
    ucs_assert(__src_index != UCT_CUDA_COPY_LOC_LAST \
               && __dst_index != UCT_CUDA_COPY_LOC_LAST); \
}

#define UCT_CUDA_COPY_GET_STREAM(_iface, _stream, _src, _dst) { \
    int __src_index; \
    int __dst_index; \
    UCT_CUDA_COPY_GET_STREAM_EVENT_INDEX(__src_index, __dst_index, _src, _dst); \
    if ((_iface)->stream[__src_index][__dst_index] == 0) { \
        ucs_status_t __status; \
        __status = \
            UCT_CUDA_FUNC_LOG_ERR(cudaStreamCreateWithFlags(&(_iface)->stream[__src_index][__dst_index], \
                                                                   cudaStreamNonBlocking)); \
        if (UCS_OK != __status) { \
            return UCS_ERR_IO_ERROR; \
        } \
    } \
    _stream = &(_iface)->stream[__src_index][__dst_index]; \
}

#define UCT_CUDA_COPY_GET_EVENT_Q(_iface, _event_q, _src, _dst) { \
    int __src_index; \
    int __dst_index; \
    UCT_CUDA_COPY_GET_STREAM_EVENT_INDEX(__src_index, __dst_index, _src, _dst); \
    _event_q = &(_iface)->outstanding_event_q[__src_index][__dst_index]; \
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

    if (!length) {
        return UCS_OK;
    }

    cuda_event = ucs_mpool_get(&iface->cuda_event_desc);
    if (ucs_unlikely(cuda_event == NULL)) {
        ucs_error("Failed to allocate cuda event object");
        return UCS_ERR_NO_MEMORY;
    }

    UCT_CUDA_COPY_GET_STREAM(iface, stream, src, dst);

    status = UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(dst, src, length, cudaMemcpyDefault,
                                                   *stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_CUDA_FUNC_LOG_ERR(cudaEventRecord(cuda_event->event, *stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }
    UCT_CUDA_COPY_GET_EVENT_Q(iface, event_q, src, dst);
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
    cudaStream_t *stream;
    ucs_status_t status;

    UCT_CUDA_COPY_GET_STREAM(iface, stream, buffer, (void*)remote_addr);

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
    cudaStream_t *stream;
    ucs_status_t status;

    UCT_CUDA_COPY_GET_STREAM(iface, stream, (void*)remote_addr, buffer);

    UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(buffer, (void*)remote_addr, length,
                                          cudaMemcpyDefault, *stream));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamSynchronize(*stream));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}

