/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_copy_ep.h"
#include "cuda_copy_iface.h"

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
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

#define uct_cuda_copy_trace_data(_name, _remote_addr, _iov, _iovcnt) \
    ucs_trace_data("%s [ptr %p len %zu] to 0x%" PRIx64, _name, (_iov)->buffer, \
                   (_iov)->length, (_remote_addr))

#define UCT_CUDA_COPY_CHECK_AND_CREATE_STREAM(_iface, _id) \
    if ((_iface)->stream[_id] == 0) { \
        ucs_status_t __status; \
        __status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamCreateWithFlags(&(_iface)->stream[_id], \
                                                                   cudaStreamNonBlocking)); \
        if (UCS_OK != __status) { \
            return UCS_ERR_IO_ERROR; \
        } \
    }

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_post_cuda_async_copy(uct_ep_h tl_ep, void *dst, void *src, size_t length,
                                   enum cudaMemcpyKind direction,
                                   uct_cuda_copy_stream_t id,
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

    UCT_CUDA_COPY_CHECK_AND_CREATE_STREAM(iface, id);

    status = UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(dst, src, length, direction,
                                                   iface->stream[id]));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    status = UCT_CUDA_FUNC_LOG_ERR(cudaEventRecord(cuda_event->event,
                                                   iface->stream[id]));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }
    ucs_queue_push(&iface->outstanding_event_q[id], &cuda_event->queue);
    cuda_event->comp = comp;

    ucs_trace_data("cuda async issued :%p dst:%p, src:%p len:%ld", cuda_event,
                   dst, src, length);
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
                                                iov[0].length, cudaMemcpyDeviceToHost,
                                                UCT_CUDA_COPY_STREAM_D2H, comp);
    if (!UCS_STATUS_IS_ERR(status)) {
        VALGRIND_MAKE_MEM_DEFINED(iov[0].buffer, iov[0].length);
    }

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_copy_trace_data("GET_ZCOPY", remote_addr, iov, iovcnt);
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
                                                iov[0].length, cudaMemcpyHostToDevice,
                                                UCT_CUDA_COPY_STREAM_H2D, comp);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), PUT, ZCOPY,
                      uct_iov_total_length(iov, iovcnt));
    uct_cuda_copy_trace_data("PUT_ZCOPY", remote_addr, iov, iovcnt);
    return status;

}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_put_short,
                 (tl_ep, buffer, length, remote_addr, rkey),
                 uct_ep_h tl_ep, const void *buffer, unsigned length,
                 uint64_t remote_addr, uct_rkey_t rkey)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    uct_cuda_copy_stream_t idx   = UCT_CUDA_COPY_STREAM_H2D;
    ucs_status_t status;

    UCT_CUDA_COPY_CHECK_AND_CREATE_STREAM(iface, idx);

    UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync((void*)remote_addr, buffer, length,
                                          cudaMemcpyHostToDevice,
                                          iface->stream[idx]));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamSynchronize(iface->stream[idx]));

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
    uct_cuda_copy_stream_t idx   = UCT_CUDA_COPY_STREAM_D2H;
    ucs_status_t status;

    UCT_CUDA_COPY_CHECK_AND_CREATE_STREAM(iface, idx);

    UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(buffer, (void*)remote_addr, length,
                                          cudaMemcpyDeviceToHost,
                                          iface->stream[idx]));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamSynchronize(iface->stream[idx]));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}

