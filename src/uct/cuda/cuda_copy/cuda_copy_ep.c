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
#include <uct/cuda/base/cuda_md.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <ucs/memory/memtype_cache.h>


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

ucs_status_t uct_cuda_copy_init_stream(cudaStream_t *stream)
{
    if (*stream != 0) {
        return UCS_OK;
    }

    return UCT_CUDA_FUNC_LOG_ERR(cudaStreamCreateWithFlags(stream,
                                                           cudaStreamNonBlocking));
}

static UCS_F_ALWAYS_INLINE cudaStream_t *
uct_cuda_copy_get_stream(uct_cuda_copy_iface_t *iface,
                         ucs_memory_type_t src_type,
                         ucs_memory_type_t dst_type)
{
    cudaStream_t *stream = NULL;
    ucs_status_t status;

    ucs_assert((src_type < UCS_MEMORY_TYPE_LAST) &&
               (dst_type < UCS_MEMORY_TYPE_LAST));

    stream = &iface->queue_desc[src_type][dst_type].stream;

    status = uct_cuda_copy_init_stream(stream);
    if (status != UCS_OK) {
        return NULL;
    }

    return stream;
}

static UCS_F_ALWAYS_INLINE ucs_memory_type_t
uct_cuda_copy_get_mem_type(uct_md_h md, void *address, size_t length)
{
    ucs_memory_info_t mem_info;
    ucs_status_t status;

    status = ucs_memtype_cache_lookup(address, length, &mem_info);
    if (status == UCS_ERR_NO_ELEM) {
        return UCS_MEMORY_TYPE_HOST;
    }

    if ((status == UCS_ERR_UNSUPPORTED) ||
        (mem_info.type == UCS_MEMORY_TYPE_UNKNOWN)) {
        status = uct_cuda_base_detect_memory_type(md, address, length,
                                                  &mem_info.type);
        if (status != UCS_OK) {
            return UCS_MEMORY_TYPE_HOST;
        }
    }

    return mem_info.type;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_post_cuda_async_copy(uct_ep_h tl_ep, void *dst, void *src,
                                   size_t length, uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_copy_iface_t);
    uct_base_iface_t *base_iface = ucs_derived_of(tl_ep->iface, uct_base_iface_t);
    uct_cuda_copy_event_desc_t *cuda_event;
    uct_cuda_copy_queue_desc_t *q_desc;
    ucs_status_t status;
    ucs_memory_type_t src_type;
    ucs_memory_type_t dst_type;
    cudaStream_t *stream;
    ucs_queue_head_t *event_q;

    if (!length) {
        return UCS_OK;
    }

    /* ensure context is set before creating events/streams */
    if (iface->cuda_context == NULL) {
        UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&iface->cuda_context));
        if (iface->cuda_context == NULL) {
            ucs_error("attempt to perform cuda memcpy without active context");
            return UCS_ERR_IO_ERROR;
        }
    }

    src_type = uct_cuda_copy_get_mem_type(base_iface->md, src, length);
    dst_type = uct_cuda_copy_get_mem_type(base_iface->md, dst, length);
    q_desc   = &iface->queue_desc[src_type][dst_type];
    event_q  = &q_desc->event_queue;
    stream   = uct_cuda_copy_get_stream(iface, src_type, dst_type);
    if (stream == NULL) {
        ucs_error("stream for src %s dst %s not available",
                   ucs_memory_type_names[src_type],
                   ucs_memory_type_names[dst_type]);
        return UCS_ERR_IO_ERROR;
    }

    cuda_event = ucs_mpool_get(&iface->cuda_event_desc);
    if (ucs_unlikely(cuda_event == NULL)) {
        ucs_error("Failed to allocate cuda event object");
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(dst, src, length, cudaMemcpyDefault,
                                                   *stream));

    status = UCT_CUDA_FUNC_LOG_ERR(cudaEventRecord(cuda_event->event, *stream));
    if (UCS_OK != status) {
        return UCS_ERR_IO_ERROR;
    }

    if (ucs_queue_is_empty(event_q)) {
        ucs_queue_push(&iface->active_queue, &q_desc->queue);
    }

    ucs_queue_push(event_q, &cuda_event->queue);
    cuda_event->comp = comp;

    ucs_trace("cuda async issued :%p dst:%p[%s], src:%p[%s] len:%ld",
              cuda_event, dst, ucs_memory_type_names[src_type], src,
              ucs_memory_type_names[dst_type], length);
    return UCS_INPROGRESS;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_get_zcopy,
                 (tl_ep, iov, iovcnt, remote_addr, rkey, comp),
                 uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                 uint64_t remote_addr, uct_rkey_t rkey,
                 uct_completion_t *comp)
{
    ucs_status_t status;


    status = uct_cuda_copy_post_cuda_async_copy(tl_ep, iov[0].buffer,
                                                (void *)remote_addr,
                                                iov[0].length, comp);
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

    status = uct_cuda_copy_post_cuda_async_copy(tl_ep, (void *)remote_addr,
                                                iov[0].buffer,
                                                iov[0].length, comp);

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
    cudaStream_t *stream         = &iface->short_stream;
    ucs_status_t status;

    status = uct_cuda_copy_init_stream(stream);
    if (status != UCS_OK) {
        return status;
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
    cudaStream_t *stream         = &iface->short_stream;
    ucs_status_t status;

    status = uct_cuda_copy_init_stream(stream);
    if (status != UCS_OK) {
        return status;
    }

    UCT_CUDA_FUNC_LOG_ERR(cudaMemcpyAsync(buffer, (void*)remote_addr, length,
                                          cudaMemcpyDefault, *stream));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaStreamSynchronize(*stream));

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}

