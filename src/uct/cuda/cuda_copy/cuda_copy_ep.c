/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_copy_ep.h"
#include "cuda_copy_iface.h"
#include "cuda_copy_md.h"

#include <uct/base/uct_log.h>
#include <uct/base/uct_iov.inl>
#include <uct/cuda/base/cuda_md.h>
#include <uct/cuda/base/cuda_ctx.inl>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/type/class.h>
#include <ucs/memory/memtype_cache.h>

typedef struct {
    ucs_memory_type_t       src_type;
    ucs_memory_type_t       dst_type;
    CUdevice                cuda_device;
    CUcontext               cuda_context;
    uct_cuda_copy_ctx_rsc_t *ctx_rsc;
} uct_cuda_copy_ep_ctx_t;

static UCS_CLASS_INIT_FUNC(uct_cuda_copy_ep_t, const uct_ep_params_t *params)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(params->iface,
                                                  uct_cuda_copy_iface_t);

    UCT_EP_PARAMS_CHECK_DEV_IFACE_ADDRS(params);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super.super);

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


static UCS_F_ALWAYS_INLINE CUstream *
uct_cuda_copy_get_stream(uct_cuda_copy_ctx_rsc_t *ctx_rsc,
                         ucs_memory_type_t src_type, ucs_memory_type_t dst_type)
{
    CUstream *stream;
    ucs_status_t status;

    ucs_assert((src_type < UCS_MEMORY_TYPE_LAST) &&
               (dst_type < UCS_MEMORY_TYPE_LAST));

    stream = &ctx_rsc->queue_desc[src_type][dst_type].stream;
    status = uct_cuda_base_init_stream(stream);
    if (ucs_unlikely(status != UCS_OK)) {
        return NULL;
    }

    return stream;
}

static UCS_F_ALWAYS_INLINE ucs_memory_type_t
uct_cuda_copy_get_mem_type(uct_md_h md, const void *address, size_t length,
                           ucs_sys_device_t *sys_dev)
{
    ucs_memory_info_t mem_info;
    uct_md_mem_attr_t mem_attr;
    ucs_status_t status;

    status = ucs_memtype_cache_lookup(address, length, &mem_info);
    if (status == UCS_ERR_NO_ELEM) {
        goto out_host;
    }

    if (ucs_unlikely((status == UCS_ERR_UNSUPPORTED) ||
                     (mem_info.type == UCS_MEMORY_TYPE_UNKNOWN))) {
        mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE |
                              UCT_MD_MEM_ATTR_FIELD_SYS_DEV;

        status = uct_cuda_copy_md_mem_query(md, address, length, &mem_attr);
        if (status != UCS_OK) {
            goto out_host;
        }

        mem_info.type    = mem_attr.mem_type;
        mem_info.sys_dev = mem_attr.sys_dev;
    }

    *sys_dev = mem_info.sys_dev;
    return mem_info.type;

out_host:
    *sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    return UCS_MEMORY_TYPE_HOST;
}

static UCS_F_ALWAYS_INLINE void
uct_cuda_copy_get_mem_types(uct_md_h md, const void *src, const void *dst,
                            size_t length, ucs_memory_type_t *src_mem_type_p,
                            ucs_memory_type_t *dst_mem_type_p,
                            ucs_sys_device_t *sys_dev_p,
                            CUdeviceptr *cuda_deviceptr_p)
{
    ucs_sys_device_t src_sys_dev, dst_sys_dev;

    *src_mem_type_p = uct_cuda_copy_get_mem_type(md, src, length, &src_sys_dev);
    *dst_mem_type_p = uct_cuda_copy_get_mem_type(md, dst, length, &dst_sys_dev);
    if (src_sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        *sys_dev_p        = src_sys_dev;
        *cuda_deviceptr_p = (CUdeviceptr)src;
    } else {
        *sys_dev_p        = dst_sys_dev;
        *cuda_deviceptr_p = (CUdeviceptr)dst;
    }

    ucs_assertv((src_sys_dev == dst_sys_dev) ||
                (src_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) ||
                (dst_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN),
                "src mtype %s, sys_dev %s; dst mtype %s, sys_dev %s",
                ucs_memory_type_names[*src_mem_type_p],
                ucs_topo_sys_device_get_name(src_sys_dev),
                ucs_memory_type_names[*dst_mem_type_p],
                ucs_topo_sys_device_get_name(dst_sys_dev));
}

static ucs_status_t uct_cuda_copy_ep_push_memory_ctx(CUdeviceptr cuda_deviceptr,
                                                     CUcontext *cuda_context_p)
{
    CUcontext cuda_context;
    ucs_status_t status;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerGetAttribute(&cuda_context, CU_POINTER_ATTRIBUTE_CONTEXT,
                                  cuda_deviceptr));
    if (status != UCS_OK) {
        return status;
    }

    if (cuda_context == NULL) {
        ucs_error("failed to query cuda context for 0x%llx", cuda_deviceptr);
        return UCS_ERR_UNSUPPORTED;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(cuda_context));
    if (status != UCS_OK) {
        return status;
    }

    *cuda_context_p = cuda_context;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_cuda_copy_ctx_rsc_get(
        uct_cuda_copy_iface_t *iface, ucs_sys_device_t sys_dev,
        CUdeviceptr cuda_deviceptr, CUdevice *cuda_device_p,
        CUcontext *cuda_context_p, uct_cuda_copy_ctx_rsc_t **ctx_rsc_p)
{
    CUcontext cuda_context = NULL;
    unsigned long long ctx_id;
    CUresult result;
    CUdevice cuda_device;
    ucs_status_t status;
    uct_cuda_ctx_rsc_t *ctx_rsc;

    if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        /* If valid sys_dev is provided we need to retain and push primary context
         * of the device. This is because of some limitation when using VMM and
         * cuMemcpyAsync - the current context should match the device VMM has
         * access to. */
        cuda_device = uct_cuda_get_cuda_device(sys_dev);
        if (ucs_unlikely(cuda_device == CU_DEVICE_INVALID)) {
            goto err;
        }

        status = uct_cuda_ctx_primary_push(cuda_device, 0, UCS_LOG_LEVEL_ERROR);
        if (ucs_unlikely(status == UCS_ERR_NO_DEVICE)) {
            /* Device primary context of `cuda_device` is inactive. The memory
             * was probably allocated on the context created with cuCtxCreate.
             * Fallback to push context based on memory address. */
            status = uct_cuda_copy_ep_push_memory_ctx(cuda_deviceptr,
                                                      &cuda_context);
            if (status != UCS_OK) {
                goto err;
            }

            cuda_device = CU_DEVICE_INVALID;
        }

        if (ucs_unlikely(status != UCS_OK)) {
            goto err;
        }
    } else {
        /* If there is a current context set, the CU_DEVICE_INVALID is returned
         * in cuda_device_p */
        cuda_device = CU_DEVICE_INVALID;
    }

    result = uct_cuda_ctx_get_id(NULL, &ctx_id);
    if (ucs_unlikely(result != CUDA_SUCCESS)) {
        if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
            /* Context is pushed, but ctx_get_id failed, which means that some
             * CUDA error occurred.*/
            ucs_error("failed to get context id of device %s (%d)",
                      ucs_topo_sys_device_get_name(sys_dev), cuda_device);
            status = UCS_ERR_IO_ERROR;
            goto err_pop_and_release;
        }

        /* Specific GPU device was not requested, push the first active primary
         * context as current context. The caller must pop, and release the
         * primary context on the device returned in cuda_device_p. */
        status = uct_cuda_ctx_primary_push_first_active(&cuda_device);
        if (status != UCS_OK) {
            goto err;
        }

        result = uct_cuda_ctx_get_id(NULL, &ctx_id);
        if (result != CUDA_SUCCESS) {
            UCT_CUDADRV_LOG(cuCtxGetId, UCS_LOG_LEVEL_ERROR, result);
            status = UCS_ERR_IO_ERROR;
            goto err_pop_and_release;
        }
    }

    status = uct_cuda_base_ctx_rsc_get(&iface->super, ctx_id, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        goto err_pop_and_release;
    }

    *cuda_device_p  = cuda_device;
    *cuda_context_p = cuda_context;
    *ctx_rsc_p      = ucs_derived_of(ctx_rsc, uct_cuda_copy_ctx_rsc_t);
    return UCS_OK;

err_pop_and_release:
    uct_cuda_ctx_pop_and_release(cuda_device, cuda_context);
err:
    return status;
}

static UCS_F_ALWAYS_INLINE ucs_status_t uct_cuda_copy_ep_get_ctx(
        uct_cuda_copy_iface_t *iface, const void *src, const void *dst,
        size_t length, uct_cuda_copy_ep_ctx_t *ctx_p)
{
    ucs_memory_type_t src_type;
    ucs_memory_type_t dst_type;
    ucs_sys_device_t sys_dev;
    CUdeviceptr cuda_deviceptr;
    CUdevice cuda_device;
    CUcontext cuda_context;
    uct_cuda_copy_ctx_rsc_t *ctx_rsc;
    ucs_status_t status;

    uct_cuda_copy_get_mem_types(iface->super.super.md, src, dst, length,
                                &src_type, &dst_type, &sys_dev,
                                &cuda_deviceptr);

    status = uct_cuda_copy_ctx_rsc_get(iface, sys_dev, cuda_deviceptr,
                                       &cuda_device, &cuda_context, &ctx_rsc);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ctx_p->src_type     = src_type;
    ctx_p->dst_type     = dst_type;
    ctx_p->cuda_device  = cuda_device;
    ctx_p->cuda_context = cuda_context;
    ctx_p->ctx_rsc      = ctx_rsc;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_copy_post_cuda_async_copy(uct_ep_h tl_ep, void *dst, void *src,
                                   size_t length, uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                  uct_cuda_copy_iface_t);
    uct_cuda_copy_ep_ctx_t ctx;
    ucs_status_t status;
    uct_cuda_queue_desc_t *q_desc;
    ucs_queue_head_t *event_q;
    CUstream *stream;
    uct_cuda_event_desc_t *cuda_event;

    if (!length) {
        return UCS_OK;
    }

    status = uct_cuda_copy_ep_get_ctx(iface, src, dst, length, &ctx);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    q_desc  = &ctx.ctx_rsc->queue_desc[ctx.src_type][ctx.dst_type];
    event_q = &q_desc->event_queue;
    stream = uct_cuda_copy_get_stream(ctx.ctx_rsc, ctx.src_type, ctx.dst_type);
    if (ucs_unlikely(stream == NULL)) {
        ucs_error("stream for src %s dst %s not available",
                  ucs_memory_type_names[ctx.src_type],
                  ucs_memory_type_names[ctx.dst_type]);
        status = UCS_ERR_IO_ERROR;
        goto out_pop_and_release;
    }

    cuda_event = ucs_mpool_get(&ctx.ctx_rsc->super.event_mp);
    if (ucs_unlikely(cuda_event == NULL)) {
        ucs_error("failed to allocate cuda event object");
        status = UCS_ERR_NO_MEMORY;
        goto out_pop_and_release;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemcpyAsync((CUdeviceptr)dst, (CUdeviceptr)src, length, *stream));
    if (ucs_unlikely(UCS_OK != status)) {
        goto err_mpool_put;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuEventRecord(cuda_event->event, *stream));
    if (ucs_unlikely(UCS_OK != status)) {
        goto err_mpool_put;
    }

    if (ucs_queue_is_empty(event_q)) {
        ucs_queue_push(&iface->super.active_queue, &q_desc->queue);
    }

    ucs_queue_push(event_q, &cuda_event->queue);
    cuda_event->comp = comp;

    UCS_STATIC_BITMAP_SET(&iface->streams_to_sync,
                          uct_cuda_copy_flush_bitmap_idx(ctx.src_type,
                                                         ctx.dst_type));

    ucs_trace("cuda async issued: %p dst:%p[%s], src:%p[%s] len:%ld",
              cuda_event, dst, ucs_memory_type_names[ctx.dst_type], src,
              ucs_memory_type_names[ctx.src_type], length);
    status = UCS_INPROGRESS;

out_pop_and_release:
    uct_cuda_ctx_pop_and_release(ctx.cuda_device, ctx.cuda_context);
out:
    return status;
err_mpool_put:
    ucs_mpool_put(cuda_event);
    goto out_pop_and_release;
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

static UCS_F_ALWAYS_INLINE ucs_status_t uct_cuda_copy_ep_rma_short(
        uct_ep_h tl_ep, CUdeviceptr dst, CUdeviceptr src, unsigned length)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                  uct_cuda_copy_iface_t);
    uct_cuda_copy_ep_ctx_t ctx;
    ucs_status_t status;
    CUstream *stream;

    status = uct_cuda_copy_ep_get_ctx(iface, (void*)src, (void*)dst, length,
                                      &ctx);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out;
    }

    stream = &ctx.ctx_rsc->short_stream;
    status = uct_cuda_base_init_stream(stream);
    if (ucs_unlikely(status != UCS_OK)) {
        goto out_pop_and_release;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemcpyAsync(dst, src, length, *stream));
    if (ucs_unlikely(status != UCS_OK)) {
        goto out_pop_and_release;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuStreamSynchronize(*stream));

out_pop_and_release:
    uct_cuda_ctx_pop_and_release(ctx.cuda_device, ctx.cuda_context);
out:
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_copy_ep_put_short,
                 (tl_ep, buffer, length, remote_addr, rkey), uct_ep_h tl_ep,
                 const void *buffer, unsigned length, uint64_t remote_addr,
                 uct_rkey_t rkey)
{
    ucs_status_t status;

    status = uct_cuda_copy_ep_rma_short(tl_ep, (CUdeviceptr)remote_addr,
                                        (CUdeviceptr)buffer, length);

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
    ucs_status_t status;

    status = uct_cuda_copy_ep_rma_short(tl_ep, (CUdeviceptr)buffer,
                                        (CUdeviceptr)remote_addr, length);

    UCT_TL_EP_STAT_OP(ucs_derived_of(tl_ep, uct_base_ep_t), GET, SHORT, length);
    ucs_trace_data("GET_SHORT size %d from %p to %p",
                   length, (void *)remote_addr, buffer);
    return status;
}

