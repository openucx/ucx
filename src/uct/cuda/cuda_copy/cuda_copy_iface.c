/**
 * Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_copy_iface.h"
#include "cuda_copy_md.h"
#include "cuda_copy_ep.h"

#include <uct/cuda/base/cuda_iface.h>
#include <uct/cuda/base/cuda_md.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>
#include <ucs/async/async.h>
#include <ucs/arch/cpu.h>


#define UCT_CUDA_COPY_IFACE_OVERHEAD 0
#define UCT_CUDA_COPY_IFACE_LATENCY  ucs_linear_func_make(8e-6, 0)


static ucs_config_field_t uct_cuda_copy_iface_config_table[] = {

    {"", "", NULL,
     ucs_offsetof(uct_cuda_copy_iface_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_iface_config_table)},

    {"MAX_POLL", "16",
     "Max number of event completions to pick during cuda events polling",
     ucs_offsetof(uct_cuda_copy_iface_config_t, max_poll), UCS_CONFIG_TYPE_UINT},

    {"MAX_EVENTS", "inf",
     "Max number of cuda events. -1 is infinite",
     ucs_offsetof(uct_cuda_copy_iface_config_t, max_cuda_events), UCS_CONFIG_TYPE_UINT},

    {"BW", "10000MBs",
     "Effective memory bandwidth",
     ucs_offsetof(uct_cuda_copy_iface_config_t, bandwidth), UCS_CONFIG_TYPE_BW},

    {NULL}
};


/* Forward declaration for the delete function */
static void UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_copy_iface_t)(uct_iface_t*);


static ucs_status_t uct_cuda_copy_iface_get_address(uct_iface_h tl_iface,
                                                    uct_iface_addr_t *iface_addr)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);

    *(uct_cuda_copy_iface_addr_t*)iface_addr = iface->id;
    return UCS_OK;
}

static int uct_cuda_copy_iface_is_reachable(const uct_iface_h tl_iface,
                                            const uct_device_addr_t *dev_addr,
                                            const uct_iface_addr_t *iface_addr)
{
    uct_cuda_copy_iface_t  *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);
    uct_cuda_copy_iface_addr_t *addr = (uct_cuda_copy_iface_addr_t*)iface_addr;

    return (addr != NULL) && (iface->id == *addr);
}

static ucs_status_t uct_cuda_copy_iface_query(uct_iface_h tl_iface,
                                              uct_iface_attr_t *iface_attr)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);

    uct_base_iface_query(&iface->super, iface_attr);

    iface_attr->iface_addr_len          = sizeof(uct_cuda_copy_iface_addr_t);
    iface_attr->device_addr_len         = 0;
    iface_attr->ep_addr_len             = 0;
    iface_attr->cap.flags               = UCT_IFACE_FLAG_CONNECT_TO_IFACE |
                                          UCT_IFACE_FLAG_GET_SHORT |
                                          UCT_IFACE_FLAG_PUT_SHORT |
                                          UCT_IFACE_FLAG_GET_ZCOPY |
                                          UCT_IFACE_FLAG_PUT_ZCOPY |
                                          UCT_IFACE_FLAG_PENDING;

    iface_attr->cap.event_flags         = UCT_IFACE_FLAG_EVENT_SEND_COMP |
                                          UCT_IFACE_FLAG_EVENT_RECV      |
                                          UCT_IFACE_FLAG_EVENT_ASYNC_CB;

    iface_attr->cap.put.max_short       = UINT_MAX;
    iface_attr->cap.put.max_bcopy       = 0;
    iface_attr->cap.put.min_zcopy       = 0;
    iface_attr->cap.put.max_zcopy       = SIZE_MAX;
    iface_attr->cap.put.opt_zcopy_align = 1;
    iface_attr->cap.put.align_mtu       = iface_attr->cap.put.opt_zcopy_align;
    iface_attr->cap.put.max_iov         = 1;

    iface_attr->cap.get.max_short       = UINT_MAX;
    iface_attr->cap.get.max_bcopy       = 0;
    iface_attr->cap.get.min_zcopy       = 0;
    iface_attr->cap.get.max_zcopy       = SIZE_MAX;
    iface_attr->cap.get.opt_zcopy_align = 1;
    iface_attr->cap.get.align_mtu       = iface_attr->cap.get.opt_zcopy_align;
    iface_attr->cap.get.max_iov         = 1;

    iface_attr->cap.am.max_short        = 0;
    iface_attr->cap.am.max_bcopy        = 0;
    iface_attr->cap.am.min_zcopy        = 0;
    iface_attr->cap.am.max_zcopy        = 0;
    iface_attr->cap.am.opt_zcopy_align  = 1;
    iface_attr->cap.am.align_mtu        = iface_attr->cap.am.opt_zcopy_align;
    iface_attr->cap.am.max_hdr          = 0;
    iface_attr->cap.am.max_iov          = 1;

    iface_attr->latency                 = UCT_CUDA_COPY_IFACE_LATENCY;
    iface_attr->bandwidth.dedicated     = 0;
    iface_attr->bandwidth.shared        = iface->config.bandwidth;
    iface_attr->overhead                = UCT_CUDA_COPY_IFACE_OVERHEAD;
    iface_attr->priority                = 0;

    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                              uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);
    uct_cuda_copy_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        if (!ucs_queue_is_empty(&q_desc->event_queue)) {
            UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface,
                                                        uct_base_iface_t));
            return UCS_INPROGRESS;
        }
    }

    UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;

}

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_copy_queue_head_ready(ucs_queue_head_t *queue_head)
{
    uct_cuda_copy_event_desc_t *cuda_event;

    if (ucs_queue_is_empty(queue_head)) {
        return 0;
    }

    cuda_event = ucs_queue_head_elem_non_empty(queue_head,
                                               uct_cuda_copy_event_desc_t,
                                               queue);
    return (cudaSuccess == cudaEventQuery(cuda_event->event));
}

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_copy_progress_event_queue(uct_cuda_copy_iface_t *iface,
                                   ucs_queue_head_t *queue_head,
                                   unsigned max_events)
{
    unsigned count = 0;
    uct_cuda_copy_event_desc_t *cuda_event;

    ucs_queue_for_each_extract(cuda_event, queue_head, queue,
                               cudaEventQuery(cuda_event->event) == cudaSuccess) {
        ucs_queue_remove(queue_head, &cuda_event->queue);
        if (cuda_event->comp != NULL) {
            ucs_trace_data("cuda_copy event %p completed", cuda_event);
            uct_invoke_completion(cuda_event->comp, UCS_OK);
        }
        ucs_trace_poll("CUDA Event Done :%p", cuda_event);
        ucs_mpool_put(cuda_event);
        count++;
        if (count >= max_events) {
            break;
        }
    }

    return count;
}

static unsigned uct_cuda_copy_iface_progress(uct_iface_h tl_iface)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);
    unsigned max_events = iface->config.max_poll;
    unsigned count      = 0;
    ucs_queue_head_t *event_q;
    uct_cuda_copy_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        count  += uct_cuda_copy_progress_event_queue(iface, event_q,
                                                     max_events - count);
        if (ucs_queue_is_empty(event_q)) {
            ucs_queue_del_iter(&iface->active_queue, iter);
        }
    }

    return count;
}

#if (__CUDACC_VER_MAJOR__ >= 100000)
static void CUDA_CB myHostFn(void *cuda_copy_iface)
#else
static void CUDA_CB myHostCallback(CUstream hStream,  CUresult status,
                                   void *cuda_copy_iface)
#endif
{
    uct_cuda_copy_iface_t *iface = cuda_copy_iface;

    ucs_assert(iface->async.event_cb != NULL);
    /* notify user */
    UCS_ASYNC_BLOCK(iface->super.worker->async);
    iface->async.event_cb(iface->async.event_arg, 0);
    UCS_ASYNC_UNBLOCK(iface->super.worker->async);
}

static ucs_status_t uct_cuda_copy_iface_event_fd_arm(uct_iface_h tl_iface,
                                                    unsigned events)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);
    ucs_status_t status;
    cudaStream_t *stream;
    ucs_queue_head_t *event_q;
    uct_cuda_copy_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        if (uct_cuda_copy_queue_head_ready(event_q)) {
            return UCS_ERR_BUSY;
        }
    }

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        stream  = &q_desc->stream;
        if (!ucs_queue_is_empty(event_q)) {
            status =
#if (__CUDACC_VER_MAJOR__ >= 100000)
                UCT_CUDADRV_FUNC_LOG_ERR(cuLaunchHostFunc(*stream,
                                                          myHostFn, iface));
#else
                UCT_CUDADRV_FUNC_LOG_ERR(cuStreamAddCallback(*stream,
                                                             myHostCallback,
                                                             iface, 0));
#endif
            if (UCS_OK != status) {
                return status;
            }
        }
    }

    return UCS_OK;
}

static uct_iface_ops_t uct_cuda_copy_iface_ops = {
    .ep_get_short             = uct_cuda_copy_ep_get_short,
    .ep_put_short             = uct_cuda_copy_ep_put_short,
    .ep_get_zcopy             = uct_cuda_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_cuda_copy_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_base_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_copy_ep_t),
    .iface_flush              = uct_cuda_copy_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_cuda_copy_iface_progress,
    .iface_event_fd_get       = (uct_iface_event_fd_get_func_t)ucs_empty_function_return_success,
    .iface_event_arm          = uct_cuda_copy_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_copy_iface_t),
    .iface_query              = uct_cuda_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_cuda_copy_iface_get_address,
    .iface_is_reachable       = uct_cuda_copy_iface_is_reachable,
};

static void uct_cuda_copy_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cuda_copy_event_desc_t *base = (uct_cuda_copy_event_desc_t *) obj;
    ucs_status_t status;

    memset(base, 0 , sizeof(*base));
    status = UCT_CUDA_FUNC_LOG_ERR(cudaEventCreateWithFlags(&base->event,
                                                            cudaEventDisableTiming));
    if (UCS_OK != status) {
        ucs_error("cudaEventCreateWithFlags Failed");
    }
}

static void uct_cuda_copy_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_cuda_copy_event_desc_t *base = (uct_cuda_copy_event_desc_t *) obj;
    uct_cuda_copy_iface_t *iface     = ucs_container_of(mp,
                                                        uct_cuda_copy_iface_t,
                                                        cuda_event_desc);
    CUcontext cuda_context;

    UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&cuda_context));
    if (uct_cuda_base_context_match(cuda_context, iface->cuda_context)) {
        UCT_CUDA_FUNC_LOG_ERR(cudaEventDestroy(base->event));
    }
}

static ucs_status_t
uct_cuda_copy_estimate_perf(uct_iface_h tl_iface, uct_perf_attr_t *perf_attr)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {
        perf_attr->bandwidth.dedicated = 0;
        if (!(perf_attr->field_mask & UCT_PERF_ATTR_FIELD_OPERATION)) {
            perf_attr->bandwidth.shared = iface->config.bandwidth;
        } else {
            switch (perf_attr->operation) {
            case UCT_EP_OP_GET_SHORT:
                perf_attr->bandwidth.shared = 9320.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_GET_ZCOPY:
                perf_attr->bandwidth.shared = 11660.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_PUT_SHORT:
                perf_attr->bandwidth.shared = 8110.0 * UCS_MBYTE;
                break;
            case UCT_EP_OP_PUT_ZCOPY:
                perf_attr->bandwidth.shared = 9980.0 * UCS_MBYTE;
                break;
            default:
                perf_attr->bandwidth.shared = iface->config.bandwidth;
                break;
            }
        }
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD) {
        perf_attr->send_pre_overhead = UCT_CUDA_COPY_IFACE_OVERHEAD;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD) {
        perf_attr->send_post_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_RECV_OVERHEAD) {
        perf_attr->recv_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_LATENCY) {
        perf_attr->latency = UCT_CUDA_COPY_IFACE_LATENCY;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
        perf_attr->max_inflight_eps = SIZE_MAX;
    }

    return UCS_OK;
}

static ucs_mpool_ops_t uct_cuda_copy_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_cuda_copy_event_desc_init,
    .obj_cleanup   = uct_cuda_copy_event_desc_cleanup,
    .obj_str       = NULL
};

static uct_iface_internal_ops_t uct_cuda_copy_iface_internal_ops = {
    .iface_estimate_perf = uct_cuda_copy_estimate_perf,
    .iface_vfs_refresh   = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query            = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate       = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported
};

static UCS_CLASS_INIT_FUNC(uct_cuda_copy_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cuda_copy_iface_config_t *config = ucs_derived_of(tl_config,
                                                          uct_cuda_copy_iface_config_t);
    ucs_status_t status;
    ucs_memory_type_t src, dst;

    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, &uct_cuda_copy_iface_ops,
                              &uct_cuda_copy_iface_internal_ops, md, worker,
                              params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG("cuda_copy"));

    if (strncmp(params->mode.device.dev_name,
                UCT_CUDA_DEV_NAME, strlen(UCT_CUDA_DEV_NAME)) != 0) {
        ucs_error("no device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    self->id                     = ucs_generate_uuid((uintptr_t)self);
    self->config.max_poll        = config->max_poll;
    self->config.max_cuda_events = config->max_cuda_events;
    self->config.bandwidth       = config->bandwidth;

    status = ucs_mpool_init(&self->cuda_event_desc,
                            0,
                            sizeof(uct_cuda_copy_event_desc_t),
                            0,
                            UCS_SYS_CACHE_LINE_SIZE,
                            128,
                            self->config.max_cuda_events,
                            &uct_cuda_copy_event_desc_mpool_ops,
                            "CUDA EVENT objects");

    if (UCS_OK != status) {
        ucs_error("mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }

    uct_iface_set_async_event_params(params, &self->async.event_cb,
                                     &self->async.event_arg);

    ucs_queue_head_init(&self->active_queue);

    for (src = 0; src < UCS_MEMORY_TYPE_LAST; ++src) {
        for (dst = 0; dst < UCS_MEMORY_TYPE_LAST; ++dst) {
            self->queue_desc[src][dst].stream = 0;
            ucs_queue_head_init(&self->queue_desc[src][dst].event_queue);
        }
    }

    self->short_stream = 0;
    self->cuda_context = 0;

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_copy_iface_t)
{
    cudaStream_t *stream;
    CUcontext cuda_context;
    ucs_queue_head_t *event_q;
    ucs_memory_type_t src, dst;

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&cuda_context));
    if (uct_cuda_base_context_match(cuda_context, self->cuda_context)) {

        for (src = 0; src < UCS_MEMORY_TYPE_LAST; ++src) {
            for (dst = 0; dst < UCS_MEMORY_TYPE_LAST; ++dst) {
                stream  = &self->queue_desc[src][dst].stream;
                event_q = &self->queue_desc[src][dst].event_queue;

                if (!ucs_queue_is_empty(event_q)) {
                    ucs_warn("stream destroyed but queue not empty");
                }

                if (*stream == 0) {
                    continue;
                }

                UCT_CUDA_FUNC_LOG_ERR(cudaStreamDestroy(*stream));
            }
        }

        if (self->short_stream) {
            UCT_CUDA_FUNC_LOG_ERR(cudaStreamDestroy(self->short_stream));
        }
    }

    ucs_mpool_cleanup(&self->cuda_event_desc, 1);
}

UCS_CLASS_DEFINE(uct_cuda_copy_iface_t, uct_base_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_copy_iface_t, uct_iface_t);


UCT_TL_DEFINE(&uct_cuda_copy_component, cuda_copy, uct_cuda_base_query_devices,
              uct_cuda_copy_iface_t, "CUDA_COPY_",
              uct_cuda_copy_iface_config_table, uct_cuda_copy_iface_config_t);
