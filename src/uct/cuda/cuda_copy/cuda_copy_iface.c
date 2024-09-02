/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2017-2019. ALL RIGHTS RESERVED.
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
#include <ucs/async/eventfd.h>
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

static int uct_cuda_copy_iface_is_reachable_v2(
        const uct_iface_h tl_iface,
        const uct_iface_is_reachable_params_t *params)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface,
                                                  uct_cuda_copy_iface_t);
    uct_cuda_copy_iface_addr_t *addr;

    if (!uct_iface_is_reachable_params_addrs_valid(params)) {
        return 0;
    }

    addr = (uct_cuda_copy_iface_addr_t*)params->iface_addr;
    if (addr == NULL) {
        uct_iface_fill_info_str_buf(params, "device address is empty");
        return 0;
    }

    if (iface->id != *addr) {
        uct_iface_fill_info_str_buf(
                params, "different iface id %"PRIx64" vs %"PRIx64"",
                iface->id, *addr);
        return 0;
    }

    return uct_iface_scope_is_reachable(tl_iface, params);
}

static ucs_status_t uct_cuda_copy_iface_query(uct_iface_h tl_iface,
                                              uct_iface_attr_t *iface_attr)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);

    uct_base_iface_query(&iface->super.super, iface_attr);

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
                                          UCT_IFACE_FLAG_EVENT_FD;

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

static ucs_status_t uct_cuda_copy_sync_streams(uct_cuda_copy_iface_t *iface)
{
    CUstream stream;
    uint32_t stream_index;
    ucs_memory_type_t src_mem_type, dst_mem_type;
    ucs_status_t status;

    UCS_STATIC_BITMAP_FOR_EACH_BIT(stream_index, &iface->streams_to_sync) {
        src_mem_type = stream_index / UCS_MEMORY_TYPE_LAST;
        if ((src_mem_type >= UCS_MEMORY_TYPE_LAST)) {
            break;
        }

        dst_mem_type = stream_index % UCS_MEMORY_TYPE_LAST;
        stream       = iface->queue_desc[src_mem_type][dst_mem_type].stream;
        status       = UCT_CUDADRV_FUNC_LOG_ERR(cuStreamSynchronize(stream));
        if (status != UCS_OK) {
            return status;
        }

        UCS_STATIC_BITMAP_RESET(&iface->streams_to_sync,
                                uct_cuda_copy_flush_bitmap_idx(src_mem_type,
                                                               dst_mem_type));
    }

    return UCS_OK;
}

static ucs_status_t uct_cuda_copy_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                              uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);
    uct_cuda_copy_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;
    ucs_status_t status;

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = uct_cuda_copy_sync_streams(iface);
    if (status != UCS_OK) {
        return status;
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
    return (CUDA_SUCCESS == cuEventQuery(cuda_event->event));
}

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_copy_progress_event_queue(uct_cuda_copy_iface_t *iface,
                                   ucs_queue_head_t *queue_head,
                                   unsigned max_events)
{
    unsigned count = 0;
    uct_cuda_copy_event_desc_t *cuda_event;

    ucs_queue_for_each_extract(cuda_event, queue_head, queue,
                               cuEventQuery(cuda_event->event) ==
                                       CUDA_SUCCESS) {
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

static ucs_status_t uct_cuda_copy_iface_event_fd_arm(uct_iface_h tl_iface,
                                                    unsigned events)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_copy_iface_t);
    ucs_status_t status;
    CUstream *stream;
    ucs_queue_head_t *event_q;
    uct_cuda_copy_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        if (uct_cuda_copy_queue_head_ready(event_q)) {
            return UCS_ERR_BUSY;
        }
    }

    status = ucs_async_eventfd_poll(iface->super.eventfd);
    if (status == UCS_OK) {
        return UCS_ERR_BUSY;
    } else if (status == UCS_ERR_IO_ERROR) {
        return status;
    }

    ucs_assertv(status == UCS_ERR_NO_PROGRESS, "%s", ucs_status_string(status));

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        stream  = &q_desc->stream;
        if (!ucs_queue_is_empty(event_q)) {
            status =
#if (__CUDACC_VER_MAJOR__ >= 100000)
                UCT_CUDADRV_FUNC_LOG_ERR(
                        cuLaunchHostFunc(*stream,
                                         uct_cuda_base_iface_stream_cb_fxn,
                                         &iface->super));
#else
                UCT_CUDADRV_FUNC_LOG_ERR(
                        cuStreamAddCallback(*stream,
                                            uct_cuda_base_iface_stream_cb_fxn,
                                            &iface->super, 0));
#endif
            if (UCS_OK != status) {
                return status;
            }
        }
    }

    return UCS_OK;
}

static ucs_status_t
uct_cuda_copy_ep_flush(uct_ep_h tl_ep, unsigned flags, uct_completion_t *comp)
{
    uct_cuda_copy_iface_t *iface = ucs_derived_of(tl_ep->iface,
                                                  uct_cuda_copy_iface_t);
    ucs_status_t status;

    status = uct_cuda_copy_sync_streams(iface);
    if (status != UCS_OK) {
        return status;
    }

    return uct_base_ep_flush(tl_ep, flags, comp);
}

static uct_iface_ops_t uct_cuda_copy_iface_ops = {
    .ep_get_short             = uct_cuda_copy_ep_get_short,
    .ep_put_short             = uct_cuda_copy_ep_put_short,
    .ep_get_zcopy             = uct_cuda_copy_ep_get_zcopy,
    .ep_put_zcopy             = uct_cuda_copy_ep_put_zcopy,
    .ep_pending_add           = ucs_empty_function_return_busy,
    .ep_pending_purge         = ucs_empty_function,
    .ep_flush                 = uct_cuda_copy_ep_flush,
    .ep_fence                 = uct_base_ep_fence,
    .ep_create                = UCS_CLASS_NEW_FUNC_NAME(uct_cuda_copy_ep_t),
    .ep_destroy               = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_copy_ep_t),
    .iface_flush              = uct_cuda_copy_iface_flush,
    .iface_fence              = uct_base_iface_fence,
    .iface_progress_enable    = uct_base_iface_progress_enable,
    .iface_progress_disable   = uct_base_iface_progress_disable,
    .iface_progress           = uct_cuda_copy_iface_progress,
    .iface_event_fd_get       = uct_cuda_base_iface_event_fd_get,
    .iface_event_arm          = uct_cuda_copy_iface_event_fd_arm,
    .iface_close              = UCS_CLASS_DELETE_FUNC_NAME(uct_cuda_copy_iface_t),
    .iface_query              = uct_cuda_copy_iface_query,
    .iface_get_device_address = (uct_iface_get_device_address_func_t)ucs_empty_function_return_success,
    .iface_get_address        = uct_cuda_copy_iface_get_address,
    .iface_is_reachable       = uct_base_iface_is_reachable
};

static void uct_cuda_copy_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cuda_copy_event_desc_t *base = (uct_cuda_copy_event_desc_t *) obj;
    ucs_status_t status;

    memset(base, 0 , sizeof(*base));
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuEventCreate(&base->event, CU_EVENT_DISABLE_TIMING));
    if (UCS_OK != status) {
        ucs_error("cuEventCreate Failed");
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
        UCT_CUDADRV_FUNC_LOG_ERR(cuEventDestroy(base->event));
    }
}

static ucs_status_t
uct_cuda_copy_estimate_perf(uct_iface_h tl_iface, uct_perf_attr_t *perf_attr)
{
    uct_cuda_copy_iface_t *iface   = ucs_derived_of(tl_iface,
                                                    uct_cuda_copy_iface_t);
    uct_ep_operation_t op          = UCT_ATTR_VALUE(PERF, perf_attr, operation,
                                                    OPERATION, UCT_EP_OP_LAST);
    ucs_memory_type_t src_mem_type = UCT_ATTR_VALUE(PERF, perf_attr,
                                                    local_memory_type,
                                                    LOCAL_MEMORY_TYPE,
                                                    UCS_MEMORY_TYPE_UNKNOWN);
    ucs_memory_type_t dst_mem_type = UCT_ATTR_VALUE(PERF, perf_attr,
                                                    remote_memory_type,
                                                    REMOTE_MEMORY_TYPE,
                                                    UCS_MEMORY_TYPE_UNKNOWN);
    int zcopy                      = uct_ep_op_is_zcopy(op);
    const double latency           = 1.8e-6;
    const double overhead          = 4.0e-6;

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {
        if (uct_ep_op_is_fetch(op)) {
            ucs_swap(&src_mem_type, &dst_mem_type);
        }

        perf_attr->bandwidth.dedicated = 0;
        if ((src_mem_type == UCS_MEMORY_TYPE_HOST) &&
            (dst_mem_type == UCS_MEMORY_TYPE_CUDA)) {
            perf_attr->bandwidth.shared = (zcopy ? 8300.0 : 7900.0) * UCS_MBYTE;
        } else if ((src_mem_type == UCS_MEMORY_TYPE_CUDA) &&
                   (dst_mem_type == UCS_MEMORY_TYPE_HOST)) {
            perf_attr->bandwidth.shared = (zcopy ? 11660.0 : 9320.0) *
                                          UCS_MBYTE;
        } else if ((src_mem_type == UCS_MEMORY_TYPE_CUDA) &&
                   (dst_mem_type == UCS_MEMORY_TYPE_CUDA)) {
            perf_attr->bandwidth.shared = 320.0 * UCS_GBYTE;
        } else {
            perf_attr->bandwidth.shared = iface->config.bandwidth;
        }
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_PRE_OVERHEAD) {
        perf_attr->send_pre_overhead = overhead;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_SEND_POST_OVERHEAD) {
        /* In case of sync mem copy, the send operation CPU overhead includes
           the latency of waiting for the copy to complete */
        perf_attr->send_post_overhead = zcopy ? 0 : latency;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_RECV_OVERHEAD) {
        perf_attr->recv_overhead = 0;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_LATENCY) {
        /* In case of async mem copy, the latency is not part of the overhead
           and it's a standalone property */
        perf_attr->latency = ucs_linear_func_make(zcopy ? latency : 0.0, 0.0);
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
        perf_attr->max_inflight_eps = SIZE_MAX;
    }

    if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_FLAGS) {
        perf_attr->flags = 0;
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
    .iface_estimate_perf   = uct_cuda_copy_estimate_perf,
    .iface_vfs_refresh     = (uct_iface_vfs_refresh_func_t)ucs_empty_function,
    .ep_query              = (uct_ep_query_func_t)ucs_empty_function_return_unsupported,
    .ep_invalidate         = (uct_ep_invalidate_func_t)ucs_empty_function_return_unsupported,
    .ep_connect_to_ep_v2   = ucs_empty_function_return_unsupported,
    .iface_is_reachable_v2 = uct_cuda_copy_iface_is_reachable_v2,
    .ep_is_connected       = uct_base_ep_is_connected
};

static UCS_CLASS_INIT_FUNC(uct_cuda_copy_iface_t, uct_md_h md, uct_worker_h worker,
                           const uct_iface_params_t *params,
                           const uct_iface_config_t *tl_config)
{
    uct_cuda_copy_iface_config_t *config = ucs_derived_of(tl_config,
                                                          uct_cuda_copy_iface_config_t);
    ucs_status_t status;
    ucs_memory_type_t src, dst;
    ucs_mpool_params_t mp_params;

    UCS_CLASS_CALL_SUPER_INIT(uct_cuda_iface_t, &uct_cuda_copy_iface_ops,
                              &uct_cuda_copy_iface_internal_ops, md, worker,
                              params, tl_config, "cuda_copy");

    status = uct_cuda_base_check_device_name(params);
    if (status != UCS_OK) {
        return status;
    }

    self->id                     = ucs_generate_uuid((uintptr_t)self);
    self->config.max_poll        = config->max_poll;
    self->config.max_cuda_events = config->max_cuda_events;
    self->config.bandwidth       = config->bandwidth;
    UCS_STATIC_BITMAP_RESET_ALL(&self->streams_to_sync);

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = sizeof(uct_cuda_copy_event_desc_t);
    mp_params.elems_per_chunk = 128;
    mp_params.max_elems       = self->config.max_cuda_events;
    mp_params.ops             = &uct_cuda_copy_event_desc_mpool_ops;
    mp_params.name            = "CUDA EVENT objects";
    status = ucs_mpool_init(&mp_params, &self->cuda_event_desc);
    if (UCS_OK != status) {
        ucs_error("mpool creation failed");
        return UCS_ERR_IO_ERROR;
    }

    ucs_queue_head_init(&self->active_queue);

    ucs_memory_type_for_each(src) {
        ucs_memory_type_for_each(dst) {
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
    CUstream *stream;
    CUcontext cuda_context;
    ucs_queue_head_t *event_q;
    ucs_memory_type_t src, dst;

    uct_base_iface_progress_disable(&self->super.super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&cuda_context));
    if (uct_cuda_base_context_match(cuda_context, self->cuda_context)) {

        ucs_memory_type_for_each(src) {
            ucs_memory_type_for_each(dst) {
                stream  = &self->queue_desc[src][dst].stream;
                event_q = &self->queue_desc[src][dst].event_queue;

                if (!ucs_queue_is_empty(event_q)) {
                    ucs_warn("stream destroyed but queue not empty");
                }

                if (*stream == 0) {
                    continue;
                }

                UCT_CUDADRV_FUNC_LOG_ERR(cuStreamDestroy(*stream));
            }
        }

        if (self->short_stream) {
            UCT_CUDADRV_FUNC_LOG_ERR(cuStreamDestroy(self->short_stream));
        }
    }

    ucs_mpool_cleanup(&self->cuda_event_desc, 1);
}

UCS_CLASS_DEFINE(uct_cuda_copy_iface_t, uct_cuda_iface_t);
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_copy_iface_t, uct_iface_t, uct_md_h, uct_worker_h,
                          const uct_iface_params_t*, const uct_iface_config_t*);
static UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_copy_iface_t, uct_iface_t);


UCT_TL_DEFINE(&uct_cuda_copy_component, cuda_copy, uct_cuda_base_query_devices,
              uct_cuda_copy_iface_t, "CUDA_COPY_",
              uct_cuda_copy_iface_config_table, uct_cuda_copy_iface_config_t);
