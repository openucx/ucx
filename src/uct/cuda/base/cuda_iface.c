/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_iface.h"
#include "cuda_md.h"

#include <ucs/sys/string.h>


#define UCT_CUDA_DEV_NAME "cuda"


const char *uct_cuda_base_cu_get_error_string(CUresult result)
{
    static __thread char buf[64];
    const char *error_str;

    if (cuGetErrorString(result, &error_str) != CUDA_SUCCESS) {
        ucs_snprintf_safe(buf, sizeof(buf), "unrecognized error code %d",
                          result);
        error_str = buf;
    }

    return error_str;
}

ucs_status_t
uct_cuda_base_query_devices_common(
        uct_md_h md, uct_device_type_t dev_type,
        uct_tl_device_resource_t **tl_devices_p, unsigned *num_tl_devices_p)
{
    ucs_sys_device_t sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    CUdevice cuda_device;
    ucs_status_t status;

    if (uct_cuda_base_is_context_active()) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetDevice(&cuda_device));
        if (status != UCS_OK) {
            return status;
        }

        uct_cuda_base_get_sys_dev(cuda_device, &sys_device);
    } else {
        ucs_debug("set cuda sys_device to `unknown` as no context is"
                  " currently active");
    }

    return uct_single_device_resource(md, UCT_CUDA_DEV_NAME, dev_type,
                                      sys_device, tl_devices_p,
                                      num_tl_devices_p);
}

ucs_status_t
uct_cuda_base_query_devices(
        uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
        unsigned *num_tl_devices_p)
{
    return uct_cuda_base_query_devices_common(md, UCT_DEVICE_TYPE_ACC,
                                              tl_devices_p, num_tl_devices_p);
}

#if (__CUDACC_VER_MAJOR__ >= 100000)
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(void *arg)
#else
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(CUstream hStream, CUresult status,
                                               void *arg)
#endif
{
    uct_cuda_iface_t *cuda_iface = arg;

    ucs_async_eventfd_signal(cuda_iface->eventfd);
}

ucs_status_t uct_cuda_base_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);
    ucs_status_t status;

    if (iface->eventfd == UCS_ASYNC_EVENTFD_INVALID_FD) {
        status = ucs_async_eventfd_create(&iface->eventfd);
        if (status != UCS_OK) {
            return status;
        }
    }

    *fd_p = iface->eventfd;
    return UCS_OK;
}

ucs_status_t uct_cuda_base_check_device_name(const uct_iface_params_t *params)
{
    UCT_CHECK_PARAM(params->field_mask & UCT_IFACE_PARAM_FIELD_DEVICE,
                    "UCT_IFACE_PARAM_FIELD_DEVICE is not defined");

    if (strncmp(params->mode.device.dev_name, UCT_CUDA_DEV_NAME,
                strlen(UCT_CUDA_DEV_NAME)) != 0) {
        ucs_error("no device was found: %s", params->mode.device.dev_name);
        return UCS_ERR_NO_DEVICE;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_base_queue_head_ready(ucs_queue_head_t *queue_head)
{
    uct_cuda_event_desc_t *cuda_event;

    if (ucs_queue_is_empty(queue_head)) {
        return 0;
    }

    cuda_event = ucs_queue_head_elem_non_empty(queue_head,
                                               uct_cuda_event_desc_t, queue);
    return (CUDA_SUCCESS == cuEventQuery(cuda_event->event));
}

ucs_status_t uct_cuda_base_iface_event_fd_arm(uct_iface_h tl_iface,
                                              unsigned events)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);
    ucs_status_t status;
    CUstream *stream;
    ucs_queue_head_t *event_q;
    uct_cuda_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        if (uct_cuda_base_queue_head_ready(event_q)) {
            return UCS_ERR_BUSY;
        }
    }

    status = ucs_async_eventfd_poll(iface->eventfd);
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
                                         iface));
#else
                UCT_CUDADRV_FUNC_LOG_ERR(
                        cuStreamAddCallback(*stream,
                                            uct_cuda_base_iface_stream_cb_fxn,
                                            iface, 0));
#endif
            if (UCS_OK != status) {
                return status;
            }
        }
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE unsigned
uct_cuda_base_progress_event_queue(uct_cuda_iface_t *iface,
                                   ucs_queue_head_t *queue_head,
                                   unsigned max_events)
{
    unsigned count = 0;
    uct_cuda_event_desc_t *cuda_event;

    ucs_queue_for_each_extract(cuda_event, queue_head, queue,
                               (count < max_events) &&
                               (cuEventQuery(cuda_event->event) == CUDA_SUCCESS)) {
        ucs_trace_data("cuda event %p completed", cuda_event);
        if (cuda_event->comp != NULL) {
            uct_invoke_completion(cuda_event->comp, UCS_OK);
        }

        iface->ops->complete_event(&iface->super.super, cuda_event);
        ucs_mpool_put(cuda_event);
        count++;
    }

    return count;
}

unsigned uct_cuda_base_iface_progress(uct_iface_h tl_iface)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);
    unsigned max_events     = iface->config.max_poll;
    unsigned count          = 0;
    ucs_queue_head_t *event_q;
    uct_cuda_queue_desc_t *q_desc;
    ucs_queue_iter_t iter;

    ucs_queue_for_each_safe(q_desc, iter, &iface->active_queue, queue) {
        event_q = &q_desc->event_queue;
        count  += uct_cuda_base_progress_event_queue(iface, event_q,
                                                     max_events - count);
        if (ucs_queue_is_empty(event_q)) {
            ucs_queue_del_iter(&iface->active_queue, iter);
        }
    }

    return count;
}

ucs_status_t uct_cuda_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                       uct_completion_t *comp)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);

    if (comp != NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (!ucs_queue_is_empty(&iface->active_queue)) {
        UCT_TL_IFACE_STAT_FLUSH_WAIT(ucs_derived_of(tl_iface, uct_base_iface_t));
        return UCS_INPROGRESS;
    }

    UCT_TL_IFACE_STAT_FLUSH(ucs_derived_of(tl_iface, uct_base_iface_t));
    return UCS_OK;
}

void uct_cuda_base_stream_destroy(CUstream *stream)
{
    if (*stream != NULL) {
        (void)UCT_CUDADRV_FUNC_LOG_WARN(cuStreamDestroy(*stream));
    }
}

static void
uct_cuda_base_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cuda_event_desc_t *event_desc = obj;

    UCT_CUDADRV_FUNC_LOG_ERR(cuEventCreate(&event_desc->event,
                                           CU_EVENT_DISABLE_TIMING));
}

static void uct_cuda_base_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_cuda_event_desc_t *event_desc = obj;

    (void)UCT_CUDADRV_FUNC_LOG_WARN(cuEventDestroy(event_desc->event));
}

void uct_cuda_base_queue_desc_init(uct_cuda_queue_desc_t *qdesc)
{
    qdesc->stream = NULL;
    ucs_queue_head_init(&qdesc->event_queue);
}

void uct_cuda_base_queue_desc_destroy(const uct_cuda_ctx_rsc_t *ctx_rsc,
                                      uct_cuda_queue_desc_t *qdesc)
{
    if (!ucs_queue_is_empty(&qdesc->event_queue)) {
        ucs_warn("cuda context %llu stream being destroyed with  %zu "
                 "outstanding events", ctx_rsc->ctx_id,
                 ucs_queue_length(&qdesc->event_queue));
    }

    uct_cuda_base_stream_destroy(&qdesc->stream);
}

static ucs_mpool_ops_t uct_cuda_event_desc_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = uct_cuda_base_event_desc_init,
    .obj_cleanup   = uct_cuda_base_event_desc_cleanup,
    .obj_str       = NULL
};

ucs_status_t uct_cuda_base_ctx_rsc_create(uct_cuda_iface_t *iface,
                                          unsigned long long ctx_id,
                                          uct_cuda_ctx_rsc_t **ctx_rsc_p)
{
    CUcontext ctx;
    ucs_status_t status;
    int ret;
    khiter_t iter;
    uct_cuda_ctx_rsc_t *ctx_rsc;
    ucs_mpool_params_t mp_params;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&ctx));
    if (status != UCS_OK) {
        return status;
    } else if (ctx == NULL) {
        ucs_error("no cuda context bound to calling thread");
        return UCS_ERR_IO_ERROR;
    }

    iter = kh_put(cuda_ctx_rscs, &iface->ctx_rscs, ctx_id, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to allocate cuda context resource hash entry");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_assertv_always(ret != UCS_KH_PUT_KEY_PRESENT,
                       "the key has already been added iface=%p key=%llu",
                       iface, ctx_id);

    ctx_rsc = iface->ops->create_rsc(&iface->super.super);
    if (ctx_rsc == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_del_iter;
    }

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = iface->config.event_desc_size;
    mp_params.elems_per_chunk = 128;
    mp_params.max_elems       = iface->config.max_events;
    mp_params.ops             = &uct_cuda_event_desc_mpool_ops;
    mp_params.name            = "cuda_event_descriptors";

    status = ucs_mpool_init(&mp_params, &ctx_rsc->event_mp);
    if (status != UCS_OK) {
        goto err_free_ctx_rsc;
    }

    ctx_rsc->ctx                     = ctx;
    ctx_rsc->ctx_id                  = ctx_id;
    kh_value(&iface->ctx_rscs, iter) = ctx_rsc;
    *ctx_rsc_p                       = ctx_rsc;
    return UCS_OK;

err_free_ctx_rsc:
    iface->ops->destroy_rsc(&iface->super.super, ctx_rsc);
err_del_iter:
    kh_del(cuda_ctx_rscs, &iface->ctx_rscs, iter);
    return UCS_ERR_NO_MEMORY;
}

static void uct_cuda_base_ctx_rsc_destroy(uct_cuda_iface_t *iface,
                                          uct_cuda_ctx_rsc_t *ctx_rsc)
{
    ucs_mpool_cleanup(&ctx_rsc->event_mp, 1);
    iface->ops->destroy_rsc(&iface->super.super, ctx_rsc);
}

UCS_CLASS_INIT_FUNC(uct_cuda_iface_t, uct_iface_ops_t *tl_ops,
                    uct_iface_internal_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config,
                    const char *dev_name)
{
    UCS_CLASS_CALL_SUPER_INIT(uct_base_iface_t, tl_ops, ops, md, worker, params,
                              tl_config UCS_STATS_ARG(params->stats_root)
                              UCS_STATS_ARG(dev_name));

    self->eventfd = UCS_ASYNC_EVENTFD_INVALID_FD;
    kh_init_inplace(cuda_ctx_rscs, &self->ctx_rscs);
    ucs_queue_head_init(&self->active_queue);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_iface_t)
{
    uct_cuda_ctx_rsc_t *ctx_rsc;

    uct_base_iface_progress_disable(&self->super.super,
                                    UCT_PROGRESS_SEND | UCT_PROGRESS_RECV);

    kh_foreach_value(&self->ctx_rscs, ctx_rsc, {
        uct_cuda_base_ctx_rsc_destroy(self, ctx_rsc);
    });

    kh_destroy_inplace(cuda_ctx_rscs, &self->ctx_rscs);
    ucs_async_eventfd_destroy(self->eventfd);
}

UCS_CLASS_DEFINE(uct_cuda_iface_t, uct_base_iface_t);
