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

static void
uct_cuda_base_event_desc_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_cuda_event_desc_t *base = obj;

    memset(base, 0 , sizeof(*base));
    UCT_CUDADRV_FUNC_LOG_ERR(cuEventCreate(&base->event, CU_EVENT_DISABLE_TIMING));
}

int uct_cuda_base_is_ctx_rsc_valid(const uct_cuda_ctx_rsc_t *ctx_rsc)
{
#if CUDA_VERSION >= 12000
    unsigned long long ctx_id;
    CUresult result;

    result = uct_cuda_base_ctx_get_id(ctx_rsc->ctx, &ctx_id);
    if (result == CUDA_ERROR_CONTEXT_IS_DESTROYED) {
        return 0;
    } else if (result != CUDA_SUCCESS) {
        UCT_CUDADRV_LOG(cuCtxGetId, UCS_LOG_LEVEL_WARN, result);
        return 0;
    }

    return ctx_id == ctx_rsc->ctx_id;
#else
    /* Best effort check on older Cuda versions */
    return uct_cuda_base_is_context_valid(ctx_rsc->ctx);
#endif
}

static void uct_cuda_base_event_desc_cleanup(ucs_mpool_t *mp, void *obj)
{
    uct_cuda_event_desc_t *base = obj;
    uct_cuda_ctx_rsc_t *ctx_rsc = ucs_container_of(mp, uct_cuda_ctx_rsc_t,
                                                   event_mp);

    if (uct_cuda_base_is_ctx_rsc_valid(ctx_rsc)) {
        UCT_CUDADRV_FUNC_LOG_WARN(cuEventDestroy(base->event));
    }
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
    ucs_kh_put_t ret;
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

    ctx_rsc = iface->alloc_rsc();
    if (ctx_rsc == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_del_iter;
    }

    ucs_mpool_params_reset(&mp_params);
    mp_params.elem_size       = sizeof(uct_cuda_event_desc_t);
    mp_params.elems_per_chunk = 128;
    mp_params.max_elems       = iface->max_events;
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
    ucs_free(ctx_rsc);
err_del_iter:
    kh_del(cuda_ctx_rscs, &iface->ctx_rscs, iter);
    return UCS_ERR_NO_MEMORY;
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

    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_iface_t)
{
    ucs_async_eventfd_destroy(self->eventfd);
}

UCS_CLASS_DEFINE(uct_cuda_iface_t, uct_base_iface_t);
