/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/profile/profile.h>
#include <ucs/async/eventfd.h>
#include <ucs/datastruct/khash.h>

#include <cuda.h>


const char *uct_cuda_base_cu_get_error_string(CUresult result);


#define UCT_CUDADRV_LOG(_func, _log_level, _result) \
    ucs_log((_log_level), "%s failed: %s", UCS_PP_MAKE_STRING(_func), \
            uct_cuda_base_cu_get_error_string(_result))


#define UCT_CUDADRV_FUNC(_func, _log_level) \
    ({ \
        CUresult _result = (_func); \
        ucs_status_t _status; \
        if (ucs_likely(_result == CUDA_SUCCESS)) { \
            _status = UCS_OK; \
        } else { \
            UCT_CUDADRV_LOG(_func, _log_level, _result); \
            _status = UCS_ERR_IO_ERROR; \
        } \
        _status; \
    })


#define UCT_CUDADRV_FUNC_LOG_ERR(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_ERROR)


#define UCT_CUDADRV_FUNC_LOG_WARN(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_WARN)


#define UCT_CUDADRV_FUNC_LOG_DEBUG(_func) \
    UCT_CUDADRV_FUNC(_func, UCS_LOG_LEVEL_DEBUG)


static UCS_F_ALWAYS_INLINE int uct_cuda_base_is_context_active()
{
    CUcontext ctx;

    return (CUDA_SUCCESS == cuCtxGetCurrent(&ctx)) && (ctx != NULL);
}


static UCS_F_ALWAYS_INLINE int uct_cuda_base_is_context_valid(CUcontext ctx)
{
    unsigned version;
    ucs_status_t status;

    /* Check if CUDA context is valid by running a dummy operation on it */
    status = UCT_CUDADRV_FUNC_LOG_DEBUG(cuCtxGetApiVersion(ctx, &version));
    return (status == UCS_OK);
}


static UCS_F_ALWAYS_INLINE int uct_cuda_base_context_match(CUcontext ctx1,
                                                           CUcontext ctx2)
{
    return ((ctx1 != NULL) && (ctx1 == ctx2) &&
            uct_cuda_base_is_context_valid(ctx1));
}


static UCS_F_ALWAYS_INLINE CUresult
uct_cuda_base_ctx_get_id(CUcontext ctx, unsigned long long *ctx_id_p)
{
    unsigned long long ctx_id = 0;

#if CUDA_VERSION >= 12000
    CUresult result = cuCtxGetId(ctx, &ctx_id);
    if (ucs_unlikely(result != CUDA_SUCCESS)) {
        return result;
    }
#endif

    *ctx_id_p = ctx_id;
    return CUDA_SUCCESS;
}


typedef enum uct_cuda_base_gen {
    UCT_CUDA_BASE_GEN_P100 = 6,
    UCT_CUDA_BASE_GEN_V100 = 7,
    UCT_CUDA_BASE_GEN_A100 = 8,
    UCT_CUDA_BASE_GEN_H100 = 9,
    UCT_CUDA_BASE_GEN_B100 = 10
} uct_cuda_base_gen_t;


typedef struct {
    /* Needed to allow queue descriptor to be added to iface queue */
    ucs_queue_elem_t queue;
    /* Stream on which asynchronous memcpy operations are enqueued */
    CUstream         stream;
    /* Queue of cuda events */
    ucs_queue_head_t event_queue;
} uct_cuda_queue_desc_t;


typedef struct {
    ucs_queue_elem_t queue;
    CUevent          event;
    uct_completion_t *comp;
} uct_cuda_event_desc_t;


typedef struct {
    /* CUDA context handle */
    CUcontext          ctx;
    /* CUDA context id */
    unsigned long long ctx_id;
    /* pool of cuda events to check completion of memcpy operations */
    ucs_mpool_t        event_mp;
} uct_cuda_ctx_rsc_t;


/* Hash map for CUDA context resources. The key is the CUDA context Id. */
KHASH_INIT(cuda_ctx_rscs, unsigned long long, uct_cuda_ctx_rsc_t*, 1,
           kh_int64_hash_func, kh_int64_hash_equal);


typedef uct_cuda_ctx_rsc_t* (*uct_cuda_create_rsc_fn_t)(uct_iface_h);
typedef void (*uct_cuda_destroy_rsc_fn_t)(uct_iface_h, uct_cuda_ctx_rsc_t*);
typedef void (*uct_cuda_complete_event_fn_t)(uct_iface_h, uct_cuda_event_desc_t*);


typedef struct {
    uct_cuda_create_rsc_fn_t     create_rsc;
    uct_cuda_destroy_rsc_fn_t    destroy_rsc;
    uct_cuda_complete_event_fn_t complete_event;
} uct_cuda_iface_ops_t;


typedef struct {
    uct_base_iface_t          super;
    int                       eventfd;
    /* CUDA resources per context */
    khash_t(cuda_ctx_rscs)    ctx_rscs;
    /* list of queues which require progress */
    ucs_queue_head_t          active_queue;
    uct_cuda_iface_ops_t      *ops;

    struct {
        unsigned              max_events;
        unsigned              max_poll;
        size_t                event_desc_size;
    } config;
} uct_cuda_iface_t;

ucs_status_t uct_cuda_base_iface_event_fd_get(uct_iface_h tl_iface, int *fd_p);

ucs_status_t uct_cuda_base_iface_event_fd_arm(uct_iface_h tl_iface,
                                              unsigned events);

unsigned uct_cuda_base_iface_progress(uct_iface_h tl_iface);

ucs_status_t uct_cuda_base_iface_flush(uct_iface_h tl_iface, unsigned flags,
                                       uct_completion_t *comp);

ucs_status_t
uct_cuda_base_query_devices_common(
        uct_md_h md, uct_device_type_t dev_type,
        uct_tl_device_resource_t **tl_devices_p, unsigned *num_tl_devices_p);

void
uct_cuda_base_get_sys_dev(CUdevice cuda_device, ucs_sys_device_t *sys_dev_p);

ucs_status_t
uct_cuda_base_get_cuda_device(ucs_sys_device_t sys_dev, CUdevice *device);

ucs_status_t uct_cuda_base_ctx_rsc_create(uct_cuda_iface_t *iface,
                                          unsigned long long ctx_id,
                                          uct_cuda_ctx_rsc_t **ctx_rsc_p);

void uct_cuda_base_queue_desc_init(uct_cuda_queue_desc_t *qdesc);

void uct_cuda_base_queue_desc_destroy(const uct_cuda_ctx_rsc_t *ctx_rsc,
                                      uct_cuda_queue_desc_t *qdesc);

void uct_cuda_base_stream_destroy(const uct_cuda_ctx_rsc_t *ctx_rsc,
                                  CUstream *stream);

#if (__CUDACC_VER_MAJOR__ >= 100000)
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(void *arg);
#else
void CUDA_CB uct_cuda_base_iface_stream_cb_fxn(CUstream hStream,  CUresult status,
                                               void *arg);
#endif

UCS_CLASS_INIT_FUNC(uct_cuda_iface_t, uct_iface_ops_t *tl_ops,
                    uct_iface_internal_ops_t *ops, uct_md_h md,
                    uct_worker_h worker, const uct_iface_params_t *params,
                    const uct_iface_config_t *tl_config, const char *dev_name);


/**
 * Retain the primary context on the given CUDA device.
 *
 * @param [in]  cuda_device Device for which primary context is requested.
 * @param [in]  force       Retain the primary context regardless of its state.
 * @param [out] cuda_ctx_p  Returned context handle of the retained context.
 *
 * @return UCS_OK if the method completes successfully. UCS_ERR_NO_DEVICE if the
 *         primary device context is inactive on the given CUDA device and
 *         retaining is not forced. UCS_ERR_IO_ERROR if the CUDA driver API
 *         methods called inside failed with an error.
 */
ucs_status_t uct_cuda_primary_ctx_retain(CUdevice cuda_device, int force,
                                         CUcontext *cuda_ctx_p);


static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_base_ctx_rsc_get(uct_cuda_iface_t *iface, unsigned long long ctx_id,
                          uct_cuda_ctx_rsc_t **ctx_rsc_p)
{
    khiter_t iter;

    iter = kh_get(cuda_ctx_rscs, &iface->ctx_rscs, ctx_id);
    if (ucs_likely(iter != kh_end(&iface->ctx_rscs))) {
        *ctx_rsc_p = kh_value(&iface->ctx_rscs, iter);
        return UCS_OK;
    }

    return uct_cuda_base_ctx_rsc_create(iface, ctx_id, ctx_rsc_p);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_base_init_stream(CUstream *stream)
{
    if (ucs_likely(*stream != NULL)) {
        return UCS_OK;
    }

    return UCT_CUDADRV_FUNC_LOG_ERR(
            cuStreamCreate(stream, CU_STREAM_NON_BLOCKING));
}

#endif
