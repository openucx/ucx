/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_cft.h"
#include "cuda_ipc_md.h"

#include <uct/cuda/base/cuda_ctx.h>
#include <uct/cuda/base/cuda_util.h>
#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/list.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/ptr_arith.h>

#include <inttypes.h>
#include <pthread.h>
#include <string.h>


#if HAVE_CUDA_LOGICAL_ENDPOINT

/* Minimal driver version which provides the logical endpoint API */
#define UCT_CUDA_IPC_CFT_DRIVER_VERSION 13030

/* Logical endpoint driver functions resolved at runtime */
#define UCT_CUDA_IPC_CFT_FOREACH_OP(_op) \
    _op(IdReserve) _op(IdRelease) _op(Create) _op(Destroy) _op(Query) \
    _op(GetLimits) _op(BindAddr) _op(Unbind) _op(Export) _op(Import)


/* Range of free offsets in a logical endpoint */
typedef struct {
    ucs_list_link_t list;
    uint64_t        offset;
    uint64_t        length;
} uct_cuda_ipc_cft_range_t;


/* Logical endpoint owned by this process on a device */
typedef struct {
    uint32_t                      le_id;
    uint64_t                      bind_align;
    CUlogicalEndpointFabricHandle handle;
    /* Free offset ranges, sorted by offset */
    ucs_list_link_t               free_list;
} uct_cuda_ipc_cft_le_t;


static UCS_F_ALWAYS_INLINE int
uct_cuda_ipc_cft_handle_equal(CUlogicalEndpointFabricHandle handle1,
                              CUlogicalEndpointFabricHandle handle2)
{
    return !memcmp(&handle1, &handle2, sizeof(handle1));
}


static UCS_F_ALWAYS_INLINE khint_t
uct_cuda_ipc_cft_handle_hash(CUlogicalEndpointFabricHandle handle)
{
    return uct_cuda_ipc_fabric_handle_hash(handle.data, sizeof(handle.data));
}


KHASH_INIT(uct_cuda_ipc_cft_import, CUlogicalEndpointFabricHandle, uint32_t, 1,
           uct_cuda_ipc_cft_handle_hash, uct_cuda_ipc_cft_handle_equal)


#define UCT_CUDA_IPC_CFT_OP_FIELD(_name) \
    __typeof__(cuLogicalEndpoint##_name) *_name;


static struct {
    pthread_mutex_t                  lock;
    /* Driver API availability: UCS_TRY until resolved */
    int                              supported;
    size_t                           le_size;
    uct_cuda_ipc_cft_le_t            *le[UCT_CUDA_MAX_DEVICES];
    /* Fabric handle -> local logical endpoint id, including the logical
     * endpoints owned by this process */
    khash_t(uct_cuda_ipc_cft_import) import_hash;
    struct {
        UCT_CUDA_IPC_CFT_FOREACH_OP(UCT_CUDA_IPC_CFT_OP_FIELD)
    } ops;
} uct_cuda_ipc_cft = {
    .lock        = PTHREAD_MUTEX_INITIALIZER,
    .supported   = UCS_TRY,
    .import_hash = KHASH_STATIC_INITIALIZER
};


static ucs_status_t uct_cuda_ipc_cft_load_ops(void)
{
#define UCT_CUDA_IPC_CFT_OP_ENTRY(_name) \
    {"cuLogicalEndpoint" #_name, (void**)&uct_cuda_ipc_cft.ops._name},

    struct {
        const char *name;
        void       **func_p;
    } funcs[] = {UCT_CUDA_IPC_CFT_FOREACH_OP(UCT_CUDA_IPC_CFT_OP_ENTRY)};

#undef UCT_CUDA_IPC_CFT_OP_ENTRY
    CUdriverProcAddressQueryResult sym_status;
    CUresult cu_err;
    int i;

    for (i = 0; i < ucs_static_array_size(funcs); i++) {
        cu_err = cuGetProcAddress(funcs[i].name, funcs[i].func_p,
                                  UCT_CUDA_IPC_CFT_DRIVER_VERSION,
                                  CU_GET_PROC_ADDRESS_DEFAULT, &sym_status);
        if ((cu_err != CUDA_SUCCESS) ||
            (sym_status != CU_GET_PROC_ADDRESS_SUCCESS)) {
            ucs_debug("%s is not provided by the CUDA driver", funcs[i].name);
            return UCS_ERR_UNSUPPORTED;
        }
    }

    return UCS_OK;
}

static int uct_cuda_ipc_cft_device_supported(CUdevice cu_dev)
{
    int supported;

    if (UCT_CUDADRV_FUNC_LOG_DEBUG(cuDeviceGetAttribute(
                &supported, CU_DEVICE_ATTRIBUTE_LOGICAL_ENDPOINT_UNICAST_SUPPORTED,
                cu_dev)) != UCS_OK) {
        return 0;
    }

    return supported;
}

ucs_status_t uct_cuda_ipc_cft_init(size_t le_size)
{
    CUdevice cu_dev;

    pthread_mutex_lock(&uct_cuda_ipc_cft.lock);
    if (uct_cuda_ipc_cft.supported == UCS_TRY) {
        /* md_open can run without a current context, so query device 0 */
        uct_cuda_ipc_cft.supported =
                (uct_cuda_ipc_cft_load_ops() == UCS_OK) &&
                (UCT_CUDADRV_FUNC_LOG_DEBUG(cuDeviceGet(&cu_dev, 0)) == UCS_OK) &&
                uct_cuda_ipc_cft_device_supported(cu_dev);
        uct_cuda_ipc_cft.le_size   = le_size;
        ucs_debug("CUDA fabric transport support is %s",
                  uct_cuda_ipc_cft.supported ? "enabled" : "disabled");
    }
    pthread_mutex_unlock(&uct_cuda_ipc_cft.lock);

    return uct_cuda_ipc_cft_is_enabled() ? UCS_OK : UCS_ERR_UNSUPPORTED;
}

int uct_cuda_ipc_cft_is_enabled(void)
{
    return uct_cuda_ipc_cft.supported == 1;
}

static ucs_status_t uct_cuda_ipc_cft_wait_ready(uint32_t le_id)
{
    ucs_status_t status;
    int ready;

    do {
        status = UCT_CUDADRV_FUNC_LOG_ERR(
                uct_cuda_ipc_cft.ops.Query(le_id, 1, &ready));
    } while ((status == UCS_OK) && !ready);

    return status;
}

static void uct_cuda_ipc_cft_id_destroy(uint32_t le_id)
{
    UCT_CUDADRV_FUNC_LOG_WARN(uct_cuda_ipc_cft.ops.Destroy(le_id));
    UCT_CUDADRV_FUNC_LOG_WARN(uct_cuda_ipc_cft.ops.IdRelease(le_id, 1));
}

static ucs_status_t
uct_cuda_ipc_cft_import_add(CUlogicalEndpointFabricHandle handle,
                            uint32_t le_id)
{
    khiter_t iter;
    int ret;

    iter = kh_put(uct_cuda_ipc_cft_import, &uct_cuda_ipc_cft.import_hash,
                  handle, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to add logical endpoint %u to import hash", le_id);
        return UCS_ERR_NO_MEMORY;
    }

    kh_val(&uct_cuda_ipc_cft.import_hash, iter) = le_id;
    return UCS_OK;
}

/* Must be called with the lock held and a context of the device current */
static ucs_status_t
uct_cuda_ipc_cft_le_get(CUdevice cu_dev, uct_cuda_ipc_cft_le_t **le_p)
{
    CUlogicalEndpointProp prop = {};
    uct_cuda_ipc_cft_range_t *range;
    uct_cuda_ipc_cft_le_t *le;
    cuuint64_t bind_align, max_size;
    ucs_status_t status;

    if (uct_cuda_ipc_cft.le[cu_dev] != NULL) {
        *le_p = uct_cuda_ipc_cft.le[cu_dev];
        return UCS_OK;
    }

    if (!uct_cuda_ipc_cft_device_supported(cu_dev)) {
        return UCS_ERR_UNSUPPORTED;
    }

    prop.type           = CU_LOGICAL_ENDPOINT_TYPE_UNICAST;
    prop.flags          = CU_LOGICAL_ENDPOINT_FLAG_NONE;
    prop.unicast.device = cu_dev;
    prop.ipcHandleTypes = CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC;
    status = UCT_CUDADRV_FUNC_LOG_ERR(
            uct_cuda_ipc_cft.ops.GetLimits(&bind_align, &max_size, &prop));
    if (status != UCS_OK) {
        return status;
    }

    le = ucs_calloc(1, sizeof(*le), "uct_cuda_ipc_cft_le");
    range = ucs_malloc(sizeof(*range), "uct_cuda_ipc_cft_range");
    if ((le == NULL) || (range == NULL)) {
        status = UCS_ERR_NO_MEMORY;
        goto err_free;
    }

    le->bind_align = bind_align;
    prop.size      = ucs_align_down(ucs_min(uct_cuda_ipc_cft.le_size, max_size),
                                    bind_align);

    status = UCT_CUDADRV_FUNC_LOG_ERR(uct_cuda_ipc_cft.ops.IdReserve(&le->le_id, 1));
    if (status != UCS_OK) {
        goto err_free;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(uct_cuda_ipc_cft.ops.Create(le->le_id, &prop));
    if (status != UCS_OK) {
        UCT_CUDADRV_FUNC_LOG_WARN(uct_cuda_ipc_cft.ops.IdRelease(le->le_id, 1));
        goto err_free;
    }

    status = uct_cuda_ipc_cft_wait_ready(le->le_id);
    if (status != UCS_OK) {
        goto err_destroy;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(uct_cuda_ipc_cft.ops.Export(
            &le->handle, le->le_id, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    if (status != UCS_OK) {
        goto err_destroy;
    }

    /* The owner accesses its logical endpoint by its own id */
    status = uct_cuda_ipc_cft_import_add(le->handle, le->le_id);
    if (status != UCS_OK) {
        goto err_destroy;
    }

    range->offset = 0;
    range->length = prop.size;
    ucs_list_head_init(&le->free_list);
    ucs_list_add_tail(&le->free_list, &range->list);

    uct_cuda_ipc_cft.le[cu_dev] = le;
    ucs_debug("created logical endpoint %u on device %d size %zu bind_align %"
              PRIu64, le->le_id, cu_dev, (size_t)prop.size, le->bind_align);
    *le_p = le;
    return UCS_OK;

err_destroy:
    uct_cuda_ipc_cft_id_destroy(le->le_id);
err_free:
    ucs_free(range);
    ucs_free(le);
    return status;
}

static ucs_status_t
uct_cuda_ipc_cft_range_alloc(uct_cuda_ipc_cft_le_t *le, uint64_t length,
                             uint64_t *offset_p)
{
    uct_cuda_ipc_cft_range_t *range;

    ucs_list_for_each(range, &le->free_list, list) {
        if (range->length < length) {
            continue;
        }

        *offset_p      = range->offset;
        range->offset += length;
        range->length -= length;
        if (range->length == 0) {
            ucs_list_del(&range->list);
            ucs_free(range);
        }
        return UCS_OK;
    }

    return UCS_ERR_NO_RESOURCE;
}

static void uct_cuda_ipc_cft_range_free(uct_cuda_ipc_cft_le_t *le,
                                        uint64_t offset, uint64_t length)
{
    uct_cuda_ipc_cft_range_t *range, *prev = NULL, *next = NULL;

    ucs_list_for_each(range, &le->free_list, list) {
        if (range->offset > offset) {
            next = range;
            break;
        }
        prev = range;
    }

    if ((prev != NULL) && ((prev->offset + prev->length) == offset)) {
        prev->length += length;
        if ((next != NULL) && ((prev->offset + prev->length) == next->offset)) {
            prev->length += next->length;
            ucs_list_del(&next->list);
            ucs_free(next);
        }
        return;
    }

    if ((next != NULL) && ((offset + length) == next->offset)) {
        next->offset  = offset;
        next->length += length;
        return;
    }

    range = ucs_malloc(sizeof(*range), "uct_cuda_ipc_cft_range");
    if (range == NULL) {
        ucs_warn("failed to allocate free range, leaking logical endpoint "
                 "offset %" PRIu64 " length %" PRIu64, offset, length);
        return;
    }

    range->offset = offset;
    range->length = length;
    if (next != NULL) {
        ucs_list_insert_before(&next->list, &range->list);
    } else {
        ucs_list_add_tail(&le->free_list, &range->list);
    }
}

ucs_status_t uct_cuda_ipc_cft_bind(CUdevice cu_dev, CUdeviceptr d_bptr,
                                   size_t b_len,
                                   CUlogicalEndpointFabricHandle *handle,
                                   uint64_t *le_offset_p)
{
    uct_cuda_ipc_cft_le_t *le;
    ucs_status_t status;
    uint64_t le_offset;

    ucs_assert(uct_cuda_ipc_cft_is_enabled());
    ucs_assert(cu_dev < UCT_CUDA_MAX_DEVICES);

    status = uct_cuda_ctx_primary_push(cu_dev, 1, UCS_LOG_LEVEL_ERROR);
    if (status != UCS_OK) {
        return status;
    }

    pthread_mutex_lock(&uct_cuda_ipc_cft.lock);
    status = uct_cuda_ipc_cft_le_get(cu_dev, &le);
    if (status != UCS_OK) {
        goto out;
    }

    if ((d_bptr % le->bind_align) || (b_len % le->bind_align)) {
        ucs_debug("allocation 0x%llx..0x%llx is not aligned to %" PRIu64
                  ", logical endpoint is not used", d_bptr, d_bptr + b_len,
                  le->bind_align);
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    status = uct_cuda_ipc_cft_range_alloc(le, b_len, &le_offset);
    if (status != UCS_OK) {
        ucs_debug("no space for %zu bytes in logical endpoint %u", b_len,
                  le->le_id);
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(uct_cuda_ipc_cft.ops.BindAddr(
            le->le_id, cu_dev, le_offset, (void*)d_bptr, b_len, 0));
    if (status != UCS_OK) {
        uct_cuda_ipc_cft_range_free(le, le_offset, b_len);
        goto out;
    }

    ucs_trace("bound 0x%llx..0x%llx to logical endpoint %u offset %" PRIu64,
              d_bptr, d_bptr + b_len, le->le_id, le_offset);
    *handle      = le->handle;
    *le_offset_p = le_offset;

out:
    pthread_mutex_unlock(&uct_cuda_ipc_cft.lock);
    uct_cuda_ctx_primary_pop_and_release(cu_dev);
    return status;
}

void uct_cuda_ipc_cft_unbind(CUdevice cu_dev, uint64_t le_offset,
                             size_t b_len)
{
    uct_cuda_ipc_cft_le_t *le;

    if (uct_cuda_ctx_primary_push(cu_dev, 1, UCS_LOG_LEVEL_WARN) != UCS_OK) {
        return;
    }

    pthread_mutex_lock(&uct_cuda_ipc_cft.lock);
    le = uct_cuda_ipc_cft.le[cu_dev];
    ucs_assert(le != NULL);

    UCT_CUDADRV_FUNC_LOG_WARN(uct_cuda_ipc_cft.ops.Unbind(le->le_id, cu_dev,
                                                          le_offset, b_len));
    uct_cuda_ipc_cft_range_free(le, le_offset, b_len);
    pthread_mutex_unlock(&uct_cuda_ipc_cft.lock);
    uct_cuda_ctx_primary_pop_and_release(cu_dev);
}

ucs_status_t
uct_cuda_ipc_cft_import(const CUlogicalEndpointFabricHandle *handle,
                        uint32_t *le_id_p)
{
    ucs_status_t status = UCS_OK;
    CUlogicalEndpointFabricHandle handle_copy;
    khiter_t iter;
    uint32_t le_id;

    ucs_assert(uct_cuda_ipc_cft_is_enabled());

    pthread_mutex_lock(&uct_cuda_ipc_cft.lock);
    iter = kh_get(uct_cuda_ipc_cft_import, &uct_cuda_ipc_cft.import_hash,
                  *handle);
    if (iter != kh_end(&uct_cuda_ipc_cft.import_hash)) {
        *le_id_p = kh_val(&uct_cuda_ipc_cft.import_hash, iter);
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(uct_cuda_ipc_cft.ops.IdReserve(&le_id, 1));
    if (status != UCS_OK) {
        goto out;
    }

    handle_copy = *handle;
    status      = UCT_CUDADRV_FUNC_LOG_ERR(uct_cuda_ipc_cft.ops.Import(
            le_id, &handle_copy, CU_LOGICAL_ENDPOINT_IPC_HANDLE_TYPE_FABRIC));
    if (status != UCS_OK) {
        UCT_CUDADRV_FUNC_LOG_WARN(uct_cuda_ipc_cft.ops.IdRelease(le_id, 1));
        goto out;
    }

    status = uct_cuda_ipc_cft_wait_ready(le_id);
    if (status == UCS_OK) {
        status = uct_cuda_ipc_cft_import_add(*handle, le_id);
    }

    if (status != UCS_OK) {
        uct_cuda_ipc_cft_id_destroy(le_id);
        goto out;
    }

    ucs_debug("imported logical endpoint as id %u", le_id);
    *le_id_p = le_id;

out:
    pthread_mutex_unlock(&uct_cuda_ipc_cft.lock);
    return status;
}

UCS_STATIC_CLEANUP {
    uct_cuda_ipc_cft_range_t *range, *tmp;
    uct_cuda_ipc_cft_le_t *le;
    CUdevice cu_dev;
    uint32_t le_id;
    int num_devices;

    /* The import hash holds every logical endpoint id, including the owned
     * ones. The driver releases them itself if it is already shut down. */
    if ((uct_cuda_ipc_cft.supported == 1) &&
        (cuDeviceGetCount(&num_devices) != CUDA_ERROR_DEINITIALIZED)) {
        kh_foreach_value(&uct_cuda_ipc_cft.import_hash, le_id, {
            uct_cuda_ipc_cft_id_destroy(le_id);
        })
    }

    for (cu_dev = 0; cu_dev < UCT_CUDA_MAX_DEVICES; cu_dev++) {
        le = uct_cuda_ipc_cft.le[cu_dev];
        if (le == NULL) {
            continue;
        }

        ucs_list_for_each_safe(range, tmp, &le->free_list, list) {
            ucs_free(range);
        }
        ucs_free(le);
    }

    kh_destroy_inplace(uct_cuda_ipc_cft_import, &uct_cuda_ipc_cft.import_hash);
}

#else /* HAVE_CUDA_LOGICAL_ENDPOINT */

ucs_status_t uct_cuda_ipc_cft_init(size_t le_size)
{
    ucs_debug("CUDA fabric transport support is disabled because UCX was "
              "built without CUDA logical endpoint support");
    return UCS_ERR_UNSUPPORTED;
}

int uct_cuda_ipc_cft_is_enabled(void)
{
    return 0;
}

#endif /* HAVE_CUDA_LOGICAL_ENDPOINT */
