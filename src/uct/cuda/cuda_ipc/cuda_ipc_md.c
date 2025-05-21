/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc.inl"
#include "cuda_ipc_cache.h"
#include "cuda_ipc_md.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/class.h>
#include <uct/api/v2/uct_v2.h>
#include <uct/cuda/base/cuda_nvml.h>

#include <limits.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t uct_cuda_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"ENABLE_MNNVL", "try",
     "Enable multi-node NVLINK capabilities.",
     ucs_offsetof(uct_cuda_ipc_md_config_t, enable_mnnvl), UCS_CONFIG_TYPE_TERNARY},

    {NULL}
};

static uct_cuda_ipc_dev_cache_t *uct_cuda_ipc_create_dev_cache(int dev_num)
{
    uct_cuda_ipc_dev_cache_t *cache;
    ucs_status_t status;
    int i, num_devices;

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetCount(&num_devices));
    if (UCS_OK != status) {
        return NULL;
    }

    cache = ucs_malloc(sizeof(*cache) + (num_devices * sizeof(uint8_t)),
                       "uct_cuda_ipc_dev_cache_t");
    if (cache == NULL) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_dev_cache_t");
        return NULL;
    }

    cache->dev_num = dev_num;
    for (i = 0; i < num_devices; ++i) {
        cache->accessible[i] = UCS_TRY;
    }

    return cache;
}

static uct_cuda_ipc_dev_cache_t *
uct_cuda_ipc_get_dev_cache(uct_cuda_ipc_component_t *component,
                           uct_cuda_ipc_rkey_t *rkey)
{
    khash_t(cuda_ipc_uuid_hash) *hash = &component->uuid_hash;
    uct_cuda_ipc_uuid_hash_key_t key;
    uct_cuda_ipc_dev_cache_t *cache;
    khiter_t iter;
    int ret;

    key.uuid = rkey->uuid;
    key.type = rkey->ph.handle_type;

    iter = kh_put(cuda_ipc_uuid_hash, hash, key, &ret);
    if (ret == UCS_KH_PUT_KEY_PRESENT) {
        return kh_val(hash, iter);
    } else if ((ret == UCS_KH_PUT_BUCKET_EMPTY) ||
               (ret == UCS_KH_PUT_BUCKET_CLEAR)) {
        cache = uct_cuda_ipc_create_dev_cache(kh_size(hash) - 1);
        if (NULL != cache) {
            kh_val(hash, iter) = cache;
        }
        return cache;
    } else {
        ucs_error("kh_put(cuda_ipc_uuid_hash) failed with %d", ret);
        return NULL;
    }
}

static ucs_status_t
uct_cuda_ipc_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG |
                                UCT_MD_FLAG_NEED_RKEY |
                                UCT_MD_FLAG_INVALIDATE |
                                UCT_MD_FLAG_INVALIDATE_RMA |
                                UCT_MD_FLAG_INVALIDATE_AMO |
                                UCT_MD_FLAG_MEMTYPE_COPY;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->rkey_packed_size = sizeof(uct_cuda_ipc_rkey_t);
    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_mem_add_reg(void *addr, uct_cuda_ipc_memh_t *memh,
                         uct_cuda_ipc_lkey_t **key_p)
{
    uct_cuda_ipc_lkey_t *key;
    ucs_status_t status;
    int is_ctx_pushed;
    CUdevice cuda_device;
#if HAVE_CUDA_FABRIC
#define UCT_CUDA_IPC_QUERY_NUM_ATTRS 3
    CUmemGenericAllocationHandle handle;
    CUmemoryPool mempool;
    CUpointer_attribute attr_type[UCT_CUDA_IPC_QUERY_NUM_ATTRS];
    void *attr_data[UCT_CUDA_IPC_QUERY_NUM_ATTRS];
    int legacy_capable;
    uint64_t allowed_handle_types;
#endif

    key = ucs_calloc(1, sizeof(*key), "uct_cuda_ipc_lkey_t");
    if (key == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_cuda_ipc_check_and_push_ctx((CUdeviceptr)addr, &cuda_device,
                                             &is_ctx_pushed);
    if (status != UCS_OK) {
        goto out;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuMemGetAddressRange(&key->d_bptr, &key->b_len, (CUdeviceptr)addr));
    if (status != UCS_OK) {
        goto out_pop_ctx;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuPointerGetAttribute(&key->ph.buffer_id,
                                      CU_POINTER_ATTRIBUTE_BUFFER_ID,
                                      (CUdeviceptr)addr));
    if (status != UCS_OK) {
        goto out_pop_ctx;
    }

#if HAVE_CUDA_FABRIC
    /* cuda_ipc can handle VMM, mallocasync, and legacy pinned device so need to
     * pack appropriate handle */

    attr_type[0] = CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE;
    attr_data[0] = &legacy_capable;
    attr_type[1] = CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES;
    attr_data[1] = &allowed_handle_types;
    attr_type[2] = CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE;
    attr_data[2] = &mempool;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerGetAttributes(ucs_static_array_size(attr_data), attr_type,
                attr_data, (CUdeviceptr)addr));
    if (status != UCS_OK) {
        goto out_pop_ctx;
    }

    if (legacy_capable) {
        goto legacy_path;
    }

    if (!(allowed_handle_types & CU_MEM_HANDLE_TYPE_FABRIC)) {
        goto non_ipc;
    }

    status =
        UCT_CUDADRV_FUNC(cuMemRetainAllocationHandle(&handle, addr),
                UCS_LOG_LEVEL_DIAG);
    if (status == UCS_OK) {
        status =
            UCT_CUDADRV_FUNC_LOG_ERR(cuMemExportToShareableHandle(
                        &key->ph.handle.fabric_handle, handle,
                        CU_MEM_HANDLE_TYPE_FABRIC, 0));
        if (status != UCS_OK) {
            cuMemRelease(handle);
            ucs_debug("unable to export handle for VMM ptr: %p", addr);
            goto non_ipc;
        }

        status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemRelease(handle));
        if (status != UCS_OK) {
            goto out_pop_ctx;
        }

        key->ph.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_VMM;
        ucs_trace("packed vmm fabric handle for %p", addr);
        goto common_path;
    }

    if (mempool == 0) {
        /* cuda_ipc can only handle UCS_MEMORY_TYPE_CUDA, which has to be either
         * legacy type, or VMM type, or mempool type. Return error if memory
         * does not belong to any of the three types */
        status = UCS_ERR_INVALID_ADDR;
        goto out_pop_ctx;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemPoolExportToShareableHandle(
                (void *)&key->ph.handle.fabric_handle, mempool,
                CU_MEM_HANDLE_TYPE_FABRIC, 0));
    if (status != UCS_OK) {
        ucs_debug("unable to export handle for mempool ptr: %p", addr);
        goto non_ipc;
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuMemPoolExportPointer(&key->ph.ptr,
                (CUdeviceptr)key->d_bptr));
    if (status != UCS_OK) {
        goto out_pop_ctx;
    }

    key->ph.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_MEMPOOL;
    ucs_trace("packed mempool handle and export pointer for %p", addr);
    goto common_path;

non_ipc:
    key->ph.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_NO_IPC;
    goto common_path;
#endif
legacy_path:
    key->ph.handle_type = UCT_CUDA_IPC_KEY_HANDLE_TYPE_LEGACY;
    status              = UCT_CUDADRV_FUNC_LOG_ERR(
            cuIpcGetMemHandle(&key->ph.handle.legacy, (CUdeviceptr)addr));
    if (status != UCS_OK) {
        goto out_pop_ctx;
    }

common_path:
    ucs_list_add_tail(&memh->list, &key->link);
    ucs_trace("registered addr:%p/%p length:%zd type:%u dev_num:%d "
              "buffer_id:%llu",
              addr, (void*)key->d_bptr, key->b_len, key->ph.handle_type,
              cuda_device, key->ph.buffer_id);

    memh->dev_num = cuda_device;
    *key_p        = key;
    status        = UCS_OK;

out_pop_ctx:
    uct_cuda_ipc_check_and_pop_ctx(is_ctx_pushed);
out:
    if (status != UCS_OK) {
        ucs_free(key);
    }

    return status;
}

static ucs_status_t
uct_cuda_ipc_mkey_pack(uct_md_h md, uct_mem_h tl_memh, void *address,
                       size_t length, const uct_md_mkey_pack_params_t *params,
                       void *mkey_buffer)
{
    uct_cuda_ipc_rkey_t *packed = mkey_buffer;
    uct_cuda_ipc_memh_t *memh   = tl_memh;
    uct_cuda_ipc_lkey_t *key;
    ucs_status_t status;

    ucs_list_for_each(key, &memh->list, link) {
        if (((uintptr_t)address >= key->d_bptr) &&
            ((uintptr_t)address < (key->d_bptr + key->b_len))) {
            goto found;
        }
    }

    status = uct_cuda_ipc_mem_add_reg(address, memh, &key);
    if (status != UCS_OK) {
        return status;
    }

found:
    ucs_assertv(((uintptr_t)address + length) <= (key->d_bptr + key->b_len),
                "buffer 0x%lx..0x%lx region 0x%llx..0x%llx", (uintptr_t)address,
                (uintptr_t)address + length, key->d_bptr, key->d_bptr +
                key->b_len);

    packed->pid    = memh->pid;
    packed->ph     = key->ph;
    packed->d_bptr = key->d_bptr;
    packed->b_len  = key->b_len;

    return UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetUuid(&packed->uuid,
                                                    memh->dev_num));
}

static ucs_status_t
uct_cuda_ipc_is_peer_accessible(uct_cuda_ipc_component_t *component,
                                uct_cuda_ipc_unpacked_rkey_t *rkey,
                                ucs_sys_device_t sys_dev)
{
    CUdevice cu_dev;
    ucs_status_t status;
    void *d_mapped;
    uct_cuda_ipc_dev_cache_t *cache;
    uint8_t *accessible;

    if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        /* Use device of the current context */
        if (UCT_CUDADRV_FUNC_LOG_DEBUG(cuCtxGetDevice(&cu_dev)) != UCS_OK) {
            return UCS_ERR_UNREACHABLE;
        }
    } else {
        status = uct_cuda_base_get_cuda_device(sys_dev, &cu_dev);
        if (status != UCS_OK) {
            ucs_warn("failed to map sys device [%d] to cuda device: %s",
                     sys_dev, ucs_status_string(status));
            return UCS_ERR_UNREACHABLE;
        }
    }

    pthread_mutex_lock(&component->lock);

    cache = uct_cuda_ipc_get_dev_cache(component, &rkey->super);
    if (ucs_unlikely(NULL == cache)) {
        status = UCS_ERR_NO_RESOURCE;
        goto err;
    }

    /* use unique id for stream id; this means that relative remote
     * device number of multiple peers do not map on the same stream and reduces
     * stream sequentialization */
    rkey->stream_id = cache->dev_num;
    accessible      = &cache->accessible[cu_dev];
    if (ucs_unlikely(*accessible == UCS_TRY)) { /* unchecked, add to cache */

        /* Check if peer is reachable by trying to open memory handle. This is
         * necessary when the device is not visible through CUDA_VISIBLE_DEVICES
         * and checking peer accessibility through CUDA driver API is not
         * possible.
         * Previously, reachability was checked by opening a memory handle
         * and immediately closing it as the handle to memory handle cache
         * was not not globally visible. Doing this with multiple threads is an
         * issue as a thread may first check reachability, and later open the
         * handle, and save mapped pointer in cache as part of a put/get
         * operation. At this point another thread can then close the same
         * memory handle as part of reachability check. This leads to a
         * cuMemcpyAsync error when accessing the mapped pointer as part of
         * put/get operation.
         * Now, we immediately insert into cache to save on calling
         * OpenMemHandle for the same handle because the cache is globally
         * accessible using rkey->pid. */
        status = uct_cuda_ipc_map_memhandle(&rkey->super, cu_dev, &d_mapped);

        *accessible = ((status == UCS_OK) || (status == UCS_ERR_ALREADY_EXISTS))
                      ? UCS_YES : UCS_NO;
    }

    status = (*accessible == UCS_YES) ? UCS_OK : UCS_ERR_UNREACHABLE;

err:
    pthread_mutex_unlock(&component->lock);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_rkey_unpack,
                 (component, rkey_buffer, params, rkey_p, handle_p),
                 uct_component_t *component, const void *rkey_buffer,
                 const uct_rkey_unpack_params_t *params, uct_rkey_t *rkey_p,
                 void **handle_p)
{
    uct_cuda_ipc_component_t *com = ucs_derived_of(component,
                                                   uct_cuda_ipc_component_t);
    uct_cuda_ipc_rkey_t *packed   = (uct_cuda_ipc_rkey_t *)rkey_buffer;
    uct_cuda_ipc_unpacked_rkey_t *unpacked;
    ucs_sys_device_t sys_dev;
    ucs_status_t status;

    sys_dev = UCS_PARAM_VALUE(UCT_RKEY_UNPACK_FIELD, params, sys_device,
                              SYS_DEVICE, UCS_SYS_DEVICE_ID_UNKNOWN);

    unpacked = ucs_malloc(sizeof(*unpacked), "uct_cuda_ipc_unpacked_rkey_t");
    if (NULL == unpacked) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_unpacked_rkey_t");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    unpacked->super = *packed;

    status = uct_cuda_ipc_is_peer_accessible(com, unpacked, sys_dev);
    if (status != UCS_OK) {
        goto err_free_key;
    }

    *handle_p = NULL;
    *rkey_p   = (uintptr_t) unpacked;
    return UCS_OK;

err_free_key:
    ucs_free(unpacked);
err:
    return status;
}

static ucs_status_t uct_cuda_ipc_rkey_release(uct_component_t *component,
                                              uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                     const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_cuda_ipc_memh_t *memh;

    memh = ucs_malloc(sizeof(*memh), "uct_cuda_ipc_memh_t");
    if (NULL == memh) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_memh_t");
        return UCS_ERR_NO_MEMORY;
    }

    /* dev_num is initialized during pack in uct_cuda_ipc_mem_add_reg */
    memh->dev_num = CU_DEVICE_INVALID;
    memh->pid     = getpid();
    ucs_list_head_init(&memh->list);

    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_mem_dereg(uct_md_h md, const uct_md_mem_dereg_params_t *params)
{
    uct_cuda_ipc_memh_t *memh = params->memh;
    uct_cuda_ipc_lkey_t *key, *tmp;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    ucs_list_for_each_safe(key, tmp, &memh->list, link) {
        ucs_free(key);
    }

    ucs_free(memh);
    return UCS_OK;
}

static int
uct_cuda_ipc_md_check_fabric_info(uct_cuda_ipc_md_t *md,
                                  ucs_ternary_auto_value_t mnnvl_enable)
{
#if !HAVE_NVML_FABRIC_INFO
    static int mnnvl_supported = 0;
#else
    static int mnnvl_supported = -1;
    nvmlGpuFabricInfoV_t fabric_info;
    nvmlDevice_t device;
    ucs_status_t status;
    char buf[64];
    int64_t *cluster_uuid;

    if (mnnvl_supported != -1) {
        goto out;
    }

    if (mnnvl_enable == UCS_NO) {
        mnnvl_supported = 0;
        goto out;
    }

    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, 0, &device);
    if (status != UCS_OK) {
        goto out_not_supported;
    }

    fabric_info.version = nvmlGpuFabricInfo_v2;
    status              = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetGpuFabricInfoV,
                                                  device, &fabric_info);
    if (status != UCS_OK) {
        goto out_not_supported;
    }

    ucs_debug("fabric_info: state=%u status=%u uuid=%s", fabric_info.state,
              fabric_info.status,
              ucs_str_dump_hex(fabric_info.clusterUuid,
                               NVML_GPU_FABRIC_UUID_LEN, buf, sizeof(buf),
                               SIZE_MAX));

    cluster_uuid = (int64_t*)fabric_info.clusterUuid;
    if ((fabric_info.state == NVML_GPU_FABRIC_STATE_COMPLETED) &&
        (fabric_info.status == NVML_SUCCESS) &&
        (cluster_uuid[0] | cluster_uuid[1]) != 0) {
        mnnvl_supported = 1;
        goto out;
    }

out_not_supported:
    mnnvl_supported = 0;
out:
#endif
    if ((mnnvl_enable == UCS_YES) && !mnnvl_supported) {
        ucs_error("multi-node NVLINK support is requested but not supported");
    } else {
        ucs_debug("multi-node NVLINK support is %s", mnnvl_supported ?
                  "enabled" : "disabled");
    }

    return mnnvl_supported;
}

static void uct_cuda_ipc_md_close(uct_md_h md)
{
    ucs_free(md);
}

static ucs_status_t
uct_cuda_ipc_md_open(uct_component_t *component, const char *md_name,
                     const uct_md_config_t *config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = uct_cuda_ipc_md_close,
        .query              = uct_cuda_ipc_md_query,
        .mem_alloc          = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
        .mem_free           = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
        .mem_advise         = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg            = uct_cuda_ipc_mem_reg,
        .mem_dereg          = uct_cuda_ipc_mem_dereg,
        .mem_query          = (uct_md_mem_query_func_t)ucs_empty_function_return_unsupported,
        .mkey_pack          = uct_cuda_ipc_mkey_pack,
        .mem_attach         = (uct_md_mem_attach_func_t)ucs_empty_function_return_unsupported,
        .detect_memory_type = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported
    };
    uct_cuda_ipc_md_config_t *ipc_config = ucs_derived_of(config,
                                                          uct_cuda_ipc_md_config_t);
    uct_cuda_ipc_md_t* md;

    md = ucs_calloc(1, sizeof(*md), "uct_cuda_ipc_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_cuda_ipc_component.super;
    md->enable_mnnvl    = uct_cuda_ipc_md_check_fabric_info(
                                                  md, ipc_config->enable_mnnvl);
    *md_p               = &md->super;

    return UCS_OK;
}

uct_cuda_ipc_component_t uct_cuda_ipc_component = {
    .super = {
        .query_md_resources = uct_cuda_base_query_md_resources,
        .md_open            = uct_cuda_ipc_md_open,
        .cm_open            = (uct_component_cm_open_func_t)ucs_empty_function_return_unsupported,
        .rkey_unpack        = uct_cuda_ipc_rkey_unpack,
        .rkey_ptr           = (uct_component_rkey_ptr_func_t)ucs_empty_function_return_unsupported,
        .rkey_release       = uct_cuda_ipc_rkey_release,
        .rkey_compare       = uct_base_rkey_compare,
        .name               = "cuda_ipc",
        .md_config          = {
            .name           = "Cuda-IPC memory domain",
            .prefix         = "CUDA_IPC_",
            .table          = uct_cuda_ipc_md_config_table,
            .size           = sizeof(uct_cuda_ipc_md_config_t),
        },
        .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
        .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_cuda_ipc_component.super),
        .flags              = 0,
        .md_vfs_init        =
                (uct_component_md_vfs_init_func_t)ucs_empty_function
    },
    .uuid_hash              = KHASH_STATIC_INITIALIZER,
    .lock                   = PTHREAD_MUTEX_INITIALIZER
};
UCT_COMPONENT_REGISTER(&uct_cuda_ipc_component.super);

UCS_STATIC_CLEANUP {
    uct_cuda_ipc_dev_cache_t *cache;

    kh_foreach_value(&uct_cuda_ipc_component.uuid_hash, cache, {
        ucs_free(cache);
    })
    kh_destroy_inplace(cuda_ipc_uuid_hash, &uct_cuda_ipc_component.uuid_hash);
    pthread_mutex_destroy(&uct_cuda_ipc_component.lock);
}
