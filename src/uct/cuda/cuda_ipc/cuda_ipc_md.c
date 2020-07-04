/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2019.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t uct_cuda_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t uct_cuda_ipc_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags            = UCT_MD_FLAG_REG |
                                    UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_CUDA);
    md_attr->cap.access_mem_type  = UCS_MEMORY_TYPE_CUDA;
    md_attr->cap.detect_mem_types = 0;
    md_attr->cap.max_alloc        = 0;
    md_attr->cap.max_reg          = ULONG_MAX;
    md_attr->rkey_packed_size     = sizeof(uct_cuda_ipc_key_t);
    md_attr->reg_cost             = ucs_linear_func_make(0, 0);
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_cuda_ipc_key_t *packed   = (uct_cuda_ipc_key_t *) rkey_buffer;
    uct_cuda_ipc_key_t *mem_hndl = (uct_cuda_ipc_key_t *) memh;

    *packed          = *mem_hndl;

    return UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGetUuid(&packed->uuid,
                                                    mem_hndl->dev_num));
}

static inline int uct_cuda_ipc_uuid_equals(const CUuuid* a, const CUuuid* b)
{
    int64_t *a0 = (int64_t *) a->bytes;
    int64_t *b0 = (int64_t *) b->bytes;
    return (a0[0] == b0[0]) && (a0[1] == b0[1]) ? 1 : 0;
}

static inline void uct_cuda_ipc_uuid_copy(CUuuid* dst, const CUuuid* src)
{
    int64_t *a = (int64_t *) src->bytes;
    int64_t *b = (int64_t *) dst->bytes;
    *b++ = *a++;
    *b   = *a;
}

ucs_status_t uct_cuda_ipc_get_unique_index_for_uuid(int* idx,
                                                    uct_cuda_ipc_md_t* md,
                                                    uct_cuda_ipc_key_t *rkey)
{
    int i;

    for (i = 0; i < md->uuid_map_size; i++) {
        if (uct_cuda_ipc_uuid_equals(&rkey->uuid, &md->uuid_map[i])) {
            *idx = i;
            return UCS_OK; /* found */
        }
    }

    if (ucs_unlikely(md->uuid_map_size == md->uuid_map_capacity)) {
        /* reallocate on demand */
        int num_devices;
        int original_cache_size, new_cache_size;
        int new_capacity = md->uuid_map_capacity * 2;

        UCT_CUDA_IPC_DEVICE_GET_COUNT(num_devices);
        original_cache_size   = md->uuid_map_capacity * num_devices;
        new_cache_size        = new_capacity * num_devices;
        md->uuid_map_capacity = new_capacity;
        md->uuid_map          = ucs_realloc(md->uuid_map,
                                            new_capacity * sizeof(CUuuid),
                                            "uct_cuda_ipc_uuid_map");
        if (md->uuid_map == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        md->peer_accessible_cache = ucs_realloc(md->peer_accessible_cache,
                                                new_cache_size,
                                                "uct_cuda_ipc_peer_accessible_cache");
        if (md->peer_accessible_cache == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        memset(md->peer_accessible_cache + original_cache_size, 0xFF,
               new_cache_size - original_cache_size);
    }

    /* Add new mapping */
    uct_cuda_ipc_uuid_copy(&md->uuid_map[md->uuid_map_size], &rkey->uuid);
    *idx = md->uuid_map_size++;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_is_peer_accessible(uct_cuda_ipc_component_t *mdc,
                                                    uct_cuda_ipc_key_t *rkey)
{
    CUdevice this_device;
    ucs_status_t status;
    int peer_idx;
    int num_devices;
    char* accessible;
    CUdeviceptr d_mapped;

    status = uct_cuda_ipc_get_unique_index_for_uuid(&peer_idx, mdc->md, rkey);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    /* overwrite dev_num with a unique ID; this means that relative remote
     * device number of multiple peers do not map on the same stream and reduces
     * stream sequentialization */
    rkey->dev_num = peer_idx;

    UCT_CUDA_IPC_GET_DEVICE(this_device);
    UCT_CUDA_IPC_DEVICE_GET_COUNT(num_devices);

    accessible = &mdc->md->peer_accessible_cache[peer_idx * num_devices + this_device];
    if (*accessible == (char)0xFF) { /* unchecked, add to cache */
        CUresult result = cuIpcOpenMemHandle(&d_mapped,
                                             rkey->ph,
                                             CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
        *accessible = ((result != CUDA_SUCCESS) && (result != CUDA_ERROR_ALREADY_MAPPED))
                    ? 0 : 1;
        if (result == CUDA_SUCCESS) {
            result = cuIpcCloseMemHandle(d_mapped);
            if (result != CUDA_SUCCESS) ucs_fatal("Unable to close memhandle");
        }
    }

    return (*accessible == 1) ? UCS_OK : UCS_ERR_UNREACHABLE;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_rkey_unpack,
                 (component, rkey_buffer, rkey_p, handle_p),
                 uct_component_t *component, const void *rkey_buffer,
                 uct_rkey_t *rkey_p, void **handle_p)
{
    uct_cuda_ipc_component_t *com = ucs_derived_of(component, uct_cuda_ipc_component_t);
    uct_cuda_ipc_key_t *packed    = (uct_cuda_ipc_key_t *) rkey_buffer;
    uct_cuda_ipc_key_t *key;
    ucs_status_t status;

    status = uct_cuda_ipc_is_peer_accessible(com, packed);
    if (status != UCS_OK) {
        return status;
    }

    key = ucs_malloc(sizeof(uct_cuda_ipc_key_t), "uct_cuda_ipc_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    *key      = *packed;
    *handle_p = NULL;
    *rkey_p   = (uintptr_t) key;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_rkey_release(uct_component_t *component,
                                              uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_mem_reg_internal(uct_md_h uct_md, void *addr, size_t length,
                              unsigned flags, uct_cuda_ipc_key_t *key)
{
    ucs_log_level_t log_level;
    CUdevice cu_device;
    ucs_status_t status;

    if (!length) {
        return UCS_OK;
    }

    log_level = (flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ? UCS_LOG_LEVEL_DEBUG :
                UCS_LOG_LEVEL_ERROR;
    status    = UCT_CUDADRV_FUNC(cuIpcGetMemHandle(&key->ph, (CUdeviceptr)addr),
                                 log_level);
    if (UCS_OK != status) {
        return status;
    }

    UCT_CUDA_IPC_GET_DEVICE(cu_device);

    UCT_CUDADRV_FUNC(cuMemGetAddressRange(&key->d_bptr, &key->b_len,
                                          (CUdeviceptr)addr),
                     log_level);

    key->dev_num  = (int) cu_device;
    ucs_trace("registered memory:%p..%p length:%lu dev_num:%d",
              addr, UCS_PTR_BYTE_OFFSET(addr, length), length, (int) cu_device);
    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_cuda_ipc_key_t *key;
    ucs_status_t status;

    key = ucs_malloc(sizeof(uct_cuda_ipc_key_t), "uct_cuda_ipc_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_cuda_ipc_mem_reg_internal(md, address, length, 0, key);
    if (status != UCS_OK) {
        ucs_free(key);
        return status;
    }
    *memh_p = key;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    ucs_free(memh);
    return UCS_OK;
}


static void uct_cuda_ipc_md_close(uct_md_h uct_md)
{
    uct_cuda_ipc_md_t *md = ucs_derived_of(uct_md, uct_cuda_ipc_md_t);

    ucs_free(md->uuid_map);
    ucs_free(md->peer_accessible_cache);
    ucs_free(md);
}

static ucs_status_t
uct_cuda_ipc_md_open(uct_component_t *component, const char *md_name,
                     const uct_md_config_t *config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = uct_cuda_ipc_md_close,
        .query              = uct_cuda_ipc_md_query,
        .mkey_pack          = uct_cuda_ipc_mkey_pack,
        .mem_reg            = uct_cuda_ipc_mem_reg,
        .mem_dereg          = uct_cuda_ipc_mem_dereg,
        .detect_memory_type = ucs_empty_function_return_unsupported,
    };

    int num_devices;
    uct_cuda_ipc_md_t* md;
    uct_cuda_ipc_component_t* com;

    UCS_STATIC_ASSERT(sizeof(md->peer_accessible_cache[0]) == sizeof(char));
    UCT_CUDA_IPC_DEVICE_GET_COUNT(num_devices);

    md = ucs_calloc(1, sizeof(uct_cuda_ipc_md_t), "uct_cuda_ipc_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_cuda_ipc_component.super;

    /* allocate uuid map and peer accessible cache */
    md->uuid_map_size     = 0;
    md->uuid_map_capacity = 16;
    md->uuid_map          = ucs_malloc(md->uuid_map_capacity * sizeof(CUuuid),
                                       "uct_cuda_ipc_uuid_map");
    if (md->uuid_map == NULL) {
        free(md);
        return UCS_ERR_NO_MEMORY;
    }

    /* Initially support caching accessibility of up to 16 other peers */
    md->peer_accessible_cache = ucs_malloc(num_devices * md->uuid_map_capacity,
                                           "uct_cuda_ipc_peer_accessible_cache");
    if (md->peer_accessible_cache == NULL) {
        free(md->uuid_map);
        free(md);
        return UCS_ERR_NO_MEMORY;
    }

    /* 0xFF = !cached, 1 = accessible, 0 = !accessible */
    memset(md->peer_accessible_cache, 0xFF, num_devices * md->uuid_map_capacity);

    com     = ucs_derived_of(md->super.component, uct_cuda_ipc_component_t);
    com->md = md;
    *md_p   = &md->super;
    return UCS_OK;
}

uct_cuda_ipc_component_t uct_cuda_ipc_component = {
    .super = {
        .query_md_resources = uct_cuda_base_query_md_resources,
        .md_open            = uct_cuda_ipc_md_open,
        .cm_open            = ucs_empty_function_return_unsupported,
        .rkey_unpack        = uct_cuda_ipc_rkey_unpack,
        .rkey_ptr           = ucs_empty_function_return_unsupported,
        .rkey_release       = uct_cuda_ipc_rkey_release,
        .name               = "cuda_ipc",
        .md_config          = {
            .name           = "Cuda-IPC memory domain",
            .prefix         = "CUDA_IPC_",
            .table          = uct_cuda_ipc_md_config_table,
            .size           = sizeof(uct_cuda_ipc_md_config_t),
        },
        .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
        .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_cuda_ipc_component.super),
        .flags              = 0
    },
    .md                     = NULL,
};
UCT_COMPONENT_REGISTER(&uct_cuda_ipc_component.super);

