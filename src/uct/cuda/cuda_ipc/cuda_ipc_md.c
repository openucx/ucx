/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2018-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "cuda_ipc_md.h"
#include "cuda_ipc_cache.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/type/class.h>
#include <ucs/profile/profile.h>
#include <uct/api/v2/uct_v2.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t uct_cuda_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_cuda_ipc_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {NULL}
};

static ucs_status_t
uct_cuda_ipc_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->flags            = UCT_MD_FLAG_REG |
                                UCT_MD_FLAG_NEED_RKEY |
                                UCT_MD_FLAG_INVALIDATE |
                                UCT_MD_FLAG_INVALIDATE_RMA |
                                UCT_MD_FLAG_INVALIDATE_AMO;
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

    key = ucs_malloc(sizeof(*key), "uct_cuda_ipc_lkey_t");
    if (key == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    status = UCT_CUDADRV_FUNC(cuIpcGetMemHandle(&key->ph, (CUdeviceptr)addr),
                              UCS_LOG_LEVEL_ERROR);
    if (UCS_OK != status) {
        goto err;
    }

    UCT_CUDADRV_FUNC(cuMemGetAddressRange(&key->d_bptr, &key->b_len,
                                          (CUdeviceptr)addr),
                     UCS_LOG_LEVEL_ERROR);

    ucs_list_add_tail(&memh->list, &key->link);
    ucs_trace("registered addr:%p/%p length:%zd dev_num:%d",
              addr, (void *)key->d_bptr, key->b_len, (int)memh->dev_num);

    *key_p = key;
    return UCS_OK;

err:
    ucs_free(key);
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
                                                    uct_cuda_ipc_rkey_t *rkey)
{
    int i;
    int num_devices;
    int original_capacity, new_capacity;
    int original_count, new_count;

    for (i = 0; i < md->uuid_map_size; i++) {
        if (uct_cuda_ipc_uuid_equals(&rkey->uuid, &md->uuid_map[i])) {
            *idx = i;
            return UCS_OK; /* found */
        }
    }

    if (ucs_unlikely(md->uuid_map_size == md->uuid_map_capacity)) {
        /* reallocate on demand */
        UCT_CUDA_IPC_DEVICE_GET_COUNT(num_devices);
        original_capacity     = md->uuid_map_capacity;
        new_capacity          = md->uuid_map_capacity ?
                                (md->uuid_map_capacity * 2) : 16;
        original_count        = original_capacity * num_devices;
        new_count             = new_capacity * num_devices;
        md->uuid_map_capacity = new_capacity;
        md->uuid_map          = ucs_realloc(md->uuid_map,
                                            new_capacity * sizeof(CUuuid),
                                            "uct_cuda_ipc_uuid_map");
        if (md->uuid_map == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        md->peer_accessible_cache = ucs_realloc(md->peer_accessible_cache,
                                                new_count *
                                                sizeof(ucs_ternary_auto_value_t),
                                                "uct_cuda_ipc_peer_accessible_cache");
        if (md->peer_accessible_cache == NULL) {
            return UCS_ERR_NO_MEMORY;
        }

        for (i = original_count; i < new_count; i++) {
            md->peer_accessible_cache[i] = UCS_TRY;
        }
    }

    /* Add new mapping */
    uct_cuda_ipc_uuid_copy(&md->uuid_map[md->uuid_map_size], &rkey->uuid);
    *idx = md->uuid_map_size++;

    return UCS_OK;
}

static ucs_status_t uct_cuda_ipc_is_peer_accessible(uct_cuda_ipc_component_t *mdc,
                                                    uct_cuda_ipc_rkey_t *rkey)
{
    CUdevice this_device;
    ucs_status_t status;
    int peer_idx;
    int num_devices;
    ucs_ternary_auto_value_t *accessible;
    void *d_mapped;

    status = uct_cuda_ipc_get_unique_index_for_uuid(&peer_idx, mdc->md, rkey);
    if (ucs_unlikely(status != UCS_OK)) {
        goto err;
    }

    /* overwrite dev_num with a unique ID; this means that relative remote
     * device number of multiple peers do not map on the same stream and reduces
     * stream sequentialization */
    rkey->dev_num = peer_idx;

    UCT_CUDA_IPC_DEVICE_GET_COUNT(num_devices);
    if (UCT_CUDADRV_FUNC_LOG_DEBUG(cuCtxGetDevice(&this_device)) != UCS_OK) {
        status = UCS_ERR_UNREACHABLE;
        goto err;
    }

    accessible = &mdc->md->peer_accessible_cache[peer_idx * num_devices + this_device];
    if (*accessible == UCS_TRY) { /* unchecked, add to cache */

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
        status = uct_cuda_ipc_map_memhandle(rkey, &d_mapped);

        *accessible = ((status == UCS_OK) || (status == UCS_ERR_ALREADY_EXISTS))
                      ? UCS_YES : UCS_NO;
    }

    return (*accessible == UCS_YES) ? UCS_OK : UCS_ERR_UNREACHABLE;

err:
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, uct_cuda_ipc_rkey_unpack,
                 (component, rkey_buffer, rkey_p, handle_p),
                 uct_component_t *component, const void *rkey_buffer,
                 uct_rkey_t *rkey_p, void **handle_p)
{
    uct_cuda_ipc_component_t *com = ucs_derived_of(component, uct_cuda_ipc_component_t);
    uct_cuda_ipc_rkey_t *packed   = (uct_cuda_ipc_rkey_t *) rkey_buffer;
    uct_cuda_ipc_rkey_t *key;
    ucs_status_t status;

    status = uct_cuda_ipc_is_peer_accessible(com, packed);
    if (status != UCS_OK) {
        return status;
    }

    key = ucs_malloc(sizeof(*key), "uct_cuda_ipc_rkey_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_rkey_t");
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
uct_cuda_ipc_mem_reg(uct_md_h md, void *address, size_t length,
                     const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_cuda_ipc_memh_t *memh;
    CUdevice cu_device;

    UCT_CUDA_IPC_GET_DEVICE(cu_device);

    memh = ucs_malloc(sizeof(*memh), "uct_cuda_ipc_rkey_t");
    if (NULL == memh) {
        ucs_error("failed to allocate memory for uct_cuda_ipc_rkey_t");
        return UCS_ERR_NO_MEMORY;
    }

    memh->dev_num = (int) cu_device;
    memh->pid     = getpid();
    ucs_list_head_init(&memh->list);

    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t
uct_cuda_ipc_mem_dereg(uct_md_h md,
                       const uct_md_mem_dereg_params_t *params)
{
    uct_cuda_ipc_memh_t *memh = params->memh;
    uct_cuda_ipc_lkey_t *key, *tmp;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    ucs_list_for_each_safe(key, tmp, &memh->list, link) {
        ucs_free(key);
    }

    ucs_free(params->memh);
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
        .mem_attach         = ucs_empty_function_return_unsupported,
        .detect_memory_type = ucs_empty_function_return_unsupported
    };

    uct_cuda_ipc_md_t* md;
    uct_cuda_ipc_component_t* com;

    md = ucs_calloc(1, sizeof(uct_cuda_ipc_md_t), "uct_cuda_ipc_md");
    if (md == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_cuda_ipc_component.super;

    /* allocate uuid map and peer accessible cache */
    md->uuid_map_size         = 0;
    md->uuid_map_capacity     = 0;
    md->uuid_map              = NULL;
    md->peer_accessible_cache = NULL;

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
    .md                     = NULL,
};
UCT_COMPONENT_REGISTER(&uct_cuda_ipc_component.super);

