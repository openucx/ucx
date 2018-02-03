/**
 * Copyright (C) Mellanox Technologies Ltd. 2017.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "gdr_copy_md.h"

#include <string.h>
#include <limits.h>
#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define UCT_GDR_COPY_MD_RCACHE_DEFAULT_ALIGN 65536

static ucs_config_field_t uct_gdr_copy_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_gdr_copy_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"RCACHE", "try", "Enable using memory registration cache",
     ucs_offsetof(uct_gdr_copy_md_config_t, enable_rcache), UCS_CONFIG_TYPE_TERNARY},

    {"", "RCACHE_ADDR_ALIGN=" UCS_PP_MAKE_STRING(UCT_GDR_COPY_MD_RCACHE_DEFAULT_ALIGN), NULL,
     ucs_offsetof(uct_gdr_copy_md_config_t, rcache),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_rcache_table)},

    {"MEM_REG_OVERHEAD", "16us", "Memory registration overhead", /* TODO take default from device */
     ucs_offsetof(uct_gdr_copy_md_config_t, uc_reg_cost.overhead), UCS_CONFIG_TYPE_TIME},

    {"MEM_REG_GROWTH", "0.06ns", "Memory registration growth rate", /* TODO take default from device */
     ucs_offsetof(uct_gdr_copy_md_config_t, uc_reg_cost.growth), UCS_CONFIG_TYPE_TIME},

    {NULL}
};

static ucs_status_t uct_gdr_copy_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_CUDA);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_CUDA;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->rkey_packed_size  = sizeof(uct_gdr_copy_key_t);
    md_attr->reg_cost.overhead = 0;
    md_attr->reg_cost.growth   = 0;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_gdr_copy_key_t *packed      = (uct_gdr_copy_key_t *)rkey_buffer;
    uct_gdr_copy_mem_t *mem_hndl    = (uct_gdr_copy_mem_t *)memh;

    packed->vaddr   = mem_hndl->info.va;
    packed->bar_ptr = mem_hndl->bar_ptr;

    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_rkey_unpack(uct_md_component_t *mdc,
                                             const void *rkey_buffer, uct_rkey_t *rkey_p,
                                             void **handle_p)
{
    uct_gdr_copy_key_t *packed = (uct_gdr_copy_key_t *)rkey_buffer;
    uct_gdr_copy_key_t *key;

    key = ucs_malloc(sizeof(uct_gdr_copy_key_t), "uct_gdr_copy_key_t");
    if (NULL == key) {
        ucs_error("failed to allocate memory for uct_gdr_copy_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    key->vaddr      = packed->vaddr;
    key->bar_ptr    = packed->bar_ptr;

    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;

    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                              void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t
uct_gdr_copy_mem_reg_internal(uct_md_h uct_md, void *address, size_t length,
                              unsigned flags, uct_gdr_copy_mem_t *mem_hndl)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    CUdeviceptr d_ptr = ((CUdeviceptr )(char *) address);
    gdr_mh_t mh;
    void *bar_ptr;
    gdr_info_t info;
    int ret;

    if (!length) {
        mem_hndl->mh = 0;
        return UCS_OK;
    }

    ret = gdr_pin_buffer(md->gdrcpy_ctx, d_ptr, length, 0, 0, &mh);
    if (ret) {
        ucs_error("gdr_pin_buffer failed. length :%lu ret:%d", length, ret);
        goto err;
    }

    ret = gdr_map(md->gdrcpy_ctx, mh, &bar_ptr, length);
    if (ret) {
        ucs_error("gdr_map failed. length :%lu ret:%d", length, ret);
        goto unpin_buffer;
    }

    ret = gdr_get_info(md->gdrcpy_ctx, mh, &info);
    if (ret) {
        ucs_error("gdr_get_info failed. ret:%d", ret);
        goto unmap_buffer;
    }

    mem_hndl->mh        = mh;
    mem_hndl->info      = info;
    mem_hndl->bar_ptr   = bar_ptr;
    mem_hndl->reg_size  = length;

    ucs_trace("registered memory:%p..%p length:%lu info.va:0x%"PRIx64" bar_ptr:%p",
              address, address + length, length, info.va, bar_ptr);

    return UCS_OK;

unmap_buffer:
    ret = gdr_unmap(md->gdrcpy_ctx, mem_hndl->mh, mem_hndl->bar_ptr, mem_hndl->reg_size);
    if (ret) {
        ucs_warn("gdr_unmap failed. unpin_size:%lu ret:%d", mem_hndl->reg_size, ret);
    }
unpin_buffer:
    ret = gdr_unpin_buffer(md->gdrcpy_ctx, mh);
    if (ret) {
        ucs_warn("gdr_unpin_buffer failed. ret;%d", ret);
    }
err:
    return UCS_ERR_IO_ERROR;
}

static ucs_status_t uct_gdr_copy_mem_dereg_internal(uct_md_h uct_md, uct_gdr_copy_mem_t *mem_hndl)
{

    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    int ret;

    ret = gdr_unmap(md->gdrcpy_ctx, mem_hndl->mh, mem_hndl->bar_ptr, mem_hndl->reg_size);
    if (ret) {
        ucs_error("gdr_unmap failed. unpin_size:%lu ret:%d", mem_hndl->reg_size, ret);
        return UCS_ERR_IO_ERROR;
    }

    ret = gdr_unpin_buffer(md->gdrcpy_ctx, mem_hndl->mh);
    if (ret) {
        ucs_error("gdr_unpin_buffer failed. ret:%d", ret);
        return UCS_ERR_IO_ERROR;
    }

    ucs_trace("deregistered memorory. info.va:0x%"PRIx64" bar_ptr:%p",
              mem_hndl->info.va, mem_hndl->bar_ptr);
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_reg(uct_md_h uct_md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_gdr_copy_mem_t *mem_hndl = NULL;
    size_t reg_size;
    void *ptr;
    ucs_status_t status;

    mem_hndl = ucs_malloc(sizeof(uct_gdr_copy_mem_t), "gdr_copy handle");
    if (NULL == mem_hndl) {
        ucs_error("failed to allocate memory for gdr_copy_mem_t");
        return UCS_ERR_NO_MEMORY;
    }

    reg_size = (length + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    ptr = (void *) ((uintptr_t)address & GPU_PAGE_MASK);

    status = uct_gdr_copy_mem_reg_internal(uct_md, ptr, reg_size, 0, mem_hndl);
    if (status != UCS_OK) {
        ucs_free(mem_hndl);
        return status;
    }

    *memh_p = mem_hndl;
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_gdr_copy_mem_t *mem_hndl = memh;
    ucs_status_t status;

    status = uct_gdr_copy_mem_dereg_internal(uct_md, mem_hndl);
    if (status != UCS_OK) {
        ucs_warn("failed to deregister memory handle");
    }

    ucs_free(mem_hndl);
    return status;
}

static int uct_is_gdr_copy_mem_type_owned(uct_md_h md, void *addr, size_t length)
{
    int memory_type;
    struct cudaPointerAttributes attributes;
    cudaError_t cuda_err;
    CUresult cu_err;

    if (addr == NULL) {
        return 0;
    }

    cu_err = cuPointerGetAttribute(&memory_type,
                                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)addr);
    if (cu_err != CUDA_SUCCESS) {
        cuda_err = cudaPointerGetAttributes (&attributes, addr);
        if (cuda_err == cudaSuccess) {
            if (attributes.memoryType == cudaMemoryTypeDevice) {
                return 1;
            }
        }
    } else if (memory_type == CU_MEMORYTYPE_DEVICE) {
        return 1;
    }
    return 0;
}

static ucs_status_t uct_gdr_copy_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                    unsigned *num_resources_p)
{
    int num_gpus;
    gdr_t ctx;
    cudaError_t cudaErr;

    cudaErr = cudaGetDeviceCount(&num_gpus);
    if ((cudaErr != cudaSuccess) || (num_gpus == 0)) {
        ucs_debug("not found cuda devices");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    ctx = gdr_open();
    if (ctx == NULL) {
        ucs_debug("could not open gdr copy. disabling gdr copy resource");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }
    gdr_close(ctx);

    return uct_single_md_resource(&uct_gdr_copy_md_component, resources_p,
                                  num_resources_p);
}

static void uct_gdr_copy_md_close(uct_md_h uct_md)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    int ret;

    if (md->rcache != NULL) {
        ucs_rcache_destroy(md->rcache);
    }

    ret = gdr_close(md->gdrcpy_ctx);
    if (ret) {
        ucs_warn("failed to close gdrcopy. ret:%d", ret);
    }

    ucs_free(md);
}

static uct_md_ops_t md_ops = {
    .close              = uct_gdr_copy_md_close,
    .query              = uct_gdr_copy_md_query,
    .mkey_pack          = uct_gdr_copy_mkey_pack,
    .mem_reg            = uct_gdr_copy_mem_reg,
    .mem_dereg          = uct_gdr_copy_mem_dereg,
    .is_mem_type_owned  = uct_is_gdr_copy_mem_type_owned,
};

static inline uct_gdr_copy_rcache_region_t*
uct_gdr_copy_rache_region_from_memh(uct_mem_h memh)
{
    return ucs_container_of(memh, uct_gdr_copy_rcache_region_t, memh);
}

static ucs_status_t
uct_gdr_copy_mem_rcache_reg(uct_md_h uct_md, void *address, size_t length,
                            unsigned flags, uct_mem_h *memh_p)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    ucs_rcache_region_t *rregion;
    ucs_status_t status;
    uct_gdr_copy_mem_t *memh;

    status = ucs_rcache_get(md->rcache, address, length, PROT_READ|PROT_WRITE,
                            &flags, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(rregion->refcount > 0);
    memh = &ucs_derived_of(rregion, uct_gdr_copy_rcache_region_t)->memh;
    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t uct_gdr_copy_mem_rcache_dereg(uct_md_h uct_md, uct_mem_h memh)
{
    uct_gdr_copy_md_t *md = ucs_derived_of(uct_md, uct_gdr_copy_md_t);
    uct_gdr_copy_rcache_region_t *region = uct_gdr_copy_rache_region_from_memh(memh);

    ucs_rcache_region_put(md->rcache, &region->super);
    return UCS_OK;
}

static uct_md_ops_t md_rcache_ops = {
    .close              = uct_gdr_copy_md_close,
    .query              = uct_gdr_copy_md_query,
    .mkey_pack          = uct_gdr_copy_mkey_pack,
    .mem_reg            = uct_gdr_copy_mem_rcache_reg,
    .mem_dereg          = uct_gdr_copy_mem_rcache_dereg,
    .is_mem_type_owned  = uct_is_gdr_copy_mem_type_owned,
};

static ucs_status_t
uct_gdr_copy_rcache_mem_reg_cb(void *context, ucs_rcache_t *rcache,
                               void *arg, ucs_rcache_region_t *rregion)
{
    uct_gdr_copy_md_t *md = context;
    int *flags = arg;
    uct_gdr_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_gdr_copy_rcache_region_t);
    return uct_gdr_copy_mem_reg_internal(&md->super, (void*)region->super.super.start,
                                         region->super.super.end -
                                         region->super.super.start,
                                         *flags, &region->memh);
}

static void uct_gdr_copy_rcache_mem_dereg_cb(void *context, ucs_rcache_t *rcache,
                                             ucs_rcache_region_t *rregion)
{
    uct_gdr_copy_md_t *md = context;
    uct_gdr_copy_rcache_region_t *region;

    region = ucs_derived_of(rregion, uct_gdr_copy_rcache_region_t);
    (void)uct_gdr_copy_mem_dereg_internal(&md->super, &region->memh);
}

static void uct_gdr_copy_rcache_dump_region_cb(void *context, ucs_rcache_t *rcache,
                                               ucs_rcache_region_t *rregion, char *buf,
                                               size_t max)
{
    uct_gdr_copy_rcache_region_t *region = ucs_derived_of(rregion,
                                                          uct_gdr_copy_rcache_region_t);
    uct_gdr_copy_mem_t *memh = &region->memh;

    snprintf(buf, max, "bar ptr:%p", memh->bar_ptr);
}

static ucs_rcache_ops_t uct_gdr_copy_rcache_ops = {
    .mem_reg     = uct_gdr_copy_rcache_mem_reg_cb,
    .mem_dereg   = uct_gdr_copy_rcache_mem_dereg_cb,
    .dump_region = uct_gdr_copy_rcache_dump_region_cb
};

static ucs_status_t uct_gdr_copy_md_open(const char *md_name,
                                         const uct_md_config_t *uct_md_config,
                                         uct_md_h *md_p)
{
    const uct_gdr_copy_md_config_t *md_config = ucs_derived_of(uct_md_config,
                                                               uct_gdr_copy_md_config_t);
    ucs_status_t status;
    uct_gdr_copy_md_t *md;
    ucs_rcache_params_t rcache_params;

    md = ucs_malloc(sizeof(uct_gdr_copy_md_t), "uct_gdr_copy_md_t");
    if (NULL == md) {
        ucs_error("failed to allocate memory for uct_gdr_copy_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->super.ops = &md_ops;
    md->super.component = &uct_gdr_copy_md_component;
    md->rcache = NULL;
    md->reg_cost = md_config->uc_reg_cost;

    md->gdrcpy_ctx = gdr_open();
    if (md->gdrcpy_ctx == NULL) {
        ucs_error("failed to open gdr copy");
        status = UCS_ERR_IO_ERROR;
        goto err_free_md;
    }

    if (md_config->enable_rcache != UCS_NO) {
        rcache_params.region_struct_size = sizeof(uct_gdr_copy_rcache_region_t);
        rcache_params.alignment          = md_config->rcache.alignment;
        rcache_params.max_alignment      = UCT_GDR_COPY_MD_RCACHE_DEFAULT_ALIGN;
        rcache_params.ucm_event_priority = md_config->rcache.event_prio;
        rcache_params.context            = md;
        rcache_params.ops                = &uct_gdr_copy_rcache_ops;
        status = ucs_rcache_create(&rcache_params, "gdr_copy", NULL, &md->rcache);
        if (status == UCS_OK) {
            md->super.ops         = &md_rcache_ops;
            md->reg_cost.overhead = 0;
            md->reg_cost.growth   = 0;
        } else {
            ucs_assert(md->rcache == NULL);
            if (md_config->enable_rcache == UCS_YES) {
                status = UCS_ERR_IO_ERROR;
                goto err_close_gdr;
            } else {
                ucs_debug("could not create registration cache for: %s",
                          ucs_status_string(status));
            }
        }
    }

    *md_p = (uct_md_h) md;
    status = UCS_OK;
out:
    return status;
err_close_gdr:
    gdr_close(md->gdrcpy_ctx);
err_free_md:
    ucs_free(md);
    goto out;
}

UCT_MD_COMPONENT_DEFINE(uct_gdr_copy_md_component, UCT_GDR_COPY_MD_NAME,
                        uct_gdr_copy_query_md_resources, uct_gdr_copy_md_open, NULL,
                        uct_gdr_copy_rkey_unpack, uct_gdr_copy_rkey_release, "GDR_COPY_",
                        uct_gdr_copy_md_config_table, uct_gdr_copy_md_config_t);
