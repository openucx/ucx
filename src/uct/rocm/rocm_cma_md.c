/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "rocm_cma_md.h"
#include "rocm_common.h"

#include <ucs/debug/log.h>
#include <ucs/sys/sys.h>
#include <ucs/debug/memtrack.h>


static ucs_config_field_t uct_rocm_cma_md_config_table[] = {
    {"", "", NULL,
    ucs_offsetof(uct_rocm_cma_md_config_t, super), UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"ANY_MEM", "y",
    "Whether or not to use ROCm CMA support to deal with any memory\n"
    "Default: Use ROCm CMA for any memory",
    ucs_offsetof(uct_rocm_cma_md_config_t, any_memory), UCS_CONFIG_TYPE_BOOL},

    {NULL}
};

static ucs_status_t uct_rocm_cma_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    ucs_trace_func("");

    md_attr->rkey_packed_size  = sizeof(uct_rocm_cma_key_t);
    md_attr->cap.flags         = UCT_MD_FLAG_REG |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;

    /** @todo: Put the real numbers. Copied from cma md */
    md_attr->reg_cost.overhead = 9e-9;
    md_attr->reg_cost.growth   = 0;

    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_rocm_cma_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                    unsigned *num_resources_p)
{
    ucs_status_t status;

    ucs_trace_func("");

    /* Initialize ROCm helper library.
     * If needed HSA RT  will be initialized as part of library
     * initialization.
     *
     * Note: We do not support PnP so ROCm configuration will be the same.
    */
    if (uct_rocm_init() != HSA_STATUS_SUCCESS) {
        ucs_error("Could not initialize ROCm support");
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }

    status = uct_single_md_resource(&uct_rocm_cma_md_component, resources_p,
                                    num_resources_p);


    ucs_trace("md name: %s, resources %d", (*resources_p)->md_name, *num_resources_p);

    return status;
}

static void uct_rocm_cma_md_close(uct_md_h md)
{
    uct_rocm_cma_md_t *rocm_md = (uct_rocm_cma_md_t *)md;

    ucs_free(rocm_md);
}

static ucs_status_t uct_rocm_cma_mem_reg(uct_md_h md, void *address, size_t length,
                                         unsigned flags, uct_mem_h *memh_p)
{
    uct_rocm_cma_key_t *key;
    uct_rocm_cma_md_t *rocm_md = (uct_rocm_cma_md_t *)md;
    hsa_status_t  status;
    void *gpu_address;

    ucs_trace("register memory address %p length 0x%lx", address, length);

    key = ucs_malloc(sizeof(uct_rocm_cma_key_t), "uct_rocm_cma_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_rocm_cma_key_t");
        return UCS_ERR_NO_MEMORY;
    }
    ucs_trace("allocated key %p", key);

    /* Assume memory is already GPU accessible */
    key->is_locked = 0;

    /* Check if memory is already GPU accessible. If yes then GPU
     * address will be returned.
     * Note that we could have case of "malloc"-ed memory which
     * was "locked" outside of UCX. In this case CPU address may be 
     * not the same as GPU one.
     */
    if (!uct_rocm_is_ptr_gpu_accessible(address, &gpu_address)) {
        if (!rocm_md->any_memory) {
            ucs_warn("Address %p is not GPU allocated.", address);
            return UCS_ERR_INVALID_ADDR;
        } else {
            status = uct_rocm_memory_lock(address, length, &gpu_address);

            if (status != HSA_STATUS_SUCCESS) {
                ucs_error("Could not lock  %p. Status %d", address, status);
                return UCS_ERR_INVALID_ADDR;
            } else {
                ucs_trace("Lock address %p as GPU %p", address, gpu_address);
                key->is_locked = 1; /* Set flag that memory was locked by us */
            }
        }
    }

    key->length         = length;
    key->gpu_address    = (uintptr_t) gpu_address;
    key->md_address     = (uintptr_t) address;

    *memh_p = key;

    ucs_trace("Success");

    return UCS_OK;
}

static ucs_status_t uct_rocm_cma_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_rocm_cma_key_t *key = (uct_rocm_cma_key_t *)memh;
    ucs_trace("free key %p", key);

    if (key->is_locked) {
        /* Memory was locked by us. Need to unlock to free resource. */
        hsa_status_t status = hsa_amd_memory_unlock((void *)key->gpu_address);

        if (status != HSA_STATUS_SUCCESS) {
            ucs_warn("Failed to unlock memory (%p): 0x%x\n",
                     (void *)key->gpu_address, status);
        }
    }

    ucs_free(key);
    return UCS_OK;
}

static ucs_status_t uct_rocm_cma_rkey_pack(uct_md_h md, uct_mem_h memh,
                                           void *rkey_buffer)
{
    uct_rocm_cma_key_t *packed = (uct_rocm_cma_key_t *)rkey_buffer;
    uct_rocm_cma_key_t *key    = (uct_rocm_cma_key_t *)memh;

    packed->length      = key->length;
    packed->gpu_address = key->gpu_address;
    packed->md_address  = key->md_address;

    ucs_trace("packed (%p) rkey (%p): length 0x%lx gpu address %"PRIxPTR" md address %"PRIxPTR,
              packed, key, key->length, key->gpu_address, key->md_address);

    return UCS_OK;
}

static ucs_status_t uct_rocm_cma_rkey_unpack(uct_md_component_t *mdc,
                                             const void *rkey_buffer, uct_rkey_t *rkey_p,
                                             void **handle_p)
{
    uct_rocm_cma_key_t *packed = (uct_rocm_cma_key_t *)rkey_buffer;
    uct_rocm_cma_key_t *key;

    key = ucs_malloc(sizeof(uct_rocm_cma_key_t), "uct_rocm_cma_key_t");
    if (NULL == key) {
        ucs_error("Failed to allocate memory for uct_rocm_cma_key_t");
        return UCS_ERR_NO_MEMORY;
    }
    key->length      = packed->length;
    key->gpu_address = packed->gpu_address;
    key->md_address  = packed->md_address;

    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;
    ucs_trace("unpacked rkey: key %p length 0x%x gpu address %"PRIxPTR" md address %"PRIxPTR,
              key, (int) key->length, key->gpu_address, key->md_address);
    return UCS_OK;
}

static ucs_status_t uct_rocm_cma_rkey_release(uct_md_component_t *mdc,
                                              uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_trace("free rkey %p", (void *)rkey);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_rocm_cma_md_open(const char *md_name,
                                         const uct_md_config_t *uct_md_config,
                                         uct_md_h *md_p)
{
    uct_rocm_cma_md_t *rocm_md;
    const uct_rocm_cma_md_config_t *md_config =
                        ucs_derived_of(uct_md_config, uct_rocm_cma_md_config_t);

    static uct_md_ops_t md_ops = {
        .close        = uct_rocm_cma_md_close,
        .query        = uct_rocm_cma_md_query,
        .mkey_pack    = uct_rocm_cma_rkey_pack,
        .mem_reg      = uct_rocm_cma_mem_reg,
        .mem_dereg    = uct_rocm_cma_mem_dereg,
        .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
    };

    ucs_trace("deal with any memory = %d\n", md_config->any_memory);

    rocm_md = ucs_malloc(sizeof(uct_rocm_cma_md_t), "uct_rocm_cma_md_t");
    if (NULL == rocm_md) {
        ucs_error("Failed to allocate memory for uct_rocm_cma_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    rocm_md->super.ops       = &md_ops;
    rocm_md->super.component = &uct_rocm_cma_md_component;
    rocm_md->any_memory      = md_config->any_memory;

    *md_p = (uct_md_h)rocm_md;
    return UCS_OK;
}

UCT_MD_COMPONENT_DEFINE(uct_rocm_cma_md_component, UCT_ROCM_CMA_MD_NAME,
                        uct_rocm_cma_query_md_resources, uct_rocm_cma_md_open, 0,
                        uct_rocm_cma_rkey_unpack,
                        uct_rocm_cma_rkey_release, "ROCM_MD_",
                        uct_rocm_cma_md_config_table,
                        uct_rocm_cma_md_config_t);
