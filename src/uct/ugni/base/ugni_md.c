/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ugni_device.h"
#include "ugni_iface.h"
#include "ugni_md.h"
#include <uct/api/v2/uct_v2.h>

/* Forward declarations */

UCS_CONFIG_DEFINE_ARRAY(ugni_alloc_methods, sizeof(uct_alloc_method_t),
                        UCS_CONFIG_TYPE_ENUM(uct_alloc_method_names));

pthread_mutex_t uct_ugni_global_lock = PTHREAD_MUTEX_INITIALIZER;

/* For Cray devices we have only one MD */
static ucs_status_t
uct_ugni_query_md_resources(uct_component_h component,
                            uct_md_resource_desc_t **resources_p,
                            unsigned *num_resources_p)
{
    if (getenv("PMI_GNI_PTAG") == NULL) {
        return uct_md_query_empty_md_resource(resources_p, num_resources_p);
    }

    return uct_md_query_single_md_resource(component, resources_p,
                                           num_resources_p);
}

static ucs_status_t uct_ugni_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->rkey_packed_size = 3 * sizeof(uint64_t);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_MEMH |
                                UCT_MD_FLAG_NEED_RKEY;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_HOST);
    md_attr->reg_cost         = ucs_linear_func_make(1000.0e-9, 0.007e-9);
    return UCS_OK;
}

static ucs_status_t uct_ugni_mem_reg(uct_md_h md, void *address, size_t length,
                                     const uct_md_mem_reg_params_t *params,
                                     uct_mem_h *memh_p)
{
    ucs_status_t status;
    gni_return_t ugni_rc;
    uct_ugni_md_t *ugni_md = ucs_derived_of(md, uct_ugni_md_t);
    gni_mem_handle_t * mem_hndl = NULL;

    mem_hndl = ucs_malloc(sizeof(gni_mem_handle_t), "gni_mem_handle_t");
    if (NULL == mem_hndl) {
        ucs_error("Failed to allocate memory for gni_mem_handle_t");
        status = UCS_ERR_NO_MEMORY;
        goto mem_err;
    }

    uct_ugni_cdm_lock(&ugni_md->cdm);
    ugni_rc = GNI_MemRegister(ugni_md->cdm.nic_handle, (uint64_t)address,
                              length, NULL,
                              GNI_MEM_READWRITE | GNI_MEM_RELAXED_PI_ORDERING,
                              -1, mem_hndl);
    uct_ugni_cdm_unlock(&ugni_md->cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_MemRegister failed (addr %p, size %zu), Error status: %s %d",
                 address, length, gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_IO_ERROR;
        goto mem_err;
    }

    ucs_debug("Memory registration address %p, len %lu, keys [%"PRIx64" %"PRIx64"]",
              address, length, mem_hndl->qword1, mem_hndl->qword2);
    *memh_p = mem_hndl;
    return UCS_OK;

mem_err:
    ucs_free(mem_hndl);
    return status;
}

static ucs_status_t uct_ugni_mem_dereg(uct_md_h md,
                                       const uct_md_mem_dereg_params_t *params)
{
    uct_ugni_md_t *ugni_md = ucs_derived_of(md, uct_ugni_md_t);
    gni_mem_handle_t *mem_hndl;
    gni_return_t ugni_rc;
    ucs_status_t status = UCS_OK;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    uct_ugni_cdm_lock(&ugni_md->cdm);
    mem_hndl = (gni_mem_handle_t *)params->memh;
    ugni_rc  = GNI_MemDeregister(ugni_md->cdm.nic_handle, mem_hndl);
    uct_ugni_cdm_unlock(&ugni_md->cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_MemDeregister failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_IO_ERROR;
    }
    ucs_free(mem_hndl);

    return status;
}

static ucs_status_t
uct_ugni_mkey_pack(uct_md_h md, uct_mem_h memh, void *address, size_t length,
                   const uct_md_mkey_pack_params_t *params,
                   void *mkey_buffer)
{
    gni_mem_handle_t *mem_hndl = memh;
    uint64_t *ptr              = mkey_buffer;

    ptr[0] = UCT_UGNI_RKEY_MAGIC;
    ptr[1] = mem_hndl->qword1;
    ptr[2] = mem_hndl->qword2;
    ucs_debug("Packed [ %"PRIx64" %"PRIx64" %"PRIx64"]", ptr[0], ptr[1], ptr[2]);
    return UCS_OK;
}

static ucs_status_t uct_ugni_rkey_release(uct_component_t *component,
                                          uct_rkey_t rkey, void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_ugni_rkey_unpack(uct_component_t *component,
                                         const void *rkey_buffer,
                                         const uct_rkey_unpack_params_t *params,
                                         uct_rkey_t *rkey_p, void **handle_p)
{
    const uint64_t *ptr = rkey_buffer;
    gni_mem_handle_t *mem_hndl = NULL;
    uint64_t magic = 0;

    ucs_debug("Unpacking [ %"PRIx64" %"PRIx64" %"PRIx64"]", ptr[0], ptr[1], ptr[2]);
    magic = ptr[0];
    if (magic != UCT_UGNI_RKEY_MAGIC) {
        ucs_error("Failed to identify key. Expected %llx but received %"PRIx64"",
                  UCT_UGNI_RKEY_MAGIC, magic);
        return UCS_ERR_UNSUPPORTED;
    }

    mem_hndl = ucs_malloc(sizeof(gni_mem_handle_t), "gni_mem_handle_t");
    if (NULL == mem_hndl) {
        ucs_error("Failed to allocate memory for gni_mem_handle_t");
        return UCS_ERR_NO_MEMORY;
    }

    mem_hndl->qword1 = ptr[1];
    mem_hndl->qword2 = ptr[2];
    *rkey_p = (uintptr_t)mem_hndl;
    *handle_p = NULL;
    return UCS_OK;
}

static void uct_ugni_md_close(uct_md_h md)
{
    uct_ugni_md_t *ugni_md = ucs_derived_of(md, uct_ugni_md_t);

    pthread_mutex_lock(&uct_ugni_global_lock);
    ugni_md->ref_count--;
    if (!ugni_md->ref_count) {
        ucs_debug("Tearing down MD CDM");
        uct_ugni_destroy_cdm(&ugni_md->cdm);
    }
    pthread_mutex_unlock(&uct_ugni_global_lock);
}

static ucs_status_t
uct_ugni_md_open(uct_component_h component,const char *md_name,
                 const uct_md_config_t *md_config, uct_md_h *md_p)
{
    ucs_status_t status = UCS_OK;
    static uct_md_ops_t md_ops;
    static uct_ugni_md_t md;

    pthread_mutex_lock(&uct_ugni_global_lock);
    md_ops.close              = uct_ugni_md_close;
    md_ops.query              = uct_ugni_md_query;
    md_ops.mem_alloc          = (void*)ucs_empty_function;
    md_ops.mem_free           = (void*)ucs_empty_function;
    md_ops.mem_reg            = uct_ugni_mem_reg;
    md_ops.mem_dereg          = uct_ugni_mem_dereg;
    md_ops.mem_attach         = (uct_md_mem_attach_func_t)ucs_empty_function_return_unsupported;
    md_ops.mkey_pack          = uct_ugni_mkey_pack;
    md_ops.detect_memory_type = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported;

    md.super.ops              = &md_ops;
    md.super.component        = &uct_ugni_component;
    md.ref_count              = 0;

    *md_p = &md.super;

    if (!md.ref_count) {
        status = init_device_list();
        if (UCS_OK != status) {
            ucs_error("Failed to init device list, Error status: %d", status);
            goto error;
        }
        status = uct_ugni_create_md_cdm(&md.cdm);
        if (UCS_OK != status) {
            ucs_error("Failed to UGNI NIC, Error status: %d", status);
            goto error;
        }
    }

    md.ref_count++;

error:
    pthread_mutex_unlock(&uct_ugni_global_lock);
    return status;
}

uct_component_t uct_ugni_component = {
    .query_md_resources = uct_ugni_query_md_resources,
    .md_open            = uct_ugni_md_open,
    .cm_open            = (uct_component_cm_open_func_t)ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ugni_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)ucs_empty_function_return_unsupported,
    .rkey_release       = uct_ugni_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = UCT_UGNI_MD_NAME,
    .md_config          = {
        .name           = "UGNI memory domain",
        .prefix         = "UGNI_",
        .table          = uct_md_config_table,
        .size           = sizeof(uct_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ugni_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_ugni_component);
