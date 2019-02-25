/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ugni_device.h"
#include "ugni_iface.h"
#include "ugni_md.h"

/* Forward declarations */

UCS_CONFIG_DEFINE_ARRAY(ugni_alloc_methods, sizeof(uct_alloc_method_t),
                        UCS_CONFIG_TYPE_ENUM(uct_alloc_method_names));

pthread_mutex_t uct_ugni_global_lock = PTHREAD_MUTEX_INITIALIZER;

/* For Cray devices we have only one MD */
static ucs_status_t uct_ugni_query_md_resources(uct_md_resource_desc_t **resources_p,
                                                unsigned *num_resources_p)
{
    if (getenv("PMI_GNI_PTAG") != NULL) {
        return uct_single_md_resource(&uct_ugni_md_component, resources_p, num_resources_p);
    } else {
        *resources_p     = NULL;
        *num_resources_p = 0;
        return UCS_OK;
    }
}

static ucs_status_t uct_ugni_md_query(uct_md_h md, uct_md_attr_t *md_attr)
{
    md_attr->rkey_packed_size  = 3 * sizeof(uint64_t);
    md_attr->cap.flags         = UCT_MD_FLAG_REG       |
                                 UCT_MD_FLAG_NEED_MEMH |
                                 UCT_MD_FLAG_NEED_RKEY;
    md_attr->cap.reg_mem_types = UCS_BIT(UCT_MD_MEM_TYPE_HOST);
    md_attr->cap.mem_type      = UCT_MD_MEM_TYPE_HOST;
    md_attr->cap.max_alloc     = 0;
    md_attr->cap.max_reg       = ULONG_MAX;
    md_attr->reg_cost.overhead = 1000.0e-9;
    md_attr->reg_cost.growth   = 0.007e-9;
    memset(&md_attr->local_cpus, 0xff, sizeof(md_attr->local_cpus));
    return UCS_OK;
}

static ucs_status_t uct_ugni_mem_reg(uct_md_h md, void *address, size_t length,
                                     unsigned flags, uct_mem_h *memh_p)
{
    ucs_status_t status;
    gni_return_t ugni_rc;
    uct_ugni_md_t *ugni_md = ucs_derived_of(md, uct_ugni_md_t);
    gni_mem_handle_t * mem_hndl = NULL;

    if (0 == length) {
        ucs_error("Unexpected length %zu", length);
        return UCS_ERR_INVALID_PARAM;
    }

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
    free(mem_hndl);
    return status;
}

static ucs_status_t uct_ugni_mem_dereg(uct_md_h md, uct_mem_h memh)
{
    uct_ugni_md_t *ugni_md = ucs_derived_of(md, uct_ugni_md_t);
    gni_mem_handle_t *mem_hndl = (gni_mem_handle_t *) memh;
    gni_return_t ugni_rc;
    ucs_status_t status = UCS_OK;

    uct_ugni_cdm_lock(&ugni_md->cdm);
    ugni_rc = GNI_MemDeregister(ugni_md->cdm.nic_handle, mem_hndl);
    uct_ugni_cdm_unlock(&ugni_md->cdm);
    if (GNI_RC_SUCCESS != ugni_rc) {
        ucs_error("GNI_MemDeregister failed, Error status: %s %d",
                 gni_err_str[ugni_rc], ugni_rc);
        status = UCS_ERR_IO_ERROR;
    }
    ucs_free(mem_hndl);

    return status;
}

static ucs_status_t uct_ugni_rkey_pack(uct_md_h md, uct_mem_h memh,
                                       void *rkey_buffer)
{
    gni_mem_handle_t *mem_hndl = (gni_mem_handle_t *) memh;
    uint64_t *ptr = rkey_buffer;

    ptr[0] = UCT_UGNI_RKEY_MAGIC;
    ptr[1] = mem_hndl->qword1;
    ptr[2] = mem_hndl->qword2;
    ucs_debug("Packed [ %"PRIx64" %"PRIx64" %"PRIx64"]", ptr[0], ptr[1], ptr[2]);
    return UCS_OK;
}

static ucs_status_t uct_ugni_rkey_release(uct_md_component_t *mdc, uct_rkey_t rkey,
                                          void *handle)
{
    ucs_assert(NULL == handle);
    ucs_free((void *)rkey);
    return UCS_OK;
}

static ucs_status_t uct_ugni_rkey_unpack(uct_md_component_t *mdc, const void *rkey_buffer,
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

static ucs_status_t uct_ugni_md_open(const char *md_name, const uct_md_config_t *md_config,
                                     uct_md_h *md_p)
{
    ucs_status_t status = UCS_OK;

    pthread_mutex_lock(&uct_ugni_global_lock);
    static uct_md_ops_t md_ops = {
        .close        = uct_ugni_md_close,
        .query        = uct_ugni_md_query,
        .mem_alloc    = (void*)ucs_empty_function,
        .mem_free     = (void*)ucs_empty_function,
        .mem_reg      = uct_ugni_mem_reg,
        .mem_dereg    = uct_ugni_mem_dereg,
        .mkey_pack     = uct_ugni_rkey_pack,
        .is_mem_type_owned = (void *)ucs_empty_function_return_zero,
    };

    static uct_ugni_md_t md = {
        .super.ops          = &md_ops,
        .super.component    = &uct_ugni_md_component,
        .ref_count          = 0
    };

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


UCT_MD_COMPONENT_DEFINE(uct_ugni_md_component,
                        UCT_UGNI_MD_NAME,
                        uct_ugni_query_md_resources,
                        uct_ugni_md_open,
                        NULL,
                        uct_ugni_rkey_unpack,
                        uct_ugni_rkey_release,
                        "UGNI_",
                        uct_md_config_table,
                        uct_md_config_t,
                        ucs_empty_function_return_unsupported);
