/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ze_ipc_md.h"

#include <uct/ze/base/ze_base.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/sys.h>
#include <ucs/type/class.h>

#include <string.h>
#include <sys/types.h>
#include <unistd.h>


/* Helper function to compute a simple checksum of IPC handle for verification */
static uint32_t uct_ze_ipc_handle_checksum(const ze_ipc_mem_handle_t *handle)
{
    const unsigned char *bytes = (const unsigned char *)handle;
    uint32_t sum = 0;
    for (size_t i = 0; i < sizeof(ze_ipc_mem_handle_t); i++) {
        sum += bytes[i];
        sum = (sum << 1) | (sum >> 31);  /* rotate left */
    }
    return sum;
}


/* Helper function to print IPC handle bytes for debugging */
static void uct_ze_ipc_print_handle(const char *prefix,
                                    const ze_ipc_mem_handle_t *handle,
                                    uintptr_t addr, size_t len, int dev_num)
{
    const unsigned char *bytes = (const unsigned char *)handle;
    uint32_t checksum = uct_ze_ipc_handle_checksum(handle);

    /* Print first 16 bytes of handle + checksum for quick comparison */
    ucs_info("%s: addr=0x%lx len=%zu dev=%d checksum=0x%08x "
             "handle[0-15]=%02x%02x%02x%02x%02x%02x%02x%02x"
             "%02x%02x%02x%02x%02x%02x%02x%02x",
             prefix, (unsigned long)addr, len, dev_num, checksum,
             bytes[0], bytes[1], bytes[2], bytes[3],
             bytes[4], bytes[5], bytes[6], bytes[7],
             bytes[8], bytes[9], bytes[10], bytes[11],
             bytes[12], bytes[13], bytes[14], bytes[15]);
}


static ucs_config_field_t uct_ze_ipc_md_config_table[] = {
    {"", "", NULL,
     ucs_offsetof(uct_ze_ipc_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"DEVICE_ORDINAL", "0",
     "Ordinal of the GPU device to use for IPC.",
     ucs_offsetof(uct_ze_ipc_md_config_t, device_ordinal),
     UCS_CONFIG_TYPE_INT},

    {NULL}
};

static ucs_status_t
uct_ze_ipc_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
{
    uct_md_base_md_query(md_attr);
    md_attr->rkey_packed_size = sizeof(uct_ze_ipc_key_t);
    md_attr->flags            = UCT_MD_FLAG_REG | UCT_MD_FLAG_NEED_RKEY;
    md_attr->reg_mem_types    = UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE);
    md_attr->cache_mem_types  = UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE);
    md_attr->access_mem_types = UCS_BIT(UCS_MEMORY_TYPE_ZE_DEVICE);
    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_mkey_pack(uct_md_h uct_md, uct_mem_h memh, void *address,
                     size_t length, const uct_md_mkey_pack_params_t *params,
                     void *mkey_buffer)
{
    uct_ze_ipc_key_t *packed = mkey_buffer;
    uct_ze_ipc_key_t *key    = memh;

    *packed = *key;

    /* Print handle info for verification - compare with rkey_unpack output */
    uct_ze_ipc_print_handle("PACK(sender)", &packed->ipc_handle,
                            packed->address, packed->length, packed->dev_num);

    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_pack_key(uct_ze_ipc_md_t *md, void *address, size_t length,
                    uct_ze_ipc_key_t *key)
{
    ze_memory_allocation_properties_t props = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES
    };
    ze_device_handle_t alloc_device = NULL;
    void *base_address;
    size_t alloc_size;
    ze_result_t ret;
    ucs_status_t status;
    int dev_ordinal;

    /* Get memory allocation properties to verify this is ZE device memory
     * Also get the device where memory was allocated */
    ret = zeMemGetAllocProperties(md->ze_context, address, &props, &alloc_device);
    if ((ret != ZE_RESULT_SUCCESS) || (props.type == ZE_MEMORY_TYPE_UNKNOWN)) {
        ucs_error("failed to get allocation properties for %p", address);
        return UCS_ERR_INVALID_ADDR;
    }

    /* Get device ordinal from the device handle */
    dev_ordinal = uct_ze_base_get_device_ordinal(alloc_device);
    if (dev_ordinal < 0) {
        ucs_error("failed to get device ordinal for device %p (ptr=%p)",
                  (void*)alloc_device, address);
        return UCS_ERR_INVALID_ADDR;
    }

    /* Get the base address and allocation size */
    ret = zeMemGetAddressRange(md->ze_context, address, &base_address,
                               &alloc_size);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to get address range for %p", address);
        return UCS_ERR_INVALID_ADDR;
    }

    /* Get IPC handle for the memory */
    status = UCT_ZE_FUNC_LOG_ERR(
            zeMemGetIpcHandle(md->ze_context, base_address, &key->ipc_handle));
    if (status != UCS_OK) {
        ucs_error("failed to get IPC handle for %p", address);
        return status;
    }

    key->pid     = getpid();
    key->address = (uintptr_t)base_address;
    key->length  = alloc_size;
    key->dev_num = dev_ordinal;

    ucs_trace("packed IPC handle for %p base=%p len=%zu dev=%d pid=%d", address,
              base_address, alloc_size, dev_ordinal, key->pid);

    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_mem_reg(uct_md_h uct_md, void *address, size_t length,
                   const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_ze_ipc_md_t *md = ucs_derived_of(uct_md, uct_ze_ipc_md_t);
    uct_ze_ipc_key_t *key;
    ucs_status_t status;

    ucs_info("ze_ipc_md: mem_reg called address=%p length=%zu", address, length);

    key = ucs_malloc(sizeof(*key), "uct_ze_ipc_key_t");
    if (key == NULL) {
        ucs_error("ze_ipc_md: failed to allocate memory for uct_ze_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    status = uct_ze_ipc_pack_key(md, address, length, key);
    if (status != UCS_OK) {
        ucs_error("ze_ipc_md: pack_key failed for address=%p status=%s",
                  address, ucs_status_string(status));
        ucs_free(key);
        return status;
    }

    ucs_info("ze_ipc_md: mem_reg succeeded address=%p key->address=0x%lx key->length=%zu",
             address, (unsigned long)key->address, key->length);

    *memh_p = key;
    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_mem_dereg(uct_md_h md, const uct_md_mem_dereg_params_t *params)
{
    uct_ze_ipc_key_t *key;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    key = params->memh;
    ucs_free(key);
    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                       const uct_rkey_unpack_params_t *params,
                       uct_rkey_t *rkey_p, void **handle_p)
{
    uct_ze_ipc_key_t *packed = (uct_ze_ipc_key_t *)rkey_buffer;
    uct_ze_ipc_key_t *key;

    /* Print handle info for verification - compare with mkey_pack output */
    uct_ze_ipc_print_handle("UNPACK(receiver)", &packed->ipc_handle,
                            packed->address, packed->length, packed->dev_num);

    key = ucs_malloc(sizeof(uct_ze_ipc_key_t), "uct_ze_ipc_key_t");
    if (key == NULL) {
        ucs_error("ze_ipc_md: failed to allocate memory for uct_ze_ipc_key_t");
        return UCS_ERR_NO_MEMORY;
    }

    *key      = *packed;
    *handle_p = NULL;
    *rkey_p   = (uintptr_t)key;

    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_rkey_release(uct_component_t *component, uct_rkey_t rkey,
                        void *handle)
{
    ucs_assert(handle == NULL);
    ucs_free((void *)rkey);
    return UCS_OK;
}


static void uct_ze_ipc_md_close(uct_md_h uct_md)
{
    uct_ze_ipc_md_t *md = ucs_derived_of(uct_md, uct_ze_ipc_md_t);

    if (md->ze_context != NULL) {
        zeContextDestroy(md->ze_context);
    }
    ucs_free(md);
}


static ucs_status_t
uct_ze_ipc_md_open(uct_component_h component, const char *md_name,
                   const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops = {
        .close              = uct_ze_ipc_md_close,
        .query              = uct_ze_ipc_md_query,
        .mem_alloc          = (uct_md_mem_alloc_func_t)ucs_empty_function_return_unsupported,
        .mem_free           = (uct_md_mem_free_func_t)ucs_empty_function_return_unsupported,
        .mem_advise         = (uct_md_mem_advise_func_t)ucs_empty_function_return_unsupported,
        .mem_reg            = uct_ze_ipc_mem_reg,
        .mem_dereg          = uct_ze_ipc_mem_dereg,
        .mem_query          = (uct_md_mem_query_func_t)ucs_empty_function_return_unsupported,
        .mkey_pack          = uct_ze_ipc_mkey_pack,
        .mem_attach         = (uct_md_mem_attach_func_t)ucs_empty_function_return_unsupported,
        .detect_memory_type = (uct_md_detect_memory_type_func_t)ucs_empty_function_return_unsupported,
    };
    uct_ze_ipc_md_config_t *config = ucs_derived_of(uct_md_config,
                                                    uct_ze_ipc_md_config_t);
    uct_ze_ipc_md_t *md;
    ze_driver_handle_t ze_driver;
    ze_context_desc_t context_desc = {};
    ze_result_t ret;

    ze_driver = uct_ze_base_get_driver();
    if (ze_driver == NULL) {
        return UCS_ERR_NO_DEVICE;
    }

    md = ucs_malloc(sizeof(uct_ze_ipc_md_t), "uct_ze_ipc_md_t");
    if (md == NULL) {
        ucs_error("failed to allocate memory for uct_ze_ipc_md_t");
        return UCS_ERR_NO_MEMORY;
    }

    md->ze_device = uct_ze_base_get_device(config->device_ordinal);
    if (md->ze_device == NULL) {
        ucs_error("failed to get device at ordinal %d", config->device_ordinal);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    ret = zeContextCreate(ze_driver, &context_desc, &md->ze_context);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("zeContextCreate failed with error 0x%x", ret);
        ucs_free(md);
        return UCS_ERR_NO_DEVICE;
    }

    md->super.ops       = &md_ops;
    md->super.component = &uct_ze_ipc_component;

    *md_p = (uct_md_h)md;
    return UCS_OK;
}


uct_component_t uct_ze_ipc_component = {
    .query_md_resources = uct_ze_base_query_md_resources,
    .md_open            = uct_ze_ipc_md_open,
    .cm_open            = (uct_component_cm_open_func_t)ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ze_ipc_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)ucs_empty_function_return_unsupported,
    .rkey_release       = uct_ze_ipc_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "ze_ipc",
    .md_config          = {
        .name           = "ZE-IPC memory domain",
        .prefix         = "ZE_IPC_",
        .table          = uct_ze_ipc_md_config_table,
        .size           = sizeof(uct_ze_ipc_md_config_t),
    },
    .cm_config          = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list            = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ze_ipc_component),
    .flags              = 0,
    .md_vfs_init        = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_ze_ipc_component);
