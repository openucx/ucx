/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ze_ipc_md.h"
#include "ze_ipc_cache.h"

#include <uct/ze/base/ze_base.h>
#include <uct/api/v2/uct_v2.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/sys.h>
#include <ucs/type/class.h>

#include <fcntl.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>


static ucs_config_field_t uct_ze_ipc_md_config_table[] = {
    {"", "", NULL, ucs_offsetof(uct_ze_ipc_md_config_t, super),
     UCS_CONFIG_TYPE_TABLE(uct_md_config_table)},

    {"DEVICE_ORDINAL", "0", "Ordinal of the GPU device to use for IPC.",
     ucs_offsetof(uct_ze_ipc_md_config_t, device_ordinal), UCS_CONFIG_TYPE_INT},

    {NULL}
};

static ucs_status_t uct_ze_ipc_md_query(uct_md_h md, uct_md_attr_v2_t *md_attr)
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
    return UCS_OK;
}


static ucs_status_t uct_ze_ipc_pack_key(uct_ze_ipc_md_t *md, void *address,
                                        size_t length, uct_ze_ipc_key_t *key)
{
    ze_memory_allocation_properties_t props = {
        .stype = ZE_STRUCTURE_TYPE_MEMORY_ALLOCATION_PROPERTIES
    };
    void *base_address;
    size_t alloc_size;
    unsigned long proc_create_time;
    ze_result_t ret;
    ucs_status_t status;
    int fd;

    /* Get memory allocation properties to verify this is ZE device memory */
    ret = zeMemGetAllocProperties(md->ze_context, address, &props, NULL);
    if ((ret != ZE_RESULT_SUCCESS) || (props.type == ZE_MEMORY_TYPE_UNKNOWN)) {
        ucs_error("failed to get allocation properties for %p", address);
        return UCS_ERR_INVALID_ADDR;
    }

    /* Get the base address and allocation size */
    ret = zeMemGetAddressRange(md->ze_context, address, &base_address,
                               &alloc_size);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_error("failed to get address range for %p", address);
        return UCS_ERR_INVALID_ADDR;
    }

    /* Resolve this before exporting the handle: without it the importer cannot
     * tell our allocations from those of a dead process whose PID we reused,
     * and exporting a key it must then distrust is worse than failing here */
    proc_create_time = ucs_sys_get_proc_create_time(getpid());
    if (proc_create_time == 0) {
        ucs_error("failed to get process creation time for pid %d", getpid());
        return UCS_ERR_IO_ERROR;
    }

    /* Get IPC handle for the memory */
    status = UCT_ZE_FUNC_LOG_ERR(
            zeMemGetIpcHandle(md->ze_context, base_address, &key->ipc_handle));
    if (status != UCS_OK) {
        ucs_error("failed to get IPC handle for %p", address);
        return status;
    }

    /* Peers get our own duplicate of the fd and the handle goes back at once:
     * while a handle is held, the driver may close its fd when another device
     * accesses the allocation, and then close that fd number again on put */
    fd = fcntl(*(int*)key->ipc_handle.data, F_DUPFD_CLOEXEC, 0);
    if (fd < 0) {
        ucs_error("failed to duplicate IPC handle fd %d for %p: %m",
                  *(int*)key->ipc_handle.data, address);
    }

    ret = zeMemPutIpcHandle(md->ze_context, key->ipc_handle);
    if (ret != ZE_RESULT_SUCCESS) {
        ucs_warn("zeMemPutIpcHandle failed with error 0x%x", ret);
    }

    if (fd < 0) {
        return UCS_ERR_IO_ERROR;
    }

    *(int*)key->ipc_handle.data = fd;

    key->pid              = getpid();
    key->address          = (uintptr_t)base_address;
    key->length           = alloc_size;
    key->alloc_id         = props.id;
    key->proc_create_time = proc_create_time;

    ucs_trace("packed ipc handle for %p base %p length %zu pid %d", address,
              base_address, alloc_size, key->pid);

    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_mem_reg(uct_md_h uct_md, void *address, size_t length,
                   const uct_md_mem_reg_params_t *params, uct_mem_h *memh_p)
{
    uct_ze_ipc_md_t *md = ucs_derived_of(uct_md, uct_ze_ipc_md_t);
    uct_ze_ipc_key_t *key;
    ucs_status_t status;

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

    *memh_p = key;
    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_mem_dereg(uct_md_h uct_md, const uct_md_mem_dereg_params_t *params)
{
    uct_ze_ipc_key_t *key;

    UCT_MD_MEM_DEREG_CHECK_PARAMS(params, 0);

    key = params->memh;
    close(*(int*)key->ipc_handle.data);
    ucs_free(key);
    return UCS_OK;
}


static ucs_status_t
uct_ze_ipc_rkey_unpack(uct_component_t *component, const void *rkey_buffer,
                       const uct_rkey_unpack_params_t *params,
                       uct_rkey_t *rkey_p, void **handle_p)
{
    uct_ze_ipc_key_t *packed = (uct_ze_ipc_key_t*)rkey_buffer;
    uct_ze_ipc_key_t *key;

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


static ucs_status_t uct_ze_ipc_rkey_release(uct_component_t *component,
                                            uct_rkey_t rkey, void *handle)
{
    ucs_assert(handle == NULL);
    ucs_free((void*)rkey);
    return UCS_OK;
}


static void uct_ze_ipc_md_close(uct_md_h uct_md)
{
    uct_ze_ipc_md_t *md = ucs_derived_of(uct_md, uct_ze_ipc_md_t);

    if (md->ze_context != NULL) {
        uct_ze_ipc_purge_cache_by_context(md->ze_context);
        zeContextDestroy(md->ze_context);
    }
    ucs_free(md);
}


static ucs_status_t
uct_ze_ipc_md_open(uct_component_h component, const char *md_name,
                   const uct_md_config_t *uct_md_config, uct_md_h *md_p)
{
    static uct_md_ops_t md_ops     = {
        .close              = uct_ze_ipc_md_close,
        .query              = uct_ze_ipc_md_query,
        .mem_alloc          = (uct_md_mem_alloc_func_t)
                ucs_empty_function_return_unsupported,
        .mem_free           = (uct_md_mem_free_func_t)
                ucs_empty_function_return_unsupported,
        .mem_advise         = (uct_md_mem_advise_func_t)
                ucs_empty_function_return_unsupported,
        .mem_reg            = uct_ze_ipc_mem_reg,
        .mem_dereg          = uct_ze_ipc_mem_dereg,
        .mem_query          = (uct_md_mem_query_func_t)
                ucs_empty_function_return_unsupported,
        .mkey_pack          = uct_ze_ipc_mkey_pack,
        .mem_attach         = (uct_md_mem_attach_func_t)
                ucs_empty_function_return_unsupported,
        .detect_memory_type = (uct_md_detect_memory_type_func_t)
                ucs_empty_function_return_unsupported,
        .mem_elem_pack      = (uct_md_mem_elem_pack_func_t)
                ucs_empty_function_return_unsupported,
        .mem_elem_release   = (uct_md_mem_elem_release_func_t)
                ucs_empty_function,
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
    .cm_open            = (uct_component_cm_open_func_t)
            ucs_empty_function_return_unsupported,
    .rkey_unpack        = uct_ze_ipc_rkey_unpack,
    .rkey_ptr           = (uct_component_rkey_ptr_func_t)
            ucs_empty_function_return_unsupported,
    .rkey_release       = uct_ze_ipc_rkey_release,
    .rkey_compare       = uct_base_rkey_compare,
    .name               = "ze_ipc",
    .md_config =
            {
                    .name   = "ZE-IPC memory domain",
                    .prefix = "ZE_IPC_",
                    .table  = uct_ze_ipc_md_config_table,
                    .size   = sizeof(uct_ze_ipc_md_config_t),
            },
    .cm_config   = UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY,
    .tl_list     = UCT_COMPONENT_TL_LIST_INITIALIZER(&uct_ze_ipc_component),
    .flags       = 0,
    .md_vfs_init = (uct_component_md_vfs_init_func_t)ucs_empty_function
};
UCT_COMPONENT_REGISTER(&uct_ze_ipc_component);
