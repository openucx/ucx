/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_types.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_types.h>
#include <ucs/type/param.h>


static ucs_status_t
ucp_device_mem_list_params_check(const ucp_device_mem_list_params_t *params)
{
    ucp_md_map_t remote_md_map       = UCS_MASK(UCP_MAX_MDS);
    ucp_worker_cfg_index_t cfg_index = UCP_WORKER_CFG_INDEX_NULL;
    ucp_mem_h memh;
    ucp_rkey_h rkey;
    size_t i, num_elements, element_size;
    const ucp_device_mem_list_elem_t *elements;

    if (params == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    element_size = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params,
                                   element_size, ELEMENT_SIZE, 0);
    num_elements = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params,
                                   num_elements, NUM_ELEMENTS, 0);
    elements     = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params,
                                   elements, ELEMENTS, NULL);

    if ((element_size != sizeof(*elements)) || (num_elements == 0) ||
        (elements == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < num_elements; i++) {
        memh = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, &elements[i], memh,
                               MEMH, NULL);
        rkey = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, &elements[i], rkey,
                               RKEY, NULL);
        if ((memh == NULL) || (rkey == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }

        if (memh->mem_type != UCS_MEMORY_TYPE_CUDA) {
            ucs_debug("invalid mem_type: i=%lu mem_type=%d", i, memh->mem_type);
            return UCS_ERR_UNSUPPORTED;
        }

        if (i == 0) {
            cfg_index = rkey->cfg_index;
            if (cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
                ucs_debug("invalid first rkey: cfg_index=%d", cfg_index);
                return UCS_ERR_INVALID_PARAM;
            }
        } else {
            if (rkey->cfg_index != cfg_index) {
                ucs_debug("mismatched rkey config index: "
                          "rkey[%lu]->cfg_index=%u cfg_index=%u",
                          i, rkey->cfg_index, cfg_index);
                return UCS_ERR_UNSUPPORTED;
            }
        }

        remote_md_map &= rkey->md_map;
    }

    if (remote_md_map == 0) {
        ucs_debug("empty remote_md_map");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

ucs_status_t ucp_device_mem_list_create(ucp_ep_h ep,
                                 const ucp_device_mem_list_params_t *params,
                                 ucp_device_mem_list_handle_h *handle_p)
{
    ucs_status_t status;
    ucp_device_mem_list_handle_h handle;
    ucp_device_mem_list_handle_t host_handle;
    uct_allocated_memory_t mem;
    ucp_memory_info_t mem_info;

    status = ucp_device_mem_list_params_check(params);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_mem_do_alloc(ep->worker->context, NULL, sizeof(*handle),
                              UCT_MD_MEM_ACCESS_LOCAL_READ |
                                      UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                              UCS_MEMORY_TYPE_CUDA, UCS_SYS_DEVICE_ID_UNKNOWN,
                              "ucp_device_mem_list_handle_t", &mem);
    if (status != UCS_OK) {
        ucs_error("failed to allocate ucp_device_mem_list_handle_t: %s",
                  ucs_status_string(status));
        return status;
    }

    ucp_memory_detect(ep->worker->context, mem.address, mem.length, &mem_info);
    if (mem_info.sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_error("failed to detect sys_dev for mem list handle: %s",
                  ucs_status_string(status));
        status = UCS_ERR_NO_DEVICE;
        goto err;
    }

    /* TODO actual lane lookup and population of handle */
    memset(&host_handle, 0, sizeof(host_handle));
    host_handle.version  = UCP_DEVICE_MEM_LIST_VERSION_V1;
    host_handle.host_mem = mem;

    ucp_mem_type_unpack(ep->worker, mem.address, &host_handle,
                        sizeof(host_handle), UCS_MEMORY_TYPE_CUDA);
    *handle_p = mem.address;
    return UCS_OK;

err:
    uct_mem_free(&mem);
    return status;
}

void ucp_device_mem_list_release(ucp_ep_h ep, ucp_device_mem_list_handle_h handle)
{
    ucp_device_mem_list_handle_t host_handle;

    if (handle == NULL) {
        return;
    }

    ucp_mem_type_pack(ep->worker, &host_handle, handle, sizeof(*handle),
                      UCS_MEMORY_TYPE_CUDA);
    ucs_assertv_always(host_handle.version == UCP_DEVICE_MEM_LIST_VERSION_V1,
                       "handle->version=%u", host_handle.version);
    uct_mem_free(&host_handle.host_mem);
}
