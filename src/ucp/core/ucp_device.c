/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_types.h>
#include <ucs/type/param.h>


static ucs_status_t
ucp_mem_list_params_check(const ucp_mem_list_params_t *params)
{
    ucp_mem_h memh;
    ucp_rkey_h rkey;
    size_t num_elements;
    size_t element_size;
    const ucp_mem_list_elem_t *elements;
    size_t i;

    num_elements = UCS_PARAM_VALUE(UCP_MEM_LIST_PARAMS_FIELD,
                               params, num_elements, NUM_ELEMENTS, 0);
    element_size = UCS_PARAM_VALUE(UCP_MEM_LIST_PARAMS_FIELD,
                               params, element_size, ELEMENT_SIZE, 0);
    elements     = UCS_PARAM_VALUE(UCP_MEM_LIST_PARAMS_FIELD,
                               params, elements, ELEMENTS, NULL);

    if ((element_size != sizeof(*elements)) ||
        (num_elements == 0)) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < num_elements; i++) {
        memh = UCS_PARAM_VALUE(
                           UCP_MEM_LIST_ELEM_FIELD, &params->elements[i],
                           memh, MEMH, NULL);
        rkey = UCS_PARAM_VALUE(
                           UCP_MEM_LIST_ELEM_FIELD, &params->elements[i],
                           rkey, RKEY, NULL);
        if ((memh == NULL) || (rkey == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }
    }

    return UCS_OK;
}

ucs_status_t
ucp_mem_list_create(ucp_ep_h ep,
                    const ucp_mem_list_params_t *params,
                    ucp_device_mem_list_handle_h *handle_p)
{
    ucs_status_t status;
    ucp_device_mem_list_handle_h handle;
    uct_allocated_memory_t mem;
    ucp_memory_info_t mem_info;

    status = ucp_mem_list_params_check(params);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_mem_do_alloc(ep->worker->context, NULL, sizeof(*handle),
                              UCT_MD_MEM_ACCESS_LOCAL_READ |
                              UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                              UCS_MEMORY_TYPE_CUDA,
                              UCS_SYS_DEVICE_ID_UNKNOWN,
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

    return status;

err:
    uct_mem_free(&mem);
    return status;
}

void ucp_mem_list_release(ucp_device_mem_list_handle_h handle)
{
}
