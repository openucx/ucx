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


KHASH_TYPE(ucp_device_handle_allocs, ucp_device_mem_list_handle_h,
           uct_allocated_memory_t);
#define ucp_device_handle_hash_key(_handle) \
    kh_int64_hash_func((uintptr_t)(_handle))
KHASH_IMPL(ucp_device_handle_allocs, ucp_device_mem_list_handle_h,
           uct_allocated_memory_t, 1,
           ucp_device_handle_hash_key, kh_int64_hash_equal);

static khash_t(ucp_device_handle_allocs) ucp_device_handle_hash;
static ucs_spinlock_t ucp_device_handle_hash_lock;


void ucp_device_init(void)
{
    ucs_spinlock_init(&ucp_device_handle_hash_lock, 0);
    kh_init_inplace(ucp_device_handle_allocs, &ucp_device_handle_hash);
}

void ucp_device_cleanup(void)
{
    kh_destroy_inplace(ucp_device_handle_allocs, &ucp_device_handle_hash);
    ucs_spinlock_destroy(&ucp_device_handle_hash_lock);
}

static ucs_status_t
ucp_device_mem_list_params_check(const ucp_device_mem_list_params_t *params)
{
    ucp_worker_cfg_index_t cfg_index = UCP_WORKER_CFG_INDEX_NULL;
    ucp_rkey_h rkey;
    size_t i, num_elements, element_size;
    const ucp_device_mem_list_elem_t *elements, *element;

    if (params == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    element_size = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params,
                                   element_size, ELEMENT_SIZE, 0);
    num_elements = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params,
                                   num_elements, NUM_ELEMENTS, 0);
    elements     = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params,
                                   elements, ELEMENTS, NULL);

    if ((element_size == 0) || (num_elements == 0) || (elements == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < num_elements; i++) {
        element = UCS_PTR_BYTE_OFFSET(elements, i * element_size);
        rkey = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element, rkey,
                               RKEY, NULL);

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
    }

    return UCS_OK;
}

ucs_status_t
ucp_device_mem_list_create(ucp_ep_h ep,
                           const ucp_device_mem_list_params_t *params,
                           ucp_device_mem_list_handle_h *handle_p)
{
    ucs_status_t status;
    ucp_device_mem_list_handle_t host_handle;
    uct_allocated_memory_t mem;
    ucp_memory_info_t mem_info;
    khiter_t iter;
    int ret;

    status = ucp_device_mem_list_params_check(params);
    if (status != UCS_OK) {
        return status;
    }

    /* TODO: Allocate based on lane selection results */
    status = ucp_mem_do_alloc(ep->worker->context, NULL, sizeof(**handle_p),
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

    ucp_mem_type_unpack(ep->worker, mem.address, &host_handle,
                        sizeof(host_handle), UCS_MEMORY_TYPE_CUDA);

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_put(ucp_device_handle_allocs, &ucp_device_handle_hash,
                  mem.address, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to hash handle=%p", mem.address);
        ucs_spin_unlock(&ucp_device_handle_hash_lock);
        goto err;
    } else if (ret == UCS_KH_PUT_KEY_PRESENT) {
        ucs_fatal("handle=%p already found in hash", mem.address);
    }

    *handle_p = mem.address;
    kh_value(&ucp_device_handle_hash, iter) = mem;
    ucs_spin_unlock(&ucp_device_handle_hash_lock);

    return UCS_OK;

err:
    uct_mem_free(&mem);
    return status;
}

void ucp_device_mem_list_release(ucp_device_mem_list_handle_h handle)
{
    khiter_t iter;
    uct_allocated_memory_t mem;

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_get(ucp_device_handle_allocs, &ucp_device_handle_hash, handle);

    ucs_assertv_always((iter != kh_end(&ucp_device_handle_hash)),
                       "handle=%p", handle);
    mem = kh_value(&ucp_device_handle_hash, iter);
    kh_del(ucp_device_handle_allocs, &ucp_device_handle_hash, iter);
    ucs_spin_unlock(&ucp_device_handle_hash_lock);

    uct_mem_free(&mem);
}
