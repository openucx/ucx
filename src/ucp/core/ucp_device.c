/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_types.h>
#include <ucp/core/ucp_mm.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_types.h>
#include <ucs/type/param.h>
#include <ucp/wireup/wireup_ep.h>
#include <uct/api/v2/uct_v2.h>

#include "ucp_worker.inl"
#include "ucp_ep.inl"
#include "ucp_mm.inl"


typedef struct {
    uct_allocated_memory_t mem;
    uint32_t               mem_list_length;
} ucp_device_handle_info_t;

#define ucp_device_handle_hash_key(_handle) \
    kh_int64_hash_func((uintptr_t)(_handle))
KHASH_INIT(ucp_device_handle_allocs, void*, ucp_device_handle_info_t, 1,
           ucp_device_handle_hash_key, kh_int64_hash_equal);

/* Hash to track handle allocator, used at release time */
static khash_t(ucp_device_handle_allocs) ucp_device_handle_hash;
static ucs_spinlock_t ucp_device_handle_hash_lock;

/* Size of temporary allocation for local sys_dev detection */
#define UCP_DEVICE_LOCAL_SYS_DEV_DETECT_SIZE 64

enum {
    UCP_DEVICE_TL_TYPE_FIRST,
    UCP_DEVICE_TL_TYPE_LKEY = UCP_DEVICE_TL_TYPE_FIRST,
    UCP_DEVICE_TL_TYPE_NOLKEY,
    UCP_DEVICE_TL_TYPE_LAST,
};


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
ucp_device_mem_handle_hash_insert(const uct_allocated_memory_t *mem_handle,
                                  uint32_t mem_list_length)
{
    ucs_status_t status;
    khiter_t iter;
    int ret;
    ucp_device_handle_info_t info;

    info.mem             = *mem_handle;
    info.mem_list_length = mem_list_length;

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_put(ucp_device_handle_allocs, &ucp_device_handle_hash,
                  mem_handle->address, &ret);
    if (ret == UCS_KH_PUT_FAILED) {
        ucs_error("failed to hash handle=%p", mem_handle->address);
        status = UCS_ERR_NO_RESOURCE;
    } else if (ret == UCS_KH_PUT_KEY_PRESENT) {
        ucs_error("handle=%p already found in hash", mem_handle->address);
        status = UCS_ERR_ALREADY_EXISTS;
    } else {
        kh_value(&ucp_device_handle_hash, iter) = info;
        status                                  = UCS_OK;
    }

    ucs_spin_unlock(&ucp_device_handle_hash_lock);
    return status;
}

static uct_allocated_memory_t ucp_device_mem_handle_hash_remove(void *handle)
{
    khiter_t iter;
    uct_allocated_memory_t mem;

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_get(ucp_device_handle_allocs, &ucp_device_handle_hash, handle);
    ucs_assertv_always((iter != kh_end(&ucp_device_handle_hash)), "handle=%p",
                       handle);
    mem = kh_value(&ucp_device_handle_hash, iter).mem;
    kh_del(ucp_device_handle_allocs, &ucp_device_handle_hash, iter);
    ucs_spin_unlock(&ucp_device_handle_hash_lock);
    return mem;
}

static ucs_status_t
ucp_device_detect_local_sys_dev(ucp_context_h context,
                                ucs_memory_type_t mem_type,
                                ucs_sys_device_t *local_sys_dev)
{
    ucs_memory_info_t mem_info;
    uct_allocated_memory_t detect_mem;
    ucs_status_t status;

    status = ucp_mem_do_alloc(context, NULL,
                              UCP_DEVICE_LOCAL_SYS_DEV_DETECT_SIZE,
                              UCT_MD_MEM_ACCESS_LOCAL_READ |
                              UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                              mem_type, UCS_SYS_DEVICE_ID_UNKNOWN,
                              "local_sys_dev_detect", &detect_mem);
    if (status != UCS_OK) {
        ucs_error("failed to allocate memory for sys_dev detection: %s",
                  ucs_status_string(status));
        return status;
    }

    ucp_memory_detect_internal(context, detect_mem.address, detect_mem.length,
                               &mem_info);

    uct_mem_free(&detect_mem);

    if (mem_info.sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_error("detected unknown local_sys_dev");
        return UCS_ERR_UNSUPPORTED;
    }

    *local_sys_dev = mem_info.sys_dev;

    ucs_trace("detected local_sys_dev=%u", *local_sys_dev);
    return UCS_OK;
}

static void
ucp_device_get_tl_bitmap(const ucp_worker_h worker,
                         ucp_tl_bitmap_t tl_bitmap[UCP_DEVICE_TL_TYPE_LAST],
                         ucs_sys_device_t local_sys_dev)
{
    const ucp_worker_iface_t *wiface;
    ucp_rsc_index_t tl_id;
    int tl_type;

    /** TODO: Maybe cache results */
    for (tl_type = UCP_DEVICE_TL_TYPE_FIRST;
         tl_type < UCP_DEVICE_TL_TYPE_LAST; tl_type++) {
        UCS_STATIC_BITMAP_RESET_ALL(&tl_bitmap[tl_type]);
    }

    UCS_STATIC_BITMAP_FOR_EACH_BIT(tl_id, &worker->context->tl_bitmap) {
        wiface = ucp_worker_iface(worker, tl_id);

        if (!(wiface->attr.cap.flags & UCT_IFACE_FLAG_DEVICE_EP)) {
            continue;
        }

        if ((wiface->attr.ctl_device != UCS_SYS_DEVICE_ID_UNKNOWN) &&
            (wiface->attr.ctl_device != local_sys_dev)) {
            continue;
        }

        if (wiface->attr.cap.flags & UCT_IFACE_FLAG_DEVICE_LKEY) {
            tl_type = UCP_DEVICE_TL_TYPE_LKEY;
        } else {
            tl_type = UCP_DEVICE_TL_TYPE_NOLKEY;
        }
        UCS_STATIC_BITMAP_SET(&tl_bitmap[tl_type], tl_id);
    }
}

static ucs_status_t ucp_device_local_mem_list_element_pack(
        const ucp_worker_h worker, const ucp_worker_iface_t *wiface,
        const ucp_device_mem_list_elem_t *element,
        const ucs_memory_type_t mem_type, uct_device_mem_elem_t *mem_element)
{
    ucp_tl_resource_desc_t *resource;
    ucp_md_index_t md_index;
    ucp_mem_h memh;
    uct_mem_h uct_memh;
    ucp_tl_md_t *ucp_md;
    ucs_status_t status;

    if (wiface == NULL) {
        return UCS_OK;
    }

    resource = &worker->context->tl_rscs[wiface->rsc_index];
    md_index = resource->md_index;
    ucp_md   = &worker->context->tl_mds[md_index];
    memh = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element, memh, MEMH,
                           NULL);
    uct_memh = memh->uct[md_index];
    if (uct_memh == UCT_MEM_HANDLE_NULL) {
        ucs_error("invalid memh for md_index=%u", md_index);
        return UCS_ERR_INVALID_PARAM;
    }

    status = uct_md_mem_elem_pack(ucp_md->md, uct_memh, UCT_INVALID_RKEY,
                                  mem_element);
    if (status != UCS_OK) {
        ucs_error("failed to pack local mem element for memh=%p", memh);
    }

    return status;
}

static ucs_status_t ucp_device_mem_list_export_handle(
        ucp_worker_h worker, const void *handle, size_t handle_size,
        ucs_memory_type_t mem_type, ucs_sys_device_t local_sys_dev,
        uct_allocated_memory_t *mem, const char *alloc_name)
{
    ucs_status_t status;

    status = ucp_mem_do_alloc(worker->context, NULL, handle_size,
                              UCT_MD_MEM_ACCESS_LOCAL_READ |
                                      UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                              mem_type, local_sys_dev, alloc_name, mem);
    if (status != UCS_OK) {
        ucs_error("failed to allocate %s mem_type=%s sys_dev=%s: %s",
                  alloc_name, ucs_memory_type_names[mem_type],
                  ucs_topo_sys_device_get_name(local_sys_dev),
                  ucs_status_string(status));
        return status;
    }

    ucp_mem_type_unpack(worker, mem->address, handle, handle_size, mem_type);
    return UCS_OK;
}

static ucs_status_t ucp_device_local_mem_list_create_handle(
        const ucp_device_mem_list_params_t *params, ucs_memory_type_t mem_type,
        uct_allocated_memory_t *mem, ucs_sys_device_t local_sys_dev)
{
    const ucp_worker_h worker   = UCS_PARAM_VALUE(
            UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params, worker, WORKER, NULL);
    size_t uct_elem_size;
    size_t handle_size;
    int tl_type = UCP_DEVICE_TL_TYPE_LKEY;
    ucp_tl_bitmap_t tl_bitmap[UCP_DEVICE_TL_TYPE_LAST] = {};
    const ucp_device_mem_list_elem_t *ucp_element;
    const ucp_worker_iface_t *wiface;
    ucp_device_local_mem_list_t *handle;
    uct_device_local_mem_elem_t *uct_element;
    uct_device_mem_elem_t *tl_element;
    size_t i, num_lanes;
    ucs_status_t status;
    ucp_rsc_index_t tl_id;
    void *local_addr;

    ucp_device_get_tl_bitmap(worker, tl_bitmap, local_sys_dev);
    num_lanes = UCS_STATIC_BITMAP_POPCOUNT(tl_bitmap[tl_type]);

    uct_elem_size = sizeof(uct_device_local_mem_elem_t) +
                    (sizeof(uct_device_mem_elem_t) * num_lanes);
    handle_size   = (uct_elem_size * params->num_elements) + sizeof(*handle);
    handle        = ucs_calloc(1, handle_size, "ucp_device_local_mem_list_t");
    if (handle == NULL) {
        ucs_error("failed to allocate ucp_device_local_mem_list_t");
        return UCS_ERR_NO_MEMORY;
    }

    /* Populate element specific parameters */
    ucp_element = params->elements;
    uct_element = UCS_PTR_TYPE_OFFSET(handle, *handle);
    for (i = 0; i < params->num_elements; i++) {
        local_addr        = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD,
                                            ucp_element, local_addr, LOCAL_ADDR, NULL);
        uct_element->addr = local_addr;
        tl_element        = uct_element->tl;
        UCS_STATIC_BITMAP_FOR_EACH_BIT(tl_id, &tl_bitmap[tl_type]) {
            wiface = ucp_worker_iface(worker, tl_id);
            status = ucp_device_local_mem_list_element_pack(worker, wiface,
                                                            ucp_element,
                                                            mem_type,
                                                            tl_element);
            if (status != UCS_OK) {
                ucs_error("failed to pack local mem list element for "
                          "element=%zu",
                          i);
                goto out;
            }

            tl_element = UCS_PTR_TYPE_OFFSET(tl_element, *tl_element);
        }
        uct_element = (void*)tl_element;
        ucp_element = UCS_PTR_BYTE_OFFSET(ucp_element, params->element_size);
    }

    handle->version = UCP_DEVICE_MEM_LIST_VERSION_V1;
    handle->length  = params->num_elements;
    handle->num_lanes = num_lanes;
    status          = ucp_device_mem_list_export_handle(
            worker, handle, handle_size, mem_type, local_sys_dev, mem,
            "ucp_device_local_mem_list_handle_t");

out:
    ucs_free(handle);
    return status;
}

static ucs_status_t ucp_device_local_mem_list_params_check(
        const ucp_device_mem_list_params_t *params, ucs_memory_type_t mem_type,
        ucs_sys_device_t *local_sys_dev)
{
    const ucp_device_mem_list_elem_t *local_elements = UCS_PARAM_VALUE(
            UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params, elements, ELEMENTS, NULL);
    const ucp_worker_h worker                        = UCS_PARAM_VALUE(
            UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params, worker, WORKER, NULL);
    size_t element_size = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD,
                                          params, element_size, ELEMENT_SIZE,
                                          0);
    size_t num_elements = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD,
                                          params, num_elements, NUM_ELEMENTS,
                                          0);
    const ucp_device_mem_list_elem_t *element;
    ucp_mem_h memh;
    size_t i;
    ucs_status_t status;

    if ((local_elements == NULL) || (element_size == 0) ||
        (num_elements == 0) || (worker == NULL)) {
        ucs_error("invalid local mem list params: local_elements=%p, "
                  "element_size=%zu, num_elements=%zu, worker=%p",
                  local_elements, element_size, num_elements, worker);
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucp_device_detect_local_sys_dev(worker->context, mem_type,
                                             local_sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    for (i = 0; i < num_elements; i++) {
        element = (const ucp_device_mem_list_elem_t*)
                UCS_PTR_BYTE_OFFSET(local_elements, i * element_size);
        memh = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element, memh,
                               MEMH, NULL);
        if (memh == NULL) {
            ucs_error("missing memh for element=%zu", i);
            return UCS_ERR_INVALID_PARAM;
        }

        if (memh->sys_dev != *local_sys_dev) {
            ucs_error("mismatched local sys_dev for element=%zu", i);
            return UCS_ERR_UNSUPPORTED;
        }
    }

    return status;
}

ucs_status_t
ucp_device_local_mem_list_create(const ucp_device_mem_list_params_t *params,
                                 ucp_device_local_mem_list_h *mem_list_h)
{
    const ucs_memory_type_t export_mem_type = UCS_MEMORY_TYPE_CUDA;
    ucs_status_t status;
    uct_allocated_memory_t mem;
    ucs_sys_device_t local_sys_dev;

    status = ucp_device_local_mem_list_params_check(params, export_mem_type,
                                                    &local_sys_dev);
    if (status != UCS_OK) {
        ucs_error("failed to check local mem list params: %s",
                  ucs_status_string(status));
        return status;
    }

    status = ucp_device_local_mem_list_create_handle(params, export_mem_type,
                                                     &mem, local_sys_dev);
    if (status != UCS_OK) {
        ucs_error("failed to create local mem list handle: %s",
                  ucs_status_string(status));
        return status;
    }

    /* Track memory allocator for later release */
    status = ucp_device_mem_handle_hash_insert(&mem, params->num_elements);
    if (status != UCS_OK) {
        uct_mem_free(&mem);
    } else {
        *mem_list_h = mem.address;
    }

    return status;
}

static ucp_lane_index_t ucp_device_ep_find_lane(const ucp_ep_h ep, ucp_rsc_index_t tl_id)
{
    ucp_lane_index_t lane;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        if (ucp_ep_get_rsc_index(ep, lane) == tl_id) {
            return lane;
        }
    }

    return UCP_NULL_LANE;
}

static int
ucp_device_ep_check_lanes(const ucp_ep_h ep, ucp_tl_bitmap_t *tl_bitmap)
{
    ucp_lane_index_t lane;
    ucp_rsc_index_t tl_id;

    if (UCS_STATIC_BITMAP_POPCOUNT(*tl_bitmap) == 0) {
        return 0;
    }

    UCS_STATIC_BITMAP_FOR_EACH_BIT(tl_id, tl_bitmap) {
        lane = ucp_device_ep_find_lane(ep, tl_id);
        if (lane == UCP_NULL_LANE) {
            return 0;
        }
    }

    return 1;
}

static ucs_status_t ucp_device_remote_mem_list_element_pack(
        const ucp_device_mem_list_elem_t *element, ucp_rsc_index_t tl_id,
        uct_device_remote_tl_elem_t *mem_element)
{
    const ucp_ep_h ep     = element->ep;
    const ucp_rkey_h rkey = element->rkey;
    ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    uint8_t rkey_index;
    uct_rkey_t uct_rkey;
    uct_ep_h uct_ep;
    uct_device_ep_h device_ep;
    ucs_status_t status;
    ucp_lane_index_t lane;

    lane = ucp_device_ep_find_lane(ep, tl_id);
    if (lane == UCP_NULL_LANE) {
        ucs_error("no lane found for ep=%p", ep);
        return UCS_ERR_NO_DEVICE;
    }

    uct_ep = ucp_ep_get_lane(ep, lane);
    status = uct_ep_get_device_ep(uct_ep, &device_ep);
    if (status != UCS_OK) {
        ucs_error("failed to get device_ep for lane=%u", lane);
        return status;
    }

    rkey_index = ucs_bitmap2idx(rkey->md_map,
                                ep_config->key.lanes[lane].dst_md_index);
    uct_rkey   = ucp_rkey_get_tl_rkey(rkey, rkey_index);
    ucs_assert(uct_rkey != UCT_INVALID_RKEY);

    mem_element->ep = device_ep;
    status          = uct_md_mem_elem_pack(ucp_ep_md(ep, lane), NULL, uct_rkey,
                                           &mem_element->uct);
    if (status != UCS_OK) {
        ucs_error("failed to pack uct memory element for lane=%u", lane);
    }

    return status;
}

#define UCP_DEVICE_MEM_LIST_GET_ELEMENT(_params, _i) \
    ((ucp_device_mem_list_elem_t*)UCS_PTR_BYTE_OFFSET( \
            (_params)->elements, (_i) * (_params)->element_size))
#define UCP_DEVICE_MEM_LIST_GET_EP(_params, _i) \
    (UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, \
                     UCP_DEVICE_MEM_LIST_GET_ELEMENT(_params, _i), ep, EP, \
                     NULL))
#define UCP_DEVICE_MEM_LIST_GET_RKEY(_params, _i) \
    (UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, \
                     UCP_DEVICE_MEM_LIST_GET_ELEMENT(_params, _i), rkey, RKEY, \
                     NULL))
#define UCP_DEVICE_MEM_ELEMENT_IS_GAP(_ucp_element) \
    ((_ucp_element)->field_mask == 0)

static ucp_ep_h ucp_device_remote_mem_list_get_first_ep(
        const ucp_device_mem_list_params_t *params)
{
    const ucp_device_mem_list_elem_t *ucp_element = params->elements;
    size_t i                                      = 0;

    for (i = 0; i < params->num_elements; i++) {
        if (UCP_DEVICE_MEM_ELEMENT_IS_GAP(ucp_element)) {
            ucp_element = UCS_PTR_BYTE_OFFSET(ucp_element,
                                              params->element_size);
            continue;
        }

        return UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, ucp_element, ep,
                               EP, NULL);
    }

    return NULL;
}

static ucs_status_t
ucp_device_remote_mem_list_fill(const ucp_device_mem_list_elem_t *ucp_element,
                                ucp_tl_bitmap_t *tl_bitmap, size_t num_lanes,
                                uct_device_remote_mem_elem_t *uct_element)
{
    uct_device_remote_tl_elem_t *tl_element;
    ucp_rsc_index_t tl_id;
    ucs_status_t status;
    size_t i;

    uct_element->addr = ucp_element->remote_addr;
    tl_element        = uct_element->tl;
    for (i = 0; i < num_lanes;) {
        UCS_STATIC_BITMAP_FOR_EACH_BIT(tl_id, tl_bitmap) {
            status = ucp_device_remote_mem_list_element_pack(ucp_element, tl_id,
                                                             tl_element);
            if (status != UCS_OK) {
                return status;
            }

            tl_element = UCS_PTR_TYPE_OFFSET(tl_element, *tl_element);
            i++;
        }
    }

    return UCS_OK;
}

static ucs_status_t ucp_device_remote_mem_list_create_handle(
        const ucp_device_mem_list_params_t *params, ucs_memory_type_t mem_type,
        uct_allocated_memory_t *mem)
{
    const ucp_ep_h ep = ucp_device_remote_mem_list_get_first_ep(params);
    size_t uct_elem_size;
    size_t handle_size         = 0;
    ucp_tl_bitmap_t tl_bitmap[UCP_DEVICE_TL_TYPE_LAST] = {};
    const ucp_device_mem_list_elem_t *ucp_element;
    ucp_device_remote_mem_list_t *handle;
    uct_device_remote_mem_elem_t *uct_element;
    ucs_sys_device_t local_sys_dev;
    size_t i, num_lanes;
    ucs_status_t status;
    int tl_type;

    if (ep == NULL) {
        ucs_error("no ep found in remote mem list");
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucp_device_detect_local_sys_dev(ep->worker->context, mem_type,
                                             &local_sys_dev);
    if (status != UCS_OK) {
        return status;
    }

    ucp_device_get_tl_bitmap(ep->worker, tl_bitmap, local_sys_dev);

    /* handle->num_lanes is the least common multiple of both lane types, so:
     * - each lane is replicated num_lanes / popcount(tl_bitmap) times
     * - channel_id % num_lanes maps to the correct lane, regardless of lane type
     */
    num_lanes = UCS_STATIC_BITMAP_POPCOUNT(tl_bitmap[UCP_DEVICE_TL_TYPE_LKEY]);
    if (!num_lanes) {
        if (!UCS_STATIC_BITMAP_POPCOUNT(tl_bitmap[UCP_DEVICE_TL_TYPE_NOLKEY])) {
            ucs_error("failed to pack uct memory element for first element");
            return UCS_ERR_INVALID_PARAM;
        }

        ucs_assert(UCS_STATIC_BITMAP_POPCOUNT(
                           tl_bitmap[UCP_DEVICE_TL_TYPE_NOLKEY]) == 1);
        num_lanes = 1;
    }

    ucp_element   = params->elements;
    uct_elem_size = sizeof(uct_device_remote_mem_elem_t) +
                    (sizeof(uct_device_remote_tl_elem_t) * num_lanes);
    handle_size   = sizeof(*handle) + (params->num_elements * uct_elem_size);
    handle      = ucs_calloc(1, handle_size, "ucp_device_remote_mem_list_t");
    if (handle == NULL) {
        ucs_error("failed to allocate ucp_device_remote_mem_list_t");
        return UCS_ERR_NO_MEMORY;
    }

    uct_element = UCS_PTR_TYPE_OFFSET(handle, *handle);
    for (i = 0; i < params->num_elements; i++) {
        if (!UCP_DEVICE_MEM_ELEMENT_IS_GAP(ucp_element)) {
            for (tl_type = UCP_DEVICE_TL_TYPE_FIRST;
                 tl_type < UCP_DEVICE_TL_TYPE_LAST; tl_type++) {
                if (ucp_device_ep_check_lanes(ucp_element->ep,
                                              &tl_bitmap[tl_type])) {
                    break;
                }
            }

            if (tl_type == UCP_DEVICE_TL_TYPE_LAST) {
                ucs_error("lane not found for element %zd", i);
                status =  UCS_ERR_INVALID_PARAM;
                goto out;
            }

            status = ucp_device_remote_mem_list_fill(ucp_element,
                                                     &tl_bitmap[tl_type],
                                                     num_lanes, uct_element);
            if (status != UCS_OK) {
                goto out;
            }
        }
        uct_element = UCS_PTR_BYTE_OFFSET(uct_element, uct_elem_size);
        ucp_element = UCS_PTR_BYTE_OFFSET(ucp_element, params->element_size);
    }

    handle->version = UCP_DEVICE_MEM_LIST_VERSION_V1;
    handle->length  = params->num_elements;
    handle->num_lanes = num_lanes;
    status          = ucp_device_mem_list_export_handle(
            ep->worker, handle, handle_size, mem_type, local_sys_dev, mem,
            "ucp_device_remote_mem_list_handle_t");

out:
    ucs_free(handle);
    return status;
}

static ucs_status_t ucp_device_remote_mem_list_params_check(
        const ucp_device_mem_list_params_t *params)
{
    const ucp_device_mem_list_elem_t *remote_elements = UCS_PARAM_VALUE(
            UCP_DEVICE_MEM_LIST_PARAMS_FIELD, params, elements, ELEMENTS, NULL);
    size_t element_size = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD,
                                          params, element_size, ELEMENT_SIZE,
                                          0);
    size_t num_elements = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_PARAMS_FIELD,
                                          params, num_elements, NUM_ELEMENTS,
                                          0);
    const ucp_device_mem_list_elem_t *remote_element;
    ucp_rkey_h rkey;
    ucp_ep_h ep;
    size_t i;
    ucs_status_t status;
    void *addr_ptr;
    uint64_t remote_addr;

    if (remote_elements == NULL || element_size == 0 || num_elements == 0) {
        ucs_error("invalid remote mem list params: remote_elements=%p, "
                  "element_size=%zu, num_elements=%zu",
                  remote_elements, element_size, num_elements);
        return UCS_ERR_INVALID_PARAM;
    }

    for (i = 0; i < num_elements; i++) {
        remote_element = UCP_DEVICE_MEM_LIST_GET_ELEMENT(params, i);
        if (UCP_DEVICE_MEM_ELEMENT_IS_GAP(remote_element)) {
            continue;
        }

        rkey = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, remote_element,
                               rkey, RKEY, NULL);
        ep = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, remote_element, ep,
                             EP, NULL);
        if ((rkey == NULL) || (ep == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }

        remote_addr = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD,
                                      remote_element, remote_addr, REMOTE_ADDR,
                                      0);
        status      = ucp_rkey_ptr(rkey, remote_addr, &addr_ptr);
        /* Check if ep is connected only in case cuda_ipc can't be used for transfer */
        if (status != UCS_OK) {
            if (!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
                /*
                 * Do not log error here because UCS_ERR_NOT_CONNECTED is expected
                 * during connection establishment. Applications are expected to retry
                 * with progress.
                 */
                return UCS_ERR_NOT_CONNECTED;
            }
        }
    }

    return UCS_OK;
}


ucs_status_t
ucp_device_remote_mem_list_create(const ucp_device_mem_list_params_t *params,
                                  ucp_device_remote_mem_list_h *mem_list_h)
{
    const ucs_memory_type_t export_mem_type = UCS_MEMORY_TYPE_CUDA;
    ucs_status_t status;
    uct_allocated_memory_t mem;

    status = ucp_device_remote_mem_list_params_check(params);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_device_remote_mem_list_create_handle(params, export_mem_type,
                                                      &mem);
    if (status != UCS_OK) {
        /*
         * Do not log error for UCS_ERR_NOT_CONNECTED because it is expected
         * during connection establishment. Applications are expected to retry
         * with progress.
         */
        if (status != UCS_ERR_NOT_CONNECTED) {
            ucs_error("failed to create handle: %s", ucs_status_string(status));
        }
        return status;
    }

    /* Track memory allocator for later release */
    status = ucp_device_mem_handle_hash_insert(&mem, params->num_elements);
    if (status != UCS_OK) {
        uct_mem_free(&mem);
    } else {
        *mem_list_h = mem.address;
    }

    return status;
}

uint32_t ucp_device_get_mem_list_length(const void *mem_list_h)
{
    khiter_t iter;
    uint32_t length;

    ucs_assert(mem_list_h != NULL);

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_get(ucp_device_handle_allocs, &ucp_device_handle_hash,
                  (void*)mem_list_h);
    ucs_assertv_always((iter != kh_end(&ucp_device_handle_hash)), "handle=%p",
                       mem_list_h);
    length = kh_value(&ucp_device_handle_hash, iter).mem_list_length;
    ucs_spin_unlock(&ucp_device_handle_hash_lock);

    return length;
}

void ucp_device_mem_list_release(void *mem_list_h)
{
    uct_allocated_memory_t mem;

    mem = ucp_device_mem_handle_hash_remove(mem_list_h);
    uct_mem_free(&mem);
}

static ucs_memory_type_t
ucp_device_counter_mem_type(ucp_context_h context, const void *counter_ptr,
                            const ucp_device_counter_params_t *params)
{
    ucs_memory_info_t mem_info;

    if (params->field_mask & UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH) {
        return params->memh->mem_type;
    }

    if (params->field_mask & UCP_DEVICE_COUNTER_PARAMS_FIELD_MEM_TYPE) {
        return params->mem_type;
    }

    ucp_memory_detect_internal(context, counter_ptr, sizeof(uint64_t),
                               &mem_info);
    return mem_info.type;
}

ucs_status_t ucp_device_counter_init(ucp_worker_h worker,
                                     const ucp_device_counter_params_t *params,
                                     void *counter_ptr)
{
    uint64_t counter_value = 0;
    ucs_memory_type_t mem_type;

    mem_type = ucp_device_counter_mem_type(worker->context, counter_ptr,
                                           params);
    ucp_dt_contig_unpack(worker, counter_ptr, &counter_value,
                         sizeof(counter_value), mem_type,
                         sizeof(counter_value));
    return UCS_OK;
}

uint64_t ucp_device_counter_read(ucp_worker_h worker,
                                 const ucp_device_counter_params_t *params,
                                 void *counter_ptr)
{
    uint64_t counter_value = 0;
    ucs_memory_type_t mem_type;

    mem_type = ucp_device_counter_mem_type(worker->context, counter_ptr,
                                           params);
    ucp_dt_contig_pack(worker, &counter_value, counter_ptr,
                       sizeof(counter_value), mem_type, sizeof(counter_value));
    return counter_value;
}
