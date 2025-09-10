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

#include "ucp_worker.inl"
#include "ucp_ep.inl"


KHASH_TYPE(ucp_device_handle_allocs, ucp_device_mem_list_handle_h,
           uct_allocated_memory_t);
#define ucp_device_handle_hash_key(_handle) \
    kh_int64_hash_func((uintptr_t)(_handle))
KHASH_IMPL(ucp_device_handle_allocs, ucp_device_mem_list_handle_h,
           uct_allocated_memory_t, 1, ucp_device_handle_hash_key,
           kh_int64_hash_equal);

/* Hash to track handle allocator, used at release time */
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
ucp_device_mem_handle_hash_insert(uct_allocated_memory_t *mem_handle)
{
    ucs_status_t status;
    khiter_t iter;
    int ret;

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
        kh_value(&ucp_device_handle_hash, iter) = *mem_handle;
        status                                  = UCS_OK;
    }

    ucs_spin_unlock(&ucp_device_handle_hash_lock);
    return status;
}

static uct_allocated_memory_t
ucp_device_mem_handle_hash_remove(ucp_device_mem_list_handle_h handle)
{
    khiter_t iter;
    uct_allocated_memory_t mem;

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_get(ucp_device_handle_allocs, &ucp_device_handle_hash, handle);
    ucs_assertv_always((iter != kh_end(&ucp_device_handle_hash)), "handle=%p",
                       handle);
    mem = kh_value(&ucp_device_handle_hash, iter);
    kh_del(ucp_device_handle_allocs, &ucp_device_handle_hash, iter);
    ucs_spin_unlock(&ucp_device_handle_hash_lock);
    return mem;
}

static ucs_status_t
ucp_device_mem_list_params_check(const ucp_device_mem_list_params_t *params,
                                 ucp_worker_cfg_index_t *rkey_cfg_index,
                                 ucs_sys_device_t *local_sys_dev,
                                 ucp_md_map_t *local_md_map,
                                 ucs_memory_type_t *mem_type)
{
    ucp_rkey_h rkey;
    ucp_mem_h memh;
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

    *local_sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    for (i = 0; i < num_elements; i++) {
        element = UCS_PTR_BYTE_OFFSET(elements, i * element_size);
        memh = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element, memh,
                               MEMH, NULL);
        rkey = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element, rkey,
                               RKEY, NULL);

        /* TODO: Delegate most of checks below to proto selection */
        if ((rkey == NULL) || (memh == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }

        if (i == 0) {
            *local_sys_dev  = memh->sys_dev;
            *local_md_map   = memh->md_map;
            *mem_type       = memh->mem_type;
            *rkey_cfg_index = rkey->cfg_index;
            if (*rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
                ucs_debug("invalid first rkey: cfg_index=%d", *rkey_cfg_index);
                return UCS_ERR_INVALID_PARAM;
            }
        } else {
            *local_md_map &= memh->md_map;
            if (rkey->cfg_index != *rkey_cfg_index) {
                ucs_debug("mismatched rkey config index: "
                          "ucp_rkey[%lu]->cfg_index=%u cfg_index=%u",
                          i, rkey->cfg_index, *rkey_cfg_index);
                return UCS_ERR_UNSUPPORTED;
            }

            if (memh->sys_dev != *local_sys_dev) {
                ucs_debug("mismatched local sys_dev: ucp_memh[%zu].sys_dev=%u "
                          "first_sys_dev=%u",
                          i, memh->sys_dev, *local_sys_dev);
                return UCS_ERR_UNSUPPORTED;
            }
        }
    }

    return UCS_OK;
}

static void ucp_device_mem_list_lane_lookup(
        ucp_ep_h ep, ucp_ep_config_t *ep_config, ucs_sys_device_t local_sys_dev,
        ucp_md_map_t local_md_map, ucs_sys_device_t remote_sys_dev,
        ucp_md_map_t remote_md_map,
        ucp_lane_index_t lanes[UCP_DEVICE_MEM_LIST_MAX_EPS])
{
    double best_bw[UCP_DEVICE_MEM_LIST_MAX_EPS] = {-1., -1.};
    ucp_lane_index_t lane;
    double bandwidth;
    ucp_ep_config_key_lane_t *lane_key;
    ucs_sys_device_t src_sys_dev;
    ucp_md_index_t src_md_index;

    lanes[0] = UCP_NULL_LANE;
    lanes[1] = UCP_NULL_LANE;

    for (lane = 0; lane < ep_config->key.num_lanes; ++lane) {
        if (!(ep_config->key.lanes[lane].lane_types &
              UCS_BIT(UCP_LANE_TYPE_DEVICE))) {
            continue;
        }

        lane_key = &ep_config->key.lanes[lane];
        if (lane_key->dst_sys_dev != remote_sys_dev) {
            ucs_trace("lane[%u] wrong destination sys_dev: dst_sys_dev=%u",
                      lane, lane_key->dst_sys_dev);
            continue;
        }

        if (!(remote_md_map & UCS_BIT(lane_key->dst_md_index))) {
            ucs_trace("lane[%u] missing remote md: dst_md_index=%u", lane,
                      lane_key->dst_md_index);
            continue;
        }

        src_sys_dev = ucp_ep_get_tl_rsc(ep, lane)->sys_device;
        if (src_sys_dev != local_sys_dev) {
            ucs_trace("lane[%u] wrong source sys_dev: src_sys_dev=%u", lane,
                      src_sys_dev);
            continue;
        }

        src_md_index = ucp_ep_md_index(ep, lane);
        if (!(local_md_map & UCS_BIT(src_md_index))) {
            ucs_trace("lane[%u] missing local md: src_md_index=%u", lane,
                      src_md_index);
            continue;
        }

        bandwidth = ucp_worker_iface_bandwidth(ep->worker,
                                               ucp_ep_get_rsc_index(ep, lane));
        ucs_trace("checking lane[%u] src_md_index=%u dst_md_index=%u "
                  "src_sys_dev=%u dst_sys_dev=%u bandwidth=%lfMB/s",
                  lane, src_md_index, lane_key->dst_md_index, src_sys_dev,
                  lane_key->dst_sys_dev, bandwidth / UCS_MBYTE);

        UCS_STATIC_ASSERT(UCP_DEVICE_MEM_LIST_MAX_EPS == 2);
        if (bandwidth > best_bw[0]) {
            best_bw[1] = best_bw[0];
            lanes[1]   = lanes[0];
            best_bw[0] = bandwidth;
            lanes[0]   = lane;
        } else if (bandwidth > best_bw[1]) {
            best_bw[1] = bandwidth;
            lanes[1]   = lane;
        } else {
            continue;
        }

        ucs_trace("best lanes: lane[%u]=%lfMB/s lane[%u]=%lfMB/s", lanes[0],
                  best_bw[0] / UCS_MBYTE, lanes[1], best_bw[1] / UCS_MBYTE);
    }
}

static ucs_status_t ucp_device_mem_list_create_handle(
        ucp_ep_h ep, ucs_sys_device_t local_sys_dev,
        const ucp_device_mem_list_params_t *params,
        ucp_lane_index_t lanes[UCP_DEVICE_MEM_LIST_MAX_EPS],
        ucp_ep_config_t *ep_config, ucs_memory_type_t mem_type,
        uct_allocated_memory_t *mem)
{
    size_t handle_size = 0;
    size_t uct_elem_size[UCP_DEVICE_MEM_LIST_MAX_EPS];
    uint8_t i, j, num_uct_eps;
    uct_iface_attr_v2_t attr;
    ucs_status_t status;
    ucp_worker_iface_t *wiface;
    ucp_device_mem_list_handle_t handle;
    uct_mem_h uct_memh;
    uct_rkey_t uct_rkey;
    uct_device_mem_element_t *uct_element;
    const ucp_device_mem_list_elem_t *ucp_element;
    ucp_md_index_t local_md_index;
    uint8_t rkey_index;

    /* For each available lane */
    for (i = 0;
         (i < UCP_DEVICE_MEM_LIST_MAX_EPS) && (lanes[i] != UCP_NULL_LANE);
         i++) {
        /* Query per transport UCT memory list element size */
        wiface          = ucp_worker_iface(ep->worker,
                                           ucp_ep_get_rsc_index(ep, lanes[i]));
        attr.field_mask = UCT_IFACE_ATTR_FIELD_DEVICE_MEM_ELEMENT_SIZE;
        status          = uct_iface_query_v2(wiface->iface, &attr);
        if (status != UCS_OK) {
            ucs_error("failed to query element size lanes[%u]=%u", i, lanes[i]);
            return status;
        }

        ucs_trace("query lane=%u device_mem_element_size=%zu", lanes[i],
                  attr.device_mem_element_size);

        handle_size     += attr.device_mem_element_size;
        uct_elem_size[i] = attr.device_mem_element_size;
    }

    if (i == 0) {
        ucs_error("failed to select lane");
        return UCS_ERR_NO_RESOURCE;
    }

    /* Populate handle header */
    num_uct_eps            = i;
    handle.version         = UCP_DEVICE_MEM_LIST_VERSION_V1;
    handle.proto_idx       = 0;
    handle.num_uct_eps     = num_uct_eps;
    handle.mem_list_length = params->num_elements;
    for (i = 0; i < num_uct_eps; i++) {
        status = uct_ep_get_device_ep(ucp_ep_get_lane(ep, lanes[i]),
                                      &handle.uct_device_eps[i]);
        if (status != UCS_OK) {
            ucs_error("failed to get device_ep for lane=%u", lanes[i]);
            goto err;
            ;
        }

        handle.uct_mem_element_size[i] = uct_elem_size[i];
    }

    /* Allocate handle on the same device memory */
    handle_size *= params->num_elements;
    handle_size += sizeof(handle);
    status       = ucp_mem_do_alloc(ep->worker->context, NULL, handle_size,
                                    UCT_MD_MEM_ACCESS_LOCAL_READ |
                                            UCT_MD_MEM_ACCESS_LOCAL_WRITE,
                                    mem_type, local_sys_dev,
                                    "ucp_device_mem_list_handle_t", mem);
    if (status != UCS_OK) {
        ucs_error("failed to allocate ucp_device_mem_list_handle_t: %s",
                  ucs_status_string(status));
        return status;
    }

    /* Populate element specific parameters */
    uct_element = UCS_PTR_TYPE_OFFSET(mem->address, ucs_typeof(handle));
    for (i = 0; i < num_uct_eps; i++) {
        local_md_index = ep_config->md_index[lanes[i]];
        wiface         = ucp_worker_iface(ep->worker,
                                          ucp_ep_get_rsc_index(ep, lanes[i]));
        ucp_element    = params->elements;
        for (j = 0; j < params->num_elements; j++) {
            /* Local registration */
            uct_memh = ucp_element->memh->uct[local_md_index];
            ucs_assertv((ucp_element->memh->md_map & UCS_BIT(local_md_index)) !=
                                0,
                        "uct_memh=%p md_map=0x%lx local_md_index=%u", uct_memh,
                        ucp_element->memh->md_map, local_md_index);
            ucs_assert(uct_memh != UCT_MEM_HANDLE_NULL);

            /* Remote registration */
            rkey_index =
                    ucs_bitmap2idx(ucp_element->rkey->md_map,
                                   ep_config->key.lanes[lanes[i]].dst_md_index);
            uct_rkey = ucp_rkey_get_tl_rkey(ucp_element->rkey, rkey_index);
            ucs_assert(uct_rkey != UCT_INVALID_RKEY);

            /* Element packing, expects device memory as destination */
            status = uct_iface_mem_element_pack(wiface->iface, uct_memh,
                                                uct_rkey, uct_element);
            if (status != UCS_OK) {
                ucs_error("failed to pack uct memory element for lane=%u",
                          lanes[i]);
                goto err;
            }

            ucp_element = UCS_PTR_BYTE_OFFSET(ucp_element,
                                              params->element_size);
            uct_element = UCS_PTR_BYTE_OFFSET(uct_element, uct_elem_size[i]);
        }
    }

    ucs_assert(UCS_PTR_BYTE_OFFSET(mem->address, handle_size) == uct_element);

    /* Migrate the constructed handle header to device memory */
    ucp_mem_type_unpack(ep->worker, mem->address, &handle, sizeof(handle),
                        mem_type);
    return UCS_OK;

err:
    uct_mem_free(mem);
    return status;
}

ucs_status_t
ucp_device_mem_list_create(ucp_ep_h ep,
                           const ucp_device_mem_list_params_t *params,
                           ucp_device_mem_list_handle_h *handle_p)
{
    ucp_lane_index_t lanes[UCP_DEVICE_MEM_LIST_MAX_EPS];
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucs_status_t status;
    ucp_rkey_config_t *rkey_config;
    ucs_sys_device_t local_sys_dev, remote_sys_dev;
    ucp_md_map_t local_md_map, remote_md_map;
    ucp_ep_config_t *ep_config;
    ucs_memory_type_t mem_type;
    uct_allocated_memory_t mem;

    if (!(ep->flags & UCP_EP_FLAG_REMOTE_CONNECTED)) {
        return UCS_ERR_NOT_CONNECTED;
    }

    /* Parameter sanity checks and extraction */
    status = ucp_device_mem_list_params_check(params, &rkey_cfg_index,
                                              &local_sys_dev, &local_md_map,
                                              &mem_type);
    if (status != UCS_OK) {
        return status;
    }

    /* Perform pseudo lane selection without size */
    rkey_config    = &ep->worker->rkey_config[rkey_cfg_index];
    ep_config      = ucp_worker_ep_config(ep->worker,
                                          rkey_config->key.ep_cfg_index);
    remote_sys_dev = rkey_config->key.sys_dev;
    remote_md_map  = rkey_config->key.md_map;

    ucs_trace_req(
            "device mem_list create ep=%p num_elements=%zu element_size=%zu "
            "local_sys_dev=%u remote_sys_dev=%u "
            "local_md_map=%" PRIx64 " remote_md_map=%" PRIx64,
            ep, params->num_elements, params->element_size, local_sys_dev,
            remote_sys_dev, local_md_map, remote_md_map);

    if ((remote_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) ||
        (local_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN)) {
        ucs_error("ep %p local or remote unknown sys_dev", ep);
        return UCS_ERR_NO_DEVICE;
    }

    /* Find set of best lanes */
    ucp_device_mem_list_lane_lookup(ep, ep_config, local_sys_dev, local_md_map,
                                    remote_sys_dev, remote_md_map, lanes);

    /* Handle creation with lanes and parameters */
    status = ucp_device_mem_list_create_handle(ep, local_sys_dev, params, lanes,
                                               ep_config, mem_type, &mem);
    if (status != UCS_OK) {
        ucs_error("failed to create handle: %s", ucs_status_string(status));
        return status;
    }

    /* Track memory allocator for later release */
    status = ucp_device_mem_handle_hash_insert(&mem);
    if (status != UCS_OK) {
        uct_mem_free(&mem);
    } else {
        *handle_p = mem.address;
    }

    return status;
}

void ucp_device_mem_list_release(ucp_device_mem_list_handle_h handle)
{
    uct_allocated_memory_t mem = ucp_device_mem_handle_hash_remove(handle);
    uct_mem_free(&mem);
}
