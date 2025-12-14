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
#include <ucp/dt/dt_contig.h>
#include <ucp/api/device/ucp_host.h>
#include <ucp/api/device/ucp_device_types.h>
#include <ucs/type/param.h>
#include <ucp/wireup/wireup_ep.h>

#include "ucp_worker.inl"
#include "ucp_ep.inl"
#include "ucp_mm.inl"


typedef struct {
    uct_allocated_memory_t mem;
    uint32_t               mem_list_length;
} ucp_device_handle_info_t;

KHASH_TYPE(ucp_device_handle_allocs, ucp_device_mem_list_handle_h,
           ucp_device_handle_info_t);
#define ucp_device_handle_hash_key(_handle) \
    kh_int64_hash_func((uintptr_t)(_handle))
KHASH_IMPL(ucp_device_handle_allocs, ucp_device_mem_list_handle_h,
           ucp_device_handle_info_t, 1, ucp_device_handle_hash_key,
           kh_int64_hash_equal);

/* Hash to track handle allocator, used at release time */
static khash_t(ucp_device_handle_allocs) ucp_device_handle_hash;
static ucs_spinlock_t ucp_device_handle_hash_lock;

/* Size of temporary allocation for local sys_dev detection */
#define UCP_DEVICE_LOCAL_SYS_DEV_DETECT_SIZE 64


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

static uct_allocated_memory_t
ucp_device_mem_handle_hash_remove(ucp_device_mem_list_handle_h handle)
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

static ucp_md_map_t
ucp_device_detect_local_md_map(const ucp_context_h context,
                               ucs_sys_device_t local_sys_dev)
{
    ucp_md_map_t local_md_map = 0;
    ucp_md_index_t md_index;

    /* Build MD map from MDs that can access the local_sys_dev */
    for (md_index = 0; md_index < context->num_mds; md_index++) {
        ucp_sys_dev_map_t sys_dev_map = context->tl_mds[md_index].sys_dev_map;

        if (sys_dev_map & UCS_BIT(local_sys_dev)) {
            local_md_map |= UCS_BIT(md_index);
        }
    }

    ucs_trace("detected local_md_map=0x%" PRIx64 " for local_sys_dev=%u",
              local_md_map, local_sys_dev);
    return local_md_map;
}

static ucs_status_t
ucp_check_rkey_elem(const ucp_device_mem_list_elem_t *element, size_t i,
                    ucp_worker_cfg_index_t *rkey_cfg_index)
{
    ucp_rkey_h rkey = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element,
                                      rkey, RKEY, NULL);

    if (!rkey || (rkey->cfg_index == UCP_WORKER_CFG_INDEX_NULL)) {
        ucs_debug("invalid rkey[%zu]: %s is NULL", i,
                  rkey ? "cfg_index" : "rkey");
        return UCS_ERR_INVALID_PARAM;
    }

    if (*rkey_cfg_index == UCP_WORKER_CFG_INDEX_NULL) {
        *rkey_cfg_index = rkey->cfg_index;
    } else if (*rkey_cfg_index != rkey->cfg_index) {
        ucs_debug("mismatched rkey config index: "
                  "ucp_rkey[%zu]->cfg_index=%u cfg_index=%u",
                  i, rkey->cfg_index, *rkey_cfg_index);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static ucs_status_t ucp_check_memh_elem(const ucp_mem_h memh, size_t i,
                                        ucs_sys_device_t *local_sys_dev,
                                        ucp_md_map_t *local_md_map)
{
    if (memh->sys_dev != *local_sys_dev) {
        ucs_debug("mismatched local sys_dev: ucp_memh[%zu].sys_dev=%u "
                  "first_sys_dev=%u",
                  i, memh->sys_dev, *local_sys_dev);
        return UCS_ERR_UNSUPPORTED;
    }

    *local_md_map &= memh->md_map;

    return UCS_OK;
}

static ucs_status_t ucp_device_mem_list_params_check(
        ucp_context_h context, const ucp_device_mem_list_params_t *params,
        ucs_memory_type_t mem_type, ucp_worker_cfg_index_t *rkey_cfg_index,
        ucs_sys_device_t *local_sys_dev, ucp_md_map_t *local_md_map)
{
    int first_memh = 1;
    size_t i, num_elements, element_size;
    const ucp_device_mem_list_elem_t *elements, *element;
    ucp_mem_h memh;
    ucs_status_t status;

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

    *local_sys_dev  = UCS_SYS_DEVICE_ID_UNKNOWN;
    *local_md_map   = 0;
    *rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL;

    for (i = 0; i < num_elements; i++) {
        element = UCS_PTR_BYTE_OFFSET(elements, i * element_size);
        status  = ucp_check_rkey_elem(element, i, rkey_cfg_index);
        if (status != UCS_OK) {
            return status;
        }

        memh = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, element, memh,
                               MEMH, NULL);
        if (memh != NULL) {
            if (first_memh) {
                *local_sys_dev = memh->sys_dev;
                *local_md_map  = memh->md_map;
                first_memh     = 0;
            } else {
                status = ucp_check_memh_elem(memh, i, local_sys_dev,
                                             local_md_map);
                if (status != UCS_OK) {
                    return status;
                }
            }
        }
    }

    if (*local_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        status = ucp_device_detect_local_sys_dev(context, mem_type,
                                                 local_sys_dev);
        if (status != UCS_OK) {
            return status;
        }

        *local_md_map = ucp_device_detect_local_md_map(context,
                                                       *local_sys_dev);
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
        /* Check lane remote sys dev only when remote memory is not host */
        if ((remote_sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) &&
            (remote_sys_dev != lane_key->dst_sys_dev)) {
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
    unsigned i, j, num_uct_eps;
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
    void **local_addresses;
    uint64_t *remote_addresses;
    size_t *lengths;
    size_t length;
    void *local_addr;
    uint64_t remote_addr;
    ucp_mem_h memh;

    handle_size += sizeof(*handle.local_addrs) + sizeof(*handle.remote_addrs) +
                   sizeof(*handle.lengths);

    /* For each available lane */
    for (i = 0;
         (i < UCP_DEVICE_MEM_LIST_MAX_EPS) && (lanes[i] != UCP_NULL_LANE);
         i++) {
        if (ucp_wireup_ep_test(ucp_ep_get_lane(ep, lanes[i]))) {
            /* TODO support proxy mem_element_pack() on wireup_ep */
            return UCS_ERR_NOT_CONNECTED;
        }

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
        ucs_error("failed to select lane for local device %s",
                  ucs_topo_sys_device_get_name(local_sys_dev));
        return UCS_ERR_NO_DEVICE;
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

    /* populate elements common parameters */
    local_addresses = (void**)UCS_PTR_BYTE_OFFSET(mem->address, sizeof(handle));
    remote_addresses = (uint64_t*)
            UCS_PTR_BYTE_OFFSET(local_addresses, sizeof(*handle.local_addrs) *
                                                         params->num_elements);
    lengths          = (size_t*)UCS_PTR_BYTE_OFFSET(remote_addresses,
                                                    sizeof(*handle.remote_addrs) *
                                                            params->num_elements);
    for (i = 0; i < params->num_elements; i++) {
        ucp_element = &params->elements[i];
        local_addr  = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD,
                                      ucp_element, local_addr, LOCAL_ADDR, NULL);
        remote_addr = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD,
                                      ucp_element, remote_addr, REMOTE_ADDR, 0);
        length = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD, ucp_element,
                                 length, LENGTH, 0);
        ucp_mem_type_unpack(ep->worker, &local_addresses[i], &local_addr,
                            sizeof(local_addresses[i]), mem_type);
        ucp_mem_type_unpack(ep->worker, &remote_addresses[i], &remote_addr,
                            sizeof(remote_addresses[i]), mem_type);
        ucp_mem_type_unpack(ep->worker, &lengths[i], &length,
                            sizeof(lengths[i]), mem_type);
    }

    handle.local_addrs  = local_addresses;
    handle.remote_addrs = remote_addresses;
    handle.lengths      = lengths;

    /* Populate element specific parameters */
    handle.uct_mem_elements = uct_element = UCS_PTR_BYTE_OFFSET(
            lengths, sizeof(*handle.lengths) * params->num_elements);
    for (i = 0; i < num_uct_eps; i++) {
        local_md_index = ep_config->md_index[lanes[i]];
        wiface         = ucp_worker_iface(ep->worker,
                                          ucp_ep_get_rsc_index(ep, lanes[i]));
        ucp_element    = params->elements;
        for (j = 0; j < params->num_elements; j++) {
            memh = UCS_PARAM_VALUE(UCP_DEVICE_MEM_LIST_ELEM_FIELD,
                                   ucp_element, memh, MEMH, NULL);
            if (memh != NULL) {
                /* Local registration */
                uct_memh = memh->uct[local_md_index];
                ucs_assertv(
                    (memh->md_map & UCS_BIT(local_md_index)) != 0,
                     "uct_memh=%p md_map=0x%lx local_md_index=%u", uct_memh,
                     memh->md_map, local_md_index);
                ucs_assert(uct_memh != UCT_MEM_HANDLE_NULL);
            } else {
                uct_memh = UCT_MEM_HANDLE_NULL;
            }

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
    const ucs_memory_type_t export_mem_type = UCS_MEMORY_TYPE_CUDA;
    ucp_worker_cfg_index_t rkey_cfg_index   = UCP_WORKER_CFG_INDEX_NULL;
    ucp_lane_index_t lanes[UCP_DEVICE_MEM_LIST_MAX_EPS];
    ucs_status_t status;
    ucp_rkey_config_t *rkey_config;
    ucs_sys_device_t local_sys_dev, remote_sys_dev;
    ucp_md_map_t local_md_map, remote_md_map;
    ucp_ep_config_t *ep_config;
    uct_allocated_memory_t mem;

    /* Parameter sanity checks and extraction */
    status = ucp_device_mem_list_params_check(ep->worker->context, params,
                                              export_mem_type, &rkey_cfg_index,
                                              &local_sys_dev, &local_md_map);
    if (status != UCS_OK) {
        return status;
    }

    /* Perform pseudo lane selection without size */
    rkey_config = &ep->worker->rkey_config[rkey_cfg_index];
    ep_config = ucp_worker_ep_config(ep->worker, rkey_config->key.ep_cfg_index);
    remote_sys_dev = rkey_config->key.sys_dev;
    remote_md_map  = rkey_config->key.md_map;

    ucs_trace_req(
            "device mem_list create ep=%p num_elements=%zu element_size=%zu "
            "local_sys_dev=%u remote_sys_dev=%u "
            "local_md_map=%" PRIx64 " remote_md_map=%" PRIx64,
            ep, params->num_elements, params->element_size, local_sys_dev,
            remote_sys_dev, local_md_map, remote_md_map);

    if (local_sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_error("ep %p unknown local sys_dev", ep);
        return UCS_ERR_NO_DEVICE;
    }

    /* Find set of best lanes */
    ucp_device_mem_list_lane_lookup(ep, ep_config, local_sys_dev, local_md_map,
                                    remote_sys_dev, remote_md_map, lanes);

    /* Handle creation with lanes and parameters */
    status = ucp_device_mem_list_create_handle(ep, local_sys_dev, params, lanes,
                                               ep_config, export_mem_type,
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
        *handle_p = mem.address;
    }

    return status;
}

uint32_t
ucp_device_get_mem_list_length(const ucp_device_mem_list_handle_h handle)
{
    khiter_t iter;
    uint32_t length;

    ucs_assert(handle != NULL);

    ucs_spin_lock(&ucp_device_handle_hash_lock);
    iter = kh_get(ucp_device_handle_allocs, &ucp_device_handle_hash, handle);
    ucs_assertv_always((iter != kh_end(&ucp_device_handle_hash)), "handle=%p",
                       handle);
    length = kh_value(&ucp_device_handle_hash, iter).mem_list_length;
    ucs_spin_unlock(&ucp_device_handle_hash_lock);

    return length;
}

void ucp_device_mem_list_release(ucp_device_mem_list_handle_h handle)
{
    uct_allocated_memory_t mem = ucp_device_mem_handle_hash_remove(handle);
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
