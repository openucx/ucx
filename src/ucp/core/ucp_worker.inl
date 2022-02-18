/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_WORKER_INL_
#define UCP_WORKER_INL_

#include "ucp_worker.h"
#include "ucp_rkey.inl"

#include <ucp/core/ucp_request.h>
#include <ucp/wireup/address.h>
#include <ucs/datastruct/ptr_map.inl>


UCS_PTR_MAP_IMPL(ep, 1);


KHASH_IMPL(ucp_worker_rkey_config, ucp_rkey_config_key_t,
           ucp_worker_cfg_index_t, 1, ucp_rkey_config_hash_func,
           ucp_rkey_config_is_equal);

/**
 * Resolve remote key configuration key to a remote key configuration index.
 *
 * @param [in]  worker          UCP worker to resolve configuration on.
 * @param [in]  key             Rkey configuration key.
 * @param [out] cfg_index_p     Filled with configuration index in the worker.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t ucp_worker_rkey_config_get(
        ucp_worker_h worker, const ucp_rkey_config_key_t *key,
        const ucs_sys_dev_distance_t *lanes_distance,
        ucp_worker_cfg_index_t *cfg_index_p)
{
    khiter_t khiter = kh_get(ucp_worker_rkey_config, &worker->rkey_config_hash,
                             *key);
    if (ucs_likely(khiter != kh_end(&worker->rkey_config_hash))) {
        *cfg_index_p = kh_val(&worker->rkey_config_hash, khiter);
        return UCS_OK;
    }

    return ucp_worker_add_rkey_config(worker, key, lanes_distance, cfg_index_p);
}

static UCS_F_ALWAYS_INLINE khint_t
ucp_worker_mpool_hash_func(ucp_worker_mpool_key_t mpool_key)
{
    return (khint_t)mpool_key.mem_type ^ (mpool_key.sys_dev << 8);
}

static UCS_F_ALWAYS_INLINE int
ucp_worker_mpool_key_is_equal(ucp_worker_mpool_key_t mpool_key1,
                              ucp_worker_mpool_key_t mpool_key2)
{
    return (mpool_key1.sys_dev == mpool_key2.sys_dev) &&
           (mpool_key1.mem_type == mpool_key2.mem_type);
}

KHASH_IMPL(ucp_worker_mpool_hash, ucp_worker_mpool_key_t, ucs_mpool_t,
           1, ucp_worker_mpool_hash_func, ucp_worker_mpool_key_is_equal);


/**
 * @return Worker name
 */
static UCS_F_ALWAYS_INLINE const char*
ucp_worker_get_address_name(ucp_worker_h worker)
{
    return worker->address_name;
}

/**
 * @return endpoint by a key received from remote side
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_worker_get_ep_by_id(ucp_worker_h worker, ucs_ptr_map_key_t id,
                        ucp_ep_h *ep_p)
{
    ucs_status_t status;
    void *ptr;

    ucs_assert(id != UCS_PTR_MAP_KEY_INVALID);
    status = UCS_PTR_MAP_GET(ep, &worker->ep_map, id, 0, &ptr);
    if (ucs_unlikely((status != UCS_OK) && (status != UCS_ERR_NO_PROGRESS))) {
        *ep_p = NULL; /* To supress compiler warning */
        return status;
    }

    *ep_p = (ucp_ep_h)ptr;
    ucs_assertv((*ep_p)->worker == worker, "worker=%p ep=%p ep->worker=%p",
                worker, (*ep_p), (*ep_p)->worker);
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE int
ucp_worker_keepalive_is_enabled(ucp_worker_h worker)
{
    return (worker->context->config.ext.keepalive_num_eps != 0) &&
           (worker->context->config.ext.keepalive_interval != UCS_TIME_INFINITY);
}

/**
 * @return worker-iface struct by resource index
 */
static UCS_F_ALWAYS_INLINE ucp_worker_iface_t*
ucp_worker_iface(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    ucp_tl_bitmap_t tl_bitmap;

    if (rsc_index == UCP_NULL_RESOURCE) {
        return NULL;
    }

    tl_bitmap = worker->context->tl_bitmap;
    ucs_assert(UCS_BITMAP_GET(tl_bitmap, rsc_index));
    return worker->ifaces[UCS_BITMAP_POPCOUNT_UPTO_INDEX(tl_bitmap, rsc_index)];
}

/**
 * @return worker's iface attributes by resource index
 */
static UCS_F_ALWAYS_INLINE uct_iface_attr_t*
ucp_worker_iface_get_attr(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return &ucp_worker_iface(worker, rsc_index)->attr;
}

/**
 * @return worker's iface bandwidth resource index
 */
static UCS_F_ALWAYS_INLINE double
ucp_worker_iface_bandwidth(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    uct_iface_attr_t *iface_attr = ucp_worker_iface_get_attr(worker, rsc_index);

    return ucp_tl_iface_bandwidth(worker->context, &iface_attr->bandwidth);
}

/**
 * @return whether the worker is using unified mode
 */
static UCS_F_ALWAYS_INLINE int
ucp_worker_is_unified_mode(ucp_worker_h worker)
{
    return worker->context->config.ext.unified_mode;
}

/**
 * @return number of connection manager components on the worker
 */
static UCS_F_ALWAYS_INLINE ucp_rsc_index_t
ucp_worker_num_cm_cmpts(const ucp_worker_h worker)
{
    return worker->context->config.num_cm_cmpts;
}

/**
 * Check if interface with @a iface_attr supports point-to-point connections.
 *
 * @param [in]  iface_attr   iface attributes.
 *
 * @return 1 if iface supports point-to-point connections, otherwise 0.
 */
static UCS_F_ALWAYS_INLINE int
ucp_worker_iface_is_tl_p2p(const uct_iface_attr_t *iface_attr)
{
    return !!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP);
}

/**
 * Check if TL supports point-to-point connections.
 *
 * @param [in]  worker       UCP worker.
 * @param [in]  rsc_index    resource index.
 *
 * @return 1 if TL supports point-to-point connections, otherwise 0.
 */
static UCS_F_ALWAYS_INLINE int
ucp_worker_is_tl_p2p(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return ucp_worker_iface_is_tl_p2p(ucp_worker_iface_get_attr(worker,
                                                                rsc_index));
}

/**
 * Check if TL supports connection to interface.
 *
 * @param [in]  worker       UCP worker.
 * @param [in]  rsc_index    resource index.
 *
 * @return 1 if TL supports connection to interface, otherwise 0.
 */
static UCS_F_ALWAYS_INLINE int
ucp_worker_is_tl_2iface(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return !!(ucp_worker_iface_get_attr(worker, rsc_index)->cap.flags &
              UCT_IFACE_FLAG_CONNECT_TO_IFACE);
}

/**
 * Check if TL supports connection to sockaddr.
 *
 * @param [in]  worker       UCP worker.
 * @param [in]  rsc_index    resource index.
 *
 * @return 1 if TL supports connection to sockaddr, otherwise 0.
 */
static UCS_F_ALWAYS_INLINE UCS_F_MAYBE_UNUSED int
ucp_worker_is_tl_2sockaddr(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return !!(ucp_worker_iface_get_attr(worker, rsc_index)->cap.flags &
              UCT_IFACE_FLAG_CONNECT_TO_SOCKADDR);
}

static UCS_F_ALWAYS_INLINE unsigned
ucp_worker_common_address_pack_flags(ucp_worker_h worker)
{
    unsigned pack_flags = 0;

    if (worker->context->num_mem_type_detect_mds > 0) {
        pack_flags |= UCP_ADDRESS_PACK_FLAG_SYS_DEVICE;
    }

    return pack_flags;
}

static UCS_F_ALWAYS_INLINE unsigned
ucp_worker_default_address_pack_flags(ucp_worker_h worker)
{
    return ucp_worker_common_address_pack_flags(worker) |
           UCP_ADDRESS_PACK_FLAG_WORKER_UUID |
           UCP_ADDRESS_PACK_FLAG_WORKER_NAME |
           UCP_ADDRESS_PACK_FLAG_DEVICE_ADDR |
           UCP_ADDRESS_PACK_FLAG_IFACE_ADDR | UCP_ADDRESS_PACK_FLAG_EP_ADDR;
}

#define UCP_WORKER_GET_EP_BY_ID(_ep_p, _worker, _ep_id, _action, _fmt_str, ...) \
    { \
        ucs_status_t __status; \
        \
        __status = ucp_worker_get_ep_by_id(_worker, _ep_id, _ep_p); \
        if (ucs_unlikely(__status != UCS_OK)) { \
            ucs_trace_data("worker %p: ep id 0x%" PRIx64 \
                           " was not found, drop" _fmt_str, \
                           _worker, _ep_id, ##__VA_ARGS__); \
            _action; \
        } \
    }

#define UCP_WORKER_GET_VALID_EP_BY_ID(_ep_p, _worker, _ep_id, _action, \
                                      _fmt_str, ...) \
    { \
        UCP_WORKER_GET_EP_BY_ID(_ep_p, _worker, _ep_id, _action, _fmt_str, \
                                ##__VA_ARGS__); \
        if (ucs_unlikely((*(_ep_p))->flags & UCP_EP_FLAG_CLOSED)) { \
            ucs_trace_data("worker %p: ep id 0x%" PRIx64 " was already closed" \
                           " ep %p, drop " _fmt_str, \
                           _worker, _ep_id, *(_ep_p), ##__VA_ARGS__); \
            _action; \
        } \
    }

#endif
