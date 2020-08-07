/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_WORKER_INL_
#define UCP_WORKER_INL_

#include "ucp_worker.h"


static UCS_F_ALWAYS_INLINE khint_t
ucp_worker_rkey_config_hash_func(ucp_rkey_config_key_t rkey_config_key)
{
    return (khint_t)rkey_config_key.md_map  ^
           rkey_config_key.ep_cfg_index     ^
           (rkey_config_key.mem_type << 16) ^
           (rkey_config_key.sys_dev  << 24);
}

static UCS_F_ALWAYS_INLINE int
ucp_worker_rkey_config_is_equal(ucp_rkey_config_key_t rkey_config_key1,
                                ucp_rkey_config_key_t rkey_config_key2)
{
    return (rkey_config_key1.md_map       == rkey_config_key2.md_map) &&
           (rkey_config_key1.ep_cfg_index == rkey_config_key2.ep_cfg_index) &&
           (rkey_config_key1.sys_dev      == rkey_config_key2.sys_dev) &&
           (rkey_config_key1.mem_type     == rkey_config_key2.mem_type);
}

KHASH_IMPL(ucp_worker_rkey_config, ucp_rkey_config_key_t, ucp_worker_cfg_index_t,
           1, ucp_worker_rkey_config_hash_func, ucp_worker_rkey_config_is_equal);

/**
 * @return Worker name
 */
static UCS_F_ALWAYS_INLINE const char*
ucp_worker_get_name(ucp_worker_h worker)
{
    return worker->name;
}

/**
 * @return endpoint by a pointer received from remote side
 */
static UCS_F_ALWAYS_INLINE ucp_ep_h
ucp_worker_get_ep_by_id(ucp_worker_h worker, ucs_ptr_map_key_t id)
{
    ucp_ep_h ep;

    ucs_assert(id != UCP_EP_ID_INVALID);
    ep = (ucp_ep_h)ucs_ptr_map_get(&worker->ptr_map, id);
    ucs_assertv(ep->worker == worker, "worker=%p ep=%p ep->worker=%p", worker,
                ep, ep->worker);
    return ep;
}

/**
 * @return worker-iface struct by resource index
 */
static UCS_F_ALWAYS_INLINE ucp_worker_iface_t*
ucp_worker_iface(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    uint64_t tl_bitmap;

    if (rsc_index == UCP_NULL_RESOURCE) {
        return NULL;
    }

    tl_bitmap = worker->context->tl_bitmap;
    ucs_assert(UCS_BIT(rsc_index) & tl_bitmap);
    return worker->ifaces[ucs_bitmap2idx(tl_bitmap, rsc_index)];
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
 * @return whether the worker should be using connection manager mode
 */
static UCS_F_ALWAYS_INLINE int
ucp_worker_sockaddr_is_cm_proto(const ucp_worker_h worker)
{
    return !!ucp_worker_num_cm_cmpts(worker);
}

/**
 * Check if interface with @a iface_attr supports point to pont connections.
 *
 * @param [in]  iface_attr   iface attributes.
 *
 * @return 1 if iface supports point to pont connections, otherwise 0.
 */
static UCS_F_ALWAYS_INLINE int
ucp_worker_iface_is_tl_p2p(const uct_iface_attr_t *iface_attr)
{
    return !!(iface_attr->cap.flags & UCT_IFACE_FLAG_CONNECT_TO_EP);
}

/**
 * Check if TL supports point to pont connections.
 *
 * @param [in]  worker       UCP worker.
 * @param [in]  rsc_index    resource index.
 *
 * @return 1 if TL supports point to pont connections, otherwise 0.
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

/**
 * Resolve remote key configuration key to a remote key configuration index
 *
 * @param [in]  worker       UCP worker to resolve configuration on
 * @param [in]  key          Rkey configuration key
 * @param [out] cfg_index_p  Filled with configuration index in the worker.
 */
static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_worker_get_rkey_config(ucp_worker_h worker, const ucp_rkey_config_key_t *key,
                           ucp_worker_cfg_index_t *cfg_index_p)
{
    khiter_t khiter = kh_get(ucp_worker_rkey_config, &worker->rkey_config_hash,
                             *key);
    if (ucs_likely(khiter != kh_end(&worker->rkey_config_hash))) {
        *cfg_index_p = kh_val(&worker->rkey_config_hash, khiter);
        return UCS_OK;
    }

    return ucp_worker_add_rkey_config(worker, key, cfg_index_p);
}


#endif
