/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RKEY_INL_
#define UCP_RKEY_INL_

#include "ucp_rkey.h"
#include "ucp_worker.h"
#include "ucp_ep.h"


static UCS_F_ALWAYS_INLINE khint_t
ucp_rkey_config_hash_func(ucp_rkey_config_key_t rkey_config_key)
{
    return (khint_t)rkey_config_key.md_map ^
           (rkey_config_key.ep_cfg_index << 8) ^
           (rkey_config_key.sys_dev << 16) ^
           (rkey_config_key.mem_type << 24);
}

static UCS_F_ALWAYS_INLINE int
ucp_rkey_config_is_equal(ucp_rkey_config_key_t rkey_config_key1,
                         ucp_rkey_config_key_t rkey_config_key2)
{
    return (rkey_config_key1.md_map == rkey_config_key2.md_map) &&
           (rkey_config_key1.ep_cfg_index == rkey_config_key2.ep_cfg_index) &&
           (rkey_config_key1.sys_dev == rkey_config_key2.sys_dev) &&
           (rkey_config_key1.mem_type == rkey_config_key2.mem_type);
}

static UCS_F_ALWAYS_INLINE ucp_rkey_config_t *
ucp_rkey_config(ucp_worker_h worker, ucp_rkey_h rkey)
{
    ucs_assert(rkey->cfg_index != UCP_WORKER_CFG_INDEX_NULL);
    return &worker->rkey_config[rkey->cfg_index];
}

static UCS_F_ALWAYS_INLINE uct_rkey_t
ucp_rkey_get_tl_rkey(ucp_rkey_h rkey, ucp_md_index_t rkey_index)
{
    if (rkey_index == UCP_NULL_RESOURCE) {
        return UCT_INVALID_RKEY;
    }

    ucs_assert(rkey_index < ucs_popcount(rkey->md_map));
    return rkey->tl_rkey[rkey_index].rkey.rkey;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_ep_rkey_unpack_reachable(ucp_ep_h ep, const void *buffer, size_t length,
                             ucp_rkey_h *rkey_p)
{
    ucp_ep_config_t *config = &ep->worker->ep_config[ep->cfg_index];
    return ucp_ep_rkey_unpack_internal(ep, buffer, length,
                                       config->key.reachable_md_map, rkey_p);
}

#endif
