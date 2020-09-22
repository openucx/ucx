/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_SELECT_INL_
#define UCP_PROTO_SELECT_INL_

#include "proto_select.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/dt/datatype_iter.h>
#include <ucp/proto/proto_select.h>
#include <ucs/debug/log.h>


typedef union {
    ucp_proto_select_param_t param;
    uint64_t                 u64;
} ucp_proto_select_key_t;


KHASH_IMPL(ucp_proto_select_hash, khint64_t, ucp_proto_select_elem_t, 1,
           kh_int64_hash_func, kh_int64_hash_equal)


static UCS_F_ALWAYS_INLINE const ucp_proto_threshold_elem_t*
ucp_proto_thresholds_search(const ucp_proto_threshold_elem_t *thresholds,
                            size_t msg_length)
{
#define UCP_PROTO_THRESHOLDS_CHECK(_arg, _i) \
    if (ucs_likely(msg_length <= thresholds[_i].max_msg_length)) { \
        return &thresholds[_i]; \
    }

    UCS_PP_FOREACH(UCP_PROTO_THRESHOLDS_CHECK, _, 0, 1, 2)
#undef UCP_PROTO_THRESHOLDS_CHECK
    return ucp_proto_thresholds_search_slow(thresholds + 3, msg_length);
}

static UCS_F_ALWAYS_INLINE uint8_t
ucp_proto_select_op_attr_to_flags(uint32_t op_attr_mask)
{
    UCS_STATIC_ASSERT(UCP_PROTO_SELECT_OP_ATTR_MASK /
                      UCP_PROTO_SELECT_OP_ATTR_BASE <= UINT8_MAX);
    return op_attr_mask / UCP_PROTO_SELECT_OP_ATTR_BASE;
}

static UCS_F_ALWAYS_INLINE uint32_t
ucp_proto_select_op_attr_from_flags(uint8_t flags)
{
    return (flags * UCP_PROTO_SELECT_OP_ATTR_BASE) &
           UCP_PROTO_SELECT_OP_ATTR_MASK;
}

static UCS_F_ALWAYS_INLINE const ucp_proto_threshold_elem_t*
ucp_proto_select_lookup(ucp_worker_h worker, ucp_proto_select_t *proto_select,
                        ucp_worker_cfg_index_t ep_cfg_index,
                        ucp_worker_cfg_index_t rkey_cfg_index,
                        const ucp_proto_select_param_t *select_param,
                        size_t msg_length)
{
    const ucp_proto_select_elem_t *select_elem;
    ucp_proto_select_key_t key;
    khiter_t khiter;

    UCS_STATIC_ASSERT(sizeof(key.param) == sizeof(key.u64));
    key.param = *select_param;

    if (ucs_likely(proto_select->cache.key == key.u64)) {
        select_elem = proto_select->cache.value;
    } else {
        khiter = kh_get(ucp_proto_select_hash, &proto_select->hash, key.u64);
        if (ucs_likely(khiter != kh_end(&proto_select->hash))) {
            /* key was found in hash - select by message size */
            select_elem = &kh_value(&proto_select->hash, khiter);
        } else {
            select_elem = ucp_proto_select_lookup_slow(worker, proto_select,
                                                       ep_cfg_index,
                                                       rkey_cfg_index,
                                                       &key.param);
            if (ucs_unlikely(select_elem == NULL)) {
                return NULL;
            }
        }

        proto_select->cache.key   = key.u64;
        proto_select->cache.value = select_elem;
    }

    return ucp_proto_thresholds_search(select_elem->thresholds, msg_length);
}

#endif
