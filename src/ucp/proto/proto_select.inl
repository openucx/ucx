/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
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


static UCS_F_ALWAYS_INLINE const ucp_proto_threshold_elem_t *
ucp_proto_select_thresholds_search(const ucp_proto_select_elem_t *select_elem,
                                   size_t msg_length)
{
    const ucp_proto_threshold_elem_t *thresholds = select_elem->thresholds;

#define UCP_PROTO_THRESHOLDS_CHECK(_arg, _i) \
    if (ucs_likely(msg_length <= thresholds[_i].max_msg_length)) { \
        return &thresholds[_i]; \
    }

    UCS_PP_FOREACH(UCP_PROTO_THRESHOLDS_CHECK, _, 0, 1, 2, 3)
#undef UCP_PROTO_THRESHOLDS_CHECK
    return ucp_proto_thresholds_search_slow(thresholds + 4, msg_length);
}

static UCS_F_ALWAYS_INLINE uint8_t
ucp_proto_select_op_attr_pack(uint32_t op_attr_mask)
{
    UCS_STATIC_ASSERT(
            (UCP_PROTO_SELECT_OP_ATTR_MASK / UCP_PROTO_SELECT_OP_ATTR_BASE) <
            UCP_PROTO_SELECT_OP_FLAGS_BASE);
    return (op_attr_mask & UCP_PROTO_SELECT_OP_ATTR_MASK) /
           UCP_PROTO_SELECT_OP_ATTR_BASE;
}

static UCS_F_ALWAYS_INLINE uint32_t
ucp_proto_select_op_attr_unpack(uint8_t op_attr)
{
    return op_attr * UCP_PROTO_SELECT_OP_ATTR_BASE;
}

static UCS_F_ALWAYS_INLINE ucp_operation_id_t
ucp_proto_select_op_id(const ucp_proto_select_param_t *select_param)
{
    return (ucp_operation_id_t)(select_param->op_id_flags &
                                (UCP_PROTO_SELECT_OP_FLAGS_BASE - 1));
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_select_check_op(const ucp_proto_select_param_t *select_param,
                          uint64_t op_id_mask)
{
    return !!(UCS_BIT(ucp_proto_select_op_id(select_param)) & op_id_mask);
}

static UCS_F_ALWAYS_INLINE uint8_t
ucp_proto_select_op_flags(const ucp_proto_select_param_t *select_param)
{
    return select_param->op_id_flags & ~(UCP_PROTO_SELECT_OP_FLAGS_BASE - 1);
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
        khiter = kh_get(ucp_proto_select_hash, proto_select->hash, key.u64);
        if (ucs_likely(khiter != kh_end(proto_select->hash))) {
            /* key was found in hash - select by message size */
            select_elem = &kh_value(proto_select->hash, khiter);
        } else {
            select_elem = ucp_proto_select_lookup_slow(worker, proto_select, 0,
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

    return ucp_proto_select_thresholds_search(select_elem, msg_length);
}

/*
 * @note op_attr_mask is from @ref ucp_request_param_t, defined by @ref ucp_op_attr_t.
 */
static UCS_F_ALWAYS_INLINE void ucp_proto_select_param_init_common(
        ucp_proto_select_param_t *select_param, ucp_operation_id_t op_id,
        uint32_t op_attr_mask, uint8_t op_flags, ucp_dt_class_t dt_class,
        const ucp_memory_info_t *mem_info, uint8_t sg_count)
{
    if (dt_class == UCP_DATATYPE_CONTIG) {
        ucs_assert(sg_count == 1);
    } else if (dt_class != UCP_DATATYPE_IOV) {
        ucs_assert(sg_count == 0);
    }

    ucs_assertv(!(op_id & op_flags), "op_id=0x%x op_flags=0x%x", op_id,
                op_flags);
    UCS_STATIC_ASSERT(sizeof(select_param->op.padding) ==
                      sizeof(select_param->op));

    /* construct a protocol lookup key based on all operation parameters
     * op_flags are modifiers for the operation, for now only FAST_CMPL is
     * supported */
    select_param->op_id_flags   = op_id | op_flags;
    select_param->op_attr       = ucp_proto_select_op_attr_pack(op_attr_mask);
    select_param->dt_class      = dt_class;
    select_param->mem_type      = mem_info->type;
    select_param->sys_dev       = mem_info->sys_dev;
    select_param->sg_count      = sg_count;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_select_param_init(ucp_proto_select_param_t *select_param,
                            ucp_operation_id_t op_id, uint32_t op_attr_mask,
                            uint8_t op_flags, ucp_dt_class_t dt_class,
                            const ucp_memory_info_t *mem_info, uint8_t sg_count)
{
    ucp_proto_select_param_init_common(select_param, op_id, op_attr_mask,
                                       op_flags, dt_class, mem_info, sg_count);
    select_param->op.padding[0] = 0;
    select_param->op.padding[1] = 0;
}

static UCS_F_ALWAYS_INLINE void ucp_proto_select_param_init_reply(
        ucp_proto_select_param_t *select_param, ucp_operation_id_t op_id,
        uint32_t op_attr_mask, uint8_t op_flags, ucp_dt_class_t dt_class,
        const ucp_memory_info_t *mem_info, uint8_t sg_count,
        const ucp_memory_info_t *reply_mem_info)
{
    ucp_proto_select_param_init_common(select_param, op_id, op_attr_mask,
                                       op_flags, dt_class, mem_info, sg_count);
    select_param->op.reply.mem_type = reply_mem_info->type;
    select_param->op.reply.sys_dev  = reply_mem_info->sys_dev;
}

static UCS_F_ALWAYS_INLINE int
ucp_proto_select_is_short(ucp_ep_h ep,
                          const ucp_proto_select_short_t *proto_short,
                          ssize_t length)
{
    return ucs_likely(length <= proto_short->max_length_unknown_mem) ||
           ((length <= proto_short->max_length_host_mem) &&
            ucs_memtype_cache_is_empty());
}

#endif
