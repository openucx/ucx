/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_SELECT_INL_
#define UCP_PROTO_SELECT_INL_

#include "proto_select.h"


typedef union {
    ucp_proto_select_param_t param;
    uint64_t                 u64;
} ucp_proto_select_key_t;


KHASH_IMPL(ucp_proto_select_hash, khint64_t, ucp_proto_select_elem_t, 1,
           kh_int64_hash_func, kh_int64_hash_equal)


static UCS_F_ALWAYS_INLINE uint32_t
ucp_proto_select_op_attr_from_flags(uint8_t flags)
{
    return (flags * UCP_PROTO_SELECT_OP_ATTR_BASE) &
           UCP_PROTO_SELECT_OP_ATTR_MASK;
}

#endif
