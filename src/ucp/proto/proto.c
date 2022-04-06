/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto.h"

#include <ucs/sys/string.h>

#define UCP_PROTO_AMO_FOR_EACH(_macro, _id) \
    _macro(ucp_amo_proto_##_id##32) \
    _macro(ucp_amo_proto_##_id##64) \
    _macro(ucp_amo_proto_##_id##32_mtype) \
    _macro(ucp_amo_proto_##_id##64_mtype)

#define UCP_PROTO_FOR_EACH(_macro) \
    _macro(ucp_reconfig_proto) \
    _macro(ucp_get_amo_post_proto) \
    _macro(ucp_get_amo_fetch_proto) \
    _macro(ucp_get_am_bcopy_proto) \
    _macro(ucp_get_offload_bcopy_proto) \
    _macro(ucp_get_offload_zcopy_proto) \
    _macro(ucp_put_am_bcopy_proto) \
    _macro(ucp_put_offload_short_proto) \
    _macro(ucp_put_offload_bcopy_proto) \
    _macro(ucp_put_offload_zcopy_proto) \
    _macro(ucp_eager_bcopy_multi_proto) \
    _macro(ucp_eager_sync_bcopy_multi_proto) \
    _macro(ucp_eager_zcopy_multi_proto) \
    _macro(ucp_eager_short_proto) \
    _macro(ucp_eager_bcopy_single_proto) \
    _macro(ucp_eager_zcopy_single_proto) \
    _macro(ucp_tag_rndv_proto) \
    _macro(ucp_eager_tag_offload_short_proto) \
    _macro(ucp_eager_sync_bcopy_single_proto) \
    _macro(ucp_tag_offload_eager_zcopy_single_proto) \
    _macro(ucp_eager_sync_zcopy_single_proto) \
    _macro(ucp_rndv_am_bcopy_proto) \
    _macro(ucp_rndv_get_zcopy_proto) \
    _macro(ucp_rndv_get_mtype_proto) \
    _macro(ucp_rndv_ats_proto) \
    _macro(ucp_rndv_rtr_proto) \
    _macro(ucp_rndv_rtr_mtype_proto) \
    _macro(ucp_rndv_send_ppln_proto) \
    _macro(ucp_rndv_recv_ppln_proto) \
    _macro(ucp_rndv_put_zcopy_proto) \
    _macro(ucp_rndv_put_mtype_proto) \
    _macro(ucp_rndv_rkey_ptr_proto) \
    _macro(ucp_tag_offload_eager_bcopy_single_proto) \
    _macro(ucp_am_eager_short_proto) \
    _macro(ucp_am_eager_single_bcopy_proto) \
    _macro(ucp_am_eager_single_zcopy_proto) \
    _macro(ucp_am_eager_multi_bcopy_proto) \
    _macro(ucp_am_eager_multi_zcopy_proto) \
    _macro(ucp_am_eager_short_reply_proto) \
    _macro(ucp_am_eager_single_bcopy_reply_proto) \
    _macro(ucp_am_eager_single_zcopy_reply_proto) \
    _macro(ucp_am_rndv_proto) \
    UCP_PROTO_AMO_FOR_EACH(_macro, post) \
    UCP_PROTO_AMO_FOR_EACH(_macro, fetch) \
    UCP_PROTO_AMO_FOR_EACH(_macro, cswap)

#define UCP_PROTO_DECL(_proto) extern ucp_proto_t _proto;

#define UCP_PROTO_ENTRY(_proto) &_proto,

/* Declare all proto objects */
UCP_PROTO_FOR_EACH(UCP_PROTO_DECL)

const ucp_proto_t *ucp_protocols[] = {
    UCP_PROTO_FOR_EACH(UCP_PROTO_ENTRY)
};

const char *ucp_proto_perf_types[] = {
    [UCP_PROTO_PERF_TYPE_SINGLE] = "single",
    [UCP_PROTO_PERF_TYPE_MULTI]  = "multi"
};

const char *ucp_operation_names[] = {
    [UCP_OP_ID_TAG_SEND]       = "tag_send",
    [UCP_OP_ID_TAG_SEND_SYNC]  = "tag_send_sync",
    [UCP_OP_ID_AM_SEND]        = "am_send",
    [UCP_OP_ID_AM_SEND_REPLY]  = "am_send_reply",
    [UCP_OP_ID_PUT]            = "put",
    [UCP_OP_ID_GET]            = "get",
    [UCP_OP_ID_AMO_POST]       = "amo_post",
    [UCP_OP_ID_AMO_FETCH]      = "amo_fetch",
    [UCP_OP_ID_AMO_CSWAP]      = "amo_cswap",
    [UCP_OP_ID_RNDV_SEND]      = "rndv_send",
    [UCP_OP_ID_RNDV_RECV]      = "rndv_recv",
    [UCP_OP_ID_RNDV_RECV_DROP] = "rndv_recv_drop",
    [UCP_OP_ID_LAST]           = NULL
};

const char *ucp_operation_descs[] = {
    [UCP_OP_ID_TAG_SEND]       = "tagged message by ucp_tag_send*",
    [UCP_OP_ID_TAG_SEND_SYNC]  = "synchronous tagged message by ucp_tag_send_sync*",
    [UCP_OP_ID_AM_SEND]        = "active message by ucp_am_send*",
    [UCP_OP_ID_AM_SEND_REPLY]  = "active message by ucp_am_send* with reply "
                                 "flag",
    [UCP_OP_ID_PUT]            = "remote memory write by ucp_put*",
    [UCP_OP_ID_GET]            = "remote memory read by ucp_get*",
    [UCP_OP_ID_AMO_POST]       = "posted atomic by ucp_atomic_op*",
    [UCP_OP_ID_AMO_FETCH]      = "fetching atomic by ucp_atomic_op*",
    [UCP_OP_ID_AMO_CSWAP]      = "atomic compare-and-swap by ucp_atomic_op*",
    [UCP_OP_ID_RNDV_SEND]      = "rendezvous data send",
    [UCP_OP_ID_RNDV_RECV]      = "rendezvous data fetch",
    [UCP_OP_ID_RNDV_RECV_DROP] = "rendezvous data drop",
    [UCP_OP_ID_LAST]           = NULL
};

unsigned ucp_protocols_count(void)
{
    UCS_STATIC_ASSERT(ucs_static_array_size(ucp_protocols) <
                      UCP_PROTO_MAX_COUNT);
    return ucs_static_array_size(ucp_protocols);
}

void ucp_proto_default_query(const ucp_proto_query_params_t *params,
                             ucp_proto_query_attr_t *attr)
{
    ucs_assert(params->proto->desc != NULL);

    attr->max_msg_length = SIZE_MAX;
    attr->is_estimation  = 0;
    ucs_strncpy_safe(attr->desc, params->proto->desc, sizeof(attr->desc));
    ucs_strncpy_safe(attr->config, "", sizeof(attr->config));
}
