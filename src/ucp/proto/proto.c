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


const ucp_proto_t *ucp_protocols[UCP_PROTO_MAX_COUNT] = {};
unsigned ucp_protocols_count                          = 0;

const char *ucp_proto_perf_types[] = {
    [UCP_PROTO_PERF_TYPE_SINGLE] = "single",
    [UCP_PROTO_PERF_TYPE_MULTI]  = "multi"
};

void ucp_proto_default_query(const ucp_proto_query_params_t *params,
                             ucp_proto_query_attr_t *attr)
{
    ucs_strncpy_safe(attr->config, "", sizeof(attr->config));
}
