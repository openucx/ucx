/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto.h"


const ucp_proto_t *ucp_protocols[UCP_PROTO_MAX_COUNT] = {};
unsigned ucp_protocols_count                          = 0;

