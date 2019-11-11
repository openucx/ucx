/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROTO_H_
#define UCP_PROTO_H_

#include <ucs/sys/compiler.h>
#include <ucs/type/status.h>
#include <stdint.h>


/**
 * Header segment for a transaction
 */
typedef struct {
    uintptr_t                 ep_ptr;
    uintptr_t                 reqptr;
} UCS_S_PACKED ucp_request_hdr_t;


/**
 * Header for transaction acknowledgment
 */
typedef struct {
    uint64_t                  reqptr;
    ucs_status_t              status;
} UCS_S_PACKED ucp_reply_hdr_t;


#endif
