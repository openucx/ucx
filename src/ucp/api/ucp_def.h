/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCP_DEF_H_
#define UCP_DEF_H_

#include <stddef.h>
#include <stdint.h>

typedef struct ucp_config        ucp_config_t;
typedef struct ucp_context       *ucp_context_h;
typedef struct ucp_ep            *ucp_ep_h;
typedef struct ucp_address       ucp_address_t;
typedef struct ucp_recv_request  *ucp_recv_request_h;
typedef struct ucp_rkey          *ucp_rkey_h;
typedef struct ucp_mem           *ucp_mem_h;
typedef struct ucp_worker        *ucp_worker_h;
typedef uint64_t                 ucp_tag_t;

#endif
