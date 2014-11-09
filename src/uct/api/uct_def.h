/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_DEF_H_
#define UCT_DEF_H_

#include <stdint.h>

#define UCT_MAX_NAME_LEN         64

typedef struct uct_context       *uct_context_h;
typedef struct uct_iface         *uct_iface_h;
typedef struct uct_iface_addr    uct_iface_addr_t;
typedef struct uct_ep            *uct_ep_h;
typedef struct uct_ep_addr       uct_ep_addr_t;
typedef uintptr_t                uct_lkey_t;
typedef uintptr_t                uct_rkey_t;
typedef struct uct_req           *uct_req_h;
typedef struct uct_memory_region *uct_memory_region_h;
typedef struct uct_pd            *uct_pd_h;
typedef void                     *uct_rkey_ctx_h;

#endif
