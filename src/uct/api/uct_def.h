/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_DEF_H_
#define UCT_DEF_H_

#include <stdint.h>


typedef struct uct_context       *uct_context_h;
typedef struct uct_iface         *uct_iface_h;
typedef struct uct_iface_addr    uct_iface_addr_t;
typedef struct uct_ep            *uct_ep_h;
typedef struct uct_ep_addr       uct_ep_addr_t;
typedef struct uct_ops           uct_ops_t;
typedef uint64_t                 uct_lkey_t;
typedef uint64_t                 uct_rkey_t;
typedef struct uct_req           *uct_req_h;


#endif
