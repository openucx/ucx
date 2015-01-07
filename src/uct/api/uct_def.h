/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_DEF_H_
#define UCT_DEF_H_

#include <ucs/sys/math.h>
#include <stdint.h>

#define UCT_MAX_NAME_LEN         64
#define UCT_AM_ID_BITS           5
#define UCT_AM_ID_MAX            UCS_BIT(UCT_AM_ID_BITS)


typedef struct uct_context       *uct_context_h;
typedef struct uct_iface         *uct_iface_h;
typedef struct uct_iface_addr    uct_iface_addr_t;
typedef struct uct_iface_config  uct_iface_config_t;
typedef struct uct_ep            *uct_ep_h;
typedef struct uct_ep_addr       uct_ep_addr_t;
typedef uintptr_t                uct_lkey_t;
typedef uintptr_t                uct_rkey_t;
typedef struct uct_pd            *uct_pd_h;
typedef struct uct_tl_ops        uct_tl_ops_t;
typedef struct uct_pd_ops        uct_pd_ops_t;
typedef void                     *uct_rkey_ctx_h;
typedef struct uct_iface_attr    uct_iface_attr_t;
typedef struct uct_pd_attr       uct_pd_attr_t;


#endif
