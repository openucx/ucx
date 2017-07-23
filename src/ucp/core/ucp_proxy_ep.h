/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_PROXY_EP_H_
#define UCP_PROXY_EP_H_

#include "ucp_types.h"

#include <uct/api/uct.h>
#include <ucs/type/class.h>


/**
 * Generic proxy endpoint, used to change behavior of a specific transport lane
 * without adding data-path checks when not needed.
 * By default, all transport endpoint operations are redirected to the underlying
 * UCT endpoint, and interface operations would result in a fatal error.
 * When this endpoint is destroyed, the lane in UCP endpoint is replaced with
 * the real transport endpoint.
 *
 * TODO make sure it works with err handling and print_ucp_info
 */
typedef struct ucp_proxy_ep {
    uct_ep_t                  super;    /**< Derived from uct_ep */
    uct_iface_t               iface;    /**< Embedded stub interface */
    ucp_ep_h                  ucp_ep;   /**< Pointer to UCP endpoint */
    uct_ep_h                  uct_ep;   /**< Underlying transport endpoint */
    int                       is_owner; /**< Is uct_ep owned by this proxy ep */
} ucp_proxy_ep_t;


UCS_CLASS_DECLARE(ucp_proxy_ep_t, const uct_iface_ops_t *ops, ucp_ep_h ucp_ep,
                  uct_ep_h uct_ep, int is_owner);


/**
 * Replace the proxy endpoint by the underlying transport endpoint, and destroy
 * the proxy endpoint.
 */
void ucp_proxy_ep_replace(ucp_proxy_ep_t *proxy_ep);

void ucp_proxy_ep_set_uct_ep(ucp_proxy_ep_t *proxy_ep, uct_ep_h uct_ep,
                             int is_owner);

#endif
