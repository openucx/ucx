/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_TL_H_
#define UCT_TL_H_

#if !defined(UCT_H_)
#  error "You should not include this header directly. Include uct.h instead."
#endif

#include "uct_def.h"

#include <ucs/type/status.h>
#include <sys/socket.h>
#include <stddef.h>
#include <sched.h>


/**
 * Transport iface operations.
 */
typedef struct uct_iface_ops {

    ucs_status_t (*iface_query)(uct_iface_h iface,
                                uct_iface_attr_t *iface_attr);
    ucs_status_t (*iface_get_address)(uct_iface_h iface,
                                      uct_iface_addr_t *iface_addr);

    ucs_status_t (*iface_flush)(uct_iface_h iface);

    void         (*iface_close)(uct_iface_h iface);

    ucs_status_t (*ep_create)(uct_iface_h iface, uct_ep_h *ep_p);
    void         (*ep_destroy)(uct_ep_h ep);

    ucs_status_t (*ep_get_address)(uct_ep_h ep, uct_ep_addr_t *ep_addr);
    ucs_status_t (*ep_connect_to_iface)(uct_ep_h ep, uct_iface_addr_t *iface_addr);
    ucs_status_t (*ep_connect_to_ep)(uct_ep_h ep, uct_iface_addr_t *iface_addr,
                                     uct_ep_addr_t *ep_addr);

    ucs_status_t (*ep_put_short)(uct_ep_h ep, void *buffer, unsigned length,
                                 uint64_t remote_addr, uct_rkey_t rkey);

    ucs_status_t (*ep_am_short)(uct_ep_h ep, uint8_t id, uint64_t header,
                                void *payload, unsigned length);

    ucs_status_t (*ep_flush)(uct_ep_h ep);

} uct_iface_ops_t;


/**
 * Protection domain
 */
typedef struct uct_pd {
    uct_pd_ops_t             *ops;
    uct_context_h            context;
} uct_pd_t;


/**
 * Communication interface context
 */
typedef struct uct_iface {
    uct_iface_ops_t          ops;
    uct_pd_h                 pd;
} uct_iface_t;


/**
 * Remote endpoint
 */
typedef struct uct_ep {
    uct_iface_h              iface;
} uct_ep_t;


#endif
