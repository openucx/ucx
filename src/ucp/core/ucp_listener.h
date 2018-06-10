/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCP_LISTENER_H_
#define UCP_LISTENER_H_

#include "ucp_worker.h"


/**
 * UCP listener
 */
typedef struct ucp_listener {
    ucp_worker_iface_t                  wiface;  /* UCT iface to listen on */
    ucp_listener_accept_callback_t      cb;      /* Listen accept callback which
                                                    creates an endpoint */
    ucp_listener_accept_addr_callback_t addr_cb; /* Listen accept callback which
                                                    creates an address to the
                                                    remote endpoint */
    void                                *arg;    /* User's arg for the accept callback */
    uct_worker_cb_id_t                  prog_id; /* Slow-path callback */
} ucp_listener_t;


/**
 * Accepted connection on a listener
 */
typedef struct ucp_listener_accept {
    ucp_listener_h                  listener; /* Listener on which the connection
                                                 was accepted */
    int                             is_ep;    /* flag to indicate which field is
                                                 valid, if not 0 then 
                                                 @ref ucp_listener_accept_t::ep,
                                                 is valid, otherwise @ref
                                                 ucp_listener_accept_t::addr. */
    union {
        ucp_ep_h                    ep;       /* New endpoint which was created
                                                 for the connection */
        ucp_ep_address_t            *addr;    /* EP address which was created
                                                 for the connection */
    };
} ucp_listener_accept_t;

void ucp_listener_schedule_accept_cb(ucp_ep_h ep);

int ucp_listener_accept_cb_remove_filter(const ucs_callbackq_elem_t *elem,
                                         void *arg);

#endif
