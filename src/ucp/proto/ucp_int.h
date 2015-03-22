/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_INT_H
#define UCP_INT_H

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue.h>


/**
 * UCP context
 */
typedef struct ucp_context {
    uct_context_h       uct;           /* UCT context */
    uct_resource_desc_t *resources;    /* Array of resources */
    unsigned            num_resources; /* Number of resources in the array*/

    struct {
        ucs_mpool_h      rreq_mp;       /* Receive requests */
        ucs_queue_head_t expected;
        ucs_queue_head_t unexpected;
    } tag;

} ucp_context_t;


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    uct_ep_h            uct;           /* UCT endpoint. TODO use multiple transports */
    ucp_worker_h        worker;        /* Worker this endpoint belongs to */

    struct {
        size_t          max_short_tag;
    } config;

} ucp_ep_t;


struct ucp_ep_addr {
};


typedef struct ucp_rkey {
} ucp_rkey_t;


typedef struct ucp_worker {
    ucp_context_h       context;
    uct_worker_h        uct;           /* UCT worker */
    unsigned            num_ifaces;    /* Number of interfaces in the array */
    ucs_queue_head_t    completed;     /* Queue of completed requests */
    uct_iface_h         ifaces[0];     /* Array of interfaces */
} ucp_worker_t;


typedef struct ucp_recv_desc {
    ucs_queue_elem_t    queue;
    size_t              length;
} ucp_recv_desc_t;


ucs_status_t ucp_tag_init(ucp_context_h context);

void ucp_tag_cleanup(ucp_context_h context);

ucs_status_t ucp_tag_set_am_handlers(ucp_context_h context, uct_iface_h iface);

#endif
