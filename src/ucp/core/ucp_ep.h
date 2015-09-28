/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_context.h"
#include "ucp_request.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>


#define UCP_PEER_NAME_MAX         16


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */
    uct_ep_h                      uct_ep;        /* Current transport for operations */

    struct {
        size_t                    max_short_egr; /* TODO should be unsigned */
        size_t                    max_short_put; /* TODO should be unsigned */
        size_t                    max_bcopy_egr; /* TODO should be unsigned */
        size_t                    max_bcopy_put;
        size_t                    max_bcopy_get;
    } config;

    uint64_t                      dest_uuid;     /* Destination worker uuid */
    ucp_ep_h                      next;          /* Next in hash table linked list */

    ucp_rsc_index_t               rsc_index;     /* Resource index the endpoint uses */
    ucp_rsc_index_t               dst_pd_index;  /* Destination protection domain index */
    volatile uint32_t             state;         /* Endpoint state */

    struct {
        uct_ep_h                  aux_ep;        /* Used to wireup the "real" endpoint */
        uct_ep_h                  next_ep;       /* Next transport being wired up */
    } wireup;

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_PEER_NAME_MAX];
#endif

} ucp_ep_t;


ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *peer_name, const char *message, ucp_ep_h *ep_p);

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep);


void ucp_ep_add_pending(ucp_ep_h ep, uct_ep_h uct_ep, ucp_request_t *req);


static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "??";
#endif
}


#endif
