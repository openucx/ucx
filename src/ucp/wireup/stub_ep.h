/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WIREUP_STUB_EP_H_
#define UCP_WIREUP_STUB_EP_H_

#include "address.h"

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#include <ucp/core/ucp_ep.h>
#include <ucs/datastruct/queue_types.h>


/**
 * Stub endpoint, to hold off send requests until wireup process completes.
 * It is placed instead UCT endpoint before it's fully connected, and for AM
 * endpoint it also contains an auxiliary endpoint which can send wireup messages.
 */
struct ucp_stub_ep {
    uct_ep_t            super;         /* Derive from uct_ep */
    ucp_ep_h            ep;            /* Pointer to the ucp_ep we're wiring */
    ucs_queue_head_t    pending_q;     /* Queue of pending operations */
    uct_ep_h            aux_ep;        /* Used to wireup the "real" endpoint */
    uct_ep_h            next_ep;       /* Next transport being wired up */
    ucs_list_link_t     list;

    ucp_ep_op_t         optype;        /* Which operation type inside the ucp_ep */
    ucp_rsc_index_t     aux_rsc_index; /* Index of auxiliary transport */
    volatile uint32_t   pending_count; /* Number of pending wireup operations */
    volatile int        connected;     /* next_ep is fully connected */
};


ucs_status_t ucp_stub_ep_create(ucp_ep_h ep, ucp_ep_op_t optype,
                                unsigned address_count,
                                const ucp_address_entry_t *address_list,
                                uct_ep_h *ep_p);

/**
 * @return Auxiliary resource index used by the stub endpoint.
 *   If the endpoint is not a stub endpoint, return UCP_NULL_RESOURCE.
 */
ucp_rsc_index_t ucp_stub_ep_get_aux_rsc_index(uct_ep_h uct_ep);

/* create endpoint for the real transport, which we would eventually connect */
ucs_status_t ucp_stub_ep_connect(uct_ep_h uct_ep, unsigned address_count,
                                 const ucp_address_entry_t *address_list);

void ucp_stub_ep_set_next_ep(uct_ep_h uct_ep, uct_ep_h next_ep);

void ucp_stub_ep_remote_connected(uct_ep_h uct_ep);

void ucp_stub_ep_progress(ucp_stub_ep_t *stub_ep);

#endif
