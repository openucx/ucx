/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WIREUP_STUB_EP_H_
#define UCP_WIREUP_STUB_EP_H_

#include <uct/api/uct.h>
#include <ucp/api/ucp.h>
#include <ucs/type/class.h>


/**
 * Stub endpoint, to hold off send requests until wireup process completes.
 */
typedef struct ucp_stub_ep {
    uct_ep_t          super;         /* Derive from uct_ep */
    uct_iface_t       iface;         /* Stub uct_iface structure */
    ucp_ep_h          ep;            /* Pointer to the ucp_ep we're wiring */
    ucs_queue_head_t  pending_q;     /* Queue of pending operations */
    volatile uint32_t pending_count; /* Number of pending wireup operations */
    uct_ep_h          aux_ep;        /* Used to wireup the "real" endpoint */
    uct_ep_h          next_ep;       /* Next transport being wired up */
} ucp_stub_ep_t;
UCS_CLASS_DECLARE(ucp_stub_ep_t, ucp_ep_h);

static inline ucs_queue_elem_t* ucp_stub_ep_req_priv(uct_pending_req_t *req)
{
    UCS_STATIC_ASSERT(sizeof(ucs_queue_elem_t) <= UCT_PENDING_REQ_PRIV_LEN);
    return (ucs_queue_elem_t*)req->priv;
}

void ucp_stub_ep_progress(void *arg);


#endif
