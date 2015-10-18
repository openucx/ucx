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
 * Dummy endpoint, to hold off send requests until wireup process completes.
 */
typedef struct ucp_dummy_ep {
    uct_ep_t          super;
    ucp_ep_h          ep;
    uct_iface_t       iface;
    ucs_queue_head_t  pending_q;
    volatile uint32_t refcount;
} ucp_dummy_ep_t;
UCS_CLASS_DECLARE(ucp_dummy_ep_t, ucp_ep_h);


void ucp_dummy_ep_progress(void *arg);


#endif
