/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_H_
#define UCP_H_

#include "ucp_def.h"
#include <uct/api/uct.h>
#include <ucs/type/status.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>


typedef struct ucp_context {
    uct_context_h       uct_context;
    uct_resource_desc_t *resources;    //should be list/array of resources?
} ucp_context_t;

/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_iface_h       ucp_iface;    // local interface?
    uct_ep_h          uct_ep;  // remote eps - one per transport
} ucp_ep_t;

/**
 * Local protocol layer interface
 */
typedef struct ucp_iface {
    ucp_context_h     context;
    uct_iface_h       uct_iface;
} ucp_iface_t;


ucs_status_t ucp_init(ucp_context_h *context_p);

void ucp_cleanup(ucp_context_h context_p);

ucs_status_t ucp_iface_create(ucp_context_h ucp_context, ucp_iface_h *ucp_iface);

void ucp_iface_close(ucp_iface_h ucp_iface);

ucs_status_t ucp_ep_create(ucp_iface_h ucp_iface, ucp_ep_h *ucp_ep);

void ucp_ep_destroy(ucp_ep_h ucp_ep);


#endif /* UCP_H_ */
