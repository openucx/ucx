/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_WIREUP_H_
#define UCP_WIREUP_H_

#include "address.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <uct/api/uct.h>


/**
 * Wireup message types
 */
enum {
    UCP_WIREUP_MSG_REQUEST,
    UCP_WIREUP_MSG_REPLY,
    UCP_WIREUP_MSG_ACK,
    UCP_WIREUP_MSG_LAST
};


/**
 * Packet structure for wireup requests.
 */
typedef struct ucp_wireup_msg {
    uint8_t          type;                /* Message type */
    uint8_t          tli[UCP_MAX_LANES];  /* Index of runtime address for every operation.
                                             We need this in order to connect the
                                             transports correctly when getting a reply
                                             message. */
    /* packed addresses follow */
} UCS_S_PACKED ucp_wireup_msg_t;


/**
 * Calculates a score of specific wireup.
 */
typedef double (*ucp_wireup_score_function_t)(ucp_worker_h worker,
                                              const uct_iface_attr_t *iface_attr,
                                              char *reason, size_t max);


ucs_status_t ucp_wireup_send_request(ucp_ep_h ep);

ucs_status_t ucp_wireup_select_aux_transport(ucp_ep_h ep,
                                             const ucp_address_entry_t *address_list,
                                             unsigned address_count,
                                             ucp_rsc_index_t *rsc_index_p,
                                             unsigned *addr_index_p);

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self);

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, unsigned address_count,
                                   const ucp_address_entry_t *address_list);

ucs_status_t ucp_wireup_select_transports(ucp_ep_h ep, unsigned address_count,
                                          const ucp_address_entry_t *address_list,
                                          unsigned *addr_indices);


static inline int ucp_worker_is_tl_p2p(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return !(worker->iface_attrs[rsc_index].cap.flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE);
}


#endif
