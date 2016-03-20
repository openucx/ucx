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
 * Calculates a score of specific wireup.
 */
typedef double (*ucp_wireup_score_function_t)(ucp_worker_h worker,
                                              uct_iface_attr_t *iface_attr,
                                              char *reason, size_t max);


/**
 * Packet structure for wireup requests.
 */
typedef struct ucp_wireup_msg {
    uint8_t          type;                /* Message type */
    uint8_t          tli[UCP_EP_OP_LAST]; /* Index of runtime address for every operation */
    /* packed addresses follow */
} UCS_S_PACKED ucp_wireup_msg_t;


typedef struct ucp_wireup_ep_op {
    const char                  *title;
    uint64_t                    features;
    ucp_wireup_score_function_t score_func;
} ucp_wireup_ep_op_t;


extern ucp_wireup_ep_op_t ucp_wireup_ep_ops[];


ucs_status_t ucp_wireup_send_request(ucp_ep_h ep);

ucs_status_t ucp_select_transport(ucp_ep_h ep,
                                  const ucp_address_entry_t *address_list,
                                  unsigned address_count, ucp_rsc_index_t pd_index,
                                  ucp_rsc_index_t *rsc_index_p,
                                  unsigned *dst_addr_index_p,
                                  ucp_wireup_score_function_t score_func,
                                  const char *title);

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self);

ucs_status_t ucp_ep_init_trasports(ucp_ep_h ep, unsigned address_count,
                                   const ucp_address_entry_t *address_list);

#endif
