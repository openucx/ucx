/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WIREUP_H_
#define UCP_WIREUP_H_

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_context.h>
#include <uct/api/uct.h>



/**
 * Endpoint wire-up state
 */
enum {
    UCP_EP_STATE_READY_TO_SEND            = UCS_BIT(0), /* uct_ep is ready to go */
    UCP_EP_STATE_AUX_EP                   = UCS_BIT(1), /* aux_ep was created */
    UCP_EP_STATE_NEXT_EP                  = UCS_BIT(2), /* next_ep was created */
    UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED  = UCS_BIT(3), /* next_ep connected to remote */
    UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED = UCS_BIT(4), /* remote also connected to our next_ep */
    UCP_EP_STATE_WIREUP_REPLY_SENT        = UCS_BIT(5), /* wireup reply message has been sent */
    UCP_EP_STATE_WIREUP_ACK_SENT          = UCS_BIT(6), /* wireup ack message has been sent */
};


/**
 * Flags in the wireup message
 */
enum {
    UCP_WIREUP_FLAG_REQUSET               = UCS_BIT(0),
    UCP_WIREUP_FLAG_REPLY                 = UCS_BIT(1),
    UCP_WIREUP_FLAG_ACK                   = UCS_BIT(2),
    UCP_WIREUP_FLAG_ADDR                  = UCS_BIT(3),
    UCP_WIREUP_FLAG_AUX_ADDR              = UCS_BIT(4)
};


/**
 * Calculates a score of specific wireup.
 */
typedef double (*ucp_wireup_score_function_t)(ucp_worker_h worker,
                                              uct_tl_resource_desc_t *resource,
                                              uct_iface_h iface,
                                              uct_iface_attr_t *iface_attr);


/**
 * Packet structure for wireup requests.
 */
typedef struct ucp_wireup_msg {
    uint64_t                      src_uuid;         /* Sender uuid */
    ucp_rsc_index_t               src_pd_index;     /* Sender PD index */
    ucp_rsc_index_t               src_rsc_index;    /* Index of sender resource */
    ucp_rsc_index_t               dst_rsc_index;    /* Index of receiver resource */
    ucp_rsc_index_t               dst_aux_index;    /* Index of receiver wireup resource */
    uint16_t                      flags;            /* Wireup flags */
    uint8_t                       addr_len;         /* Length of first address */
    /* addresses follow */
} UCS_S_PACKED ucp_wireup_msg_t;


ucs_status_t ucp_wireup_start(ucp_ep_h ep, ucp_address_t *address);

void ucp_wireup_stop(ucp_ep_h ep);

void ucp_wireup_progress(ucp_ep_h ep);


static inline uint64_t ucp_address_uuid(ucp_address_t *address)
{
    return *(uint64_t*)address;
}


#endif
