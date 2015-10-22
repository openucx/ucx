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

void ucp_address_peer_name(ucp_address_t *address, char *peer_name);

static inline uint64_t ucp_address_uuid(ucp_address_t *address)
{
    return *(uint64_t*)address;
}

static inline void *ucp_address_iter_start(ucp_address_t *address)
{
    uint8_t name_length;
    name_length = *(uint8_t*)(address + sizeof(uint64_t));
    return address + sizeof(uint64_t) + 1 + name_length;
}

#endif
