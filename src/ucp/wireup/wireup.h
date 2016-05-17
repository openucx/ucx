/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WIREUP_H_
#define UCP_WIREUP_H_

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.h>
#include <uct/api/uct.h>


/* Transport capabilities used for wireup decision */
#define UCP_WIREUP_TL_CAP_FLAGS \
    (UCT_IFACE_FLAG_AM_SHORT | \
     UCT_IFACE_FLAG_AM_BCOPY | \
     UCT_IFACE_FLAG_AM_ZCOPY | \
     UCT_IFACE_FLAG_PUT_SHORT | \
     UCT_IFACE_FLAG_PUT_BCOPY | \
     UCT_IFACE_FLAG_PUT_ZCOPY | \
     UCT_IFACE_FLAG_GET_SHORT | \
     UCT_IFACE_FLAG_GET_BCOPY | \
     UCT_IFACE_FLAG_GET_ZCOPY | \
     UCT_IFACE_FLAG_ATOMIC_ADD32 | \
     UCT_IFACE_FLAG_ATOMIC_ADD64 | \
     UCT_IFACE_FLAG_ATOMIC_FADD32 | \
     UCT_IFACE_FLAG_ATOMIC_FADD64 | \
     UCT_IFACE_FLAG_ATOMIC_SWAP32 | \
     UCT_IFACE_FLAG_ATOMIC_SWAP64 | \
     UCT_IFACE_FLAG_ATOMIC_CSWAP32 | \
     UCT_IFACE_FLAG_ATOMIC_CSWAP64)


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
 * Criteria for transport selection.
 */
typedef struct {
    const char  *title;            /* Name of the criteria for debugging */
    uint64_t    local_pd_flags;    /* Required local PD flags */
    uint64_t    remote_pd_flags;   /* Required remote PD flags */
    uint64_t    local_iface_flags; /* Required local iface flags */
    uint64_t    remote_iface_flags;/* Required remote iface flags */

    /**
     * Calculates score of a potential transport.
     *
     * @param [in]  pd_attr      Local PD attributes.
     * @param [in]  iface_attr   Local interface attributes.
     * @param [in]  remote_info  Remote peer attributes.
     *
     * @return Transport score, the higher the better.
     */
    double      (*calc_score)(const uct_pd_attr_t *pd_attr,
                              const uct_iface_attr_t *iface_attr,
                              const ucp_wireup_tl_info_t *remote_info);

} ucp_wireup_criteria_t;


/**
 * Transport capabilities and performance.
 */
struct ucp_wireup_tl_info {
    uint32_t                   tl_caps;
    float                      overhead;
    float                      bandwidth;
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


ucs_status_t ucp_wireup_send_request(ucp_ep_h ep);

ucs_status_t ucp_wireup_select_aux_transport(ucp_ep_h ep,
                                             const ucp_address_entry_t *address_list,
                                             unsigned address_count,
                                             ucp_rsc_index_t *rsc_index_p,
                                             unsigned *addr_index_p);

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self);

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, unsigned address_count,
                                   const ucp_address_entry_t *address_list);

#endif
