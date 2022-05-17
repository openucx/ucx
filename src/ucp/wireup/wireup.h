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
#include <ucs/arch/bitops.h>


/* Peer name to show when we don't have debug information, or the name was not
 * packed in the worker address */
#define UCP_WIREUP_EMPTY_PEER_NAME  "<no debug data>"


/**
 * Wireup message types
 */
enum {
    UCP_WIREUP_MSG_PRE_REQUEST,
    UCP_WIREUP_MSG_REQUEST,
    UCP_WIREUP_MSG_REPLY,
    UCP_WIREUP_MSG_ACK,
    UCP_WIREUP_MSG_EP_CHECK,
    UCP_WIREUP_MSG_EP_REMOVED,
    UCP_WIREUP_MSG_LAST
};


/**
 * Criteria for transport selection.
 */
typedef struct {
    const char  *title;             /* Name of the criteria for debugging */
    uint64_t    local_md_flags;     /* Required local MD flags */
    uint64_t    local_iface_flags;  /* Required local interface flags */
    uint64_t    remote_iface_flags; /* Required remote interface flags */
    uint64_t    local_event_flags;  /* Required local event flags */
    uint64_t    remote_event_flags; /* Required remote event flags */
    uint64_t    alloc_mem_types;    /* Mandatory memory types for allocation */
    uint64_t    reg_mem_types;      /* Mandatory memory types for registration */
    int         is_keepalive;       /* Required support of keepalive mechanism */

    /**
     * Calculates score of a potential transport.
     *
     * @param [in]  wiface       UCP worker iface.
     * @param [in]  md_attr      Local MD attributes.
     * @param [in]  remote_info  Remote peer attributes.
     *
     * @return Transport score, the higher the better.
     */
    double      (*calc_score)(const ucp_worker_iface_t *wiface,
                              const uct_md_attr_t *md_attr,
                              const ucp_address_iface_attr_t *remote_iface_attr);
    uint8_t     tl_rsc_flags; /* Flags that describe TL specifics */

    ucp_tl_iface_atomic_flags_t local_atomic_flags;
    ucp_tl_iface_atomic_flags_t remote_atomic_flags;
} ucp_wireup_criteria_t;


/**
 * Packet structure for wireup requests.
 */
typedef struct ucp_wireup_msg {
    uint8_t                type; /* Message type */
    uint8_t                err_mode; /* Peer error handling mode defined in
                                        @ucp_err_handling_mode_t */
    ucp_ep_match_conn_sn_t conn_sn; /* Connection sequence number */
    uint64_t               src_ep_id; /* Endpoint ID of source */
    uint64_t               dst_ep_id; /* Endpoint ID of destination, can be
                                         UCS_PTR_MAP_KEY_INVALID */
    /* packed addresses follow */
} UCS_S_PACKED ucp_wireup_msg_t;


typedef struct {
    double          score;
    unsigned        addr_index;
    unsigned        path_index;
    ucp_rsc_index_t rsc_index;
    uint8_t         priority;
} ucp_wireup_select_info_t;


ucs_status_t ucp_wireup_send_request(ucp_ep_h ep);

ucs_status_t ucp_wireup_send_pre_request(ucp_ep_h ep);

ucs_status_t ucp_wireup_connect_remote(ucp_ep_h ep, ucp_lane_index_t lane);

ucs_status_t
ucp_wireup_select_aux_transport(ucp_ep_h ep, unsigned ep_init_flags,
                                ucp_tl_bitmap_t tl_bitmap,
                                const ucp_unpacked_address_t *remote_address,
                                ucp_wireup_select_info_t *select_info);

double ucp_wireup_amo_score_func(const ucp_worker_iface_t *wiface,
                                 const uct_md_attr_t *md_attr,
                                 const ucp_address_iface_attr_t *remote_iface_attr);

size_t ucp_wireup_msg_pack(void *dest, void *arg);

const char* ucp_wireup_msg_str(uint8_t msg_type);

ucs_status_t ucp_wireup_msg_progress(uct_pending_req_t *self);

ucs_status_t
ucp_wireup_msg_prepare(ucp_ep_h ep, uint8_t type,
                       const ucp_tl_bitmap_t *tl_bitmap,
                       const ucp_lane_index_t *lanes2remote,
                       ucp_wireup_msg_t *msg_hdr, void **address_p,
                       size_t *address_length_p);

int ucp_wireup_msg_ack_cb_pred(const ucs_callbackq_elem_t *elem, void *arg);

int ucp_wireup_is_reachable(ucp_ep_h ep, unsigned ep_init_flags,
                            ucp_rsc_index_t rsc_index,
                            const ucp_address_entry_t *ae);

ucs_status_t ucp_wireup_init_lanes(ucp_ep_h ep, unsigned ep_init_flags,
                                   const ucp_tl_bitmap_t *local_tl_bitmap,
                                   const ucp_unpacked_address_t *remote_address,
                                   unsigned *addr_indices);

ucs_status_t
ucp_wireup_select_lanes(ucp_ep_h ep, unsigned ep_init_flags,
                        ucp_tl_bitmap_t tl_bitmap,
                        const ucp_unpacked_address_t *remote_address,
                        unsigned *addr_indices, ucp_ep_config_key_t *key,
                        int show_error);

void ucp_wireup_replay_pending_requests(ucp_ep_h ucp_ep,
                                        ucs_queue_head_t *tmp_pending_queue);

/* Set lanes which are wireup_ep as remote connected.
   If 'ready' is true - also mark them as ready and switch them to the real
   transport uct_ep in the next progress call */
void ucp_wireup_remote_connect_lanes(ucp_ep_h ep, int ready);

void ucp_wireup_remote_connected(ucp_ep_h ep);

unsigned ucp_ep_init_flags(const ucp_worker_h worker,
                           const ucp_ep_params_t *params);

int ucp_wireup_connect_p2p(ucp_worker_h worker, ucp_rsc_index_t rsc_index,
                           int has_cm_lane);

ucs_status_t
ucp_wireup_connect_local(ucp_ep_h ep,
                         const ucp_unpacked_address_t *remote_address,
                         const ucp_lane_index_t *lanes2remote);

uct_ep_h ucp_wireup_extract_lane(ucp_ep_h ep, ucp_lane_index_t lane);

#endif
