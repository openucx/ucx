/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
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


/**
 * Flags for wireup select criteria, that include mandatory and optional flags
 */
typedef struct {
    /* All flags specified by this field must be set. */
    uint64_t mandatory;

    /* In addition to all mandatory flags, at least one of the flags
       defined by it must be present. */
    uint64_t optional;
} ucp_wireup_select_flags_t;


/* Peer name to show when we don't have debug information, or the name was not
 * packed in the worker address */
#define UCP_WIREUP_EMPTY_PEER_NAME  "<no debug data>"

#define UCP_RELEASE_LEGACY 0

#define UCP_WIREUP_UCT_EVENT_CAP_FLAGS \
    (UCT_IFACE_FLAG_EVENT_SEND_COMP | UCT_IFACE_FLAG_EVENT_RECV)


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
    /* Name of the criteria for debugging */
    const char                 *title;

    /* Required local MD flags */
    uint64_t                    local_md_flags;

    /* Required local component flags */
    uint64_t                    local_cmpt_flags;

    /* Required local interface flags */
    ucp_wireup_select_flags_t   local_iface_flags;

    /* Required remote interface flags */
    ucp_wireup_select_flags_t   remote_iface_flags;

    /* Required local event flags */
    uint64_t                    local_event_flags;

    /* Required remote event flags */
    uint64_t                    remote_event_flags;

    /* Mandatory memory types for allocation */
    uint64_t                    alloc_mem_types;

    /* Mandatory memory types for registration */
    uint64_t                    reg_mem_types;

    /* Required support of keepalive mechanism */
    int                         is_keepalive;

    /**
     * Calculates score of a potential transport.
     *
     * @param [in]  wiface        UCP worker iface.
     * @param [in]  md_attr       Local MD attributes.
     * @param [in]  unpacked_addr The whole remote address unpacked.
     * @param [in]  remote_addr   Remote transport address info and attributes.
     * @param [in]  arg           Custom argument.
     *
     * @return Transport score, the higher the better.
     */
    double                      (*calc_score)(const ucp_worker_iface_t *wiface,
                                              const uct_md_attr_v2_t *md_attr,
                                              const ucp_unpacked_address_t *unpacked_addr,
                                              const ucp_address_entry_t *remote_addr,
                                              void *arg);

    /* Custom argument of @a calc_score function */
    void                       *arg;

    /* Flags that describe TL specifics */
    uint8_t                     tl_rsc_flags;

    ucp_tl_iface_atomic_flags_t local_atomic_flags;

    ucp_tl_iface_atomic_flags_t remote_atomic_flags;
    ucp_lane_type_t             lane_type;
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
                                 const uct_md_attr_v2_t *md_attr,
                                 const ucp_unpacked_address_t *unpacked_address,
                                 const ucp_address_entry_t *remote_addr,
                                 void *arg);

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
                            const ucp_address_entry_t *ae,
                            char *info_str, size_t info_str_size);

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

/* add flags to all wireup_ep->flags */
void ucp_wireup_update_flags(ucp_ep_h ep, uint32_t new_flags);

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

unsigned ucp_wireup_eps_progress(void *arg);

double ucp_wireup_iface_lat_distance_v1(const ucp_worker_iface_t *wiface);

double ucp_wireup_iface_lat_distance_v2(const ucp_worker_iface_t *wiface);

double ucp_wireup_iface_bw_distance(const ucp_worker_iface_t *wiface);

int ucp_wireup_is_lane_connected(ucp_ep_h ep, ucp_lane_index_t lane,
                                 const ucp_address_entry_t *addr_entry);

static inline int ucp_wireup_lane_types_has_fast_path(ucp_lane_map_t lane_types)
{
    return lane_types &
           (UCS_BIT(UCP_LANE_TYPE_AM) | UCS_BIT(UCP_LANE_TYPE_RMA) |
            UCS_BIT(UCP_LANE_TYPE_AMO) | UCS_BIT(UCP_LANE_TYPE_CM) |
            UCS_BIT(UCP_LANE_TYPE_TAG));
}

static inline int ucp_wireup_lane_type_is_fast_path(ucp_lane_type_t lane_type)
{
    return ucp_wireup_lane_types_has_fast_path(UCS_BIT(lane_type));
}

static inline double ucp_wireup_fp8_pack_unpack_latency(double latency)
{
    return UCS_FP8_PACK_UNPACK(LATENCY, latency * UCS_NSEC_PER_SEC) /
           UCS_NSEC_PER_SEC;
}

#endif
