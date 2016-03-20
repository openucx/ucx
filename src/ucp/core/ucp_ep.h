/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_context.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>
#include <ucs/debug/log.h>
#include <limits.h>


/**
 * Endpoint flags
 */
enum {
    UCP_EP_FLAG_LOCAL_CONNECTED  = UCS_BIT(0), /* All local endpoints are connected */
    UCP_EP_FLAG_REMOTE_CONNECTED = UCS_BIT(1), /* All remote endpoints are connected */
    UCP_EP_FLAG_CONNECT_REQ_SENT = UCS_BIT(2), /* Connection request was sent */
};


/**
 * Endpoint operation types
 */
typedef enum ucp_ep_op {
    UCP_EP_OP_AM,      /* Active messages */
    UCP_EP_OP_RMA,     /* Remote memory access */
    UCP_EP_OP_AMO,     /* Atomic operations */
    UCP_EP_OP_LAST
} ucp_ep_op_t;


typedef struct ucp_ep_config {
    /* Transport configuration */
    ucp_rsc_index_t        rscs[UCP_EP_OP_LAST]; /* Resource index for every operation */
    ucp_ep_op_t            dups[UCP_EP_OP_LAST]; /* List of which resources are
                                                    duplicate of others. if an
                                                    entry is not UCP_EP_OP_LAST,
                                                    it's the index of the first
                                                    instance of the resource. */

    /* Limits for protocols using short message only */
    size_t                 max_eager_short;  /* Maximal payload of eager short */
    size_t                 max_put_short;    /* Maximal payload of put short */
    size_t                 max_am_short;     /* Maximal payload of am short */

    /* Limits for bcopy operations */
    size_t                 max_am_bcopy;     /* Maximal total size of am_bcopy */
    size_t                 max_put_bcopy;    /* Maximal total size of put_bcopy */
    size_t                 max_get_bcopy;    /* Maximal total size of get_bcopy */

    /* Limits for zero-copy operations */
    size_t                 max_am_zcopy;     /* Maximal total size of am_zcopy */
    size_t                 max_put_zcopy;    /* Maximal total size of put_zcopy */
    size_t                 max_get_zcopy;    /* Maximal total size of get_zcopy */

    /* Threshold for switching from put_short to put_bcopy */
    size_t                 bcopy_thresh;

    /* Threshold for switching from eager to rendezvous */
    size_t                 rndv_thresh;

    /* threshold for switching from eager-sync to rendezvous */
    size_t                 sync_rndv_thresh;

    /* zero-copy threshold for operations which do not have to wait for remote side */
    size_t                 zcopy_thresh;

    /* zero-copy threshold for operations which anyways have to wait for remote side */
    size_t                 sync_zcopy_thresh;

} ucp_ep_config_t;


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_worker_h                  worker;        /* Worker this endpoint belongs to */

    uct_ep_h                      uct_eps[UCP_EP_OP_LAST]; /* Transports for operations */

    ucp_rsc_index_t               rma_dst_pdi;   /* Destination protection domain index for RMA */
    ucp_rsc_index_t               amo_dst_pdi;   /* Destination protection domain index for AMO */
    uint8_t                       cfg_index;     /* Configuration index */
    uint8_t                       flags;         /* Endpoint flags */

    uint64_t                      dest_uuid;     /* Destination worker uuid */
    ucp_ep_h                      next;          /* Next in hash table linked list */

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_NAME_MAX];
#endif

} ucp_ep_t;


ucs_status_t ucp_ep_create_connected(ucp_worker_h worker, uint64_t dest_uuid,
                                     const char *peer_name, unsigned address_count,
                                     const ucp_address_entry_t *address_list,
                                     const char *message, ucp_ep_h *ep_p);

ucs_status_t ucp_ep_create_stub(ucp_worker_h worker, uint64_t dest_uuid,
                                const char *message, ucp_ep_h *ep_p);

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep);

ucs_status_t ucp_ep_add_pending_uct(ucp_ep_h ep, uct_ep_h uct_ep,
                                    uct_pending_req_t *req);

void ucp_ep_add_pending(ucp_ep_h ep, uct_ep_h uct_ep, ucp_request_t *req,
                        int progress);

ucs_status_t ucp_ep_pending_req_release(uct_pending_req_t *self);

void ucp_ep_send_reply(ucp_request_t *req, ucp_ep_op_t optype, int progress);

int ucp_ep_is_op_primary(ucp_ep_h ep, ucp_ep_op_t optype);

static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "??";
#endif
}


#endif
