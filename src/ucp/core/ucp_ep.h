/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_EP_H_
#define UCP_EP_H_

#include "ucp_context.h"
#include "ucp_request.h"

#include <uct/api/uct.h>
#include <ucs/debug/log.h>
#include <ucs/debug/log.h>
#include <limits.h>




/**
 * Endpoint wire-up state
 */
enum {
    UCP_EP_STATE_READY_TO_SEND            = UCS_BIT(0), /* uct_ep can send to remote */
    UCP_EP_STATE_READY_TO_RECEIVE         = UCS_BIT(1), /* remote can send to me */
    UCP_EP_STATE_AUX_EP                   = UCS_BIT(2), /* aux_ep was created */
    UCP_EP_STATE_NEXT_EP                  = UCS_BIT(3), /* next_ep was created */
    UCP_EP_STATE_NEXT_EP_LOCAL_CONNECTED  = UCS_BIT(4), /* next_ep connected to remote */
    UCP_EP_STATE_NEXT_EP_REMOTE_CONNECTED = UCS_BIT(5), /* remote also connected to our next_ep */
    UCP_EP_STATE_WIREUP_REPLY_SENT        = UCS_BIT(6), /* wireup reply message has been sent */
    UCP_EP_STATE_WIREUP_ACK_SENT          = UCS_BIT(7), /* wireup ack message has been sent */
    UCP_EP_STATE_STUB_EP                  = UCS_BIT(8), /* the current ep is a stub */
};


typedef struct ucp_ep_config {
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
    uct_ep_h                      uct_ep;        /* Current transport for operations */

    ucp_rsc_index_t               rsc_index;     /* Resource index the endpoint uses */
    ucp_rsc_index_t               dst_pd_index;  /* Destination protection domain index */
    volatile uint32_t             state;         /* Endpoint state */

    uint64_t                      dest_uuid;     /* Destination worker uuid */
    ucp_ep_h                      next;          /* Next in hash table linked list */

#if ENABLE_DEBUG_DATA
    char                          peer_name[UCP_WORKER_NAME_MAX];
#endif

} ucp_ep_t;


ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *peer_name, const char *message, ucp_ep_h *ep_p);

void ucp_ep_delete(ucp_ep_h ep);

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep);

ucs_status_t ucp_ep_add_pending_uct(ucp_ep_h ep, uct_ep_h uct_ep,
                                    uct_pending_req_t *req);

void ucp_ep_add_pending(ucp_ep_h ep, uct_ep_h uct_ep, ucp_request_t *req,
                        int progress);

void ucp_ep_send_reply(ucp_request_t *req, int progress);


static inline const char* ucp_ep_peer_name(ucp_ep_h ep)
{
#if ENABLE_DEBUG_DATA
    return ep->peer_name;
#else
    return "??";
#endif
}


#endif
