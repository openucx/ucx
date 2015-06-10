/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCP_INT_H
#define UCP_INT_H

#include <ucp/api/ucp.h>
#include <uct/api/uct.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/sglib_wrapper.h>


#define UCP_MAX_TLS             UINT8_MAX
#define UCP_CONFIG_ENV_PREFIX   "UCX_"
#define UCP_EP_HASH_SIZE        32767

typedef uint8_t                 ucp_rsc_index_t;


/**
 * Active message codes
 */
enum {
    UCP_AM_ID_EAGER_ONLY,
    UCP_AM_ID_CONN_REQ,
    UCP_AM_ID_CONN_REP,
    UCP_AM_ID_CONN_ACK,
    UCP_AM_ID_LAST
};


/**
 * Endpoint wire-up state
 */
enum {
    UCP_EP_STATE_LOCAL_CONNECTED  = UCS_BIT(0), /* Local endpoint connected to remote */
    UCP_EP_STATE_REMOTE_CONNECTED = UCS_BIT(1), /* Remove also connected to local */
    UCP_EP_STATE_CONN_REP_SENT    = UCS_BIT(2), /* CONN_REP message has been sent */
    UCP_EP_STATE_CONN_ACK_SENT    = UCS_BIT(3), /* CONN_ACK message has been sent */

    UCP_EP_STATE_PENDING          = UCS_BIT(16) /* Resource-available notification
                                                   has been requested from one of
                                                   the UCT endpoints */
};


/**
 * UCP communication resource descriptor
 */
typedef struct ucp_tl_resource_desc {
    uct_tl_resource_desc_t  tl_rsc;      /* UCT resource descriptor */
    ucp_rsc_index_t         pd_index;    /* Protection domain index (within the context) */
} ucp_tl_resource_desc_t;


/**
 * UCP context
 */
typedef struct ucp_context {
    uct_pd_resource_desc_t  *pd_rscs;     /* Protection domain resources */
    uct_pd_h                *pds;         /* Protection domain handles */
    ucp_rsc_index_t         num_pds;      /* Number of protection domains */


    ucp_tl_resource_desc_t  *tl_rscs;     /* Array of communication resources */
    ucp_rsc_index_t         num_tls;      /* Number of resources in the array*/

    struct {
        ucs_mpool_h         rreq_mp;       /* Receive requests */
        ucs_queue_head_t    expected;
        ucs_queue_head_t    unexpected;
    } tag;

} ucp_context_t;


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_worker_h        worker;        /* Worker this endpoint belongs to */

    struct {
        uct_ep_h        ep;            /* Current transport for operations */
        uct_ep_h        next_ep;       /* Next transport being wired up */
        ucp_rsc_index_t rsc_index;     /* Resource index the endpoint uses */
        ucp_rsc_index_t dst_pd_index;  /* Destination rotection domain index */
    } uct;

    struct {
        size_t          max_short_tag;
    } config;

    uct_ep_h            wireup_ep;     /* Used to wireup the "real" endpoint */
    uint64_t            dest_uuid;     /* Destination worker uuid */
    ucp_ep_h            next;          /* Next in hash table linked list */

    volatile uint32_t   state;         /* Endpoint state */
    ucs_queue_head_t    pending_q;     /* Queue of pending operations - protected by the async worker lock */
    uct_completion_t    notify_comp;   /* Completion token for progressing pending queue */

} ucp_ep_t;


/**
 * Endpoint pending operation.
 */
typedef struct ucp_ep_pending_op ucp_ep_pending_op_t;
struct ucp_ep_pending_op {
    ucs_queue_elem_t    queue;

    /**
     * Progress callback for the pending operation.
     * It attempts to initiate the operation, and returns the status.
     * If the operation cannot be started, the function returns UCS_ERR_NO_RESOURCE
     * and *uct_ep_p is filled with the transport endpoint on which the send
     * was attempted.
     */
    ucs_status_t        (*progress)(ucp_ep_h ep, ucp_ep_pending_op_t *op,
                                    uct_ep_h *uct_ep_p);
};


/**
 * Pending operation of wireup message.
 */
typedef struct ucp_ep_wireup_op {
    ucp_ep_pending_op_t super;
    uint8_t             am_id;
    ucp_rsc_index_t     dest_rsc_index;
} ucp_ep_wireup_op_t;


/**
 * Remote memory key structure.
 */
typedef struct ucp_rkey {
    /* TODO */
} ucp_rkey_t;


/**
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    ucs_async_context_t async;
    ucp_context_h       context;
    uint64_t            uuid;
    uct_worker_h        uct;           /* UCT worker */
    ucs_queue_head_t    completed;     /* Queue of completed requests */
    size_t              uct_comp_priv; /* Max. length of UCT completion private area */

    ucp_ep_t            **ep_hash;
    uct_iface_attr_t    *iface_attrs;  /* Array of interface attributes */
    uct_iface_h         ifaces[0];     /* Array of interfaces, one for each resource */
} ucp_worker_t;


typedef struct ucp_recv_desc {
    ucs_queue_elem_t    queue;
    size_t              length;
} ucp_recv_desc_t;


/**
 * Packet structure for wireup requests.
 */
typedef struct ucp_wireup_msg {
    uint64_t            src_uuid;         /* Sender uuid */
    ucp_rsc_index_t     src_pd_index;     /* Sender PD index */
    ucp_rsc_index_t     src_rsc_index;    /* Index of sender resource */
    ucp_rsc_index_t     dst_rsc_index;    /* Index of receiver resource */
    /* EP address follows */
} ucp_wireup_msg_t;


/**
 * Calculates a score of specific wireup.
 */
typedef double (*ucp_wireup_score_function_t)(uct_tl_resource_desc_t *resource,
                                              uct_iface_h iface,
                                              uct_iface_attr_t *iface_attr);



ucs_status_t ucp_tag_init(ucp_context_h context);

void ucp_tag_cleanup(ucp_context_h context);

ucs_status_t ucp_tag_set_am_handlers(ucp_worker_h worker, uct_iface_h iface);

ucs_status_t ucp_ep_wireup_start(ucp_ep_h ep, ucp_address_t *address);

void ucp_ep_wireup_stop(ucp_ep_h ep);

ucs_status_t ucp_wireup_set_am_handlers(ucp_worker_h worker, uct_iface_h iface);


#define ucp_ep_compare(_ep1, _ep2) ((int64_t)(_ep1)->dest_uuid - (int64_t)(_ep2)->dest_uuid)
#define ucp_ep_hash(_ep)           ((_ep)->dest_uuid)

SGLIB_DEFINE_LIST_PROTOTYPES(ucp_ep_t, ucp_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(ucp_ep_t, UCP_EP_HASH_SIZE, ucp_ep_hash);


static inline void ucp_ep_add_pending_op(ucp_ep_h ep, uct_ep_h uct_ep,
                                         ucp_ep_pending_op_t *op)
{
    UCS_ASYNC_BLOCK(&ep->worker->async);
    ucs_queue_push(&ep->pending_q, &op->queue);
    if (!(ep->state & UCP_EP_STATE_PENDING)) {
        uct_ep_req_notify(uct_ep, &ep->notify_comp);
        ep->state |= UCP_EP_STATE_PENDING;
    }
    UCS_ASYNC_UNBLOCK(&ep->worker->async);
}

static inline ucp_ep_h ucp_worker_find_ep(ucp_worker_h worker, uint64_t dest_uuid)
{
    ucp_ep_t search;

    search.dest_uuid = dest_uuid;
    return sglib_hashed_ucp_ep_t_find_member(worker->ep_hash, &search);
}


#endif
