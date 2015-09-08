/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
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
#define UCP_MAX_PDS             (sizeof(uint64_t) * 8)
#define UCP_CONFIG_ENV_PREFIX   "UCX_"
#define UCP_EP_HASH_SIZE        32767

typedef uint8_t                 ucp_rsc_index_t;


/**
 * Active message codes
 */
enum {
    UCP_AM_ID_EAGER_ONLY,
    UCP_AM_ID_WIREUP,
    UCP_AM_ID_LAST
};


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
    UCP_WIREUP_FLAG_REQUSET     = UCS_BIT(0),
    UCP_WIREUP_FLAG_REPLY       = UCS_BIT(1),
    UCP_WIREUP_FLAG_ACK         = UCS_BIT(2),
    UCP_WIREUP_FLAG_ADDR        = UCS_BIT(3),
    UCP_WIREUP_FLAG_AUX_ADDR    = UCS_BIT(4)
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
    uct_pd_attr_t           *pd_attrs;    /* Protection domain attributes */
    ucp_rsc_index_t         num_pds;      /* Number of protection domains */

    ucp_tl_resource_desc_t  *tl_rscs;     /* Array of communication resources */
    ucp_rsc_index_t         num_tls;      /* Number of resources in the array*/

    struct {
        ucs_queue_head_t    expected;
        ucs_queue_head_t    unexpected;
    } tag;

    struct {

        /* Bitmap of features supported by the context */
        uint64_t            features;

        /* Array of allocation methods, a mix of PD allocation methods and non-PD */
        struct {
            /* Allocation method */
            uct_alloc_method_t method;

            /* PD name to use, if method is PD */
            char               pdc_name[UCT_PD_COMPONENT_NAME_MAX];
        } *alloc_methods;
        unsigned            num_alloc_methods;

        /* Threshold for switching from short to bcopy protocol */
        size_t              bcopy_thresh;

    } config;

} ucp_context_t;


/**
 * Remote protocol layer endpoint
 */
typedef struct ucp_ep {
    ucp_worker_h        worker;        /* Worker this endpoint belongs to */
    uct_ep_h            uct_ep;        /* Current transport for operations */

    struct {
        size_t          max_short_tag; /* TODO should be unsigned */
        size_t          max_short_put; /* TODO should be unsigned */
        size_t          max_bcopy_put;
        size_t          max_bcopy_get;
    } config;

    uint64_t            dest_uuid;     /* Destination worker uuid */
    ucp_ep_h            next;          /* Next in hash table linked list */

    ucp_rsc_index_t     rsc_index;     /* Resource index the endpoint uses */
    ucp_rsc_index_t     dst_pd_index;  /* Destination protection domain index */
    volatile uint32_t   state;         /* Endpoint state */

    struct {
        uct_ep_h        aux_ep;        /* Used to wireup the "real" endpoint */
        uct_ep_h        next_ep;       /* Next transport being wired up */
    } wireup;

} ucp_ep_t;


/**
 * Endpoint pending operation.
 */
typedef struct ucp_ep_pending_op ucp_ep_pending_op_t;
struct ucp_ep_pending_op {
    uct_pending_req_t   uct;
    ucp_ep_h            ep;
};


/**
 * Pending operation of wireup message.
 */
typedef struct ucp_ep_wireup_op {
    ucp_ep_pending_op_t super;
    uint32_t            flags;
    ucp_rsc_index_t     dst_rsc_index;
    ucp_rsc_index_t     dst_aux_rsc_index;
} ucp_ep_wireup_op_t;


/**
 * Remote memory key structure.
 * Contains remote keys for UCT PDs.
 * pd_map specifies which PDs from the current context are present in the array.
 * The array itself contains only the PDs specified in pd_map, without gaps.
 */
typedef struct ucp_rkey {
    uint64_t            pd_map;  /* Which *remote* PDs have valid memory handles */
    uct_rkey_bundle_t   uct[0];  /* Remote key for every PD */
} ucp_rkey_t;


/**
 * Memory handle.
 * Contains general information, and a list of UCT handles.
 * pd_map specifies which PDs from the current context are present in the array.
 * The array itself contains only the PDs specified in pd_map, without gaps.
 */
typedef struct ucp_mem {
    void                *address;
    size_t              length;
    uct_alloc_method_t  alloc_method;
    uct_pd_h            alloc_pd;
    uint64_t            pd_map; /* Which PDs have valid memory handles */
    uct_mem_h           uct[0]; /* Valid memory handles, as popcount(pd_map) */
} ucp_mem_t;


/**
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    ucs_async_context_t       async;
    ucp_context_h             context;
    uint64_t                  uuid;
    uct_worker_h              uct;           /* UCT worker */
    ucp_ep_t                  **ep_hash;
    uct_iface_attr_t          *iface_attrs;  /* Array of interface attributes */
    ucp_user_progress_func_t  user_cb;
    void                      *user_cb_arg;
    uct_iface_h               ifaces[0];     /* Array of interfaces, one for each resource */
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
    ucp_rsc_index_t     dst_aux_index; /* Index of receiver wireup resource */
    uint16_t            flags;            /* Wireup flags */
    uint8_t             addr_len;         /* Length of first address */
    /* addresses follow */
} UCS_S_PACKED ucp_wireup_msg_t;


/**
 * Calculates a score of specific wireup.
 */
typedef double (*ucp_wireup_score_function_t)(ucp_worker_h worker,
                                              uct_tl_resource_desc_t *resource,
                                              uct_iface_h iface,
                                              uct_iface_attr_t *iface_attr);



ucs_status_t ucp_tag_init(ucp_context_h context);

void ucp_tag_cleanup(ucp_context_h context);

ucs_status_t ucp_tag_set_am_handlers(ucp_worker_h worker, uct_iface_h iface);



ucs_status_t ucp_ep_new(ucp_worker_h worker, uint64_t dest_uuid,
                        const char *message, ucp_ep_h *ep_p);

void ucp_ep_ready_to_send(ucp_ep_h ep);

void ucp_ep_destroy_uct_ep_safe(ucp_ep_h ep, uct_ep_h uct_ep);

ucs_status_t ucp_wireup_start(ucp_ep_h ep, ucp_address_t *address);

void ucp_wireup_stop(ucp_ep_h ep);

ucs_status_t ucp_wireup_set_am_handlers(ucp_worker_h worker, uct_iface_h iface);


#define ucp_ep_compare(_ep1, _ep2) ((int64_t)(_ep1)->dest_uuid - (int64_t)(_ep2)->dest_uuid)
#define ucp_ep_hash(_ep)           ((_ep)->dest_uuid)

SGLIB_DEFINE_LIST_PROTOTYPES(ucp_ep_t, ucp_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(ucp_ep_t, UCP_EP_HASH_SIZE, ucp_ep_hash);


#define UCP_RMA_RKEY_LOOKUP(_ep, _rkey) \
    ({ \
        if (ENABLE_PARAMS_CHECK && \
            !((_rkey)->pd_map & UCS_BIT((_ep)->dst_pd_index))) \
        { \
            ucs_fatal("Remote key does not support current transport " \
                       "(remote pd index: %d rkey map: 0x%"PRIx64")", \
                       (_ep)->dst_pd_index, (_rkey)->pd_map); \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        ucp_lookup_uct_rkey(_ep, _rkey); \
    })


static inline void ucp_ep_add_pending_op(ucp_ep_h ep, uct_ep_h uct_ep,
                                         ucp_ep_pending_op_t *op)
{
    ucs_trace_data("add pending operation %p to uct ep %p", op, uct_ep);
    op->ep = ep;
    uct_ep_pending_add(uct_ep, &op->uct);
}

static inline uint64_t ucp_address_uuid(ucp_address_t *address)
{
    return *(uint64_t*)address;
}

static inline ucp_ep_h ucp_worker_find_ep(ucp_worker_h worker, uint64_t dest_uuid)
{
    ucp_ep_t search;

    search.dest_uuid = dest_uuid;
    return sglib_hashed_ucp_ep_t_find_member(worker->ep_hash, &search);
}

static inline uct_rkey_t ucp_lookup_uct_rkey(ucp_ep_h ep, ucp_rkey_h rkey)
{
    unsigned rkey_index;

    /*
     * Calculate the rkey index inside the compact array. This is actually the
     * number of PDs in the map with index less-than ours. So mask pd_map to get
     * only the less-than indices, and then count them using popcount operation.
     * TODO save the mask in ep->uct, to avoid the shift operation.
     */
    rkey_index = ucs_count_one_bits(rkey->pd_map & UCS_MASK(ep->dst_pd_index));
    return rkey->uct[rkey_index].rkey;
}


#endif
