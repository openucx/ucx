/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WORKER_H_
#define UCP_WORKER_H_

#include "ucp_ep.h"
#include "ucp_context.h"
#include "ucp_thread.h"
#include "ucp_rkey.h"

#include <ucp/core/ucp_am.h>
#include <ucp/tag/tag_match.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/datastruct/strided_alloc.h>
#include <ucs/datastruct/conn_match.h>
#include <ucs/datastruct/ptr_map.h>
#include <ucs/arch/bitops.h>


/* The size of the private buffer in UCT descriptor headroom, which UCP may
 * use for its own needs. This size does not include ucp_recv_desc_t length,
 * because it is common for all cases and protocols (TAG, STREAM). */
#define UCP_WORKER_HEADROOM_PRIV_SIZE 32


#if ENABLE_MT

#define UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(_worker)                 \
    do {                                                                \
        if ((_worker)->flags & UCP_WORKER_FLAG_MT) {                    \
            UCS_ASYNC_BLOCK(&(_worker)->async);                         \
        }                                                               \
    } while (0)


#define UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(_worker)                  \
    do {                                                                \
        if ((_worker)->flags & UCP_WORKER_FLAG_MT) {                    \
            UCS_ASYNC_UNBLOCK(&(_worker)->async);                       \
        }                                                               \
    } while (0)


#else

#define UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(_worker)
#define UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(_worker)

#endif


/**
 * UCP worker flags
 */
enum {
    UCP_WORKER_FLAG_EXTERNAL_EVENT_FD = UCS_BIT(0), /**< worker event fd is external */
    UCP_WORKER_FLAG_EDGE_TRIGGERED    = UCS_BIT(1), /**< events are edge-triggered */
    UCP_WORKER_FLAG_MT                = UCS_BIT(2)  /**< MT locking is required */
};


/**
 * UCP iface flags
 */
enum {
    UCP_WORKER_IFACE_FLAG_OFFLOAD_ACTIVATED = UCS_BIT(0), /**< UCP iface receives tag
                                                               offload messages */
    UCP_WORKER_IFACE_FLAG_ON_ARM_LIST       = UCS_BIT(1), /**< UCP iface is an element
                                                               of arm_ifaces list, so
                                                               it needs to be armed
                                                               in ucp_worker_arm(). */
    UCP_WORKER_IFACE_FLAG_UNUSED            = UCS_BIT(2)  /**< There is another UCP iface
                                                               with the same caps, but
                                                               with better performance */
};


/**
 * UCP worker statistics counters
 */
enum {
    /* Total number of received eager messages */
    UCP_WORKER_STAT_TAG_RX_EAGER_MSG,
    UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG,

    /* Total number of  received eager chunks (every message
     * can be split into a bunch of chunks). It is possible that
     * some chunks  of the message arrived unexpectedly and then
     * receive had been posted and the rest arrived expectedly */
    UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_EXP,
    UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_UNEXP,

    UCP_WORKER_STAT_TAG_RX_RNDV_EXP,
    UCP_WORKER_STAT_TAG_RX_RNDV_UNEXP,

    UCP_WORKER_STAT_TAG_RX_RNDV_GET_ZCOPY,
    UCP_WORKER_STAT_TAG_RX_RNDV_SEND_RTR,
    UCP_WORKER_STAT_TAG_RX_RNDV_RKEY_PTR,

    UCP_WORKER_STAT_LAST
};


/**
 * UCP worker tag offload statistics counters
 */
enum {
    UCP_WORKER_STAT_TAG_OFFLOAD_POSTED,
    UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED,
    UCP_WORKER_STAT_TAG_OFFLOAD_MATCHED_SW_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_CANCELED,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_TAG_EXCEED,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NON_CONTIG,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_WILDCARD,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_SW_PEND,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_NO_IFACE,
    UCP_WORKER_STAT_TAG_OFFLOAD_BLOCK_MEM_REG,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_EGR,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_LAST
};


#define UCP_WORKER_STAT_EAGER_MSG(_worker, _flags) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             ((_flags) & UCP_RECV_DESC_FLAG_EAGER_SYNC) ? \
                             UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG : \
                             UCP_WORKER_STAT_TAG_RX_EAGER_MSG, 1);

#define UCP_WORKER_STAT_EAGER_CHUNK(_worker, _is_exp) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_##_is_exp, 1);

#define UCP_WORKER_STAT_RNDV(_worker, _is_exp, _value) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_RNDV_##_is_exp, _value);

#define UCP_WORKER_STAT_TAG_OFFLOAD(_worker, _name) \
    UCS_STATS_UPDATE_COUNTER((_worker)->tm_offload_stats, \
                             UCP_WORKER_STAT_TAG_OFFLOAD_##_name, 1);

#define ucp_worker_mpool_get(_mp) \
    ({ \
        ucp_mem_desc_t *_rdesc = ucs_mpool_get_inline(_mp); \
        if (_rdesc != NULL) { \
            VALGRIND_MAKE_MEM_DEFINED(_rdesc, sizeof(*_rdesc)); \
        } \
        _rdesc; \
    })


/* Hash map to find rkey config index by rkey config key, for fast rkey unpack */
KHASH_TYPE(ucp_worker_rkey_config, ucp_rkey_config_key_t, ucp_worker_cfg_index_t);
typedef khash_t(ucp_worker_rkey_config) ucp_worker_rkey_config_hash_t;


/* Hash map of UCT EPs that are being discarded on UCP Worker */
KHASH_TYPE(ucp_worker_discard_uct_ep_hash, uct_ep_h, ucp_request_t*);
typedef khash_t(ucp_worker_discard_uct_ep_hash) ucp_worker_discard_uct_ep_hash_t;


/**
 * UCP worker iface, which encapsulates UCT iface, its attributes and
 * some auxiliary info needed for tag matching offloads.
 */
struct ucp_worker_iface {
    uct_iface_h                   iface;         /* UCT interface */
    uct_iface_attr_t              attr;          /* UCT interface attributes */
    ucp_worker_h                  worker;        /* The parent worker */
    ucs_list_link_t               arm_list;      /* Element in arm_ifaces list */
    ucp_rsc_index_t               rsc_index;     /* Resource index */
    int                           event_fd;      /* Event FD, or -1 if undefined */
    unsigned                      activate_count;/* How many times this iface has
                                                    been activated */
    int                           check_events_id;/* Callback id for check_events */
    unsigned                      proxy_recv_count;/* Counts active messages on proxy handler */
    unsigned                      post_count;    /* Counts uncompleted requests which are
                                                    offloaded to the transport */
    uint8_t                       flags;         /* Interface flags */
};


/**
 * UCP worker CM, which encapsulates UCT CM and its auxiliary info.
 */
struct ucp_worker_cm {
    uct_cm_h                      cm;            /* UCT CM handle */
    uct_cm_attr_t                 attr;          /* UCT CM attributes */
    ucp_rsc_index_t               cmpt_idx;      /* Index of corresponding
                                                    component */
};


/**
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    unsigned                         flags;               /* Worker flags */
    ucs_async_context_t              async;               /* Async context for this worker */
    ucp_context_h                    context;             /* Back-reference to UCP context */
    uint64_t                         uuid;                /* Unique ID for wireup */
    uct_worker_h                     uct;                 /* UCT worker handle */
    ucs_mpool_t                      req_mp;              /* Memory pool for requests */
    ucs_mpool_t                      rkey_mp;             /* Pool for small memory keys */
    uint64_t                         atomic_tls;          /* Which resources can be used for atomics */

    int                              inprogress;
    char                             name[UCP_WORKER_NAME_MAX]; /* Worker name */

    unsigned                         flush_ops_count;     /* Number of pending operations */

    int                              event_fd;            /* Allocated (on-demand) event fd for wakeup */
    ucs_sys_event_set_t              *event_set;          /* Allocated UCS event set for wakeup */
    int                              eventfd;             /* Event fd to support signal() calls */
    unsigned                         uct_events;          /* UCT arm events */
    ucs_list_link_t                  arm_ifaces;          /* List of interfaces to arm */

    void                             *user_data;          /* User-defined data */
    ucs_strided_alloc_t              ep_alloc;            /* Endpoint allocator */
    ucs_list_link_t                  stream_ready_eps;    /* List of EPs with received stream data */
    ucs_list_link_t                  all_eps;             /* List of all endpoints */
    ucs_conn_match_ctx_t             conn_match_ctx;      /* Endpoint-to-endpoint matching context */
    ucp_worker_iface_t               **ifaces;            /* Array of pointers to interfaces,
                                                             one for each resource */
    unsigned                         num_ifaces;          /* Number of elements in ifaces array  */
    unsigned                         num_active_ifaces;   /* Number of activated ifaces  */
    uint64_t                         scalable_tl_bitmap;  /* Map of scalable tl resources */
    ucp_worker_cm_t                  *cms;                /* Array of CMs, one for each component */
    ucs_mpool_t                      am_mp;               /* Memory pool for AM receives */
    ucs_mpool_t                      reg_mp;              /* Registered memory pool */
    ucs_mpool_t                      rndv_frag_mp;        /* Memory pool for RNDV fragments */
    ucs_queue_head_t                 rkey_ptr_reqs;       /* Queue of submitted RKEY PTR requests that
                                                           * are in-progress */
    uct_worker_cb_id_t               rkey_ptr_cb_id;      /* RKEY PTR worker callback queue ID */
    ucp_tag_match_t                  tm;                  /* Tag-matching queues and offload info */
    ucs_array_t(ucp_am_cbs)          am;                  /* Array of AM callbacks and their data */
    uint64_t                         am_message_id;       /* For matching long AMs */
    ucp_ep_h                         mem_type_ep[UCS_MEMORY_TYPE_LAST]; /* Memory type EPs */

    UCS_STATS_NODE_DECLARE(stats)
    UCS_STATS_NODE_DECLARE(tm_offload_stats)

    ucs_cpu_set_t                    cpu_mask;            /* Save CPU mask for subsequent calls to
                                                             ucp_worker_listen */

    ucp_worker_rkey_config_hash_t    rkey_config_hash;    /* RKEY config key -> index */
    ucp_worker_discard_uct_ep_hash_t discard_uct_ep_hash; /* Hash of discarded UCT EPs */
    ucs_ptr_map_t                    ptr_map;             /* UCP objects key to ptr mapping */

    unsigned                         ep_config_count;     /* Current number of ep configurations */
    ucp_ep_config_t                  ep_config[UCP_WORKER_MAX_EP_CONFIG];

    unsigned                         rkey_config_count;   /* Current number of rkey configurations */
    ucp_rkey_config_t                rkey_config[UCP_WORKER_MAX_RKEY_CONFIG];

    struct {
        uct_worker_cb_id_t           cb_id;               /* Keepalive callback id */
        ucs_time_t                   last_round;          /* Last round timespamp */
        ucs_list_link_t              *iter;               /* Last EP processed keepalive */
        ucp_lane_map_t               lane_map;            /* Lane map used to retry after no-resources */
        unsigned                     ep_count;            /* Number if EPs processed in current time slot */
    } keepalive;
} ucp_worker_t;


/**
 * UCP worker argument for the error handling callback
 */
typedef struct ucp_worker_err_handle_arg {
    ucp_ep_h         ucp_ep;
    ucs_status_t     status;
} ucp_worker_err_handle_arg_t;


ucs_status_t
ucp_worker_get_ep_config(ucp_worker_h worker, const ucp_ep_config_key_t *key,
                         int print_cfg, ucp_worker_cfg_index_t *cfg_index_p);

ucs_status_t
ucp_worker_add_rkey_config(ucp_worker_h worker, const ucp_rkey_config_key_t *key,
                           ucp_worker_cfg_index_t *cfg_index_p);

ucs_status_t ucp_worker_iface_open(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   uct_iface_params_t *iface_params,
                                   ucp_worker_iface_t **wiface);

ucs_status_t ucp_worker_iface_init(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   ucp_worker_iface_t *wiface);

void ucp_worker_iface_cleanup(ucp_worker_iface_t *wiface);

void ucp_worker_iface_progress_ep(ucp_worker_iface_t *wiface);

void ucp_worker_iface_unprogress_ep(ucp_worker_iface_t *wiface);

void ucp_worker_signal_internal(ucp_worker_h worker);

void ucp_worker_iface_activate(ucp_worker_iface_t *wiface, unsigned uct_flags);

int ucp_worker_err_handle_remove_filter(const ucs_callbackq_elem_t *elem,
                                        void *arg);

ucs_status_t ucp_worker_set_ep_failed(ucp_worker_h worker, ucp_ep_h ucp_ep,
                                      uct_ep_h uct_ep, ucp_lane_index_t lane,
                                      ucs_status_t status);

void ucp_worker_keepalive_add_ep(ucp_ep_h );

/* EP should be removed from worker all_eps prior to call this function */
void ucp_worker_keepalive_remove_ep(ucp_ep_h ep);

/* must be called with async lock held */
int ucp_worker_is_uct_ep_discarding(ucp_worker_h worker, uct_ep_h uct_ep);

/* must be called with async lock held */
void ucp_worker_discard_uct_ep(ucp_ep_h ucp_ep, uct_ep_h uct_ep,
                               unsigned ep_flush_flags,
                               uct_pending_purge_callback_t purge_cb,
                               void *purge_arg);

/* must be called with async lock held */
static UCS_F_ALWAYS_INLINE void
ucp_worker_flush_ops_count_inc(ucp_worker_h worker)
{
    ucs_assert(worker->flush_ops_count < UINT_MAX);
    ++worker->flush_ops_count;
}

/* must be called with async lock held */
static UCS_F_ALWAYS_INLINE void
ucp_worker_flush_ops_count_dec(ucp_worker_h worker)
{
    ucs_assert(worker->flush_ops_count > 0);
    --worker->flush_ops_count;
}

#endif
