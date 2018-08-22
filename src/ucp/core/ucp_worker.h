/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WORKER_H_
#define UCP_WORKER_H_

#include "ucp_ep.h"
#include "ucp_thread.h"

#include <ucp/proto/proto.h>
#include <ucp/tag/tag_match.h>
#include <ucp/wireup/ep_match.h>
#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/datastruct/strided_alloc.h>


/* The size of the private buffer in UCT descriptor headroom, which UCP may
 * use for its own needs. This size does not include ucp_recv_desc_t length,
 * because it is common for all cases and protocols (TAG, STREAM). */
#define UCP_WORKER_HEADROOM_PRIV_SIZE 24


#define UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(_worker)                 \
    do {                                                                   \
        ucs_assert(!UCP_THREAD_IS_REQUIRED(&(_worker)->mt_lock) ||      \
                   UCP_THREAD_CS_IS_RECURSIVELY_LOCKED(&(_worker)->mt_lock) || \
                   !UCS_ASYNC_IS_RECURSIVELY_BLOCKED(&(_worker)->async)); \
        UCP_THREAD_CS_ENTER_CONDITIONAL(&(_worker)->mt_lock);           \
    } while (0)


#define UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(_worker)                  \
    UCP_THREAD_CS_EXIT_CONDITIONAL(&(_worker)->mt_lock)


/**
 * UCP worker flags
 */
enum {
    UCP_WORKER_FLAG_EXTERNAL_EVENT_FD = UCS_BIT(0), /**< worker event fd is external */
    UCP_WORKER_FLAG_EDGE_TRIGGERED    = UCS_BIT(1)  /**< events are edge-triggered */
};


/**
 * UCP iface flags
 */
enum {
    UCP_WORKER_IFACE_FLAG_OFFLOAD_ACTIVATED = UCS_BIT(0), /**< UCP iface receives tag
                                                               offload messages */
    UCP_WORKER_IFACE_FLAG_ON_ARM_LIST       = UCS_BIT(1)  /**< UCP iface is an element
                                                               of arm_ifaces list, so
                                                               it needs to be armed
                                                               in ucp_worker_arm(). */
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
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_EGR,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_RX_UNEXP_SW_RNDV,
    UCP_WORKER_STAT_TAG_OFFLOAD_LAST
};


#define UCP_WORKER_UCT_RECV_EVENT_ARM_FLAGS  (UCT_EVENT_RECV | \
                                              UCT_EVENT_RECV_SIG)
#define UCP_WORKER_UCT_RECV_EVENT_CAP_FLAGS  (UCT_IFACE_FLAG_EVENT_RECV | \
                                              UCT_IFACE_FLAG_EVENT_RECV_SIG)
#define UCP_WORKER_UCT_ALL_EVENT_CAP_FLAGS   (UCT_IFACE_FLAG_EVENT_SEND_COMP | \
                                              UCT_IFACE_FLAG_EVENT_RECV | \
                                              UCT_IFACE_FLAG_EVENT_RECV_SIG)
#define UCP_WORKER_UCT_UNSIG_EVENT_CAP_FLAGS (UCT_IFACE_FLAG_EVENT_SEND_COMP | \
                                              UCT_IFACE_FLAG_EVENT_RECV)


#define UCP_WORKER_STAT_EAGER_MSG(_worker, _flags) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             (_flags & UCP_RECV_DESC_FLAG_EAGER_SYNC) ? \
                             UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG : \
                             UCP_WORKER_STAT_TAG_RX_EAGER_MSG, 1);

#define UCP_WORKER_STAT_EAGER_CHUNK(_worker, _is_exp) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_##_is_exp, 1);

#define UCP_WORKER_STAT_RNDV(_worker, _is_exp) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_RNDV_##_is_exp, 1);

#define UCP_WORKER_STAT_TAG_OFFLOAD(_worker, _name) \
    UCS_STATS_UPDATE_COUNTER((_worker)->tm_offload_stats, \
                             UCP_WORKER_STAT_TAG_OFFLOAD_##_name, 1);

#define ucp_worker_mpool_get(_worker) \
    ({ \
        ucp_mem_desc_t *rdesc = ucs_mpool_get_inline(&(_worker)->reg_mp); \
        if (rdesc != NULL) { \
            VALGRIND_MAKE_MEM_DEFINED(rdesc, sizeof(*rdesc)); \
        } \
        rdesc; \
    })


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
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    ucs_async_context_t           async;         /* Async context for this worker */
    ucp_context_h                 context;       /* Back-reference to UCP context */
    uint64_t                      uuid;          /* Unique ID for wireup */
    uct_worker_h                  uct;           /* UCT worker handle */
    ucs_mpool_t                   req_mp;        /* Memory pool for requests */
    uint64_t                      atomic_tls;    /* Which resources can be used for atomics */

    int                           inprogress;
    char                          name[UCP_WORKER_NAME_MAX]; /* Worker name */

    unsigned                      flush_ops_count;/* Number of pending operations */

    unsigned                      flags;         /* Worker flags */
    int                           epfd;          /* Allocated (on-demand) epoll fd for wakeup */
    int                           eventfd;       /* Event fd to support signal() calls */
    unsigned                      uct_events;    /* UCT arm events */
    ucs_list_link_t               arm_ifaces;    /* List of interfaces to arm */

    void                          *user_data;    /* User-defined data */
    ucs_strided_alloc_t           ep_alloc;      /* Endpoint allocator */
    ucs_list_link_t               stream_ready_eps; /* List of EPs with received stream data */
    ucs_list_link_t               all_eps;       /* List of all endpoints */
    ucp_ep_match_ctx_t            ep_match_ctx;  /* Endpoint-to-endpoint matching context */
    ucp_worker_iface_t            *ifaces;       /* Array of interfaces, one for each resource */
    ucs_mpool_t                   am_mp;         /* Memory pool for AM receives */
    ucs_mpool_t                   reg_mp;        /* Registered memory pool */
    ucs_mpool_t                   rndv_frag_mp;  /* Memory pool for RNDV fragments */
    ucp_mt_lock_t                 mt_lock;       /* Configuration of multi-threading support */
    ucp_tag_match_t               tm;            /* Tag-matching queues and offload info */
    ucp_ep_h                      mem_type_ep[UCT_MD_MEM_TYPE_LAST];/* memory type eps */

    UCS_STATS_NODE_DECLARE(stats);
    UCS_STATS_NODE_DECLARE(tm_offload_stats);

    ucs_cpu_set_t                 cpu_mask;        /* Save CPU mask for subsequent calls to ucp_worker_listen */
    unsigned                      ep_config_max;   /* Maximal number of configurations */
    unsigned                      ep_config_count; /* Current number of configurations */
    ucp_ep_config_t               ep_config[0];    /* Array of transport limits and thresholds */
} ucp_worker_t;


/**
 * UCP worker argument for the error handling callback
 */
typedef struct ucp_worker_err_handle_arg {
    ucp_worker_h     worker;
    ucp_ep_h         ucp_ep;
    uct_ep_h         uct_ep;
    ucp_lane_index_t failed_lane;
    ucs_status_t     status;
} ucp_worker_err_handle_arg_t;


unsigned ucp_worker_get_ep_config(ucp_worker_h worker,
                                  const ucp_ep_config_key_t *key);

ucs_status_t ucp_worker_iface_init(ucp_worker_h worker, ucp_rsc_index_t tl_id,
                                   uct_iface_params_t *iface_params,
                                   ucp_worker_iface_t *wiface);

void ucp_worker_iface_cleanup(ucp_worker_iface_t *wiface);

void ucp_worker_iface_progress_ep(ucp_worker_iface_t *wiface);

void ucp_worker_iface_unprogress_ep(ucp_worker_iface_t *wiface);

void ucp_worker_signal_internal(ucp_worker_h worker);

void ucp_worker_iface_activate(ucp_worker_iface_t *wiface, unsigned uct_flags);

int ucp_worker_err_handle_remove_filter(const ucs_callbackq_elem_t *elem,
                                        void *arg);

static inline const char* ucp_worker_get_name(ucp_worker_h worker)
{
    return worker->name;
}

/* get ep by pointer received from remote side, do some debug checks */
static inline ucp_ep_h ucp_worker_get_ep_by_ptr(ucp_worker_h worker,
                                                uintptr_t ep_ptr)
{
    ucp_ep_h ep = (ucp_ep_h)ep_ptr;

    ucs_assert(ep != NULL);
    ucs_assertv(ep->worker == worker, "worker=%p ep=%p ep->worker=%p", worker,
                ep, ep->worker);
    return ep;
}

#endif
