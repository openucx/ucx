/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WORKER_H_
#define UCP_WORKER_H_

#include "ucp_ep.h"
#include "ucp_thread.h"

#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/async/async.h>

KHASH_MAP_INIT_INT64(ucp_worker_ep_hash, ucp_ep_t *);
KHASH_MAP_INIT_INT64(ucp_ep_errh_hash,   ucp_err_handler_t);


enum {
    UCP_UCT_IFACE_ATOMIC32_FLAGS =
        UCT_IFACE_FLAG_ATOMIC_ADD32  |
        UCT_IFACE_FLAG_ATOMIC_FADD32 |
        UCT_IFACE_FLAG_ATOMIC_SWAP32 |
        UCT_IFACE_FLAG_ATOMIC_CSWAP32,
    UCP_UCT_IFACE_ATOMIC64_FLAGS =
        UCT_IFACE_FLAG_ATOMIC_ADD64  |
        UCT_IFACE_FLAG_ATOMIC_FADD64 |
        UCT_IFACE_FLAG_ATOMIC_SWAP64 |
        UCT_IFACE_FLAG_ATOMIC_CSWAP64
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


#define UCP_WORKER_STAT_EAGER_MSG(_worker, _flags) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             (_flags & UCP_RECV_DESC_FLAG_SYNC) ? \
                             UCP_WORKER_STAT_TAG_RX_EAGER_SYNC_MSG : \
                             UCP_WORKER_STAT_TAG_RX_EAGER_MSG, 1);

#define UCP_WORKER_STAT_EAGER_CHUNK(_worker, _is_exp) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_EAGER_CHUNK_##_is_exp, 1);

#define UCP_WORKER_STAT_RNDV(_worker, _is_exp) \
    UCS_STATS_UPDATE_COUNTER((_worker)->stats, \
                             UCP_WORKER_STAT_TAG_RX_RNDV_##_is_exp, 1);

/**
 * UCP worker iface, which encapsulates UCT iface, its attributes and
 * some auxiliary info needed for tag matching offloads.
 */
typedef struct ucp_worker_iface {
    uct_iface_h                   iface;
    uct_iface_attr_t              attr;
    ucs_queue_elem_t              queue;
    ucp_rsc_index_t               rsc_index;
} ucp_worker_iface_t;


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

    unsigned                      stub_pend_count;/* Number of pending requests on stub endpoints*/

    int                           epfd;          /* Allocated (on-demand) epoll fd for wakeup */
    int                           wakeup_pipe[2];/* Pipe to support signal() calls */
    unsigned                      uct_events;    /* UCT arm events */

    khash_t(ucp_worker_ep_hash)   ep_hash;       /* Hash table of all endpoints */
    khash_t(ucp_ep_errh_hash)     ep_errh_hash;  /* Hash table of error handlers associated with endpoints */
    ucp_worker_iface_t            *ifaces;       /* Array of interfaces, one for each resource */
    ucs_mpool_t                   am_mp;         /* Memory pool for AM receives */
    UCS_STATS_NODE_DECLARE(stats);
    unsigned                      ep_config_max; /* Maximal number of configurations */
    unsigned                      ep_config_count; /* Current number of configurations */
    ucp_mt_lock_t                 mt_lock; /* All configurations about multithreading support */
    ucp_ep_config_t               ep_config[0];  /* Array of transport limits and thresholds */
} ucp_worker_t;


ucp_ep_h ucp_worker_get_reply_ep(ucp_worker_h worker, uint64_t dest_uuid);

ucp_request_t *ucp_worker_allocate_reply(ucp_worker_h worker, uint64_t dest_uuid);

unsigned ucp_worker_get_ep_config(ucp_worker_h worker,
                                  const ucp_ep_config_key_t *key);

static inline const char* ucp_worker_get_name(ucp_worker_h worker)
{
    return worker->name;
}

static inline ucp_ep_h ucp_worker_ep_find(ucp_worker_h worker, uint64_t dest_uuid)
{
    khiter_t hash_it;

    hash_it = kh_get(ucp_worker_ep_hash, &worker->ep_hash, dest_uuid);
    if (ucs_unlikely(hash_it == kh_end(&worker->ep_hash))) {
        return NULL;
    }

    return kh_value(&worker->ep_hash, hash_it);
}

static UCS_F_ALWAYS_INLINE
uint64_t ucp_worker_is_tl_tag_offload(ucp_worker_h worker, ucp_rsc_index_t rsc_index)
{
    return (worker->ifaces[rsc_index].attr.cap.flags &
            (UCT_IFACE_FLAG_TAG_EAGER_SHORT | UCT_IFACE_FLAG_TAG_EAGER_BCOPY |
             UCT_IFACE_FLAG_TAG_EAGER_ZCOPY | UCT_IFACE_FLAG_TAG_RNDV_ZCOPY));
}

#endif
