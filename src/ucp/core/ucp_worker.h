/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WORKER_H_
#define UCP_WORKER_H_

#include "ucp_ep.h"

#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/khash.h>
#include <ucs/async/async.h>

KHASH_MAP_INIT_INT64(ucp_worker_ep_hash, ucp_ep_t *);


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
 * UCP worker wake-up context.
 */
typedef struct ucp_worker_wakeup {
    int                           wakeup_efd;     /* Allocated (on-demand) epoll fd for wakeup */
    int                           wakeup_pipe[2]; /* Pipe to support signal() calls */
    uct_wakeup_h                  *iface_wakeups; /* Array of interface wake-up handles */
} ucp_worker_wakeup_t;


/**
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    ucs_async_context_t           async;         /* Async context for this worker */
    ucp_context_h                 context;       /* Back-reference to UCP context */
    uint64_t                      uuid;          /* Unique ID for wireup */
    uct_worker_h                  uct;           /* UCT worker handle */
    ucs_mpool_t                   req_mp;        /* Memory pool for requests */
    ucp_worker_wakeup_t           wakeup;        /* Wakeup-related context */
    uint64_t                      atomic_tls;    /* Which resources can be used for atomics */

    int                           inprogress;
    char                          name[UCP_WORKER_NAME_MAX]; /* Worker name */

    unsigned                      stub_pend_count;/* Number of pending requests on stub endpoints*/
    ucs_list_link_t               stub_ep_list;  /* List of stub endpoints to progress */

    khash_t(ucp_worker_ep_hash)   ep_hash;       /* Hash table of all endpoints */
    uct_iface_h                   *ifaces;       /* Array of interfaces, one for each resource */
    uct_iface_attr_t              *iface_attrs;  /* Array of interface attributes */
    unsigned                      ep_config_max; /* Maximal number of configurations */
    unsigned                      ep_config_count; /* Current number of configurations */
    ucp_ep_config_t               ep_config[0];  /* Array of transport limits and thresholds */
} ucp_worker_t;


ucp_ep_h ucp_worker_get_reply_ep(ucp_worker_h worker, uint64_t dest_uuid);

ucp_request_t *ucp_worker_allocate_reply(ucp_worker_h worker, uint64_t dest_uuid);

unsigned ucp_worker_get_ep_config(ucp_worker_h worker,
                                  const ucp_ep_config_key_t *key);

void ucp_worker_progress_stub_eps(void *arg);

void ucp_worker_stub_ep_add(ucp_worker_h worker, ucp_stub_ep_t *stub_ep);

void ucp_worker_stub_ep_remove(ucp_worker_h worker, ucp_stub_ep_t *stub_ep);


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

#endif
