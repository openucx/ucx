/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_WORKER_H_
#define UCP_WORKER_H_

#include "ucp_ep.h"

#include <ucs/datastruct/mpool.h>
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/async/async.h>


/**
 * UCP worker (thread context).
 */
typedef struct ucp_worker {
    ucs_async_context_t           async;         /* Async context for this worker */
    ucp_context_h                 context;       /* Back-reference to UCP context */
    uint64_t                      uuid;          /* Unique ID for wireup */
    uct_worker_h                  uct;           /* UCT worker handle */
#if ENABLE_ASSERT
    int                           inprogress;
#endif
    ucs_mpool_t                   req_mp;        /* Memory pool for requests */
    ucp_ep_t                      **ep_hash;     /* Hash table of all endpoints */
    uct_iface_h                   *ifaces;       /* Array of interfaces, one for each resource */
    uct_iface_attr_t              *iface_attrs;  /* Array of interface attributes */
    ucp_ep_config_t               ep_config[0];  /* Array of transport limits and thresholds */
} ucp_worker_t;


#define UCP_WORKER_EP_HASH_SIZE            32767
#define ucp_worker_ep_compare(_ep1, _ep2)  ((int64_t)(_ep1)->dest_uuid - (int64_t)(_ep2)->dest_uuid)
#define ucp_worker_ep_hash(_ep)            ((_ep)->dest_uuid)

SGLIB_DEFINE_LIST_PROTOTYPES(ucp_ep_t, ucp_worker_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(ucp_ep_t, UCP_WORKER_EP_HASH_SIZE, ucp_worker_ep_hash);


static inline ucp_ep_h ucp_worker_ep_find(ucp_worker_h worker, uint64_t dest_uuid)
{
    ucp_ep_t search;

    search.dest_uuid = dest_uuid;
    return sglib_hashed_ucp_ep_t_find_member(worker->ep_hash, &search);
}

static inline ucp_ep_config_t *ucp_ep_config(ucp_ep_h ep)
{
    return &ep->worker->ep_config[ep->rsc_index];
}

#endif
