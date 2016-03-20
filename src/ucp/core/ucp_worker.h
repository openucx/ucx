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

    int                           inprogress;
    char                          name[UCP_WORKER_NAME_MAX]; /* Worker name */

    unsigned                      stub_pend_count;/* Number of pending requests on stub endpoints*/
    ucs_list_link_t               stub_ep_list;  /* List of stub endpoints to progress */

    ucp_ep_t                      **ep_hash;     /* Hash table of all endpoints */
    uct_iface_h                   *ifaces;       /* Array of interfaces, one for each resource */
    uct_iface_attr_t              *iface_attrs;  /* Array of interface attributes */
    unsigned                      ep_config_max; /* Maximal number of configurations */
    unsigned                      ep_config_count; /* Current number of configurations */
    ucp_ep_config_t               ep_config[0];  /* Array of transport limits and thresholds */
} ucp_worker_t;


#define UCP_WORKER_EP_HASH_SIZE            32767
#define ucp_worker_ep_compare(_ep1, _ep2)  ((int64_t)(_ep1)->dest_uuid - (int64_t)(_ep2)->dest_uuid)
#define ucp_worker_ep_hash(_ep)            ((_ep)->dest_uuid)

SGLIB_DEFINE_LIST_PROTOTYPES(ucp_ep_t, ucp_worker_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(ucp_ep_t, UCP_WORKER_EP_HASH_SIZE, ucp_worker_ep_hash);


ucp_ep_h ucp_worker_get_reply_ep(ucp_worker_h worker, uint64_t dest_uuid);

ucp_request_t *ucp_worker_allocate_reply(ucp_worker_h worker, uint64_t dest_uuid);

unsigned ucp_worker_get_ep_config(ucp_worker_h worker, const ucp_rsc_index_t *rscs);

void ucp_worker_progress_stub_eps(void *arg);

void ucp_worker_stub_ep_add(ucp_worker_h worker, ucp_stub_ep_t *stub_ep);

void ucp_worker_stub_ep_remove(ucp_worker_h worker, ucp_stub_ep_t *stub_ep);


static inline const char* ucp_worker_get_name(ucp_worker_h worker)
{
    return worker->name;
}

static inline ucp_ep_h ucp_worker_ep_find(ucp_worker_h worker, uint64_t dest_uuid)
{
    ucp_ep_t search;

    search.dest_uuid = dest_uuid;
    return sglib_hashed_ucp_ep_t_find_member(worker->ep_hash, &search);
}

static inline ucp_ep_config_t *ucp_ep_config(ucp_ep_h ep)
{
    return &ep->worker->ep_config[ep->cfg_index];
}

static inline ucp_rsc_index_t ucp_ep_pd_index(ucp_ep_h ep, ucp_ep_op_t optype)
{
    ucp_context_h context = ep->worker->context;
    return context->tl_rscs[ucp_ep_config(ep)->rscs[optype]].pd_index;
}

static inline uct_pd_h ucp_ep_pd(ucp_ep_h ep, ucp_ep_op_t optype)
{
    ucp_context_h context = ep->worker->context;
    return context->pds[ucp_ep_pd_index(ep, optype)];
}

#endif
