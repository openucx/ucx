/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_GROUP_H_
#define UCG_GROUP_H_

#include "ucg_plan.h"

#include <ucs/stats/stats.h>

#define UCG_GROUP_MAX_IFACES (8)

extern size_t ucg_ctx_worker_offset;
#define UCG_WORKER_TO_GROUPS_CTX(worker) \
    ((ucg_groups_t*)((char*)(worker) + ucg_ctx_worker_offset))
#define UCG_GROUP_COLLECTIVE_MODIFIER_MASK UCS_MASK(7)
#define UCG_FLAG_MASK(params) \
    ((params)->type.modifiers & UCG_GROUP_COLLECTIVE_MODIFIER_MASK)

/*
 * To enable the "Groups" feature in UCX - it's registered as part of the UCX
 * context - and allocated a context slot in each UCP Worker at a certain offset.
 */
typedef struct ucg_groups {
    ucs_list_link_t       groups_head;
    ucg_group_id_t        next_id;

    unsigned              iface_cnt;
    uct_iface_h           ifaces[UCG_GROUP_MAX_IFACES];

    size_t                total_planner_sizes;
    unsigned              num_planners;
    ucg_plan_desc_t      *planners;
} ucg_groups_t;

struct ucg_group {
    /*
     * Whether a current barrier is waited upon. If so, new collectives cannot
     * start until this barrier is cleared, so it is put in the pending queue.
     */
    int                is_barrier_outstanding;

    ucg_worker_h       worker;       /* for conn. est. and progress calls */
    ucg_coll_id_t      next_id;      /* for the next collective operation */
    ucg_group_id_t     group_id;     /* group identifier (order of creation) */
    ucs_queue_head_t   pending;      /* requests currently pending execution */
    ucg_group_params_t params;       /* parameters, for future connections */
    ucs_list_link_t    list;         /* worker's group list */

    UCS_STATS_NODE_DECLARE(stats);

    unsigned           iface_cnt;
    uct_iface_h        ifaces[UCG_GROUP_MAX_IFACES];

    /* per-group cache of previous plans/operations, arranged as follows:
     * for each collective type (e.g. Allreduce) there is a plan with a list of
     * operations. To re-use a past operation it must be available and match the
     * requested collective parameters.
     */
    ucg_plan_t        *cache[UCG_GROUP_COLLECTIVE_MODIFIER_MASK];

    /* Below this point - the private per-planner data is allocated/stored */
};

#endif /* UCG_GROUP_H_ */
