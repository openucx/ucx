/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <math.h>
#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack.h>
#include <uct/api/uct_def.h>

#include "builtin_plan.h"

#define MAX_PEERS (100)
/*
 * max number of phases are determined by both tree & recursive
 * MAX_PHASES for tree plan      is 4
 * MAX_PHASES for recursive plan is 4 (namely it support 2^4 nodes !)
 */

#define MAX_PHASES (10) /* till now, binomial tree can only support 2^MAX_PHASES process at most */

ucs_config_field_t ucg_builtin_binomial_tree_config_table[] = {
    {"DEGREE", "8", "k-normial tree degree.\n",
     ucs_offsetof(ucg_builtin_binomial_tree_config_t, degree), UCS_CONFIG_TYPE_UINT},

    {NULL}
};

typedef struct ucg_builtin_binomial_tree_params {
    ucg_builtin_group_ctx_t *ctx;
    const ucg_group_params_t *group_params;
    const ucg_collective_type_t *coll_type;
    enum ucg_builtin_plan_topology_type topo_type;
    ucg_group_member_index_t root;
    int tree_degree;
} ucg_builtin_binomial_tree_params_t;

static inline ucs_status_t ucg_builtin_binomial_tree_connect_phase(ucg_builtin_plan_phase_t *phase,
                                                                   const ucg_builtin_binomial_tree_params_t *params,
                                                                   ucg_step_idx_t step_index,
                                                                   uct_ep_h **eps,
                                                                   ucg_group_member_index_t *peers,
                                                                   unsigned peer_cnt,
                                                                   enum ucg_builtin_plan_method_type method)
{
    /* Initialization */
    ucs_assert(peer_cnt > 0);
    ucs_status_t status = UCS_OK;
    phase->method = method;
    phase->ep_cnt = peer_cnt;
    phase->step_index = step_index;
#if ENABLE_DEBUG_DATA || ENABLE_FAULT_TOLERANCE
    phase->indexes     = UCS_ALLOC_CHECK(peer_cnt * sizeof(*peers),
                                         "binomial tree topology indexes");
#endif
    if (peer_cnt == 1) {
        status = ucg_builtin_connect(params->ctx, peers[0], phase, UCG_BUILTIN_CONNECT_SINGLE_EP);
    }
    else {
        phase->multi_eps = *eps;
        *eps += peer_cnt;

        /* connect every endpoint, by group member index */
        unsigned idx;
        for (idx = 0; (idx < peer_cnt) && (status == UCS_OK); idx++, peers++) {
            status = ucg_builtin_connect(params->ctx, *peers, phase, idx);
        }
    }
    return status;
}

static inline ucs_status_t ucg_builtin_binomial_tree_connect(ucg_builtin_plan_t *tree,
                                                    const ucg_builtin_binomial_tree_params_t *params,
                                                    size_t *alloc_size,
                                                    ucg_group_member_index_t *up,
                                                    unsigned up_cnt,
                                                    ucg_group_member_index_t *down,
                                                    unsigned down_cnt)
{
    enum ucg_collective_modifiers mod = params->coll_type->modifiers;
    ucs_status_t status               = UCS_OK;
    uct_ep_h *first_ep                = (uct_ep_h*)(&tree->phss[MAX_PHASES]);
    uct_ep_h *eps                     = first_ep;
    tree->phs_cnt                     = 0;

    ucs_assert(up_cnt + down_cnt > 0);

    enum ucg_builtin_plan_method_type fanout_method;
    if (down_cnt) {
        fanout_method = (mod & UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST) ?
                     (up_cnt ? UCG_PLAN_METHOD_BCAST_WAYPOINT :
                               UCG_PLAN_METHOD_SEND_TERMINAL) :
                     (up_cnt ? UCG_PLAN_METHOD_SCATTER_WAYPOINT :
                               UCG_PLAN_METHOD_SCATTER_TERMINAL);
    }
    else {
        fanout_method = UCG_PLAN_METHOD_RECV_TERMINAL;
    }

    if (params->topo_type == UCG_PLAN_TREE_FANOUT){
        /* only recv (recv_terminal) */
        /* Receive from parents */
        if (up_cnt == 1 && down_cnt == 0) {
            /* Connect this phase to its peers */
            ucs_assert(up_cnt == 1); /* sanity check: not multi-root */
            status = ucg_builtin_binomial_tree_connect_phase(&tree->phss[tree->phs_cnt++], params, 0,
                &eps, up, up_cnt, fanout_method);
        }

        /* only send (send_terminal) */
        /* Send to children */
        if (up_cnt == 0 && down_cnt >0) {
            /* Connect this phase to its peers */
            ucg_group_member_index_t  member_idx;
            ucg_group_member_index_t *down_another;
            for (member_idx = 0; member_idx < down_cnt; member_idx++) {
                //down_another = down + sizeof(ucg_group_member_index_t)*member_idx;
                down_another = down + member_idx;
                status = ucg_builtin_binomial_tree_connect_phase(&tree->phss[tree->phs_cnt++], params, 0,
                    &eps, down_another, 1, fanout_method);
            }
        }

        /* first recv then send (waypoint) */
        if (up_cnt == 1 && down_cnt > 0) {
            /* Connect this phase to its peers */
            status = ucg_builtin_binomial_tree_connect_phase(&tree->phss[tree->phs_cnt++], params, 0,
                &eps, up, up_cnt, UCG_PLAN_METHOD_RECV_TERMINAL);

            ucg_group_member_index_t member_idx;
            ucg_group_member_index_t *down_another;
            for (member_idx = 0; member_idx < down_cnt; member_idx++) {
                down_another = down + member_idx;
                status = ucg_builtin_binomial_tree_connect_phase(&tree->phss[tree->phs_cnt++], params, 0,
                    &eps, down_another, 1, UCG_PLAN_METHOD_SEND_TERMINAL);
            }
        }
    }else{
        status = UCS_ERR_INVALID_PARAM;
    }

    if (status != UCS_OK) {
        //free(tree);
        return status;
    }


    tree->ep_cnt = eps - first_ep;
    size_t ep_size = tree->ep_cnt * sizeof(*eps);
    memmove(&tree->phss[tree->phs_cnt], first_ep, ep_size);
    *alloc_size = (void*)first_ep + ep_size - (void*)tree;
    return UCS_OK;
}

/****************************************************************************
*                                                                           *
*                               Binomial tree                               *
*                                                                           *
****************************************************************************/


static ucs_status_t ucg_builtin_binomial_tree_build(const ucg_builtin_binomial_tree_params_t* params,
    ucg_builtin_plan_t* tree,
    size_t* alloc_size)
{
    ucg_group_member_index_t member_idx;
    ucg_group_member_index_t up[MAX_PEERS] = { 0 };
    ucg_group_member_index_t down[MAX_PEERS] = { 0 };
    unsigned up_cnt = 0, down_cnt = 0;

    unsigned num_child = 0, my_index, rank_shift, member_count, tree_mask = 1, peer;
    unsigned root;

    root = params->root;
    member_count = params->group_params->member_count;

    my_index = member_count;

    for (member_idx = 0; member_idx < params->group_params->member_count; member_idx++) {
        enum ucg_group_member_distance next_distance =
            params->group_params->distance[member_idx];
        ucs_assert(next_distance < UCG_GROUP_MEMBER_DISTANCE_LAST);

        /* Possibly add the next member to my list according to its distance */
        if (ucs_unlikely(next_distance == UCG_GROUP_MEMBER_DISTANCE_SELF)) {
            my_index = member_idx;
            break;
        }
    }

    if (my_index == member_count) {
        ucs_error("No member with distance==UCP_GROUP_MEMBER_DISTANCE_SELF found");
        return UCS_ERR_INVALID_PARAM;
    }

    /* Inverse Binomial Tree */
    rank_shift = (my_index - root + member_count) % member_count;

    unsigned value = rank_shift;
    for (tree_mask = 1; value > 0; value >>= 1, tree_mask <<= 1) /* empty */;

    /* find parent */
    if (root == my_index) {
        up_cnt = 0;
    }
    else {
        peer = rank_shift ^ (tree_mask >> 1);
        up[0] = (peer + root) % member_count;
        up_cnt = 1;
    }

    /* find children */
    while (tree_mask < member_count) {
        peer = rank_shift ^ tree_mask;
        if (peer >= member_count)
            break;
        down[num_child] = (peer + root) % member_count;
        num_child++;
        tree_mask <<= 1;
    }

    down_cnt = num_child;

    /* Some output, for informational purposes */
    ucs_info("Topology for member #%lu :", tree->super.my_index);
    for (member_idx = 0; member_idx < up_cnt; member_idx++) {
        ucs_info("%lu's parent #%lu/%u: %lu ", tree->super.my_index, member_idx, up_cnt, up[member_idx]);
    }
    for (member_idx = 0; member_idx < down_cnt; member_idx++) {
        ucs_info("%lu's child  #%lu/%u: %lu ", tree->super.my_index, member_idx, down_cnt, down[member_idx]);
    }

    /* fill in the tree phases while establishing the connections */
    return ucg_builtin_binomial_tree_connect(tree, params, alloc_size, up, up_cnt, down, down_cnt);
}

ucs_status_t ucg_builtin_binomial_tree_create(ucg_builtin_group_ctx_t *ctx,
        enum ucg_builtin_plan_topology_type plan_topo_type,
        const ucg_builtin_config_t *config,
        const ucg_group_params_t *group_params,
        const ucg_collective_type_t *coll_type,
        ucg_builtin_plan_t **plan_p)
{
    /* Allocate worst-case memory footprint, resized down later */
    size_t alloc_size = sizeof(ucg_builtin_plan_t) +
            MAX_PHASES * (sizeof(ucg_builtin_plan_phase_t) + (MAX_PEERS * sizeof(uct_ep_h)));
    ucg_builtin_plan_t *tree = (ucg_builtin_plan_t*)UCS_ALLOC_CHECK(alloc_size, "tree topology");
    tree->phs_cnt = 0; /* will be incremented with usage */

    /* tree discovery and construction, by phase */
    ucg_builtin_binomial_tree_params_t params = {
            .ctx = ctx,
            .coll_type = coll_type,
            .topo_type = plan_topo_type,
            .group_params = group_params,
            .root = coll_type->root,
            .tree_degree = config->bmtree.degree
    };
    ucs_status_t ret = ucg_builtin_binomial_tree_build(&params, tree, &alloc_size);
    if (ret != UCS_OK) {
        ucs_free(tree);
        return ret;
    }

    /* Reduce the allocation size according to actual usage */
    *plan_p = (ucg_builtin_plan_t*)ucs_realloc(tree, alloc_size, "tree topology");
    ucs_assert(*plan_p != NULL); /* only reduces size - should never fail */
    return UCS_OK;
}
