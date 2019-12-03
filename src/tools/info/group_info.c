/**
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucx_info.h"

#include <ucg/api/ucg_mpi.h>
#include <ucg/api/ucg_plan_component.h>
#include <ucg/api/ucg_mpi.h>
#include <ucs/debug/memtrack.h>

/* In accordance with @ref enum ucg_predefined */
const char *collective_names[] = {
    "barrier",
    "reduce",
    "gather",
    "bcast",
    "scatter",
    "allreduce",
    NULL
};

#define EMPTY UCG_GROUP_MEMBER_DISTANCE_LAST

ucg_address_t *worker_address = 0;
ucs_status_t dummy_resolve_address(void *cb_group_obj,
                                   ucg_group_member_index_t index,
                                   ucg_address_t **addr, size_t *addr_len)
{
    *addr = worker_address;
    *addr_len = 0; /* special debug flow: replace uct_ep_t with member indexes */
    return UCS_OK;
}

void dummy_release_address(ucg_address_t *addr) { }

ucs_status_t gen_ucg_topology(ucg_group_member_index_t me,
        ucg_group_member_index_t peer_count[UCG_GROUP_MEMBER_DISTANCE_LAST],
        enum ucg_group_member_distance **distance_array_p,
        ucg_group_member_index_t *distance_array_length_p)
{
    printf("UCG Processes per socket:  %lu\n", peer_count[UCG_GROUP_MEMBER_DISTANCE_SOCKET]);
    printf("UCG Sockets per host:      %lu\n", peer_count[UCG_GROUP_MEMBER_DISTANCE_HOST]);
    printf("UCG Hosts in the network:  %lu\n", peer_count[UCG_GROUP_MEMBER_DISTANCE_NET]);

    /* generate the array of distances in order to create a group */
    ucg_group_member_index_t member_count = 1;
    peer_count[UCG_GROUP_MEMBER_DISTANCE_SELF] = 1;
    enum ucg_group_member_distance distance_idx = UCG_GROUP_MEMBER_DISTANCE_SELF;
    peer_count[distance_idx] = 1; /* not initialized by the user */
    for (; distance_idx < UCG_GROUP_MEMBER_DISTANCE_LAST; distance_idx++) {
        if (peer_count[distance_idx]) {
            member_count *= peer_count[distance_idx];
            peer_count[distance_idx] = member_count;
        }
    }

    if (me >= member_count) {
        printf("<Error: index is %lu, out of %lu total>\n", me, member_count);
        return UCS_ERR_INVALID_PARAM;
    }

    /* create the distance array for group creation */
    printf("UCG Total member count:    %lu\n", member_count);
    enum ucg_group_member_distance *distance_array =
            ucs_malloc(member_count * sizeof(*distance_array), "distance array");
    if (!distance_array) {
        printf("<Error: failed to allocate the distance array>\n");
        return UCS_ERR_NO_MEMORY;
    }

    memset(distance_array, EMPTY, member_count * sizeof(*distance_array));
    distance_idx = UCG_GROUP_MEMBER_DISTANCE_SELF;
    for (; distance_idx < UCG_GROUP_MEMBER_DISTANCE_LAST; distance_idx++) {
        if (!peer_count[distance_idx]) continue;

        unsigned array_idx, array_offset = me - (me % peer_count[distance_idx]);
        for (array_idx = 0; array_idx < peer_count[distance_idx]; array_idx++) {
            if (distance_array[array_idx + array_offset] == EMPTY) {
                distance_array[array_idx + array_offset] = distance_idx;
            }
        }
    }

    *distance_array_length_p = member_count;
    *distance_array_p = distance_array;
    return UCS_OK;
}

void print_ucg_topology(const char *req_planner_name, ucg_worker_h worker,
        ucg_group_member_index_t root,
        ucg_group_member_index_t me,
        const char *collective_type_name,
        enum ucg_group_member_distance *distance_array,
        ucg_group_member_index_t member_count, int is_verbose)
{
    ucs_status_t status;

    /* print the resulting distance array*/
    unsigned array_idx;
    printf("UCG Distance array for rank #%3lu [", me);
    for (array_idx = 0; array_idx < member_count; array_idx++) {
        switch (distance_array[array_idx]) {
        case UCG_GROUP_MEMBER_DISTANCE_SELF:
            printf("M");
            break;
        case UCG_GROUP_MEMBER_DISTANCE_SOCKET:
            printf(root == array_idx ? "S" : "s");
            break;
        case UCG_GROUP_MEMBER_DISTANCE_HOST:
            printf(root == array_idx ? "H" : "h");
            break;
        case UCG_GROUP_MEMBER_DISTANCE_NET:
            printf(root == array_idx ? "N" : "n");
            break;
        default:
            printf("<Failed to generate UCG distance array>\n");
            status = UCS_ERR_INVALID_PARAM;
            goto cleanup;
        }
    }
    printf("]\n");
    if (!is_verbose) {
        return;
    }

    /* create a group with the generated paramters */
    ucg_group_h group;
    ucg_group_params_t group_params = {
            .member_count = member_count,
            .distance = distance_array,
            .resolve_address_f = dummy_resolve_address,
            .release_address_f = dummy_release_address
    };
    size_t worker_address_length;
    status = ucp_worker_get_address(worker, &worker_address, &worker_address_length);
    if (status != UCS_OK) {
        goto cleanup;
    }

    status = ucg_group_create(worker, &group_params, &group);
    if (status != UCS_OK) {
        goto address_cleanup;
    }

    /* plan a collective operation */
    ucg_plan_t *plan;
    ucg_plan_component_t *planner;
    ucg_collective_params_t coll_params = {{0}};
    coll_params.send.dt_len = 1;
    coll_params.recv.dt_len = 1;
    coll_params.send.count = 1;
    coll_params.recv.count = 1;
    coll_params.send.buf = "send-buffer";
    coll_params.recv.buf = "recv-buffer";
    coll_params.type.root = root;

    const char **name = collective_names;
    while  (*name) {
        if (!strcmp(*name, collective_type_name)) {
            coll_params.type.modifiers = ucg_predefined_modifiers[name - collective_names];
            break;
        }
        name++;
    }

    status = ucg_plan_select(group, req_planner_name, &coll_params, &planner);
    if (status != UCS_OK) {
        goto group_cleanup;
    }

    status = planner->plan(planner, &coll_params.type, group, &plan);
    if (status != UCS_OK) {
        goto group_cleanup;
    }

    plan->group      = group;
    plan->planner    = planner;
    plan->type       = coll_params.type;
    ucs_list_head_init(&plan->op_head);

    planner->print(plan, &coll_params);

group_cleanup:
    ucg_group_destroy(group);

address_cleanup:
    ucp_worker_release_address(worker, worker_address);

cleanup:
    if (status != UCS_OK) {
        printf("<Failed to plan a UCG collective: %s>\n", ucs_status_string(status));
    }

    ucs_free(distance_array);
}
