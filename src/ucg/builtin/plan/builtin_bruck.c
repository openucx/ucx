/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <string.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <uct/api/uct_def.h>
#include <ucg/api/ucg_mpi.h>

#include "builtin_plan.h"

ucs_status_t ucg_builtin_bruck_create(ucg_builtin_group_ctx_t *ctx,
        enum ucg_builtin_plan_topology_type plan_topo_type,
        const ucg_builtin_config_t *config,
        const ucg_group_params_t *group_params,
        const ucg_collective_type_t *coll_type,
        ucg_builtin_plan_t **plan_p)
{
    ucs_status_t status = UCS_OK;

    /* Calculate the number of bruck steps */
    unsigned proc_count = group_params->member_count;
    ucg_step_idx_t step_idx = 0;
    unsigned step_size = 1;

    while (step_size < proc_count) {
        step_size <<= 1;
        step_idx++;
    }

    /* Allocate memory resources */
    size_t alloc_size =            sizeof(ucg_builtin_plan_t) +
                       (step_idx * sizeof(ucg_builtin_plan_phase_t)
                   + (2*step_idx * sizeof(uct_ep_h)));/* every phase has no more than two endpoints! */

    ucg_builtin_plan_t *bruck       = (ucg_builtin_plan_t*)UCS_ALLOC_CHECK(alloc_size, "bruck topology");
    ucg_builtin_plan_phase_t *phase = &bruck->phss[0];
    uct_ep_h *next_ep               = (uct_ep_h*)(phase + step_idx);
    bruck->ep_cnt                   = step_idx * 2;
    bruck->phs_cnt                  = step_idx;

    /* Find my own index */
    ucg_group_member_index_t my_index = 0;
    while ((my_index < proc_count) &&
           (group_params->distance[my_index] !=
                   UCG_GROUP_MEMBER_DISTANCE_SELF)) {
        my_index++;
    }

    if (my_index == proc_count) {
        ucs_error("No member with distance==UCP_GROUP_MEMBER_DISTANCE_SELF found");
        return UCS_ERR_INVALID_PARAM;
    }

    /* Calculate the peers for each step */
    for (step_idx = 0, step_size = 1;
         ((step_idx < bruck->phs_cnt) && (status == UCS_OK));
         step_idx++, phase++, step_size <<= 1)
    {
        if(coll_type->modifiers & ucg_predefined_modifiers[UCG_PRIMITIVE_ALLGATHER])
            phase->method = UCG_PLAN_METHOD_ALLGATHER_BRUCK;
        else if(coll_type->modifiers & ucg_predefined_modifiers[UCG_PRIMITIVE_ALLTOALL])
           phase->method = UCG_PLAN_METHOD_ALLTOALL_BRUCK;

        phase->step_index = step_idx;

        /* In each step, there are two peers */
        ucg_group_member_index_t peer_index_src=0, peer_index_dst=0;/* src: source,  dst: destination */
        if (coll_type->modifiers & ucg_predefined_modifiers[UCG_PRIMITIVE_ALLGATHER])
        {
            peer_index_src = (my_index + step_size) % proc_count;
            peer_index_dst = (my_index - step_size + proc_count) % proc_count;
        }
        else if (coll_type->modifiers & ucg_predefined_modifiers[UCG_PRIMITIVE_ALLTOALL])
        {
            peer_index_dst = (my_index + step_size) % proc_count;
            peer_index_src = (my_index - step_size + proc_count) % proc_count;
        }

#if ENABLE_DEBUG_DATA || ENABLE_FAULT_TOLERANCE
        phase->indexes     = UCS_ALLOC_CHECK((peer_index_src == peer_index_dst ? 1 : 2) * sizeof(my_index),
                                             "bruck topology indexes");
#endif

        ucs_info("%lu's peer #%u(source) and #%u(destination) at (step #%u/%u)", my_index, (unsigned)peer_index_src,
            (unsigned)peer_index_dst, (unsigned)step_idx + 1, bruck->phs_cnt);

        if (peer_index_src != peer_index_dst)
        {
            phase->ep_cnt = 2-1;/* 1 sender and 1 receiver */

            unsigned phase_ep_index = 1; /* index: 0 for sender and 1 for receiver */

            phase->multi_eps = next_ep++;

            /* connected to receiver for second EP */
            status = ucg_builtin_connect(ctx, peer_index_src, phase, phase_ep_index);
            if (status != UCS_OK) {
                return status;
            }
            phase_ep_index--;
            next_ep++;

            /* set threshold for receiver
             * for bruck threshold for receiver and sender maybe not same!!!
             */
             phase->max_short_one_recv = phase->max_short_one;
             phase->max_short_max_recv = phase->max_short_max;
             phase->max_bcopy_one_recv = phase->max_bcopy_one;
             phase->max_bcopy_max_recv = phase->max_bcopy_max;
            phase->max_zcopy_one_recv = phase->max_zcopy_one;
            phase->md_attr_cap_max_reg_recv = phase->md_attr->cap.max_reg;

            /* connected to sender for first EP */
            status = ucg_builtin_connect(ctx, peer_index_dst, phase, phase_ep_index);
            if (status != UCS_OK) {
                return status;
            }

            /*
             * It is very important for Bruck plan that first EP is a sender
             * while phase->ep_cnt is set to be 1. So phase->single_ep should
             * point to multi_eps[0].
             */
            phase->single_ep = phase->multi_eps[0];

        }
        else
        {
            phase->ep_cnt  = 1;
            bruck->ep_cnt -= 1;
            phase->multi_eps = next_ep++;
            status = ucg_builtin_connect(ctx, peer_index_src, phase, UCG_BUILTIN_CONNECT_SINGLE_EP);
            if (status != UCS_OK) {
                return status;
            }
            /* set threshold for receiver
             * for bruck threshold for receiver and sender maybe not same!!!
             */
             phase->max_short_one_recv = phase->max_short_one;
             phase->max_short_max_recv = phase->max_short_max;
             phase->max_bcopy_one_recv = phase->max_bcopy_one;
             phase->max_bcopy_max_recv = phase->max_bcopy_max;
            phase->max_zcopy_one_recv = phase->max_zcopy_one;
            phase->md_attr_cap_max_reg_recv = phase->md_attr->cap.max_reg;
        }

    }

    bruck->super.my_index = my_index;
    *plan_p = bruck;
    return status;
}
