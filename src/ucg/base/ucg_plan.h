/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_PLAN_H_
#define UCG_PLAN_H_

#include "../api/ucg_plan_component.h"

/* Functions on all planning components */
ucs_status_t ucg_plan_query(unsigned *next_am_id, ucg_plan_desc_t **resources_p, unsigned *num_resources_p);
void ucg_plan_release_list(ucg_plan_desc_t *resources, unsigned resource_cnt);
ucs_status_t ucg_plan_select_component(ucg_plan_desc_t *planners,
                                       unsigned num_planners,
                                       const char* planner_name,
                                       const ucg_group_params_t *group_params,
                                       const ucg_collective_params_t *coll_params,
                                       ucg_plan_component_t **planc_p);

/* Functions on a specific component */
#define ucg_plan(planc, group_ctx, coll_params, plan_p) \
    (planc->plan(planc, group_ctx, coll_params, plan_p))
#define ucg_prepare(plan, params, op) ((plan)->planner->prepare(plan, params, op))
#define ucg_trigger(op, cid, req)     ((op)->plan->planner->trigger(op, cid, req))
#define ucg_discard(op)               ((op)->plan->planner->discard(op))
#define ucg_destroy(plan)             ((plan)->planner->destroy(plan))

#endif /* UCG_PLAN_H_ */
