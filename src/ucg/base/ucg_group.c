/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucg_group.h"

#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/list.h>
#include <ucs/profile/profile.h>
#include <ucs/debug/memtrack.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_proxy_ep.h> /* for @ref ucp_proxy_ep_test */

#if ENABLE_STATS
/**
 * UCG group statistics counters
 */
enum {
    UCG_GROUP_STAT_PLANS_CREATED,
    UCG_GROUP_STAT_PLANS_USED,

    UCG_GROUP_STAT_OPS_CREATED,
    UCG_GROUP_STAT_OPS_USED,
    UCG_GROUP_STAT_OPS_IMMEDIATE,

    UCG_GROUP_STAT_LAST
};

static ucs_stats_class_t ucg_group_stats_class = {
    .name           = "ucg_group",
    .num_counters   = UCG_GROUP_STAT_LAST,
    .counter_names  = {
        [UCG_GROUP_STAT_PLANS_CREATED] = "plans_created",
        [UCG_GROUP_STAT_PLANS_USED]    = "plans_reused",
        [UCG_GROUP_STAT_OPS_CREATED]   = "ops_created",
        [UCG_GROUP_STAT_OPS_USED]      = "ops_started",
        [UCG_GROUP_STAT_OPS_IMMEDIATE] = "ops_immediate"
    }
};
#endif

#define UCG_GROUP_PROGRESS_ADD(iface, ctx) {         \
    unsigned idx = 0;                                \
    if (ucs_unlikely(idx == UCG_GROUP_MAX_IFACES)) { \
        return UCS_ERR_EXCEEDS_LIMIT;                \
    }                                                \
                                                     \
    while (idx < (ctx)->iface_cnt) {                 \
        if ((ctx)->ifaces[idx] == (iface)) {         \
            break;                                   \
        }                                            \
        idx++;                                       \
    }                                                \
                                                     \
    if (idx == (ctx)->iface_cnt) {                   \
        (ctx)->ifaces[(ctx)->iface_cnt++] = (iface); \
    }                                                \
}

__KHASH_IMPL(ucg_group_ep, static UCS_F_MAYBE_UNUSED inline,
             ucg_group_member_index_t, ucp_ep_h, 1, kh_int64_hash_func,
             kh_int64_hash_equal);

unsigned ucg_worker_progress(ucg_worker_h worker)
{
    unsigned idx, ret = 0;
    ucg_groups_t *gctx = UCG_WORKER_TO_GROUPS_CTX(worker);

    /* First, try the interfaces used for collectives */
    for (idx = 0; idx < gctx->iface_cnt; idx++) {
        ret += uct_iface_progress(gctx->ifaces[idx]);
    }

    /* As a fallback (and for correctness - try all other transports */
    return ucp_worker_progress(worker);
}

unsigned ucg_group_progress(ucg_group_h group)
{
    unsigned idx, ret = 0;
    ucg_groups_t *gctx = UCG_WORKER_TO_GROUPS_CTX(group->worker);

    /* First, try the per-planner progress functions */
    for (idx = 0; idx < gctx->num_planners; idx++) {
        ucg_plan_component_t *planc = gctx->planners[idx].plan_component;
        ret += planc->progress(group);
    }
    if (ret) {
        return ret;
    }

    /* Next, try the per-group interfaces */
    for (idx = 0; idx < group->iface_cnt; idx++) {
        ret += uct_iface_progress(group->ifaces[idx]);
    }
    if (ret) {
        return ret;
    }

    /* Lastly, try the "global" progress */
    return ucg_worker_progress(group->worker);
}

size_t ucg_ctx_worker_offset;
ucs_status_t ucg_group_create(ucg_worker_h worker,
                              const ucg_group_params_t *params,
                              ucg_group_h *group_p)
{
    ucs_status_t status;
    ucg_groups_t *ctx = UCG_WORKER_TO_GROUPS_CTX(worker);
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker); // TODO: check where needed

    /* allocate a new group */
    size_t distance_size              = sizeof(*params->distance) * params->member_count;
    struct ucg_group *new_group       = ucs_malloc(sizeof(struct ucg_group) +
            ctx->total_planner_sizes + distance_size, "communicator group");
    if (new_group == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto cleanup_none;
    }

    /* fill in the group fields */
    new_group->is_barrier_outstanding = 0;
    new_group->group_id               = ctx->next_id++;
    new_group->worker                 = worker;
    new_group->next_id                = 1; /* Some Transports don't like == 0... // TODO: fix wrap-around ! */
    new_group->iface_cnt              = 0;

    ucs_queue_head_init(&new_group->pending);
    memcpy((ucg_group_params_t*)&new_group->params, params, sizeof(*params));
    new_group->params.distance = (typeof(params->distance))((char*)(new_group
            + 1) + ctx->total_planner_sizes);
    memcpy(new_group->params.distance, params->distance, distance_size);
    memset(new_group + 1, 0, ctx->total_planner_sizes);

    unsigned idx;
    for (idx = 0; idx < UCG_GROUP_COLLECTIVE_MODIFIER_MASK; idx++) {
        new_group->cache[idx] = NULL;
    }

    /* Create a loopback connection, since resolve_cb may fail loopback */
    ucp_ep_params_t ep_params = { .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS };
    status = ucp_worker_get_address(worker, (ucp_address_t**)&ep_params.address,
            &distance_size);
    if (status != UCS_OK) {
        return status;
    }
    ucp_ep_h loopback_ep;
    status = ucp_ep_create(worker, &ep_params, &loopback_ep);
    ucp_worker_release_address(worker, (ucp_address_t*)ep_params.address);
    if (status != UCS_OK) {
        return status;
    }

    /* Store this loopback endpoint, for future reference */
    ucg_group_member_index_t my_index = strlen((char*)new_group->params.distance); // TODO: replace this hack!
    ucs_assert(kh_get(ucg_group_ep, &ctx->eps, my_index) == kh_end(&ctx->eps));
    khiter_t iter = kh_put(ucg_group_ep, &ctx->eps, my_index, (int*)&idx);
    kh_value(&ctx->eps, iter) = loopback_ep;

    /* Initialize the planners (modules) */
    for (idx = 0; idx < ctx->num_planners; idx++) {
        /* Create the per-planner per-group context */
        ucg_plan_component_t *planner = ctx->planners[idx].plan_component;
        status = planner->create(planner, worker, new_group,
                new_group->group_id, &new_group->params);
        if (status != UCS_OK) {
            goto cleanup_planners;
        }
    }

    status = UCS_STATS_NODE_ALLOC(&new_group->stats,
            &ucg_group_stats_class, worker->stats, "-%p", new_group);
    if (status != UCS_OK) {
        goto cleanup_planners;
    }

    ucs_list_add_head(&ctx->groups_head, &new_group->list);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    *group_p = new_group;
    return UCS_OK;

cleanup_planners:
    while (idx) {
        ucg_plan_component_t *planner = ctx->planners[idx--].plan_component;
        planner->destroy((void*)new_group);
    }
    ucs_free(new_group);

cleanup_none:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;
}

const ucg_group_params_t* ucg_group_get_params(ucg_group_h group)
{
    return &group->params;
}

void ucg_group_destroy(ucg_group_h group)
{
    /* First - make sure all the collectives are completed */
    while (!ucs_queue_is_empty(&group->pending)) {
        ucg_group_progress(group);
    }

#if ENABLE_MT
    ucg_worker_h worker = group->worker;
#endif
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    unsigned idx;
    ucg_groups_t *gctx = UCG_WORKER_TO_GROUPS_CTX(group->worker);
    for (idx = 0; idx < gctx->num_planners; idx++) {
        ucg_plan_component_t *planc = gctx->planners[idx].plan_component;
        planc->destroy(group);
    }

    UCS_STATS_NODE_FREE(group->stats);
    ucs_list_del(&group->list);
    ucs_free(group);

    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
}

ucs_status_t ucg_request_check_status(void *request)
{
    ucg_request_t *req = (ucg_request_t*)request - 1;

    if (req->flags & UCG_REQUEST_COMMON_FLAG_COMPLETED) {
        ucs_assert(req->status != UCS_INPROGRESS);
        return req->status;
    }
    return UCS_INPROGRESS;
}

void ucg_request_cancel(ucg_worker_h worker, void *request) { }

void ucg_request_free(void *request) { }

ucs_status_t ucg_plan_select(ucg_group_h group, const char* planner_name,
                             const ucg_collective_params_t *params,
                             ucg_plan_component_t **planc_p)
{
    ucg_groups_t *ctx = UCG_WORKER_TO_GROUPS_CTX(group->worker);
    return ucg_plan_select_component(ctx->planners, ctx->num_planners,
            planner_name, &group->params, params, planc_p);
}

UCS_PROFILE_FUNC(ucs_status_t, ucg_collective_create,
        (group, params, coll), ucg_group_h group,
        ucg_collective_params_t *params, ucg_coll_h *coll)
{
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(group->worker);

    /* check the recycling/cache for this collective */
    ucg_op_t *op;
    ucs_status_t status;
    unsigned coll_mask = UCG_FLAG_MASK(params);
    ucg_plan_t *plan = group->cache[coll_mask];
    if (ucs_likely(plan != NULL)) {
        ucs_list_for_each(op, &plan->op_head, list) {
            if (!memcmp(&op->params, params, sizeof(*params))) {
                status = UCS_OK;
                goto op_found;
            }
        }

        UCS_STATS_UPDATE_COUNTER(group->stats, UCG_GROUP_STAT_PLANS_USED, 1);
        goto plan_found;
    }

    /* select which plan to use for this collective operation */
    ucg_plan_component_t *planc; // TODO: replace NULL with config value
    status = ucg_plan_select(group, NULL, params, &planc);
    if (status != UCS_OK) {
        goto out;
    }

    /* create the actual plan for the collective operation */
    UCS_PROFILE_CODE("ucg_plan") {
        ucs_trace_req("ucg_collective_create PLAN: planc=%s type=%x root=%lu",
                &planc->name[0], params->type.modifiers, (uint64_t)params->type.root);
        status = ucg_plan(planc, &params->type, group, &plan);
    }
    if (status != UCS_OK) {
        goto out;
    }

    plan->planner           = planc;
    plan->group             = group;
    plan->type              = params->type;
    plan->group_id          = group->group_id;
    plan->group_size        = group->params.member_count;
    plan->group_host_size   = group->worker->context->config.num_local_peers;
    plan->am_mp             = &group->worker->am_mp;
    group->cache[coll_mask] = plan;
    ucs_list_head_init(&plan->op_head);
    UCS_STATS_UPDATE_COUNTER(group->stats, UCG_GROUP_STAT_PLANS_CREATED, 1);

plan_found:
    UCS_STATS_UPDATE_COUNTER(group->stats, UCG_GROUP_STAT_OPS_CREATED, 1);
    UCS_PROFILE_CODE("ucg_prepare") {
        status = ucg_prepare(plan, params, &op);
    }
    if (status != UCS_OK) {
        goto out;
    }

    ucs_trace_req("ucg_collective_create OP: planc=%s "
            "params={type=%u, root=%lu, send=[%p,%i,%lu,%p,%p], "
            "recv=[%p,%i,%lu,%p,%p], cb=%p, op=%p}", &planc->name[0],
            (unsigned)params->type.modifiers, (uint64_t)params->type.root,
            params->send.buf, params->send.count, params->send.dt_len,
            params->send.dt_ext, params->send.displs,
            params->recv.buf, params->recv.count, params->recv.dt_len,
            params->recv.dt_ext, params->recv.displs,
            params->comp_cb, params->recv.op_ext);

    ucs_list_add_head(&plan->op_head, &op->list);
    memcpy(&op->params, params, sizeof(*params));
    op->plan = plan;

op_found:
    *coll = op;

out:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(group->worker);
    return status;
}

ucs_status_t static UCS_F_ALWAYS_INLINE
ucg_collective_trigger(ucg_group_h group, ucg_op_t *op, ucg_request_t **req)
{
    /* Barrier effect - all new collectives are pending */
    if (ucs_unlikely(op->params.type.modifiers & UCG_GROUP_COLLECTIVE_MODIFIER_BARRIER)) {
        ucs_assert(group->is_barrier_outstanding == 0);
        group->is_barrier_outstanding = 1;
    }

    /* Start the first step of the collective operation */
    ucs_status_t ret;
    UCS_PROFILE_CODE("ucg_trigger") {
        ret = ucg_trigger(op, group->next_id++, req);
    }

    if (ret != UCS_INPROGRESS) {
        UCS_STATS_UPDATE_COUNTER(group->stats, UCG_GROUP_STAT_OPS_IMMEDIATE, 1);
    }

    return ret;
}

ucs_status_t ucg_collective_release_barrier(ucg_group_h group)
{
    ucs_assert(group->is_barrier_outstanding);
    group->is_barrier_outstanding = 0;
    if (ucs_queue_is_empty(&group->pending)) {
        return UCS_OK;
    }

    ucs_status_t ret;
    do {
        /* Move the operation from the pending queue back to the original one */
        ucg_op_t *op = (ucg_op_t*)ucs_queue_pull_non_empty(&group->pending);
        ucg_request_t **req = op->pending_req;
        ucs_list_add_head(&op->plan->op_head, &op->list);

        /* Start this next pending operation */
        ret = ucg_collective_trigger(group, op, req);
    } while ((!ucs_queue_is_empty(&group->pending)) &&
             (!group->is_barrier_outstanding) &&
             (ret == UCS_OK));

    return ret;
}

ucs_status_t static UCS_F_ALWAYS_INLINE
ucg_collective_start(ucg_coll_h coll, ucg_request_t **req)
{
    ucs_status_t ret;
    ucg_op_t *op = (ucg_op_t*)coll;
    ucg_group_h group = op->plan->group;

    /* Since group was created - don't need UCP_CONTEXT_CHECK_FEATURE_FLAGS */
    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(group->worker);

    ucs_trace_req("ucg_collective_start: op=%p req=%p", coll, *req);

    if (ucs_unlikely(group->is_barrier_outstanding)) {
        ucs_list_del(&op->list);
        ucs_queue_push(&group->pending, &op->queue);
        op->pending_req = req;
        ret = UCS_INPROGRESS;
    } else {
        ret = ucg_collective_trigger(group, op, req);
    }

    UCS_STATS_UPDATE_COUNTER(group->stats, UCG_GROUP_STAT_OPS_USED, 1);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(group->worker);
    return ret;
}

UCS_PROFILE_FUNC(ucs_status_ptr_t, ucg_collective_start_nb,
                 (coll), ucg_coll_h coll)
{
    ucg_request_t *req = NULL;
    ucs_status_ptr_t ret = UCS_STATUS_PTR(ucg_collective_start(coll, &req));
    return UCS_PTR_IS_ERR(ret) ? ret : req;
}

UCS_PROFILE_FUNC(ucs_status_t, ucg_collective_start_nbr,
                 (coll, request), ucg_coll_h coll, void *request)
{
    return ucg_collective_start(coll, (ucg_request_t**)&request);
}

void ucg_collective_destroy(ucg_coll_h coll)
{
    ucg_discard((ucg_op_t*)coll);
}

static ucs_status_t ucg_worker_groups_init(ucp_worker_h worker,
        unsigned *next_am_id, void *groups_ctx)
{
    ucg_groups_t *gctx  = (ucg_groups_t*)groups_ctx;
    ucs_status_t status = ucg_plan_query(next_am_id, &gctx->planners, &gctx->num_planners);
    if (status != UCS_OK) {
        return status;
    }

    unsigned planner_idx;
    size_t group_ctx_offset  = sizeof(struct ucg_group);
    size_t global_ctx_offset = ucg_ctx_worker_offset + sizeof(ucg_groups_t);
    for (planner_idx = 0; planner_idx < gctx->num_planners; planner_idx++) {
        ucg_plan_desc_t* planner    = &gctx->planners[planner_idx];
        ucg_plan_component_t* planc = planner->plan_component;
        planc->global_ctx_offset    = global_ctx_offset;
        global_ctx_offset          += planc->global_context_size;
        planc->group_ctx_offset     = group_ctx_offset;
        group_ctx_offset           += planc->group_context_size;
    }

    gctx->next_id             = 0;
    gctx->iface_cnt           = 0;
    gctx->total_planner_sizes = group_ctx_offset;
    gctx->am_mp               = &worker->am_mp;
    gctx->num_local_peers     = worker->context->config.num_local_peers;
    gctx->my_local_peer_idx   = worker->context->config.my_local_peer_idx;
    kh_init_inplace(ucg_group_ep, &gctx->eps);
    ucs_list_head_init(&gctx->groups_head);
    return UCS_OK;
}

static void ucg_worker_groups_cleanup(void *groups_ctx)
{
    ucg_groups_t *gctx = (ucg_groups_t*)groups_ctx;

    ucg_group_h group, tmp;
    if (!ucs_list_is_empty(&gctx->groups_head)) {
        ucs_list_for_each_safe(group, tmp, &gctx->groups_head, list) {
            ucg_group_destroy(group);
        }
    }

    ucg_plan_release_list(gctx->planners, gctx->num_planners);
    kh_destroy_inplace(ucg_group_ep, &gctx->eps);
}

ucs_status_t ucg_init_version(unsigned api_major_version,
                              unsigned api_minor_version,
                              const ucg_params_t *params,
                              const ucg_config_t *config,
                              ucg_context_h *context_p)
{
    ucs_status_t status = ucp_init_version(api_major_version, api_minor_version,
                                           params, config, context_p);
    if (status == UCS_OK) {
        size_t ctx_size = sizeof(ucg_groups_t) +
                ucs_list_length(&ucg_plan_components_list) * sizeof(void*);
        status = ucp_extend(*context_p, ctx_size, ucg_worker_groups_init,
                ucg_worker_groups_cleanup, &ucg_ctx_worker_offset);
    }
    return status;
}

ucs_status_t ucg_init(const ucg_params_t *params,
                      const ucg_config_t *config,
                      ucg_context_h *context_p)
{
    ucs_status_t status = ucp_init(params, config, context_p);
    if (status == UCS_OK) {
        size_t ctx_size = sizeof(ucg_groups_t) +
                ucs_list_length(&ucg_plan_components_list) * sizeof(void*);
        status = ucp_extend(*context_p, ctx_size, ucg_worker_groups_init,
                ucg_worker_groups_cleanup, &ucg_ctx_worker_offset);
    }
    return status;
}

ucs_status_t ucg_plan_connect(ucg_group_h group,
                              ucg_group_member_index_t idx,
                              enum ucg_plan_connect_flags flags,
                              uct_ep_h *ep_p, const uct_iface_attr_t **ep_attr_p,
                              uct_md_h *md_p, const uct_md_attr_t    **md_attr_p)
{
    int ret;
    ucs_status_t status;
    size_t remote_addr_len;
    ucp_address_t *remote_addr = NULL;
    ucg_groups_t *gctx = UCG_WORKER_TO_GROUPS_CTX(group->worker);

    /* Look-up the UCP endpoint based on the index */
    ucp_ep_h ucp_ep;
    khiter_t iter = kh_get(ucg_group_ep, &gctx->eps, idx);
    if (iter != kh_end(&gctx->eps)) {
        /* Use the cached connection */
        ucp_ep = kh_value(&gctx->eps, iter);
    } else {
        /* fill-in UCP connection parameters */
        status = group->params.resolve_address_f(group->params.cb_group_obj,
                idx, &remote_addr, &remote_addr_len);
        if (status != UCS_OK) {
            ucs_error("failed to obtain a UCP endpoint from the external callback");
            return status;
        }

        /* special case: connecting to a zero-length address means it's "debugging" */
        if (ucs_unlikely(remote_addr_len == 0)) {
            *ep_p = NULL;
            return UCS_OK;
        }

        /* create an endpoint for communication with the remote member */
        ucp_ep_params_t ep_params = {
                .field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS,
                .address = remote_addr
        };
        status = ucp_ep_create(group->worker, &ep_params, &ucp_ep);
        group->params.release_address_f(remote_addr);
        if (status != UCS_OK) {
            return status;
        }

        /* Store this endpoint, for future reference */
        iter = kh_put(ucg_group_ep, &gctx->eps, idx, &ret);
        kh_value(&gctx->eps, iter) = ucp_ep;
    }

    /* Connect for point-to-point communication */
    ucp_lane_index_t lane;
am_retry:
    lane  = ucp_ep_get_am_lane(ucp_ep);
    *ep_p = ucp_ep_get_am_uct_ep(ucp_ep);
    if (*ep_p == NULL) {
        status = ucp_wireup_connect_remote(ucp_ep, lane);
        if (status != UCS_OK) {
            return status;
        }
    }

    if (ucp_proxy_ep_test(*ep_p)) {
        ucp_proxy_ep_t *proxy_ep = ucs_derived_of(*ep_p, ucp_proxy_ep_t);
        *ep_p = proxy_ep->uct_ep;
        ucs_assert(*ep_p != NULL);
    }

    ucs_assert((*ep_p)->iface != NULL);
    if ((*ep_p)->iface->ops.ep_am_short ==
            (typeof((*ep_p)->iface->ops.ep_am_short))
            ucs_empty_function_return_no_resource) {
        ucp_worker_progress(group->worker);
        goto am_retry;
    }

    if (flags & UCG_PLAN_CONNECT_FLAG_WANT_INCAST) {
        lane = ucp_ep_get_incast_lane(ucp_ep);
        if (lane == UCP_NULL_LANE) {
            return UCS_ERR_UNREACHABLE;
        }
        *ep_p = ucp_ep_get_incast_uct_ep(ucp_ep);
    } else if (flags & UCG_PLAN_CONNECT_FLAG_WANT_BCAST) {
        lane = ucp_ep_get_bcast_lane(ucp_ep);
        if (lane == UCP_NULL_LANE) {
            return UCS_ERR_UNREACHABLE;
        }
        *ep_p = ucp_ep_get_bcast_uct_ep(ucp_ep);
    }

    /* Register interfaces to be progressed in future calls */
    UCG_GROUP_PROGRESS_ADD((*ep_p)->iface, group);
    UCG_GROUP_PROGRESS_ADD((*ep_p)->iface, gctx);

    *md_p      = ucp_ep_md(ucp_ep, lane);
    *md_attr_p = ucp_ep_md_attr(ucp_ep, lane);
    *ep_attr_p = ucp_ep_get_iface_attr(ucp_ep, lane);

    return UCS_OK;
}

ucs_status_t ucg_plan_query_resources(ucg_group_h group,
                                      ucg_plan_resources_t **resources)
{
    return UCS_OK;
}
