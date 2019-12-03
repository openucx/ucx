/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <string.h>
#include <ucs/debug/memtrack.h>
#include <ucg/api/ucg_plan_component.h>
#include <ucs/profile/profile.h>

#include "ops/builtin_ops.h"
#include "plan/builtin_plan.h"
#include <ucg/api/ucg_mpi.h>

#define UCG_BUILTIN_SUPPORT_MASK (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE |\
                                  UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST)

static ucs_config_field_t ucg_builtin_config_table[] = {
    {"PLAN_", "", NULL, ucs_offsetof(ucg_builtin_config_t, super),
     UCS_CONFIG_TYPE_TABLE(ucg_plan_config_table)},

    {"TREE_", "", NULL, ucs_offsetof(ucg_builtin_config_t, tree),
     UCS_CONFIG_TYPE_TABLE(ucg_builtin_tree_config_table)},

    {"BMTREE_", "", NULL, ucs_offsetof(ucg_builtin_config_t, bmtree),
     UCS_CONFIG_TYPE_TABLE(ucg_builtin_binomial_tree_config_table)},

    {"RECURSIVE_", "", NULL, ucs_offsetof(ucg_builtin_config_t, recursive),
     UCS_CONFIG_TYPE_TABLE(ucg_builtin_recursive_config_table)},

    {"NEIGHBOR_", "", NULL, ucs_offsetof(ucg_builtin_config_t, neighbor),
     UCS_CONFIG_TYPE_TABLE(ucg_builtin_neighbor_config_table)},

    {"CACHE_SIZE", "1000", "Number of cached collective operations",
     ucs_offsetof(ucg_builtin_config_t, cache_size), UCS_CONFIG_TYPE_UINT},

    {"SHORT_MAX_TX_SIZE", "256", "Largest send operation to use short messages",
     ucs_offsetof(ucg_builtin_config_t, short_max_tx), UCS_CONFIG_TYPE_MEMUNITS},

    {"BCOPY_MAX_TX_SIZE", "32768", "Largest send operation to use buffer copy",
     ucs_offsetof(ucg_builtin_config_t, bcopy_max_tx), UCS_CONFIG_TYPE_MEMUNITS},

    {"MEM_REG_OPT_CNT", "10", "Operation counter before registering the memory",
     ucs_offsetof(ucg_builtin_config_t, mem_reg_opt_cnt), UCS_CONFIG_TYPE_ULUNITS},

    {NULL}
};

extern ucg_plan_component_t ucg_builtin_component;
struct ucg_builtin_group_ctx {
    ucs_list_link_t           send_head;    /* request list for (re)send */

    ucg_group_h               group;
    const ucg_group_params_t *group_params;
    ucg_group_id_t            group_id;
    uint16_t                  am_id;
    ucs_list_link_t           plan_head;    /* for resource release */
    ucg_builtin_config_t     *config;

    ucg_builtin_comp_slot_t   slots[UCG_BUILTIN_MAX_CONCURRENT_OPS];
};

typedef struct ucg_builtin_ctx {
    unsigned slots_total;
    unsigned slots_used;
    ucg_builtin_comp_slot_t *slots[];
} ucg_builtin_ctx_t;

/*
 *
 */
static ucs_status_t ucg_builtin_query(unsigned ucg_api_version,
        ucg_plan_desc_t **desc_p, unsigned *num_descs_p)
{
    ucs_status_t status              = ucg_plan_single(&ucg_builtin_component,
                                                       desc_p, num_descs_p);
    (*desc_p)[0].modifiers_supported = UCG_BUILTIN_SUPPORT_MASK;
    (*desc_p)[0].flags = 0;
    return status;
}

static enum ucg_builtin_plan_topology_type ucg_builtin_choose_type(enum ucg_collective_modifiers flags)
{
    if (flags & UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE) {
        return UCG_PLAN_TREE_FANOUT;
    }

    if (flags & UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION) {
        return UCG_PLAN_TREE_FANIN;
    }

    if (flags & UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE) {
        if (RECURSIVE)
            return UCG_PLAN_RECURSIVE;
        else
            return UCG_PLAN_TREE_FANIN_FANOUT;
    }

    if (flags & ucg_predefined_modifiers[UCG_PRIMITIVE_ALLTOALL]) {
        return UCG_PLAN_BRUCK;
    }

    if (flags & UCG_GROUP_COLLECTIVE_MODIFIER_ALLGATHER) {
        if (BRUCK)
            return UCG_PLAN_BRUCK;
        else
            return UCG_PLAN_RECURSIVE;
    }

    return UCG_PLAN_TREE_FANIN_FANOUT;
}

UCS_PROFILE_FUNC(ucs_status_t, ucg_builtin_am_handler,
                 (arg, data, length, am_flags),
                 void *arg, void *data, size_t length, unsigned am_flags)
{
    ucg_builtin_header_t* header  = data;
    ucg_builtin_ctx_t **ctx       = UCG_WORKER_TO_COMPONENT_CTX(ucg_builtin_component, arg);
    ucg_builtin_comp_slot_t *slot = &(*ctx)->slots[header->group_id]
        [header->coll_id % UCG_BUILTIN_MAX_CONCURRENT_OPS];
    ucs_assert(header->group_id < (*ctx)->slots_total);
    ucs_assert(length >= sizeof(header));

    /* Consume the message if it fits the current collective and step index */
    ucs_assert((slot->coll_id != header->coll_id) ||
               (slot->step_idx <= header->step_idx));
    if (ucs_likely(slot->cb && (header->local_id == slot->local_id))) {
        /* Make sure the packet indeed belongs to the collective currently on */
//        ucs_assert((header->remote_offset + length -
//               sizeof(ucg_builtin_header_t)) <= slot->req.step->buffer_length);
//        ucs_assert((length == sizeof(ucg_builtin_header_t)) ||
//                   (length - sizeof(ucg_builtin_header_t) == slot->req.step->buffer_length) ||
//                   ((length - sizeof(ucg_builtin_header_t) <= slot->req.step->fragment_length) &&
//                    (slot->req.step->fragments_recv >= 1)));

        ucs_trace_req("ucg_builtin_am_handler CB: coll_id %u step_idx %u cb %p pending %u",
                header->coll_id, header->step_idx, slot->cb, slot->req.pending);

        /* The packet arrived "on time" - process it */
        UCS_PROFILE_CODE("ucg_builtin_am_handler_cb") {
            (void) slot->cb(&slot->req, header->remote_offset,
                            data + sizeof(ucg_builtin_header_t),
                            length - sizeof(ucg_builtin_header_t));
        }
        return UCS_OK;
    }

    /* Store the message - use RX_headroom for @ref ucg_builtin_comp_desc_t */
    ucs_status_t ret;
    ucg_builtin_comp_desc_t* desc = NULL;
    if (am_flags & UCT_CB_PARAM_FLAG_DESC) {
        desc = (ucg_builtin_comp_desc_t*)((char*)data -
                offsetof(ucg_builtin_comp_desc_t, header));
        ret = UCS_INPROGRESS;
    } else {
        /* Cannot use existing descriptor - must allocate my own... */
        desc = (ucg_builtin_comp_desc_t*)ucs_mpool_get_inline(slot->mp);
        ucs_assert(desc != NULL); //here desc should not be null
        memcpy(&desc->header, data, length);
        ret = UCS_OK;
    }

    ucs_trace_req("ucg_builtin_am_handler STORE: group_id %u coll_id %u(%u) step_idx %u(%u)",
            header->group_id, header->coll_id, slot->coll_id, header->step_idx, slot->step_idx);

    desc->super.flags  = am_flags;
    desc->super.length = length - sizeof(ucg_builtin_header_t);
    ucs_list_add_tail(&slot->msg_head, &desc->super.tag_list[0]);
    return ret;
}

static void ucg_builtin_msg_dump(ucp_worker_h worker, uct_am_trace_type_t type,
                                 uint8_t id, const void *data, size_t length,
                                 char *buffer, size_t max)
{
    const ucg_builtin_header_t *header = (const ucg_builtin_header_t*)data;
    snprintf(buffer, max, "COLLECTIVE [coll_id %u step_idx %u offset %lu length %lu]",
             (unsigned)header->coll_id, (unsigned)header->step_idx,
             (uint64_t)header->remote_offset, length - sizeof(*header));
}

static ucs_status_t ucg_builtin_create(ucg_plan_component_t *plan_component,
                                       ucg_worker_h worker,
                                       ucg_group_h group,
                                       unsigned base_am_id,
                                       ucg_group_id_t group_id,
                                       ucs_mpool_t *group_am_mp,
                                       const ucg_group_params_t *group_params)
{
    /* Create or expand the per-worker context - for the AM-handler's sake */
    ucg_builtin_ctx_t **bctx =
            UCG_WORKER_TO_COMPONENT_CTX(ucg_builtin_component, worker);
    if ((ucs_unlikely(*bctx == NULL)) ||
        (ucs_likely((*bctx)->slots_total <= group_id))) {
        void *temp = *bctx;
        size_t bctx_size = sizeof(**bctx) + ((group_id + 1) * sizeof(void*));
        *bctx = ucs_realloc(temp, bctx_size, "builtin_context");
        if (ucs_unlikely(*bctx == NULL)) {
            *bctx = (ucg_builtin_ctx_t *)temp;
            return UCS_ERR_NO_MEMORY;
        }

        (*bctx)->slots_total = group_id + 1;
        if (temp == NULL) {
            (*bctx)->slots_used = 0;
        }
    } else {
        (*bctx)->slots_used++;
    }

    /* Fill in the information in the per-group context */
    ucg_builtin_group_ctx_t *gctx =
            UCG_GROUP_TO_COMPONENT_CTX(ucg_builtin_component, group);
    ucg_builtin_mpi_reduce_cb     = group_params->mpi_reduce_f;
    gctx->group                   = group;
    gctx->group_id                = group_id;
    gctx->group_params            = group_params;
    gctx->config                  = (ucg_builtin_config_t *)plan_component->plan_config;
    gctx->am_id                   = base_am_id;
    ucs_list_head_init(&gctx->send_head);
    ucs_list_head_init(&gctx->plan_head);

    // TODO: only do this once...
    ucp_am_handler_t* am_handler  = ucp_am_handlers + base_am_id;
    am_handler->features          = UCP_FEATURE_GROUPS;
    am_handler->cb                = ucg_builtin_am_handler;
    am_handler->tracer            = ucg_builtin_msg_dump;
    am_handler->flags             = 0;

    int i;
    for (i = 0; i < UCG_BUILTIN_MAX_CONCURRENT_OPS; i++) {
        ucs_list_head_init(&gctx->slots[i].msg_head);
        gctx->slots[i].mp       = group_am_mp;
        gctx->slots[i].cb       = NULL;
        gctx->slots[i].coll_id  = i;
        gctx->slots[i].step_idx = 0;
    }

    /* Link the two contexts */
    (*bctx)->slots[group_id] = gctx->slots;
    return UCS_OK;
}

static void ucg_builtin_destroy(ucg_group_h group)
{
    ucg_builtin_group_ctx_t *gctx =
            UCG_GROUP_TO_COMPONENT_CTX(ucg_builtin_component, group);

    int i;
    for (i = 0; i < UCG_BUILTIN_MAX_CONCURRENT_OPS; i++) {
        if (gctx->slots[i].cb != NULL) {
            ucs_warn("Collective operation #%u has been left incomplete (Group #%u)",
                    gctx->slots[i].coll_id, gctx->group_id);
        }

        while (!ucs_list_is_empty(&gctx->slots[i].msg_head)) {
            ucg_builtin_comp_desc_t *desc =
                    ucs_list_extract_head(&gctx->slots[i].msg_head,
                            ucg_builtin_comp_desc_t, super.tag_list[0]);
            ucs_warn("Collective operation #%u has %u bytes left pending for step #%u (Group #%u)",
                    desc->header.coll_id, desc->super.length, desc->header.step_idx, desc->header.group_id);
            if (desc->super.flags == UCT_CB_PARAM_FLAG_DESC) {
                uct_iface_release_desc(desc);
            } else {
                ucs_mpool_put_inline(desc);
            }
        }
    }

    while (!ucs_list_is_empty(&gctx->plan_head)) {
        ucg_builtin_plan_t *plan = ucs_list_extract_head(&gctx->plan_head,
                ucg_builtin_plan_t, list);

        while (!ucs_list_is_empty(&plan->super.op_head)) {
            ucg_op_t *op = ucs_list_extract_head(&plan->super.op_head, ucg_op_t, list);
            ucg_builtin_op_discard(op);
        }

        ucs_mpool_cleanup(&plan->op_mp, 1);
        ucs_free(plan);
    }
}

static unsigned ucg_builtin_progress(ucg_group_h group)
{
    ucg_builtin_group_ctx_t *gctx =
            UCG_GROUP_TO_COMPONENT_CTX(ucg_builtin_component, group);
    if (ucs_likely(ucs_list_is_empty(&gctx->send_head))) {
        return 0;
    }

    /*
     * Since calling @ref ucg_builtin_step_execute may place the operation in
     * the same list again, the list of pending sends is moved to a temporary
     * head, then drained - each call "resets" the state of that operation.
     */
    unsigned ret = 0;
    UCS_LIST_HEAD(temp_head);
    ucs_list_splice_tail(&temp_head, &gctx->send_head);
    ucs_list_head_init(&gctx->send_head);
    while (!ucs_list_is_empty(&temp_head)) {
        ucg_builtin_request_t *req = ucs_list_extract_head(&temp_head,
                ucg_builtin_request_t, send_list);
        ucs_status_t status = ucg_builtin_step_execute(req, NULL);
        if (status != UCS_INPROGRESS) {
            ret++;
        }
    }
    return ret;
}

ucs_mpool_ops_t ucg_builtin_plan_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucs_empty_function,
    .obj_cleanup   = ucs_empty_function
};

static ucs_status_t ucg_builtin_plan(ucg_plan_component_t *plan_component,
                                     const ucg_collective_type_t *coll_type,
                                     ucg_group_h group,
                                     ucg_plan_t **plan_p)
{
    ucg_builtin_config_t *config = (ucg_builtin_config_t *)plan_component->plan_config;
    //TODO: global variable of Binomial tree selection(should be moved to algorithm selection module!!!)
    BMTREE    = 1;
    RECURSIVE = 1;
    BRUCK     = 1;  //Just for the allgather operation judgement, need enable support non power of two
    PIPELINE  = config->pipelining;
    
    ucs_status_t status;
    ucg_builtin_plan_t *plan=NULL;
    ucg_builtin_group_ctx_t *builtin_ctx =
            UCG_GROUP_TO_COMPONENT_CTX(ucg_builtin_component, group);

    enum ucg_builtin_plan_topology_type plan_topo_type =
            ucg_builtin_choose_type(coll_type->modifiers);

    /* Build the topology according to the requested */
    switch(plan_topo_type) {
    case UCG_PLAN_RECURSIVE:
        status = ucg_builtin_recursive_create(builtin_ctx, plan_topo_type,
                (const ucg_builtin_config_t *)plan_component->plan_config, builtin_ctx->group_params, coll_type, &plan);
        break;

    case UCG_PLAN_BRUCK:
        status = ucg_builtin_bruck_create(builtin_ctx, plan_topo_type,
                (const ucg_builtin_config_t *)plan_component->plan_config, builtin_ctx->group_params, coll_type, &plan);
        break;

    default:
        //TODO :add binomial tree support for other MPI operation
        if (BMTREE == 1)
            status = ucg_builtin_binomial_tree_create(builtin_ctx, plan_topo_type,
                    (const ucg_builtin_config_t *)plan_component->plan_config, builtin_ctx->group_params, coll_type, &plan);
        else
            status = ucg_builtin_tree_create(builtin_ctx, plan_topo_type,
                    (const ucg_builtin_config_t *)plan_component->plan_config, builtin_ctx->group_params, coll_type, &plan);
    }

    if (status != UCS_OK) {
        return status;
    }

    /* Create a memory-pool for operations for this plan */
    size_t op_size = sizeof(ucg_builtin_op_t) + plan->phs_cnt * sizeof(ucg_builtin_op_step_t);
    status = ucs_mpool_init(&plan->op_mp, 0, op_size, 0, UCS_SYS_CACHE_LINE_SIZE,
            1, UINT_MAX, &ucg_builtin_plan_mpool_ops, "ucg_builtin_plan_mp");
    if (status != UCS_OK) {
        return status;
    }

    ucs_list_add_head(&builtin_ctx->plan_head, &plan->list);
    plan->resend    = &builtin_ctx->send_head;
    plan->slots     = &builtin_ctx->slots[0];
    plan->am_id     = builtin_ctx->am_id;
    *plan_p         = (ucg_plan_t*)plan;
    return UCS_OK;
}

static void ucg_builtin_print(ucg_plan_t *plan, const ucg_collective_params_t *coll_params)
{
    ucs_status_t status;
    ucg_builtin_plan_t *builtin_plan = (ucg_builtin_plan_t*)plan;
    printf("Planner:    %s\n", builtin_plan->super.planner->name);
    printf("Endpoints:  %i\n", builtin_plan->ep_cnt);
    printf("Phases:     %i\n", builtin_plan->phs_cnt);

    printf("Object memory size:\n");
    printf("\tPer-group context: %lu bytes\n", sizeof(ucg_builtin_group_ctx_t));
    printf("\tPlan: %lu bytes\n", sizeof(ucg_builtin_plan_t) +
            builtin_plan->phs_cnt * sizeof(ucg_builtin_plan_phase_t) +
            builtin_plan->ep_cnt * sizeof(uct_ep_h));
    printf("\tOperation: %lu bytes (%lu per step)\n", sizeof(ucg_builtin_op_t) +
            builtin_plan->phs_cnt * sizeof(ucg_builtin_op_step_t),
            sizeof(ucg_builtin_op_step_t));
    printf("\tRequest: %lu bytes\n", sizeof(ucg_builtin_request_t));
    printf("\tSlot: %lu bytes\n", sizeof(ucg_builtin_comp_slot_t));

    unsigned phase_idx;
    for (phase_idx = 0; phase_idx < builtin_plan->phs_cnt; phase_idx++) {
        printf("Phase #%u: ", phase_idx);
        printf("the method is ");
        switch (builtin_plan->phss[phase_idx].method) {
        case UCG_PLAN_METHOD_SEND_TERMINAL:
            printf("Send (T), ");
            break;
        case UCG_PLAN_METHOD_RECV_TERMINAL:
            printf("Recv (T), ");
            break;
        case UCG_PLAN_METHOD_BCAST_WAYPOINT:
            printf("Bcast (W), ");
            break;
        case UCG_PLAN_METHOD_SCATTER_TERMINAL:
            printf("Scatter (T), ");
            break;
        case UCG_PLAN_METHOD_SCATTER_WAYPOINT:
            printf("Scatter (W), ");
            break;
        case UCG_PLAN_METHOD_GATHER_WAYPOINT:
            printf("Gather (W), ");
            break;
        case UCG_PLAN_METHOD_REDUCE_TERMINAL:
            printf("Reduce (T), ");
            break;
        case UCG_PLAN_METHOD_REDUCE_WAYPOINT:
            printf("Reduce (W), ");
            break;
        case UCG_PLAN_METHOD_REDUCE_RECURSIVE:
            printf("Reduce (R), ");
            break;
        case UCG_PLAN_METHOD_NEIGHBOR:
            printf("Neighbors, ");
            break;
        case UCG_PLAN_METHOD_ALLGATHER_RECURSIVE:
            printf("Allgather Recursive method, ");
            break;
        case UCG_PLAN_METHOD_ALLGATHER_BRUCK:
            printf("Bruck (ALLGATHER), ");
            break;
        case UCG_PLAN_METHOD_ALLTOALL_BRUCK:
            printf("Bruck (ALLTOALL), ");
            break;
        }

#if ENABLE_DEBUG_DATA || ENABLE_FAULT_TOLERANCE
        printf("with the following peers: ");
        unsigned peer_idx;
        ucg_builtin_plan_phase_t *phase = &builtin_plan->phss[phase_idx];
        uct_ep_h *ep = (phase->ep_cnt == 1) ? &phase->single_ep : phase->multi_eps;
        for (peer_idx = 0;
             peer_idx < phase->ep_cnt;
             peer_idx++, ep++) {
            printf("%lu,", phase->indexes[peer_idx]);
        }
        printf("\n");
#else
        printf("no peer info (configured without \"--enable-debug-data\")");
#endif

        if (coll_params) {
            int flags = 0;
            if (phase_idx == 0) {
                flags |= UCG_BUILTIN_OP_STEP_FLAG_FIRST_STEP;
            }
            if (phase_idx == (builtin_plan->phs_cnt - 1)) {
                flags |= UCG_BUILTIN_OP_STEP_FLAG_LAST_STEP;
            }

            int8_t *temp_buffer = NULL;
            ucg_builtin_op_step_t step;
            step.fragment_pending = NULL;
            printf("Step #%u (actual index used: %u):", phase_idx,
                    builtin_plan->phss[phase_idx].step_index);
            status = ucg_builtin_step_create(&builtin_plan->phss[phase_idx],
                    flags, 0, plan->group_id, coll_params, &temp_buffer, &step);
            if (status != UCS_OK) {
                printf("failed to create, %s", ucs_status_string(status));
            }

            printf("\n\tBuffer Length: %lu", step.buffer_length);
            if (step.flags & UCG_BUILTIN_OP_STEP_FLAG_FRAGMENTED) {
                printf("\n\tFragment Length: %lu", step.fragment_length);
                printf("\n\tFragment Count: %u", step.fragments);
            }

            int flag;
            printf("\n\tFlags:");
            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_RECV1_BEFORE_SEND) != 0);
            printf("\n\t\t(Pre-)RECV1: %i", flag);
            if (flag)
                printf(" (buffer: %s)", strlen((char*)step.recv_buffer) ?
                        (char*)step.recv_buffer : "temp-buffer");

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_RECV_BEFORE_SEND1) != 0);
            printf("\n\t\t(Pre-)RECVn: %i", flag);
            if (flag)
                printf(" (buffer: %s)", strlen((char*)step.recv_buffer) ?
                        (char*)step.recv_buffer : "temp-buffer");

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_SEND_AM_SHORT) ||
                    (step.flags & UCG_BUILTIN_OP_STEP_FLAG_SEND_AM_BCOPY) ||
                    (step.flags & UCG_BUILTIN_OP_STEP_FLAG_SEND_AM_ZCOPY));
            printf("\n\t\t      SEND: %i", flag);
            if (flag)
                printf(" (buffer: %s)", strlen((char*)step.send_buffer) ?
                        (char*)step.send_buffer : "temp-buffer");

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_RECV_AFTER_SEND) != 0);
            printf("\n\t\t(Post)RECV: %i", flag);
            if (flag)
                printf(" (buffer: %s)", strlen((char*)step.recv_buffer) ?
                        (char*)step.recv_buffer : "temp-buffer");

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_SINGLE_ENDPOINT) != 0);
            printf("\n\t\tSINGLE_ENDPOINT: %i", flag);

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_LENGTH_PER_REQUEST) != 0);
            printf("\n\t\tLENGTH_PER_REQUEST: %i", flag);

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_FRAGMENTED) != 0);
            printf("\n\t\tFRAGMENTED: %i", flag);

            flag = ((step.flags & UCG_BUILTIN_OP_STEP_FLAG_PIPELINED) != 0);
            printf("\n\t\tPIPELINED: %i", flag);

            printf("\n\n");
        }
    }
}

ucs_status_t ucg_builtin_connect(ucg_builtin_group_ctx_t *ctx,
        ucg_group_member_index_t idx, ucg_builtin_plan_phase_t *phase,
        unsigned phase_ep_index)
{
    uct_ep_h ep;
    ucs_status_t status = ucg_plan_connect(ctx->group, idx, &ep,
            &phase->ep_attr, &phase->md, &phase->md_attr);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

#if ENABLE_DEBUG_DATA || ENABLE_FAULT_TOLERANCE
    phase->indexes[(phase_ep_index != UCG_BUILTIN_CONNECT_SINGLE_EP) ?
            phase_ep_index : 0] = idx;
#endif
    if (!ep) {
        phase->max_short_one = UCS_CONFIG_MEMUNITS_INF;
        phase->md = NULL;
        return UCS_OK;
    }

    if (phase_ep_index == UCG_BUILTIN_CONNECT_SINGLE_EP) {
        phase->single_ep = ep;
    } else {
        /* 
         * Only avoid for case of Bruck plan because phase->ep_cnt = 1 
         * with 2 endpoints(send + recv) actually
         */
        if (phase->method != UCG_PLAN_METHOD_ALLGATHER_BRUCK &&
            phase->method != UCG_PLAN_METHOD_ALLTOALL_BRUCK)
            ucs_assert(phase_ep_index < phase->ep_cnt);
        phase->multi_eps[phase_ep_index] = ep;
    }

    /* Set the thresholds */
    phase->max_short_one = phase->ep_attr->cap.am.max_short - sizeof(ucg_builtin_header_t);
    phase->max_short_max = ctx->config->short_max_tx;
    // TODO: support UCS_CONFIG_MEMUNITS_AUTO
    if (phase->max_short_one > phase->max_short_max) {
    	phase->max_short_one = phase->max_short_max;
    }

    phase->max_bcopy_one = phase->ep_attr->cap.am.max_bcopy - sizeof(ucg_builtin_header_t);
    phase->max_bcopy_max = ctx->config->bcopy_max_tx;
    // TODO: support UCS_CONFIG_MEMUNITS_AUTO
    if (phase->md_attr->cap.max_reg) {
    	if (phase->max_bcopy_one > phase->max_bcopy_max) {
    		phase->max_bcopy_one = phase->max_bcopy_max;
    	}
    	phase->max_zcopy_one = phase->ep_attr->cap.am.max_zcopy - sizeof(ucg_builtin_header_t);
    } else {
    	// TODO: issue a warning?
    	phase->max_zcopy_one = phase->max_bcopy_max = UCS_CONFIG_MEMUNITS_INF;
    }
    return status;
}

UCG_PLAN_COMPONENT_DEFINE(ucg_builtin_component, "builtin",
                          sizeof(ucg_builtin_group_ctx_t), ucg_builtin_query,
                          ucg_builtin_create, ucg_builtin_destroy,
                          ucg_builtin_progress, ucg_builtin_plan,
                          ucg_builtin_op_create, ucg_builtin_op_trigger,
                          ucg_builtin_op_discard, ucg_builtin_print, "BUILTIN_",
                          ucg_builtin_config_table, ucg_builtin_config_t);
