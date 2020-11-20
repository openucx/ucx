/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "test_op.h"

using namespace std;

ucg_op_test::ucg_op_test()
{
    num_procs = 4;
};

ucg_op_test::~ucg_op_test()
{
};

ucg_builtin_plan_phase_t *ucg_op_test::create_phase(ucg_builtin_plan_method_type method)
{
    ucg_builtin_plan_phase_t *phase = new ucg_builtin_plan_phase_t();

    init_phase(method, phase);

    return phase;
}

void ucg_op_test::init_phase(ucg_builtin_plan_method_type method, ucg_builtin_plan_phase_t *phase)
{
    uct_md_attr_t *md_attr = new uct_md_attr_t();

    md_attr->cap.max_reg = 8128;
    phase->md_attr = md_attr;

    phase->md = NULL;
    phase->single_ep = NULL;
    phase->multi_eps = NULL;

    phase->method = method;
    phase->ep_cnt = 1;
    phase->step_index = 1;

    phase->send_thresh.max_short_one = 32;
    phase->send_thresh.max_short_max = 64;
    phase->send_thresh.max_bcopy_one = 128;
    phase->send_thresh.max_bcopy_max = 256;
    phase->send_thresh.max_zcopy_one = 1024;

    phase->recv_thresh.max_short_one = phase->send_thresh.max_short_one;
    phase->recv_thresh.max_short_max = phase->send_thresh.max_short_max;
    phase->recv_thresh.max_bcopy_one = phase->send_thresh.max_bcopy_one;
    phase->recv_thresh.max_bcopy_max = phase->send_thresh.max_bcopy_max;
    phase->recv_thresh.max_zcopy_one = phase->send_thresh.max_zcopy_one;
    phase->recv_thresh.md_attr_cap_max_reg = 8128;
}

void ucg_op_test::destroy_phase(ucg_builtin_plan_phase_t *phase)
{
    if (phase != NULL) {
        if (phase->md_attr != NULL) {
            delete phase->md_attr;
            phase->md_attr = NULL;
        }

        delete phase;
    }
}

ucg_plan_t *ucg_op_test::create_plan(unsigned phs_cnt, ucg_collective_params_t *params, ucg_group_h group)
{
    ucg_builtin_plan_t *builtin_plan = (ucg_builtin_plan_t *)malloc(sizeof(ucg_builtin_plan_t) +
                                                                    phs_cnt * sizeof(ucg_builtin_plan_phase_t));
    ucg_plan_t *plan = &builtin_plan->super;
    ucg_plan_component_t *planc = NULL;

    plan->group_id = 1;
    plan->my_index = 0;
    plan->group = group;
    ucg_plan_select(group, NULL, params, &planc);
    plan->planner = planc;
    builtin_plan->am_id = 1;
    builtin_plan->resend = NULL;
    builtin_plan->slots = NULL;
    builtin_plan->phs_cnt = phs_cnt;

    ucs_mpool_ops_t ops = {
            ucs_mpool_chunk_malloc,
            ucs_mpool_chunk_free,
            NULL,
            NULL
    };
    size_t op_size = sizeof(ucg_builtin_op_t) + builtin_plan->phs_cnt * sizeof(ucg_builtin_op_step_t);

    ucs_mpool_init(&builtin_plan->op_mp, 0, op_size, 0, UCS_SYS_CACHE_LINE_SIZE,
                   1, UINT_MAX, &ops, "ucg_builtin_plan_mp");
    ucs_mpool_grow(&builtin_plan->op_mp, 1);

    for (unsigned i = 0; i < phs_cnt; i++) {
        init_phase(UCG_PLAN_METHOD_RECV_TERMINAL, &builtin_plan->phss[i]);
    }

    return (ucg_plan_t *)builtin_plan;
}

ucg_builtin_plan_t *ucg_op_test::create_method_plan(ucg_builtin_plan_method_type method)
{
    ucg_builtin_plan_t *builtin_plan = (ucg_builtin_plan_t *)malloc(sizeof(ucg_builtin_plan_t) +
                                                                    sizeof(ucg_builtin_plan_phase_t));

    init_phase(method, &builtin_plan->phss[0]);

    return builtin_plan;
}

ucg_group_h ucg_op_test::create_group()
{
    ucg_rank_info my_rank_info = {.rank = 0, .nodex_idx = 0, .socket_idx = 0};
    ucg_rank_info other_rank_info = {.rank = 1, .nodex_idx = 0, .socket_idx = 0};

    vector<ucg_rank_info> all_rank_infos;
    all_rank_infos.push_back(my_rank_info);
    all_rank_infos.push_back(other_rank_info);
    ucg_group_params_t *group_params = m_resource_factory->create_group_params(my_rank_info, all_rank_infos);

    return m_resource_factory->create_group(group_params, m_ucg_worker);
}

ucs_status_t mem_reg_mock(uct_md_h md, void *address, size_t length, unsigned flags, uct_mem_h *memh_p)
{
    // do nothing
    return UCS_OK;
}

uct_md_h ucg_op_test::create_md()
{
    uct_md_h md = new uct_md;
    uct_md_ops_t *ops = new uct_md_ops_t;

    ops->mem_reg = mem_reg_mock;
    md->ops = ops;
    return md;
}

/**
 * Test: ucg_builtin_op_create
 */

TEST_F(ucg_op_test, test_op_create_phase_1) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_bcast_params();
    ucg_plan_t *plan = create_plan(1, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);

    ASSERT_EQ(UCS_OK, ret);
    ucg_builtin_component.destroy(group);
}

TEST_F(ucg_op_test, test_op_create_phase_2) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_allreduce_params();
    ucg_plan_t *plan = create_plan(2, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);

    ASSERT_EQ(UCS_OK, ret);
    ucg_builtin_component.destroy(group);
}

/**
 * Test: ucg_builtin_op_discard
 */

TEST_F(ucg_op_test, test_op_discard) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_allreduce_params();
    ucg_plan_t *plan = create_plan(1, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);

    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_op_t *builtin_op = (ucg_builtin_op_t *) op;
    ucg_builtin_op_step_t *step = &builtin_op->steps[0];

    step->flags |= UCG_BUILTIN_OP_STEP_FLAG_PIPELINED;

    ucg_builtin_op_discard(op);
    ucg_builtin_component.destroy(group);
}

/**
 * Test: ucg_builtin_op_trigger
 */

TEST_F(ucg_op_test, test_op_trigger) {
    ucg_group_h group = create_group();
    ucg_collective_params_t *params = create_allreduce_params();
    ucg_plan_t *plan = create_plan(1, params, group);

    ucg_op_t *op = new ucg_op_t();

    ucs_status_t ret = ucg_builtin_op_create(plan, params, &op);

    ASSERT_EQ(UCS_OK, ret);

    ucg_builtin_op *builtin_op = (ucg_builtin_op_t *) op;
    ucg_builtin_comp_slot_t slot = *new ucg_builtin_comp_slot_t;
    slot.cb = NULL;
    ucs_list_head_init(&slot.msg_head);

    builtin_op->slots = &slot;

    ucg_request_t *request = new ucg_request_t;
    op->params = *params;

    ret = ucg_builtin_op_trigger(op, 0, &request);
    ASSERT_EQ(UCS_INPROGRESS, ret);
    ucg_builtin_component.destroy(group);
    m_ucg_worker = NULL;
    m_ucg_context = NULL;
}
