/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <ucg/builtin/plan/builtin_plan.h>
}
#include "ucg_test.h"
#include "ucg_plan_test.h"


ucg_plan_test::ucg_plan_test()
{
    m_builtin_ctx = NULL;
    m_planc = NULL;
    m_group_params = NULL;
    m_coll_params = NULL;
    m_group = NULL;
    m_all_rank_infos.clear();
}

ucg_plan_test::ucg_plan_test(size_t node_cnt, size_t ppn, unsigned myrank)
{
    m_planc = NULL;
    m_all_rank_infos.clear();
    m_resource_factory->create_balanced_rank_info(m_all_rank_infos, node_cnt, ppn);
    m_group_params = m_resource_factory->create_group_params(m_all_rank_infos[myrank], m_all_rank_infos);
    m_group = m_resource_factory->create_group(m_group_params, m_ucg_worker);
    m_coll_params = m_resource_factory->create_collective_params(UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
                                                                 0, NULL, 1, NULL, 4, NULL, NULL);
    m_coll_type.modifiers = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE;
    m_coll_type.root = 0;
    ucg_plan_select(m_group, NULL, m_coll_params, &m_planc);
    m_builtin_ctx = (ucg_builtin_group_ctx_t *)UCG_GROUP_TO_COMPONENT_CTX(ucg_builtin_component, m_group);
}

ucg_plan_test::~ucg_plan_test()
{
    if (m_coll_params != NULL) {
        delete m_coll_params;
        m_coll_params = NULL;
    }

    ucg_group_destroy(m_group);

    if (m_group_params != NULL) {
        delete m_group_params;
        m_group_params = NULL;
    }

    m_all_rank_infos.clear();
}


TEST(ucg_plan_test, ucg_plan_1_test) {
    ucg_plan_test example(4, 8, 0);
    size_t msg_size = UCG_GROUP_MED_MSG_SIZE - 1024;

    ucg_plan_t *plan = NULL;
    ucs_status_t ret = ucg_builtin_component.create(&ucg_builtin_component, example.m_ucg_worker,
                                                    example.m_group, 23, 0, NULL, example.m_group_params);
    EXPECT_EQ(UCS_OK, ret);
    ret = ucg_builtin_component.plan(&ucg_builtin_component, &example.m_coll_type, msg_size,
                                     example.m_group, example.m_coll_params, &plan);
    EXPECT_EQ(UCS_OK, ret);

}

TEST(ucg_plan_test, ucg_plan_2_test) {
    ucg_plan_test example(4, 8, 0);
    size_t msg_size = UCG_GROUP_MED_MSG_SIZE + 1024;

    ucg_plan_t *plan = NULL;
    ucs_status_t ret = ucg_builtin_component.create(&ucg_builtin_component, example.m_ucg_worker,
                                                    example.m_group, 23, 0, NULL, example.m_group_params);
    EXPECT_EQ(UCS_OK, ret);
    ret = ucg_builtin_component.plan(&ucg_builtin_component, &example.m_coll_type, msg_size,
                                     example.m_group, example.m_coll_params, &plan);
    EXPECT_EQ(UCS_OK, ret);
}

TEST(ucg_plan_test, ucg_plan_3_test) {
    ucg_plan_test example(4, 8, 0);
    size_t msg_size = UCG_GROUP_MED_MSG_SIZE - 1024;

    ucg_plan_t *plan = NULL;
    ucs_status_t ret = ucg_builtin_component.create(&ucg_builtin_component, example.m_ucg_worker,
                                                    example.m_group, 23, 0, NULL, example.m_group_params);
    EXPECT_EQ(UCS_OK, ret);
    ret = ucg_builtin_component.plan(&ucg_builtin_component, &example.m_coll_type, msg_size,
                                     example.m_group, example.m_coll_params, &plan);
    EXPECT_EQ(UCS_OK, ret);
}
/*
TEST(ucg_plan_test, ucg_plan_4_test) {
    ucg_plan_test example(4, 8, 0);
    size_t msg_size = UCG_GROUP_MED_MSG_SIZE + 1024;

    ucg_plan_t *plan;
    ucs_status_t ret = ucg_builtin_component.create(&ucg_builtin_component, example.m_ucg_worker,
                                                    example.m_group, 23, 0, NULL, example.m_group_params);
    EXPECT_EQ(UCS_OK, ret);
    ret = ucg_builtin_component.plan(&ucg_builtin_component, &example.m_coll_type, msg_size,
                                     example.m_group, example.m_coll_params, &plan);
    EXPECT_EQ(UCS_OK, ret);
}
*/
TEST(ucg_plan_test, algorithm_selection) {
    ucs_status_t ret;
    unsigned idx;
    for (idx = 0; idx < UCG_ALGORITHM_ALLREDUCE_LAST; idx++) {
        ret = ucg_builtin_allreduce_algo_switch((enum ucg_builtin_allreduce_algorithm) idx, &ucg_algo);
        ASSERT_EQ(UCS_OK, ret);
    }

    for (idx = 0; idx < UCG_ALGORITHM_BARRIER_LAST; idx++) {
        ret = ucg_builtin_barrier_algo_switch((enum ucg_builtin_barrier_algorithm) idx, &ucg_algo);
        ASSERT_EQ(UCS_OK, ret);
    }

    for (idx = 0; idx < UCG_ALGORITHM_BCAST_LAST; idx++) {
        ret = ucg_builtin_bcast_algo_switch((enum ucg_builtin_bcast_algorithm) idx, &ucg_algo);
        ASSERT_EQ(UCS_OK, ret);
    }

}

TEST(ucg_plan_test, topo_level) {
    ucs_status_t ret;
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_NODE;
    enum ucg_group_member_distance domain_distance = UCG_GROUP_MEMBER_DISTANCE_SELF;
    ret = choose_distance_from_topo_aware_level(&domain_distance);
    ASSERT_EQ(UCS_OK, ret);
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;
    ret = choose_distance_from_topo_aware_level(&domain_distance);
    ASSERT_EQ(UCS_OK, ret);
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_L3CACHE;
    ret = choose_distance_from_topo_aware_level(&domain_distance);
    ASSERT_EQ(UCS_OK, ret);
}

TEST(ucg_plan_test, check_continus_number) {
    ucg_group_params_t group_params;
    group_params.member_count = 4;
    group_params.topo_map = (char **)malloc(sizeof(char *) * group_params.member_count);
    group_params.topo_map[0] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_SELF, UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_NET, UCG_GROUP_MEMBER_DISTANCE_NET};
    group_params.topo_map[1] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_SELF, UCG_GROUP_MEMBER_DISTANCE_NET, UCG_GROUP_MEMBER_DISTANCE_NET};
    group_params.topo_map[2] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_NET, UCG_GROUP_MEMBER_DISTANCE_NET, UCG_GROUP_MEMBER_DISTANCE_SELF, UCG_GROUP_MEMBER_DISTANCE_HOST};
    group_params.topo_map[3] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_NET, UCG_GROUP_MEMBER_DISTANCE_NET, UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_SELF};
    unsigned discount = 0;
    ucs_status_t status = ucg_builtin_check_continuous_number(&group_params, UCG_GROUP_MEMBER_DISTANCE_HOST, &discount);
    ASSERT_EQ(UCS_OK, status);
    ASSERT_EQ(0u, discount);

    group_params.topo_map[0] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_SELF, UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_SOCKET};
    group_params.topo_map[1] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_SELF, UCG_GROUP_MEMBER_DISTANCE_SOCKET, UCG_GROUP_MEMBER_DISTANCE_HOST};
    group_params.topo_map[2] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_SOCKET, UCG_GROUP_MEMBER_DISTANCE_SELF, UCG_GROUP_MEMBER_DISTANCE_HOST};
    group_params.topo_map[3] = new char[4] {UCG_GROUP_MEMBER_DISTANCE_SOCKET, UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_HOST, UCG_GROUP_MEMBER_DISTANCE_SELF};
    discount = 0;
    status = ucg_builtin_check_continuous_number(&group_params, UCG_GROUP_MEMBER_DISTANCE_SOCKET, &discount);
    ASSERT_EQ(UCS_OK, status);
    ASSERT_EQ(1u, discount);
}

TEST(ucg_plan_test, choose_type) {

    enum ucg_collective_modifiers flags[] = \
            { UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE, UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_DESTINATION, \
            UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE, UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE, UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE, \
            UCG_GROUP_COLLECTIVE_MODIFIER_ALLGATHER, UCG_GROUP_COLLECTIVE_MODIFIER_ALLGATHER};

    enum ucg_builtin_plan_topology_type expect_result[] = {UCG_PLAN_TREE_FANOUT, UCG_PLAN_TREE_FANIN, \
            UCG_PLAN_RECURSIVE, UCG_PLAN_RING, UCG_PLAN_TREE_FANIN_FANOUT, \
            UCG_PLAN_BRUCK, UCG_PLAN_RECURSIVE};
    enum ucg_builtin_plan_topology_type  ret_type;
    /* TODO */
    unsigned case_num = 7;
    for (unsigned i = 0; i < case_num; i++) {

        switch (i)
        {
        case 2:
            ucg_algo.recursive = 1;
            ucg_algo.ring      = 0;
            ucg_algo.bruck     = 0;
            break;
        case 3:
            ucg_algo.recursive = 0;
            ucg_algo.ring      = 1;
            ucg_algo.bruck     = 0;
            break;
        case 5:
            ucg_algo.recursive = 0;
            ucg_algo.ring      = 0;
            ucg_algo.bruck     = 1;
            break;
        default:
            ucg_algo.recursive = 0;
            ucg_algo.ring      = 0;
            ucg_algo.bruck     = 0;
            break;
        }

        ret_type = ucg_builtin_choose_type(flags[i]);
        ASSERT_EQ(expect_result[i], ret_type);
    }
}

/* TODO: add verification to below functions */
/*
TEST(ucg_plan_test, plan_decision_in_discontinuous_case) {
    ucg_plan_test example(2, 2, 0);
    unsigned op_num = 3;
    enum ucg_collective_modifiers modifiers[op_num] = { (enum ucg_collective_modifiers ) (UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST | UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE), \
        (enum ucg_collective_modifiers) (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST | UCG_GROUP_COLLECTIVE_MODIFIER_BARRIER), \
        (enum ucg_collective_modifiers) (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST) };
    unsigned size_num = 2;
    size_t msg_size[size_num] = {UCG_GROUP_MED_MSG_SIZE - 10, UCG_GROUP_MED_MSG_SIZE + 10};
    for (unsigned i = 0; i < op_num; i++) {
        for (unsigned j = 0; j < size_num; j++) {
            ucg_builtin_plan_decision_in_discontinuous_case(msg_size[j], example.m_group_params, modifiers[i], example.m_coll_params);
        }
    }
}
*/
TEST(ucg_plan_test, plan_decision_fixed) {
    ucg_plan_test example(2, 2, 0);
    unsigned op_num = 3;
    enum ucg_collective_modifiers modifiers[op_num] = { (enum ucg_collective_modifiers ) (UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST | UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE), \
        (enum ucg_collective_modifiers) (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST | UCG_GROUP_COLLECTIVE_MODIFIER_BARRIER), \
        (enum ucg_collective_modifiers) (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST) };
    unsigned size_num = 2;
    size_t msg_size[size_num] = {UCG_GROUP_MED_MSG_SIZE - 10, UCG_GROUP_MED_MSG_SIZE + 10};
    unsigned data_num = 2;
    unsigned large_data[data_num] = {100, 10000};
    example.m_coll_params->send.dt_len = 200;
    enum ucg_builtin_bcast_algorithm bcast_algo_decision;
    enum ucg_builtin_allreduce_algorithm allreduce_algo_decision;
    enum ucg_builtin_barrier_algorithm barrier_algo_decision;
    for (unsigned i = 0; i < op_num; i++) {
        for (unsigned j = 0; j < size_num; j++) {
            for (unsigned k = 0; k < data_num; k++) {
                plan_decision_fixed(msg_size[j], example.m_group_params, modifiers[i], example.m_coll_params, large_data[k], 0, &bcast_algo_decision, &allreduce_algo_decision, &barrier_algo_decision);
            }
        }
    }
}

TEST(ucg_plan_test, plan_chooose_ops) {
    ucg_plan_test example(2, 2, 0);
    unsigned op_num = 3;
    enum ucg_collective_modifiers modifiers[op_num] = { (enum ucg_collective_modifiers ) (UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST | UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE), \
        (enum ucg_collective_modifiers) (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST | UCG_GROUP_COLLECTIVE_MODIFIER_BARRIER), \
        (enum ucg_collective_modifiers) (UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE | UCG_GROUP_COLLECTIVE_MODIFIER_BROADCAST) };

    for (unsigned i = 0; i < op_num; i++) {
            ucg_builtin_plan_choose_ops(example.m_planc, modifiers[i]);
    }
}

TEST(ucg_plan_test, test_algorithm_decision) {
    ucg_plan_test example(2, 2, 0);
    ucs_status_t ret = ucg_builtin_algorithm_decision(&(example.m_coll_type), 1024, example.m_group_params, example.m_coll_params, example.m_planc);
    ASSERT_EQ(UCS_OK, ret);
}
