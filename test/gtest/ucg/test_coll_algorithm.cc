/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <ucg/builtin/plan/builtin_plan.h>
#include <ucg/builtin/ops/builtin_ops.h>
#include <ucg/api/ucg_plan_component.h>
}

#include "ucg_test.h"
#include "ucg_plan_test.h"

TEST_F(ucg_plan_test, test_ring) {

    ucg_plan_test example1(4, 8, 0);
    ucg_builtin_plan_t *plan;
    ucs_status_t ret = ucg_builtin_ring_create(example1.m_builtin_ctx, UCG_PLAN_RING,
                                               (ucg_builtin_config_t *) example1.m_planc->plan_config,
                                               example1.m_group_params, &example1.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);


    ucg_plan_test example2(4, 7, 0);
    ret = ucg_builtin_ring_create(example2.m_builtin_ctx, UCG_PLAN_RING,
                                  (ucg_builtin_config_t *) example2.m_planc->plan_config, example2.m_group_params,
                                  &example2.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example3(1, 2, 0);
    ret = ucg_builtin_ring_create(example3.m_builtin_ctx, UCG_PLAN_RING,
                                  (ucg_builtin_config_t *) example3.m_planc->plan_config, example3.m_group_params,
                                  &example3.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

}

TEST_F(ucg_plan_test, test_recursive) {

    size_t node_cnt = 2;
    size_t ppn = 2;
    ucg_builtin_plan_t *plan;
    ucs_status_t ret;
    unsigned idx;

    for (idx = 0; idx < (node_cnt * ppn); idx++) {
        ucg_plan_test example1(node_cnt, ppn, idx);
        ret = ucg_builtin_recursive_create(example1.m_builtin_ctx, UCG_PLAN_RECURSIVE,
                                                     (ucg_builtin_config_t *) example1.m_planc->plan_config,
                                                     example1.m_group_params, &example1.m_coll_type, &plan);
        ASSERT_EQ(UCS_OK, ret);
    }

    node_cnt = 1;
    ppn = 3;
    for (idx = 0; idx < (node_cnt * ppn); idx++) {
        ucg_plan_test example1(node_cnt, ppn, idx);
        ret = ucg_builtin_recursive_create(example1.m_builtin_ctx, UCG_PLAN_RECURSIVE,
                                                     (ucg_builtin_config_t *) example1.m_planc->plan_config,
                                                     example1.m_group_params, &example1.m_coll_type, &plan);
        ASSERT_EQ(UCS_OK, ret);
    }

}

TEST_F(ucg_plan_test, test_binomial_tree) {

    ucg_plan_test example1(4, 8, 0);
    ucg_builtin_plan_t *plan;
    ucs_status_t ret = ucg_builtin_binomial_tree_create(example1.m_builtin_ctx, UCG_PLAN_TREE_FANOUT,
                                                        (ucg_builtin_config_t *) example1.m_planc->plan_config,
                                                        example1.m_group_params, &example1.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    size_t node_cnt = 1;
    size_t ppn = 3;
    for (unsigned idx = 0; idx < (node_cnt * ppn); idx++) {
        ucg_plan_test example(node_cnt, ppn, idx);
        ret = ucg_builtin_binomial_tree_create(example.m_builtin_ctx, UCG_PLAN_TREE_FANOUT,
                                                        (ucg_builtin_config_t *) example.m_planc->plan_config,
                                                        example.m_group_params, &example.m_coll_type, &plan);
        ASSERT_EQ(UCS_OK, ret);
    }

    ucg_plan_test example2(4, 7, 0);
    ucg_algo.topo = 1;
    ret = ucg_builtin_binomial_tree_create(example2.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example2.m_planc->plan_config,
                                           example2.m_group_params, &example2.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example3(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 1;
    ret = ucg_builtin_binomial_tree_create(example3.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example3.m_planc->plan_config,
                                           example3.m_group_params, &example3.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example4(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 1;
    ucg_algo.kmtree = 1;
    ret = ucg_builtin_binomial_tree_create(example4.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example4.m_planc->plan_config,
                                           example4.m_group_params, &example4.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example5(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 1;
    ucg_algo.kmtree = 1;
    ret = ucg_builtin_binomial_tree_create(example5.m_builtin_ctx, UCG_PLAN_TREE_FANOUT,
                                           (ucg_builtin_config_t *) example5.m_planc->plan_config,
                                           example5.m_group_params, &example5.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example6(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 0;
    ucg_algo.kmtree = 1;
    ret = ucg_builtin_binomial_tree_create(example6.m_builtin_ctx, UCG_PLAN_TREE_FANOUT,
                                           (ucg_builtin_config_t *) example6.m_planc->plan_config,
                                           example6.m_group_params, &example6.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example7(3, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 0;
    ucg_algo.kmtree = 0;
    ret = ucg_builtin_binomial_tree_create(example7.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example7.m_planc->plan_config,
                                           example7.m_group_params, &example7.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example8(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 0;
    ucg_algo.kmtree = 0;
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;
    ret = ucg_builtin_binomial_tree_create(example8.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example8.m_planc->plan_config,
                                           example8.m_group_params, &example8.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example9(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 1;
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;  
    ret = ucg_builtin_binomial_tree_create(example9.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example9.m_planc->plan_config,
                                           example9.m_group_params, &example9.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example10(4, 8, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 1;
    ucg_algo.kmtree = 1;
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;   
    ret = ucg_builtin_binomial_tree_create(example10.m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                           (ucg_builtin_config_t *) example10.m_planc->plan_config,
                                           example10.m_group_params, &example10.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);

    ucg_plan_test example11(2, 1, 0);
    ucg_algo.topo = 1;
    ucg_algo.bmtree = 1;
    ucg_algo.recursive = 0;
    ucg_algo.kmtree_intra = 0;
    ucg_algo.kmtree = 0;
    ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_NODE;     
    ret = ucg_builtin_binomial_tree_create(example11.m_builtin_ctx, UCG_PLAN_TREE_FANOUT,
                                           (ucg_builtin_config_t *) example11.m_planc->plan_config,
                                           example11.m_group_params, &example11.m_coll_type, &plan);
    ASSERT_EQ(UCS_OK, ret);
}


