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

TEST(ucg_plan_test, test_ring) {
    ucs_status_t ret;
    ucg_plan_test *obj = NULL;
    ucg_builtin_plan_t *plan = NULL;
    ucg_plan_test_data_t data[] = {
        {4, 8, 0},
        {4, 7, 0},
        {1, 2, 0},
    };

    for (unsigned i = 0; i < sizeof(data) / sizeof(data[0]); i++) {
        obj = new ucg_plan_test(data[i].node_cnt, data[i].ppn, data[i].myrank);
        ret = ucg_builtin_ring_create(obj->m_builtin_ctx, UCG_PLAN_RING,
                                      (ucg_builtin_config_t *)obj->m_planc->plan_config,
                                      obj->m_group_params, &obj->m_coll_type, &plan);
        delete obj;

        ASSERT_EQ(UCS_OK, ret);
    }
}

TEST(ucg_plan_test, test_recursive) {
    ucs_status_t ret;
    ucg_plan_test *obj = NULL;
    ucg_builtin_plan_t *plan = NULL;
    ucg_plan_test_data_t data[] = {
        {2, 2, 0}, {2, 2, 1}, {2, 2, 2}, {2, 2, 3},
        {1, 3, 0}, {1, 3, 1}, {1, 3, 2},
    };

    for (unsigned i = 0; i < sizeof(data) / sizeof(data[0]); i++) {
        obj = new ucg_plan_test(data[i].node_cnt, data[i].ppn, data[i].myrank);
        ret = ucg_builtin_recursive_create(obj->m_builtin_ctx, UCG_PLAN_RECURSIVE,
                                           (ucg_builtin_config_t *)obj->m_planc->plan_config,
                                           obj->m_group_params, &obj->m_coll_type, &plan);
        delete obj;

        ASSERT_EQ(UCS_OK, ret);
    }
}

static void ucg_algo_set(int option)
{
    switch (option) {
    case 0:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 1;
        ucg_algo.kmtree = 1;
        return;

    case 1:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 0;
        ucg_algo.kmtree = 1;
        return;

    case 2:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 0;
        ucg_algo.kmtree = 0;
        ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_NODE;
        return;

    case 3:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 0;
        ucg_algo.kmtree = 0;
        return;

    case 4:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 0;
        ucg_algo.kmtree = 0;
        ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;
        return;

    case 5:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 1;
        ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;
        return;

    case 6:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 1;
        ucg_algo.kmtree = 1;
        ucg_algo.topo_level = UCG_GROUP_HIERARCHY_LEVEL_SOCKET;
        return;

    case 7:
        ucg_algo.topo = 1;
        return;

    case 8:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 1;
        return;

    case 9:
        ucg_algo.topo = 1;
        ucg_algo.bmtree = 1;
        ucg_algo.recursive = 0;
        ucg_algo.kmtree_intra = 1;
        ucg_algo.kmtree = 1;
        return;

    default:
        return;
    }
}

TEST(ucg_plan_test, test_binomial_tree) {
    unsigned i;
    ucs_status_t ret;
    ucg_plan_test *obj = NULL;
    ucg_builtin_plan_t *plan = NULL;
    ucg_plan_test_data_algo_t fanout[] = {
        {{4, 8, 0}, -1},
        {{1, 3, 0}, -1},
        {{1, 3, 1}, -1},
        {{1, 3, 2}, -1},
        {{4, 8, 0}, 0},
        {{4, 8, 0}, 1},
        {{2, 1, 0}, 2},
    };
    ucg_plan_test_data_algo_t fanin_out[] = {
        {{3, 8, 0}, 3},
        {{4, 8, 0}, 4},
        {{4, 8, 0}, 5},
        {{4, 8, 0}, 6},
        {{4, 7, 0}, 7},
        {{4, 8, 0}, 8},
        {{4, 8, 0}, 9},
    };

    for (i = 0; i < sizeof(fanout) / sizeof(fanout[0]); i++) {
        obj = new ucg_plan_test(fanout[i].data.node_cnt, fanout[i].data.ppn, fanout[i].data.myrank);
        ucg_algo_set(fanout[i].algo_id);
        ret = ucg_builtin_binomial_tree_create(obj->m_builtin_ctx, UCG_PLAN_TREE_FANOUT,
                                               (ucg_builtin_config_t *)obj->m_planc->plan_config,
                                               obj->m_group_params, &obj->m_coll_type, &plan);
        delete obj;

        ASSERT_EQ(UCS_OK, ret);
    }

    for (i = 0; i < sizeof(fanin_out) / sizeof(fanin_out[0]); i++) {
        obj = new ucg_plan_test(fanin_out[i].data.node_cnt, fanin_out[i].data.ppn, fanin_out[i].data.myrank);
        ucg_algo_set(fanin_out[i].algo_id);
        ret = ucg_builtin_binomial_tree_create(obj->m_builtin_ctx, UCG_PLAN_TREE_FANIN_FANOUT,
                                               (ucg_builtin_config_t *)obj->m_planc->plan_config,
                                               obj->m_group_params, &obj->m_coll_type, &plan);
        delete obj;

        ASSERT_EQ(UCS_OK, ret);
    }
}

