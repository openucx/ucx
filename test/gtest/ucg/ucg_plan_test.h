/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_PLAN_TEST_H_
#define UCG_PLAN_TEST_H_

#include "ucg_test.h"

using namespace std;

/**
 * UCG_PLAN test.
 */
class ucg_plan_test : public ucg_test {
public:
    ucg_plan_test();

    ucg_plan_test(size_t node_cnt, size_t ppn, unsigned myrank);

    virtual ~ucg_plan_test();

    virtual void TestBody() {}

public:
    vector<ucg_rank_info> m_all_rank_infos;
    ucg_builtin_group_ctx_t *m_builtin_ctx;
    ucg_plan_component_t *m_planc;
    ucg_group_params_t *m_group_params;
    ucg_collective_type_t m_coll_type;
    ucg_collective_params_t *m_coll_params;
    ucg_group_h m_group;
};


typedef struct {
    size_t node_cnt;
    size_t ppn;
    unsigned myrank;
} ucg_plan_test_data_t;

typedef struct {
    ucg_plan_test_data_t data;
    int algo_id;
} ucg_plan_test_data_algo_t;


#endif