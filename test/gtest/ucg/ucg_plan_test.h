/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCG_PLAN_TEST_H_
#define UCG_PLAN_TEST_H_

#include "ucg_test.h"

using namespace std;

class ucg_plan_resource_factory;

/**
 * UCG_PLAN test.
 */
class ucg_plan_test : public ::ucg_test {

public:
    ucg_plan_test() {}

    ucg_plan_test(size_t node_cnt, size_t ppn, unsigned myrank) {

        vector<ucg_rank_info> all_rank_infos = m_resource_factory->create_balanced_rank_info(node_cnt, ppn);

        m_group = m_resource_factory->create_group(all_rank_infos[myrank], all_rank_infos, m_ucg_worker);
        m_coll_params = m_resource_factory->create_collective_params(UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
                                                                     0, NULL, 1, NULL, 4, NULL, NULL);

        m_group_params = m_resource_factory->create_group_params(all_rank_infos[myrank], all_rank_infos);

        m_coll_type.modifiers = UCG_GROUP_COLLECTIVE_MODIFIER_AGGREGATE;
        m_coll_type.root = 0;

        ucg_plan_select(m_group, NULL, m_coll_params, &m_planc);
        m_builtin_ctx = (ucg_builtin_group_ctx_t *) UCG_GROUP_TO_COMPONENT_CTX(ucg_builtin_component, m_group);
    }

    ~ucg_plan_test() {}

    virtual void TestBody() {}

protected:

public:
    ucg_builtin_group_ctx_t *m_builtin_ctx;
    ucg_plan_component_t *m_planc;
    ucg_group_params_t *m_group_params;
    ucg_collective_type_t m_coll_type;
    ucg_collective_params_t *m_coll_params;
    ucg_group_h m_group;
};

class ucg_plan_resource_factory {
public:;
};

#endif



