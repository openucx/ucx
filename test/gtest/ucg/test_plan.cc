/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucg_test.h"


using namespace std;

TEST_F(ucg_test, select_plan_component) {
    vector<ucg_rank_info> all_rank_infos;

    m_resource_factory->create_balanced_rank_info(all_rank_infos, 4, 8);
    ucg_group_params_t *group_params = m_resource_factory->create_group_params(all_rank_infos[0], all_rank_infos);
    ucg_group_h group = m_resource_factory->create_group(group_params, m_ucg_worker);
    ucg_collective_params_t *params = m_resource_factory->create_collective_params(
                                                UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
                                                0, NULL, 1, NULL, 4, NULL, NULL);
    ucg_plan_component_t *planc = NULL;
    ucs_status_t ret = ucg_plan_select(group, NULL, params, &planc);

    ASSERT_EQ(UCS_OK, ret);
    ASSERT_TRUE(planc != NULL);

    delete params;
    ucg_group_destroy(group);
    delete group_params;
    all_rank_infos.clear();
}

TEST_F(ucg_test, create_plan) {
    // vector<ucg_rank_info> all_rank_infos = m_resource_factory->create_balanced_rank_info(4,8);
    // ucg_group_h group = m_resource_factory->create_group(all_rank_infos[0],all_rank_infos,m_ucg_worker);
    // ucg_collective_params_t *params = create_allreduce_params();
    // ucg_collective_type_t coll_type = create_allreduce_coll_type();
    // size_t msg_size = 32;

    // ucg_plan_t *plan;
    // ucs_status_t ret = ucg_builtin_component.plan(&ucg_builtin_component, &coll_type, msg_size, group, params, &plan);

    // ASSERT_EQ(UCS_OK, ret);
    // ASSERT_TRUE(plan!=NULL);
}

TEST_F(ucg_test, create_plan_component) {
    vector<ucg_rank_info> all_rank_infos;

    m_resource_factory->create_balanced_rank_info(all_rank_infos, 4, 8);
    ucg_group_params_t *group_params = m_resource_factory->create_group_params(all_rank_infos[0], all_rank_infos);
    ucg_group_h group = m_resource_factory->create_group(group_params, m_ucg_worker);
    unsigned base_am_id = 23;
    ucg_group_id_t group_id = 0;

    ucs_status_t ret = ucg_builtin_component.create(&ucg_builtin_component, m_ucg_worker, group,
                                                    base_am_id, group_id, NULL, group_params);
    ASSERT_EQ(UCS_OK, ret);

    ucg_group_destroy(group);
    delete group_params;
    all_rank_infos.clear();
}

TEST_F(ucg_test, destroy_group) {
    vector<ucg_rank_info> all_rank_infos;

    m_resource_factory->create_balanced_rank_info(all_rank_infos, 4, 8);
    ucg_group_params_t *group_params = m_resource_factory->create_group_params(all_rank_infos[0], all_rank_infos);
    ucg_group_h group = m_resource_factory->create_group(group_params, m_ucg_worker);
    ucg_builtin_component.destroy(group);
    ASSERT_TRUE(true);

    ucg_group_destroy(group);
    delete group_params;
    all_rank_infos.clear();
}

TEST_F(ucg_test, progress_group) {
    vector<ucg_rank_info> all_rank_infos;

    m_resource_factory->create_balanced_rank_info(all_rank_infos, 4, 8);
    ucg_group_params_t *group_params = m_resource_factory->create_group_params(all_rank_infos[0], all_rank_infos);
    ucg_group_h group = m_resource_factory->create_group(group_params, m_ucg_worker);

    unsigned ret = ucg_builtin_component.progress(group);
    //TODO how to judge progress result?
    cout << "ucg_builtin_component.progress return: " << ret << endl;

    ucg_group_destroy(group);
    delete group_params;
    all_rank_infos.clear();
}

TEST_F(ucg_test, query_plan) {
    ucg_plan_desc_t *planners;
    unsigned num_plans;
    ucs_status_t ret = ucg_builtin_component.query(0, &planners, &num_plans);

    ASSERT_EQ(UCS_OK, ret);
    ASSERT_EQ((unsigned) 1, num_plans);
    ASSERT_STREQ("builtin", planners[0].plan_name);
}
