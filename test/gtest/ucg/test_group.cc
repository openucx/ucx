/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucg_test.h"

using namespace std;

class ucg_group_test : public ucg_test {
public:
    ucg_group_test() {
        m_all_rank_infos.clear();
        m_resource_factory->create_balanced_rank_info(m_all_rank_infos, 2, 2);
        m_params = m_resource_factory->create_group_params(m_all_rank_infos[0], m_all_rank_infos);
    }

    ~ucg_group_test() {
        if (m_params != NULL) {
            delete m_params;
            m_params = NULL;
        }
        m_all_rank_infos.clear();
    }

protected:
    vector<ucg_rank_info> m_all_rank_infos;
    ucg_group_params_t *m_params;
};

TEST_F(ucg_group_test, test_group_create) {
    ucg_group_h group;
    ucs_status_t ret = ucg_group_create(m_ucg_worker, m_params, &group);
    ucg_group_destroy(group);

    ASSERT_EQ(UCS_OK, ret);
}

TEST_F(ucg_group_test, test_group_destroy) {
    ucg_group_h group;
    ucs_status_t ret = ucg_group_create(m_ucg_worker, m_params, &group);
    EXPECT_EQ(UCS_OK, ret);

    ucg_group_destroy(group);
    //TODO
    //ASSERT_TRUE(true);
}

TEST_F(ucg_group_test, test_group_progress) {
    ucg_group_h group;
    ucs_status_t retS = ucg_group_create(m_ucg_worker, m_params, &group);
    EXPECT_EQ(UCS_OK, retS);

    unsigned retU = ucg_group_progress(group);
    ucg_group_destroy(group);
    //TODO
    cout << "ucg_group_progress return: " << retU << endl;
    ASSERT_TRUE(true) << "ucg_group_progress return " << retU;
}
