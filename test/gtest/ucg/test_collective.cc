/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucg_test.h"

using namespace std;

class ucg_collective_test : public ucg_test {
protected:
    ucg_collective_test() {
        init();
    }

    ~ucg_collective_test() {}

    void init() {
        vector<ucg_rank_info> all_rank_infos = m_resource_factory->create_balanced_rank_info(2, 2);

        m_group = m_resource_factory->create_group(all_rank_infos[0], all_rank_infos, m_ucg_worker);
        m_group2 = m_resource_factory->create_group(all_rank_infos[1], all_rank_infos, m_ucg_worker);

        m_coll_params = m_resource_factory->create_collective_params(
                UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE, 0, NULL, 1,
                NULL, 4, NULL, NULL);
    }

protected:
    ucg_group_h m_group;
    ucg_group_h m_group2;
    ucg_collective_params_t *m_coll_params;
};

TEST_F(ucg_collective_test, test_collective_create) {
    ucg_coll_h coll = NULL;

    ucs_status_t ret = ucg_collective_create(m_group, m_coll_params, &coll);
    ucg_collective_destroy(coll);

    ASSERT_EQ(UCS_OK, ret);
}

TEST_F(ucg_collective_test, test_collective_start_nb) {
    ucg_coll_h coll = NULL;

    ucs_status_t retC = ucg_collective_create(m_group2, m_coll_params, &coll);
    EXPECT_EQ(UCS_OK, retC);

    ucs_status_ptr_t retP = ucg_collective_start_nb(coll);
    ucg_collective_destroy(coll);

    ASSERT_TRUE(retP != NULL);
}

TEST_F(ucg_collective_test, test_collective_start_nbr) {
    ucg_request_t *req = NULL;
    ucg_coll_h coll = NULL;

    ucs_status_t retC = ucg_collective_create(m_group2, m_coll_params, &coll);
    EXPECT_EQ(UCS_OK, retC);

    ucg_collective_start_nbr(coll, req);
    ucg_collective_destroy(coll);

    //ASSERT_EQ(UCS_OK, retS);
}

TEST_F(ucg_collective_test, test_collective_destroy) {
    ucg_coll_h coll = NULL;

    ucs_status_t ret = ucg_collective_create(m_group, m_coll_params, &coll);
    EXPECT_EQ(UCS_OK, ret);

    ucg_collective_destroy(coll);
    //TODO
    ASSERT_TRUE(true);
}
