/*
 * Copyright (C) Huawei Technologies Co., Ltd. 2019-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucg_test.h"

using namespace std;

class ucg_collective_test : public ucg_test {
public:
    ucg_collective_test();

    virtual ~ucg_collective_test();

protected:
    vector<ucg_rank_info> m_all_rank_infos;
    ucg_group_params_t *m_group_params;
    ucg_group_params_t *m_group2_params;
    ucg_collective_params_t *m_coll_params;
    ucg_group_h m_group;
    ucg_group_h m_group2;
};

ucg_collective_test::ucg_collective_test()
{
    m_all_rank_infos.clear();
    m_resource_factory->create_balanced_rank_info(m_all_rank_infos, 2, 2);
    m_group_params = m_resource_factory->create_group_params(m_all_rank_infos[0], m_all_rank_infos);
    m_group2_params = m_resource_factory->create_group_params(m_all_rank_infos[1], m_all_rank_infos);
    m_group = m_resource_factory->create_group(m_group_params, m_ucg_worker);
    m_group2 = m_resource_factory->create_group(m_group2_params, m_ucg_worker);
    m_coll_params = m_resource_factory->create_collective_params(
                                            UCG_GROUP_COLLECTIVE_MODIFIER_SINGLE_SOURCE,
                                            0, NULL, 1, NULL, 4, NULL, NULL);
}

ucg_collective_test::~ucg_collective_test()
{
    if (m_coll_params != NULL) {
        delete m_coll_params;
        m_coll_params = NULL;
    }

    ucg_group_destroy(m_group2);
    ucg_group_destroy(m_group);

    if (m_group2_params != NULL) {
        delete m_group2_params;
        m_group2_params = NULL;
    }

    if (m_group_params != NULL) {
        delete m_group_params;
        m_group_params = NULL;
    }

    m_all_rank_infos.clear();
}


TEST_F(ucg_collective_test, test_collective_create) {
    ucg_coll_h coll = NULL;

    ucs_status_t ret = ucg_collective_create(m_group, m_coll_params, &coll);

    ASSERT_EQ(UCS_OK, ret);
}

TEST_F(ucg_collective_test, test_collective_start_nb) {
    ucg_coll_h coll = NULL;

    ucs_status_t retC = ucg_collective_create(m_group2, m_coll_params, &coll);
    EXPECT_EQ(UCS_OK, retC);

    ucs_status_ptr_t retP = ucg_collective_start_nb(coll);

    ASSERT_TRUE(retP != NULL);
}

TEST_F(ucg_collective_test, test_collective_start_nbr) {
    ucg_request_t *req = NULL;
    ucg_coll_h coll = NULL;

    ucs_status_t retC = ucg_collective_create(m_group2, m_coll_params, &coll);
    EXPECT_EQ(UCS_OK, retC);

    ucg_collective_start_nbr(coll, req);

    //ASSERT_EQ(UCS_OK, retS);
}

TEST_F(ucg_collective_test, test_collective_destroy) {
    ucg_coll_h coll = NULL;

    ucs_status_t ret = ucg_collective_create(m_group, m_coll_params, &coll);
    EXPECT_EQ(UCS_OK, ret);

    //TODO
    ASSERT_TRUE(true);
}
