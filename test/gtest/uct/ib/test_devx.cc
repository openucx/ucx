/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/api/uct.h>
#include <uct/uct_test.h>
#include <common/test.h>

class test_devx : public uct_test {
public:
    entity* m_e;

    void init() {
        uct_test::init();

        m_e = create_entity(0);
        m_entities.push_back(m_e);

        if (!(md()->super.dev.flags & UCT_IB_DEVICE_FLAG_DEVX)) {
            std::stringstream ss;
            ss << "DEVX is not supported by " << GetParam();
            UCS_TEST_SKIP_R(ss.str());
        }
    }

    uct_ib_mlx5_md_t *md() {
        return ucs_derived_of(m_e->md(), uct_ib_mlx5_md_t);
    }
};

UCT_INSTANTIATE_IB_TEST_CASE(test_devx);
