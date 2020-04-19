/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <infiniband/verbs.h>
extern "C" {
#include <uct/ib/mlx5/ib_mlx5.h>
}
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

        if (!(md()->super.dev.flags & UCT_IB_DEVICE_FLAG_MLX5_PRM &&
              md()->flags & UCT_IB_MLX5_MD_FLAG_DEVX)) {
            std::stringstream ss;
            ss << "DEVX is not supported by " << GetParam();
            UCS_TEST_SKIP_R(ss.str());
        }
    }

    uct_ib_mlx5_md_t *md() {
        return ucs_derived_of(m_e->md(), uct_ib_mlx5_md_t);
    }

    uct_priv_worker_t *worker() {
        return ucs_derived_of(m_e->worker(), uct_priv_worker_t);
    }
};

UCS_TEST_P(test_devx, dbrec)
{
    uct_ib_mlx5_dbrec_t *dbrec;

    dbrec = (uct_ib_mlx5_dbrec_t *)ucs_mpool_get_inline(&md()->dbrec_pool);
    ASSERT_FALSE(dbrec == NULL);
    ucs_mpool_put_inline(dbrec);
}

UCT_INSTANTIATE_IB_TEST_CASE(test_devx);
