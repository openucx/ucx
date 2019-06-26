/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_rc.h"

#include <uct/uct_test.h>
#include <common/test.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/rc/accel/rc_mlx5.h>
#include <uct/ib/dc/dc_mlx5.h>
#include <uct/ib/dc/dc_mlx5_ep.h>
}

class test_mlx5 : public test_rc {
public:

    virtual void init()
    {
        ASSERT_TRUE(has_transport("dc_mlx5") || has_transport("rc_mlx5"));
        set_config("RC_TX_QUEUE_LEN=32");

        // Disable MEMIC to use maximum allowed BBs with inline sends
        set_config("RC_DM_COUNT=0");
        test_rc::init();
    }

    uct_rc_mlx5_iface_common_t* rc_mlx5_iface(entity *e)
    {
        return ucs_derived_of(e->iface(), uct_rc_mlx5_iface_common_t);
    }

    unsigned get_wqe_num(entity *e)
    {
        uct_rc_txqp_t *txqp;
        uct_ib_mlx5_txwq_t *txwq;

        if (has_transport("rc_mlx5")) {
            txqp = &ucs_derived_of(e->ep(0), uct_rc_mlx5_ep_t)->super.txqp;
            txwq = &ucs_derived_of(e->ep(0), uct_rc_mlx5_ep_t)->tx.wq;
        } else {
            uct_dc_mlx5_iface_t *iface = ucs_derived_of(e->iface(),
                                                        uct_dc_mlx5_iface_t);
            uint8_t dci = ucs_derived_of(e->ep(0), uct_dc_mlx5_ep_t)->dci;

            txqp = &iface->tx.dcis[dci].txqp;
            txwq = &iface->tx.dci_wqs[dci];
        }

        return uct_rc_mlx5_common_txqp_wqes_num(txwq,
                                                uct_rc_txqp_available(txqp));
    }

    unsigned cqe_available(entity *e)
    {
        return rc_mlx5_iface(m_e1)->super.tx.cq_available;
    }

    ucs_status_t send_some_diff_len(entity *e, int num = 3)
    {
        EXPECT_TRUE(is_caps_supported(UCT_IFACE_FLAG_AM_SHORT));

        size_t max_short = e->iface_attr().cap.am.max_short - sizeof(uint64_t);
        mapped_buffer sendbuf(max_short, 0ul, *e);

        for (int i = 0; i < num; ++i) {
            size_t length = (ucs::rand() % UCT_IB_MLX5_MAX_BB) * MLX5_SEND_WQE_BB;
            size_t slen   = ucs_min(length, max_short);

            ucs_status_t status = uct_ep_am_short(e->ep(0), 0, 0ul,
                                                  sendbuf.ptr(), slen);
            if (status != UCS_OK) {
                return status;
            }
        }

        return UCS_OK;
    }

    void test_get_wqe_num(int pre_sends_num, bool fill_qp = false)
    {
        if (pre_sends_num) {
            // Move txwq PI by some value
            send_some_diff_len(m_e1, pre_sends_num);
            flush();
        }

        unsigned cq_init = cqe_available(m_e1);

        if (fill_qp) {
            ucs_status_t status;
            do {
                status = send_some_diff_len(m_e1, 16);
            } while (!UCS_STATUS_IS_ERR(status));
        } else {
            send_some_diff_len(m_e1);
        }

        EXPECT_EQ(cq_init, get_wqe_num(m_e1) + cqe_available(m_e1));
        flush();
    }

};

UCS_TEST_P(test_mlx5, wqe_num)
{
    test_get_wqe_num(3);
}

UCS_TEST_P(test_mlx5, wqe_num_from_start)
{
    test_get_wqe_num(0);
}

UCS_TEST_P(test_mlx5, wqe_num_qp_full)
{
    test_get_wqe_num(5, true);
}

UCS_TEST_P(test_mlx5, wqe_num_from_start_qp_full)
{
    test_get_wqe_num(0, true);
}

_UCT_INSTANTIATE_TEST_CASE(test_mlx5, rc_mlx5)
_UCT_INSTANTIATE_TEST_CASE(test_mlx5, dc_mlx5)
