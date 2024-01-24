/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2022. ALL RIGHTS RESERVED.
*/

#include <common/test.h>
#include <common/test_helpers.h>
#include <uct/ib/test_ib.h>

extern "C" {
#include <uct/api/uct.h>
#include <uct/ib/mlx5/ib_mlx5.h>
#include <uct/ib/base/ib_iface.h>
}

#define UCT_INSTANTIATE_MLX5_TEST_CASE(_test_case) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, rc_mlx5) \
    _UCT_INSTANTIATE_TEST_CASE(_test_case, dc_mlx5)

class test_cqe_zipping : public test_uct_ib_with_specific_port {
public:
    bool is_cqe_zipping_expected()
    {
#ifdef ENABLE_STATS
        return !RUNNING_ON_VALGRIND &&
               !IBV_PORT_IS_LINK_LAYER_ETHERNET(&m_port_attr);
#else
        return false;
#endif
    }

    void send_while_possible()
    {
        while (am_zcopy() != UCS_ERR_NO_RESOURCE) {
            ++m_send_cnt;
        }
    }

    void wait_for_completion() const
    {
        /*
         * Local deadline prevents hanging inside receive loop in case of
         * connectivity issues
         */
        const ucs_time_t deadline = ucs::get_deadline(60);
        while ((m_send_cnt != m_recv_cnt) && (ucs_get_time() < deadline)) {
            progress();
        }
        ASSERT_EQ(m_send_cnt, m_recv_cnt);
    }

    size_t get_total_zipped_count() const
    {
        size_t zipped_cqes_count  = 0;
#ifdef ENABLE_STATS
        constexpr int counters[2] = {UCT_IB_IFACE_STAT_TX_COMPLETION_ZIPPED,
                                     UCT_IB_IFACE_STAT_RX_COMPLETION_ZIPPED};

        for (auto &entity : m_entities) {
            auto stats = ucs_derived_of(entity->iface(), uct_ib_iface_t)->stats;
            for (auto &counter : counters) {
                zipped_cqes_count += UCS_STATS_GET_COUNTER(stats, counter);
            }
        }
#endif
        return zipped_cqes_count;
    }

    virtual void init()
    {
        stats_activate();
        modify_config("IB_TX_CQE_ZIP_ENABLE", "yes");
        modify_config("IB_RX_CQE_ZIP_ENABLE", "yes");

        test_uct_ib_with_specific_port::init();
        test_uct_ib::init();

        if (!check_cqe_zip_caps()) {
            cleanup();
            UCS_TEST_SKIP_R("unsupported");
        }

        uct_iface_set_am_handler(receiver()->iface(), 0, am_cb, &m_recv_cnt, 0);
        m_send_buf = new mapped_buffer(m_buf_size, 0, *sender());
    }

    void flush_and_reset()
    {
        uct_test::flush();
        m_send_cnt = m_recv_cnt = 0;
    }

    bool check_cqe_zip_caps()
    {
        uct_ib_mlx5_md_t *md = NULL;
        FOR_EACH_ENTITY(entity) {
            md = ucs_derived_of((*entity)->md(), uct_ib_mlx5_md_t);
            if (!(ucs_test_all_flags(md->flags, UCT_IB_MLX5_MD_FLAG_CQE64_ZIP |
                                     UCT_IB_MLX5_MD_FLAG_CQE128_ZIP |
                                     UCT_IB_MLX5_MD_FLAG_DEVX_CQ))) {
                return false;
            }
        }
        return true;
    }

private:
    entity *sender() const
    {
        return m_e1;
    }

    entity *receiver() const
    {
        return m_e2;
    }

    virtual void cleanup()
    {
        delete m_send_buf;
        uct_iface_set_am_handler(receiver()->iface(), 0, NULL, NULL, 0);

        test_uct_ib::cleanup();
        test_uct_ib_with_specific_port::cleanup();

        stats_restore();
    }

    ucs_status_t am_zcopy()
    {
        auto size = sender()->iface_attr().cap.am.max_zcopy;
        size      = ucs_min(size, m_buf_size);
        uct_iov_t iov{m_send_buf->ptr(), size, m_send_buf->memh(), 0, 1};

        return uct_ep_am_zcopy(sender()->ep(0), 0, m_send_buf->ptr(), 0, &iov,
                               1, 0, nullptr);
    }

    static ucs_status_t
    am_cb(void *arg, void *data, size_t length, unsigned flags)
    {
        size_t &recv_cnt = *(static_cast<size_t*>(arg));
        ++recv_cnt;
        return UCS_OK;
    }

    static constexpr size_t m_buf_size = 2048;
    size_t                  m_send_cnt = 0, m_recv_cnt = 0;
    mapped_buffer           *m_send_buf{nullptr};
};

/* We test only ZCOPY with 4K size messages due to good PCI load and stably
 * CQE zipping reproducing on all platforms.
 */
UCS_TEST_P(test_cqe_zipping, zcopy)
{
    int deadline_seconds      = is_cqe_zipping_expected() ? 90 : 5;
    const ucs_time_t deadline = ucs::get_deadline(deadline_seconds);

    while ((ucs_get_time() < deadline) && (get_total_zipped_count() == 0)) {
        /* Flush resets the transort resources which were acquired during the
         * previous iteration.
         */
        flush_and_reset();
        send_while_possible();
        wait_for_completion();
    }

    if (is_cqe_zipping_expected()) {
        EXPECT_GT(get_total_zipped_count(), 0);
    }
}

UCT_INSTANTIATE_MLX5_TEST_CASE(test_cqe_zipping)
