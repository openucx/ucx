/**
 * Copyright (C) Zhixian Li 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/rocm/base/rocm_base.h>
#include <uct/rocm/base/rocm_signal.h>
}


class test_rocm_signal : public ucs::test {
protected:
    struct completion : uct_completion_t {
        completion() :
            uct_completion_t({.func = completion_cb,
                              .count = 1,
                              .status = UCS_OK}),
            callback_count(0)
        {
        }

        static void completion_cb(uct_completion_t *comp)
        {
            reinterpret_cast<completion*>(comp)->callback_count++;
        }

        unsigned callback_count;
    };

    void init() override
    {
        ucs_mpool_params_t mp_params;

        ucs::test::init();
        if (uct_rocm_base_init() != HSA_STATUS_SUCCESS) {
            UCS_TEST_SKIP_R("ROCm is unavailable");
        }

        ucs_mpool_params_reset(&mp_params);
        mp_params.elem_size       = sizeof(uct_rocm_base_signal_desc_t);
        mp_params.elems_per_chunk = 1;
        mp_params.max_elems       = 1;
        mp_params.ops             = &uct_rocm_base_signal_desc_mpool_ops;
        mp_params.name            = "ROCM signal test objects";
        ASSERT_UCS_OK(ucs_mpool_init(&mp_params, &m_signal_pool));
        m_signal_pool_initialized = true;
        ucs_queue_head_init(&m_signal_queue);
    }

    void cleanup() override
    {
        if (m_signal_pool_initialized) {
            ucs_mpool_cleanup(&m_signal_pool, 1);
        }

        ucs::test::cleanup();
    }

    ucs_mpool_t m_signal_pool;
    ucs_queue_head_t m_signal_queue;
    bool m_signal_pool_initialized = false;
};


UCS_TEST_F(test_rocm_signal, async_error) {
    uct_rocm_base_signal_desc_t *signal_desc;
    completion comp;

    signal_desc = static_cast<uct_rocm_base_signal_desc_t*>(
            ucs_mpool_get(&m_signal_pool));
    ASSERT_NE(nullptr, signal_desc);

    signal_desc->comp        = &comp;
    signal_desc->mapped_addr = nullptr;
    hsa_signal_store_screlease(signal_desc->signal, -1);
    ucs_queue_push(&m_signal_queue, &signal_desc->queue);

    scoped_log_handler log_handler(wrap_errors_logger);
    EXPECT_EQ(1u, uct_rocm_base_progress(&m_signal_queue));
    EXPECT_EQ(1u, comp.callback_count);
    EXPECT_EQ(UCS_ERR_IO_ERROR, comp.status);
    EXPECT_TRUE(ucs_queue_is_empty(&m_signal_queue));
}
