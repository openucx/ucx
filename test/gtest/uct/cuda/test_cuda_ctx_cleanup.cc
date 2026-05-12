/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

extern "C" {
#include <uct/cuda/base/cuda_iface.h>
#include <ucs/datastruct/mpool.h>
}

#include <cuda.h>


class test_cuda_ctx_cleanup : public ucs::test {
protected:
    void init() override
    {
        int num_devices;

        ucs::test::init();

        ASSERT_EQ(CUDA_SUCCESS, cuInit(0));
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGetCount(&num_devices));
        if (num_devices == 0) {
            UCS_TEST_SKIP_R("no cuda devices available");
        }

        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&m_device, 0));
        init_iface();
        push_primary_ctx();
    }

    void cleanup() override
    {
        destroy_ctx_rsc();

        kh_destroy_inplace(cuda_ctx_rscs, &m_iface.ctx_rscs);

        if (m_ctx_pushed) {
            EXPECT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(NULL));
            m_ctx_pushed = false;
        }

        if (m_ctx_retained) {
            EXPECT_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxRelease(m_device));
            m_ctx_retained = false;
        }

        restore_current_cuda_context(m_restore_current_ctx);
        m_restore_current_ctx = 0;

        ucs::test::cleanup();
    }

    void create_ctx_rsc()
    {
        ASSERT_UCS_OK(uct_cuda_base_ctx_rsc_create(&m_iface, 1, &m_ctx_rsc));
    }

    void release_ctx_rsc_primary_ctx()
    {
        if ((m_ctx_rsc == NULL) || (m_ctx_rsc->primary_ctx == NULL)) {
            return;
        }

        EXPECT_EQ(CUDA_SUCCESS,
                  cuDevicePrimaryCtxRelease(m_ctx_rsc->cuda_device));
        m_ctx_rsc->primary_ctx = NULL;
        m_ctx_rsc->cuda_device = CU_DEVICE_INVALID;
    }

    void free_ctx_rsc()
    {
        if (m_ctx_rsc == NULL) {
            return;
        }

        release_ctx_rsc_primary_ctx();
        destroy_rsc(NULL, m_ctx_rsc);
        m_ctx_rsc = NULL;
    }

    void destroy_ctx_rsc()
    {
        if (m_ctx_rsc == NULL) {
            return;
        }

        ucs_mpool_cleanup(&m_ctx_rsc->event_mp, 1);
        free_ctx_rsc();
    }

    void release_primary_ctx()
    {
        CUcontext cuda_ctx;

        ASSERT_TRUE(m_ctx_pushed);
        ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(&cuda_ctx));
        ASSERT_EQ(m_cuda_ctx, cuda_ctx);
        m_ctx_pushed = false;

        ASSERT_TRUE(m_ctx_retained);
        ASSERT_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxRelease(m_device));
        m_ctx_retained = false;
    }

    void pop_current_cuda_contexts(int *popped)
    {
        CUcontext cuda_ctx;

        *popped = 0;
        for (;;) {
            ASSERT_EQ(CUDA_SUCCESS, cuCtxGetCurrent(&cuda_ctx));
            if (cuda_ctx == NULL) {
                return;
            }

            ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(NULL));
            *popped = 1;
        }
    }

    void restore_current_cuda_context(int popped)
    {
        CUcontext cuda_ctx;

        if (!popped) {
            return;
        }

        /* Restore the gtest CUDA guard context that was current before reset. */
        ASSERT_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxRetain(&cuda_ctx, m_device));
        ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(cuda_ctx));
    }

    void reset_primary_ctx()
    {
        CUcontext cuda_ctx;
        int popped;

        ASSERT_TRUE(m_ctx_pushed);
        ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(&cuda_ctx));
        ASSERT_EQ(m_cuda_ctx, cuda_ctx);
        m_ctx_pushed = false;

        /* Avoid resetting a primary context which is still current. */
        pop_current_cuda_contexts(&popped);
        ASSERT_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxReset(m_device));
        m_restore_current_ctx = popped;
        m_ctx_retained = false;
    }

    static uct_cuda_ctx_rsc_t *create_rsc(uct_iface_h iface)
    {
        return static_cast<uct_cuda_ctx_rsc_t*>(
                ucs_calloc(1, sizeof(uct_cuda_ctx_rsc_t),
                           "test_cuda_ctx_rsc"));
    }

    static void destroy_rsc(uct_iface_h iface, uct_cuda_ctx_rsc_t *ctx_rsc)
    {
        ucs_free(ctx_rsc);
    }

    static void complete_event(uct_iface_h iface, uct_cuda_event_desc_t *event)
    {
    }

    uct_cuda_iface_t m_iface = {};
    uct_cuda_ctx_rsc_t *m_ctx_rsc = NULL;
    CUcontext m_cuda_ctx = NULL;
    CUdevice m_device = 0;
    bool m_ctx_pushed = false;
    bool m_ctx_retained = false;
    int m_restore_current_ctx = 0;

private:
    void init_iface()
    {
        static uct_cuda_iface_ops_t iface_ops = {
            create_rsc,
            destroy_rsc,
            complete_event
        };

        m_iface.ops                    = &iface_ops;
        m_iface.config.event_desc_size = sizeof(uct_cuda_event_desc_t);
        m_iface.config.max_events      = 128;
        kh_init_inplace(cuda_ctx_rscs, &m_iface.ctx_rscs);
    }

    void push_primary_ctx()
    {
        ASSERT_EQ(CUDA_SUCCESS,
                  cuDevicePrimaryCtxRetain(&m_cuda_ctx, m_device));
        m_ctx_retained = true;
        ASSERT_EQ(CUDA_SUCCESS, cuCtxPushCurrent(m_cuda_ctx));
        m_ctx_pushed = true;
    }
};

UCS_TEST_F(test_cuda_ctx_cleanup, retain_primary_ctx_until_rsc_cleanup)
{
    create_ctx_rsc();
    EXPECT_EQ(m_cuda_ctx, m_ctx_rsc->primary_ctx);
    EXPECT_EQ(m_device, m_ctx_rsc->cuda_device);

    destroy_ctx_rsc();
}

UCS_TEST_F(test_cuda_ctx_cleanup, event_cleanup_after_primary_ctx_release)
{
    uct_cuda_event_desc_t *event_desc;

    create_ctx_rsc();

    event_desc = static_cast<uct_cuda_event_desc_t*>(
            ucs_mpool_get(&m_ctx_rsc->event_mp));
    ASSERT_NE(nullptr, event_desc);
    ucs_mpool_put(event_desc);

    release_primary_ctx();

    ucs_mpool_cleanup(&m_ctx_rsc->event_mp, 1);
    free_ctx_rsc();
}

UCS_TEST_F(test_cuda_ctx_cleanup, stream_cleanup_after_primary_ctx_release)
{
    uct_cuda_queue_desc_t qdesc;

    create_ctx_rsc();

    uct_cuda_base_queue_desc_init(&qdesc);
    ASSERT_UCS_OK(uct_cuda_base_init_stream(&qdesc.stream));

    release_primary_ctx();

    uct_cuda_base_queue_desc_destroy(m_ctx_rsc, &qdesc);
    destroy_ctx_rsc();
}

UCS_TEST_F(test_cuda_ctx_cleanup, event_cleanup_after_primary_ctx_reset)
{
    uct_cuda_event_desc_t *event_desc;

    create_ctx_rsc();

    event_desc = static_cast<uct_cuda_event_desc_t*>(
            ucs_mpool_get(&m_ctx_rsc->event_mp));
    ASSERT_NE(nullptr, event_desc);
    ucs_mpool_put(event_desc);

    reset_primary_ctx();

    ucs_mpool_cleanup(&m_ctx_rsc->event_mp, 1);
    free_ctx_rsc();
}

UCS_TEST_F(test_cuda_ctx_cleanup, stream_cleanup_after_primary_ctx_reset)
{
    uct_cuda_queue_desc_t qdesc;

    create_ctx_rsc();

    uct_cuda_base_queue_desc_init(&qdesc);
    ASSERT_UCS_OK(uct_cuda_base_init_stream(&qdesc.stream));

    reset_primary_ctx();

    uct_cuda_base_queue_desc_destroy(m_ctx_rsc, &qdesc);
    destroy_ctx_rsc();
}
