/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/uct_test.h>
#include <cuda.h>


class test_user_stream : public uct_test {
protected:
    void init() override
    {
        uct_test::init();

        CUctxCreateParams params = {};
        CUcontext ctx;
        CUdevice dev;
        ASSERT_EQ(CUDA_SUCCESS, cuInit(0));
        ASSERT_EQ(CUDA_SUCCESS, cuDeviceGet(&dev, 0));
        ASSERT_EQ(CUDA_SUCCESS, cuCtxCreate(&ctx, &params, 0, dev));

        EXPECT_EQ(CUDA_SUCCESS,
                  cuStreamCreate(&m_stream, CU_STREAM_NON_BLOCKING));
        m_iface_stream = reinterpret_cast<uct_iface_stream_h>(m_stream);

        entity *e = uct_test::create_entity(0ul);
        m_iface   = e->iface();
        m_entities.push_back(e);
    }

    void cleanup() override
    {
        EXPECT_EQ(CUDA_SUCCESS, cuStreamDestroy(m_stream));
        uct_test::cleanup();
    }

    uct_iface_stream_op_handle_h m_op_handle;
    uct_iface_h                  m_iface;
    CUstream                     m_stream;
    uct_iface_stream_h           m_iface_stream;

private:
};

namespace {
void increment_cb(void *arg)
{
    (*reinterpret_cast<int*>(arg))++;
}

void host_wait_cb(void *data)
{
    while (*reinterpret_cast<int*>(data) != 1) {
        sched_yield();
    }
}
} // namespace

UCS_TEST_P(test_user_stream, stream_already_ready)
{
    int done = 0;

    ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                            increment_cb, &done, &m_op_handle));
    wait_for_value(&done, 1, true);
    ASSERT_EQ(1, done);
    ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, m_op_handle));
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
}

UCS_TEST_P(test_user_stream, stream_not_ready_waited)
{
    int done = 0;

    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(m_stream, host_wait_cb, &done));
    ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                            increment_cb, &done, &m_op_handle));
    done = 1;
    wait_for_value(&done, 2, true);
    ASSERT_EQ(2, done);
    ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, m_op_handle));
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
}

UCS_TEST_P(test_user_stream, stream_not_ready_unblocked)
{
    int done = 0;

    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(m_stream, host_wait_cb, &done));
    ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                            increment_cb, &done, &m_op_handle));
    ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, m_op_handle));
    done = 1;
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
}

UCS_TEST_P(test_user_stream, stream_blocked_after)
{
    int done = 0, ready = 0;

    auto host_signal_cb = [](void *data) { *reinterpret_cast<int*>(data) = 1; };

    // Block user stream
    ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                            increment_cb, &ready,
                                            &m_op_handle));
    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(m_stream, host_signal_cb, &done));

    for (int i = 0; i < 1000; ++i) {
        ASSERT_EQ(CUDA_ERROR_NOT_READY, cuStreamQuery(m_stream));
    }
    ASSERT_EQ(0, done);

    ASSERT_EQ(0, ready);
    ASSERT_EQ(1, uct_iface_progress(m_iface));
    ASSERT_EQ(1, ready);

    // Unblock user stream
    ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, m_op_handle));
    wait_for_value(&done, 1, true);
    ASSERT_EQ(1, done);

    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
}

UCS_TEST_P(test_user_stream, stream_repeat)
{
    int done = 0, total = 30, ready = 0, count = 0;
    std::vector<uct_iface_stream_op_handle_h> handles;
    uct_iface_stream_op_handle_h op_handle;

    for (auto i = 0; i < total; ++i) {
        ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                                increment_cb, &ready,
                                                &op_handle));
        ASSERT_EQ(CUDA_SUCCESS,
                  cuLaunchHostFunc(m_stream, increment_cb, &done));
        handles.push_back(op_handle);
    }

    for (auto handle : handles) {
        wait_for_value(&ready, ++count, true);
        EXPECT_EQ(count, ready);
        ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, handle));
    }

    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
    ASSERT_EQ(total, done);
}

UCS_TEST_P(test_user_stream, stream_arm_busy)
{
    int ready = 0, fd;
    uct_iface_stream_op_handle_h handle;

    ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                            increment_cb, &ready, &handle));
    ASSERT_UCS_OK(uct_iface_event_fd_get(m_iface, &fd));
    ASSERT_EQ(UCS_ERR_BUSY, uct_iface_event_arm(m_iface, 0));
    ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, handle));
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
}

UCS_TEST_P(test_user_stream, stream_arm_success)
{
    int done = 0, ready = 0, fd;
    uct_iface_stream_op_handle_h handle;
    uct_test::async_event_ctx async_event_ctx;

    ASSERT_EQ(CUDA_SUCCESS, cuLaunchHostFunc(m_stream, host_wait_cb, &done));
    ASSERT_UCS_OK(uct_iface_stream_op_block(m_iface, m_iface_stream,
                                            increment_cb, &ready, &handle));
    ASSERT_UCS_OK(uct_iface_event_fd_get(m_iface, &fd));
    ASSERT_EQ(UCS_OK, uct_iface_event_arm(m_iface, 0));
    ASSERT_UCS_OK(uct_iface_stream_op_unblock(m_iface, handle));

    ASSERT_FALSE(async_event_ctx.wait_for_event(*m_entities.back(), 0.2));
    done = 1;
    EXPECT_TRUE(async_event_ctx.wait_for_event(*m_entities.back(), 1.));
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(m_stream));
}

_UCT_INSTANTIATE_TEST_CASE(test_user_stream, cuda_copy)
