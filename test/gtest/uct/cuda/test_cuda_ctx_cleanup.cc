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

#include <atomic>
#include <cstdlib>
#include <dlfcn.h>


static std::atomic<bool>     g_destroy_audit(false);
static std::atomic<bool>     g_primary_ctx_audit(false);
static std::atomic<unsigned> g_event_destroy_count(0);
static std::atomic<unsigned> g_stream_destroy_count(0);
static std::atomic<unsigned> g_ctx_push_count(0);
static std::atomic<unsigned> g_primary_ctx_retain_count(0);
static std::atomic<unsigned> g_primary_ctx_release_count(0);

template<typename func_t>
static func_t cuda_real_func(const char *symbol)
{
    static void *handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    void *func;

    if (handle == NULL) {
        abort();
    }

    func = dlsym(handle, symbol);
    if (func == NULL) {
        abort();
    }

    return reinterpret_cast<func_t>(func);
}

extern "C" CUresult cuEventDestroy_v2(CUevent event)
{
    typedef CUresult (*func_t)(CUevent);

    if (g_destroy_audit.load(std::memory_order_relaxed)) {
        ++g_event_destroy_count;
        return CUDA_SUCCESS;
    }

    return cuda_real_func<func_t>("cuEventDestroy_v2")(event);
}

extern "C" CUresult cuStreamDestroy_v2(CUstream stream)
{
    typedef CUresult (*func_t)(CUstream);

    if (g_destroy_audit.load(std::memory_order_relaxed)) {
        ++g_stream_destroy_count;
        return CUDA_SUCCESS;
    }

    return cuda_real_func<func_t>("cuStreamDestroy_v2")(stream);
}

extern "C" CUresult cuCtxPushCurrent(CUcontext ctx)
{
    typedef CUresult (*func_t)(CUcontext);
    CUresult result;

    result = cuda_real_func<func_t>("cuCtxPushCurrent_v2")(ctx);
    if (g_destroy_audit.load(std::memory_order_relaxed) &&
        (result == CUDA_SUCCESS)) {
        ++g_ctx_push_count;
    }

    return result;
}

extern "C" CUresult cuDevicePrimaryCtxRetain(CUcontext *ctx, CUdevice dev)
{
    typedef CUresult (*func_t)(CUcontext*, CUdevice);
    CUresult result;

    result = cuda_real_func<func_t>("cuDevicePrimaryCtxRetain")(ctx, dev);
    if (g_primary_ctx_audit.load(std::memory_order_relaxed) &&
        (result == CUDA_SUCCESS)) {
        ++g_primary_ctx_retain_count;
    }

    return result;
}

extern "C" CUresult cuDevicePrimaryCtxRelease(CUdevice dev)
{
    typedef CUresult (*func_t)(CUdevice);
    CUresult result;

    result = cuda_real_func<func_t>("cuDevicePrimaryCtxRelease")(dev);
    if (g_primary_ctx_audit.load(std::memory_order_relaxed) &&
        (result == CUDA_SUCCESS)) {
        ++g_primary_ctx_release_count;
    }

    return result;
}


class scoped_cuda_destroy_audit {
public:
    scoped_cuda_destroy_audit()
    {
        g_event_destroy_count.store(0, std::memory_order_relaxed);
        g_stream_destroy_count.store(0, std::memory_order_relaxed);
        g_ctx_push_count.store(0, std::memory_order_relaxed);
        g_destroy_audit.store(true, std::memory_order_relaxed);
    }

    ~scoped_cuda_destroy_audit()
    {
        g_destroy_audit.store(false, std::memory_order_relaxed);
    }

    unsigned event_destroy_count() const
    {
        return g_event_destroy_count.load(std::memory_order_relaxed);
    }

    unsigned stream_destroy_count() const
    {
        return g_stream_destroy_count.load(std::memory_order_relaxed);
    }

    unsigned ctx_push_count() const
    {
        return g_ctx_push_count.load(std::memory_order_relaxed);
    }
};

class scoped_cuda_primary_ctx_audit {
public:
    scoped_cuda_primary_ctx_audit()
    {
        g_primary_ctx_retain_count.store(0, std::memory_order_relaxed);
        g_primary_ctx_release_count.store(0, std::memory_order_relaxed);
        g_primary_ctx_audit.store(true, std::memory_order_relaxed);
    }

    ~scoped_cuda_primary_ctx_audit()
    {
        g_primary_ctx_audit.store(false, std::memory_order_relaxed);
    }

    unsigned retain_count() const
    {
        return g_primary_ctx_retain_count.load(std::memory_order_relaxed);
    }

    unsigned release_count() const
    {
        return g_primary_ctx_release_count.load(std::memory_order_relaxed);
    }
};


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

    void reset_primary_ctx()
    {
        CUcontext cuda_ctx;

        ASSERT_TRUE(m_ctx_pushed);
        ASSERT_EQ(CUDA_SUCCESS, cuCtxPopCurrent(&cuda_ctx));
        ASSERT_EQ(m_cuda_ctx, cuda_ctx);
        m_ctx_pushed = false;

        ASSERT_EQ(CUDA_SUCCESS, cuDevicePrimaryCtxReset(m_device));
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
    scoped_cuda_primary_ctx_audit audit;

    create_ctx_rsc();
    EXPECT_EQ(1u, audit.retain_count());
    EXPECT_EQ(m_cuda_ctx, m_ctx_rsc->primary_ctx);
    EXPECT_EQ(m_device, m_ctx_rsc->cuda_device);

    destroy_ctx_rsc();
    EXPECT_EQ(1u, audit.release_count());
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

    scoped_cuda_destroy_audit audit;
    ucs_mpool_cleanup(&m_ctx_rsc->event_mp, 1);
    EXPECT_LT(0u, audit.ctx_push_count());
    EXPECT_LT(0u, audit.event_destroy_count());

    free_ctx_rsc();
}

UCS_TEST_F(test_cuda_ctx_cleanup, stream_cleanup_after_primary_ctx_release)
{
    uct_cuda_queue_desc_t qdesc;

    create_ctx_rsc();

    uct_cuda_base_queue_desc_init(&qdesc);
    ASSERT_UCS_OK(uct_cuda_base_init_stream(&qdesc.stream));

    release_primary_ctx();

    scoped_cuda_destroy_audit audit;
    uct_cuda_base_queue_desc_destroy(m_ctx_rsc, &qdesc);
    EXPECT_EQ(1u, audit.ctx_push_count());
    EXPECT_EQ(1u, audit.stream_destroy_count());

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

    scoped_cuda_destroy_audit audit;
    ucs_mpool_cleanup(&m_ctx_rsc->event_mp, 1);
    EXPECT_EQ(0u, audit.ctx_push_count());
    EXPECT_EQ(0u, audit.event_destroy_count());

    free_ctx_rsc();
}

UCS_TEST_F(test_cuda_ctx_cleanup, stream_cleanup_after_primary_ctx_reset)
{
    uct_cuda_queue_desc_t qdesc;

    create_ctx_rsc();

    uct_cuda_base_queue_desc_init(&qdesc);
    ASSERT_UCS_OK(uct_cuda_base_init_stream(&qdesc.stream));

    reset_primary_ctx();

    scoped_cuda_destroy_audit audit;
    uct_cuda_base_queue_desc_destroy(m_ctx_rsc, &qdesc);
    EXPECT_EQ(0u, audit.ctx_push_count());
    EXPECT_EQ(0u, audit.stream_destroy_count());

    destroy_ctx_rsc();
}
