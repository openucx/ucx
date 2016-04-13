/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <ucs/arch/atomic.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/callbackq.h>
}

class test_callbackq_base : public ucs::test_base {
protected:

    enum {
        COMMAND_NONE,
        COMMAND_REMOVE_SELF,
        COMMAND_ADD_ANOTHER
    };

    struct callback_ctx {
        test_callbackq_base *test;
        uint32_t            count;
        int                 command;
        callback_ctx        *to_add;
    };

    test_callbackq_base() : m_async_ptr(NULL) {
        memset(&m_cbq, 0, sizeof(m_cbq)); /* Silence coverity */
    }

    virtual void init() {
        ucs::test_base::init();
        ucs_status_t status = ucs_callbackq_init(&m_cbq, 64, m_async_ptr);
        ASSERT_UCS_OK(status);
    }

    virtual void cleanup() {
        ucs_callbackq_cleanup(&m_cbq);
        ucs::test_base::cleanup();
    }

    static void callback_proxy(void *arg)
    {
        callback_ctx *ctx = reinterpret_cast<callback_ctx*>(arg);
        ctx->test->callback(ctx);
    }

    void callback(callback_ctx *ctx)
    {
        ucs_atomic_add32(&ctx->count, 1);

        switch (ctx->command) {
        case COMMAND_REMOVE_SELF:
            remove(ctx);
            break;
        case COMMAND_ADD_ANOTHER:
            add(ctx->to_add);
            break;
        case COMMAND_NONE:
        default:
            break;
        }
    }

    void init_ctx(callback_ctx *ctx)
    {
        ctx->test    = this;
        ctx->count   = 0;
        ctx->command = COMMAND_NONE;
    }

    void add(callback_ctx *ctx)
    {
        ucs_callbackq_add(&m_cbq, callback_proxy, reinterpret_cast<void*>(ctx));
    }

    void remove(callback_ctx *ctx)
    {
        ucs_status_t status = ucs_callbackq_remove(&m_cbq, callback_proxy,
                                                   reinterpret_cast<void*>(ctx));
        ASSERT_UCS_OK(status);
    }

    void remove_all(callback_ctx *ctx)
    {
        ucs_callbackq_remove_all(&m_cbq, callback_proxy,
                                 reinterpret_cast<void*>(ctx));
    }

    void add_safe(callback_ctx *ctx)
    {
        ucs_status_t status = ucs_callbackq_add_safe(&m_cbq, callback_proxy,
                                                     reinterpret_cast<void*>(ctx));
        ASSERT_UCS_OK(status);
     }

    void remove_safe(callback_ctx *ctx)
    {
        ucs_status_t status = ucs_callbackq_remove_safe(&m_cbq, callback_proxy,
                                                        reinterpret_cast<void*>(ctx));

        ASSERT_UCS_OK(status);
     }

    void dispatch(unsigned count = 1)
    {
        for (unsigned i = 0; i < count; ++i) {
            ucs_callbackq_dispatch(&m_cbq);
        }
    }

    ucs_callbackq_t     m_cbq;
    ucs_async_context_t *m_async_ptr;
};

class test_callbackq : public test_callbackq_base, public ::testing::Test {
    UCS_TEST_BASE_IMPL;
};

UCS_TEST_F(test_callbackq, single) {
    callback_ctx ctx;

    init_ctx(&ctx);
    add(&ctx);
    dispatch();
    remove(&ctx);
    EXPECT_EQ(1u, ctx.count);
}

UCS_TEST_F(test_callbackq, refcount) {
    callback_ctx ctx;

    init_ctx(&ctx);
    add(&ctx);
    add(&ctx);

    dispatch();
    EXPECT_EQ(1u, ctx.count);

    remove(&ctx);
    dispatch();
    EXPECT_EQ(2u, ctx.count);

    remove(&ctx);
    dispatch();
    EXPECT_EQ(2u, ctx.count);
}

UCS_TEST_F(test_callbackq, multi) {
    static const unsigned COUNT = 3;

    callback_ctx ctx[COUNT];

    for (unsigned i = 0; i < COUNT; ++i) {
        init_ctx(&ctx[i]);
        add(&ctx[i]);
    }

    dispatch();
    dispatch();

    for (unsigned i = 0; i < COUNT; ++i) {
        remove(&ctx[i]);
        EXPECT_EQ(2u, ctx[i].count);
    }
}

UCS_TEST_F(test_callbackq, remove_self) {
    callback_ctx ctx;

    init_ctx(&ctx);
    ctx.command = COMMAND_REMOVE_SELF;
    add(&ctx);
    dispatch();
    EXPECT_EQ(1u, ctx.count);

    dispatch();
    dispatch();
    EXPECT_EQ(1u, ctx.count);
}

UCS_TEST_F(test_callbackq, add_another) {
    callback_ctx ctx, ctx2;

    init_ctx(&ctx);
    init_ctx(&ctx2);
    ctx.command = COMMAND_ADD_ANOTHER;
    ctx.to_add  = &ctx2;

    add(&ctx);

    dispatch();
    EXPECT_EQ(1u, ctx.count);
    unsigned count = ctx.count;

    dispatch();
    EXPECT_EQ(2u, ctx.count);
    EXPECT_EQ(count + 1, ctx2.count);

    remove(&ctx);
    dispatch();
    EXPECT_EQ(2u, ctx.count);
    EXPECT_EQ(count + 2, ctx2.count);

    remove(&ctx2);
    remove(&ctx2);
    dispatch();
    EXPECT_EQ(count + 2, ctx2.count);
}

UCS_MT_TEST_F(test_callbackq, threads, 10) {

    static unsigned COUNT = 2000;

    if (barrier()) {
        for (unsigned i = 0; i < COUNT; ++i) {
            /* part 1 */
            dispatch(100); /* simulate race */
            barrier(); /*1*/
            dispatch(5);
            barrier(); /*2*/

            /* part 2 */
            dispatch(100); /* simulate race */
            barrier(); /*3*/
            dispatch(5);
            barrier(); /*4*/
            dispatch(100);
            barrier(); /*5*/
        }
    } else {
        for (unsigned i = 0; i < COUNT; ++i) {
            /* part 1 */
            callback_ctx ctx;
            init_ctx(&ctx);
            add_safe(&ctx);
            barrier(); /*1*/
            barrier(); /*2*/  /* dispatch which seen the add command already called */
            EXPECT_GE(ctx.count, 1u);

            /* part 2 */
            remove_safe(&ctx);
            barrier(); /*3*/
            barrier(); /*4*/ /* dispatch which seen the remove command already called */
            unsigned count = ctx.count;
            barrier(); /*5*/
            EXPECT_EQ(count, ctx.count);
        }
    }
}

UCS_MT_TEST_F(test_callbackq, remove_all, 10) {
    static callback_ctx ctx1;

    init_ctx(&ctx1);

    if (barrier()) {
        callback_ctx ctx2;
        init_ctx(&ctx2);

        dispatch(1000);
        add(&ctx2);
        barrier();
        remove_all(&ctx1);
        remove_all(&ctx2);

        /* this should remove all instances of 'ctx', including queued async
         * commands */
        uint32_t count1 = ctx1.count;
        dispatch(100);
        EXPECT_EQ(count1, ctx1.count);
        EXPECT_EQ(0u,     ctx2.count);
    } else {
        for (unsigned i = 0; i < 1000; ++i) {
            add_safe(&ctx1);
        }
        barrier();
    }
}

class test_callbackq_async : public test_callbackq_base,
                             public ::testing::TestWithParam<ucs_async_mode_t>
{
protected:
    test_callbackq_async() : m_add_count(0) {
        /* Silence coverity */
        memset(&m_cbctx, 0, sizeof(m_cbctx));
        memset(&m_async, 0, sizeof(m_async));
    }

    virtual void init() {
        ucs_status_t status = ucs_async_context_init(&m_async, GetParam());
        ASSERT_UCS_OK(status);
        m_async_ptr = &m_async;
        test_callbackq_base::init();
        init_ctx(&m_cbctx);
    }

    virtual void cleanup() {
        test_callbackq_base::cleanup();
        ucs_async_context_cleanup(&m_async);
    }

    void timer() {
        if ((m_add_count > 0) && ((rand() % 2) == 0)) {
            remove_safe(&m_cbctx);
            --m_add_count;
        } else {
            add_safe(&m_cbctx);
            ++m_add_count;
        }
    }

    static void timer_callback(void *arg) {
        test_callbackq_async *self = reinterpret_cast<test_callbackq_async*>(arg);
        self->timer();
    }

protected:
    ucs_async_context_t m_async;
    callback_ctx        m_cbctx;
    volatile uint32_t   m_add_count;

    UCS_TEST_BASE_IMPL
};


UCS_TEST_P(test_callbackq_async, test) {
    ucs_status_t status;
    int timer_id;

    status = ucs_async_add_timer(m_async.mode, ucs_time_from_msec(1),
                                 timer_callback, this, &m_async, &timer_id);
    ASSERT_UCS_OK(status);

    ucs_time_t end_time   = ucs_get_time() + ucs_time_from_msec(300);
    while (ucs_get_time() < end_time) {
        dispatch(10);
        add(&m_cbctx);
        dispatch(10);
        remove(&m_cbctx);
    }

    ucs_async_remove_timer(timer_id);

    remove_all(&m_cbctx);
}

INSTANTIATE_TEST_CASE_P(signal, test_callbackq_async, ::testing::Values(UCS_ASYNC_MODE_SIGNAL));
INSTANTIATE_TEST_CASE_P(thread, test_callbackq_async, ::testing::Values(UCS_ASYNC_MODE_THREAD));
