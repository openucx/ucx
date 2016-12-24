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
        test_callbackq_base       *test;
        uint32_t                  count;
        int                       command;
        callback_ctx              *to_add;
        ucs_callbackq_slow_elem_t slow_elem;
    };

    test_callbackq_base() : m_async_ptr(NULL), is_fast_path(true) {
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

    static void callback_slow_proxy(ucs_callbackq_slow_elem_t *self)
    {
        callback_ctx *ctx = ucs_container_of(self, callback_ctx, slow_elem);
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
        if (is_fast_path) {
            ucs_callbackq_add(&m_cbq, callback_proxy, reinterpret_cast<void*>(ctx));
        } else {
            ctx->slow_elem.cb = callback_slow_proxy;
            ucs_callbackq_add_slow_path(&m_cbq, &ctx->slow_elem);
        }
    }

    void remove(callback_ctx *ctx)
    {
        if (is_fast_path) {
            ucs_status_t status = ucs_callbackq_remove(&m_cbq, callback_proxy,
                                                       reinterpret_cast<void*>(ctx));
            ASSERT_UCS_OK(status);
        } else {
            ucs_callbackq_remove_slow_path(&m_cbq, &ctx->slow_elem);
        }
    }
    void dispatch(unsigned count = 1)
    {
        for (unsigned i = 0; i < count; ++i) {
            ucs_callbackq_dispatch(&m_cbq);
        }
    }

    ucs_callbackq_t     m_cbq;
    ucs_async_context_t *m_async_ptr;
    bool is_fast_path;
};

class test_callbackq : public test_callbackq_base, public ::testing::TestWithParam<bool> {
    virtual void init() {
       test_callbackq_base::init();
       is_fast_path = GetParam();
    }
    UCS_TEST_BASE_IMPL;
};

UCS_TEST_P(test_callbackq, single) {
    callback_ctx ctx;

    init_ctx(&ctx);
    add(&ctx);
    dispatch();
    remove(&ctx);
    EXPECT_EQ(1u, ctx.count);
}

UCS_TEST_P(test_callbackq, refcount) {
    if (!is_fast_path) {
        UCS_TEST_SKIP;
    }

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

UCS_TEST_P(test_callbackq, multi) {
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

UCS_TEST_P(test_callbackq, remove_self) {
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

UCS_TEST_P(test_callbackq, add_another) {
    callback_ctx ctx, ctx2;

    init_ctx(&ctx);
    init_ctx(&ctx2);
    ctx.command = COMMAND_ADD_ANOTHER;
    ctx.to_add  = &ctx2;

    add(&ctx);

    dispatch();
    EXPECT_EQ(1u, ctx.count);
    ctx.command = COMMAND_NONE;

    /* When element is added to the slow path while the last/only one is being
     * dispatched, it will be called only in the next dispatch cycle.
     * But fast path queue is updated on the fly. */
    unsigned count = is_fast_path ? ctx.count : ctx.count - 1;

    dispatch();
    EXPECT_EQ(2u, ctx.count);
    EXPECT_EQ(count + 1, ctx2.count);

    remove(&ctx);
    dispatch();
    EXPECT_EQ(2u, ctx.count);
    EXPECT_EQ(count + 2, ctx2.count);

    remove(&ctx2);
    dispatch();
    EXPECT_EQ(count + 2, ctx2.count);
}

INSTANTIATE_TEST_CASE_P(fast_path, test_callbackq, ::testing::Values(true));
INSTANTIATE_TEST_CASE_P(slow_path, test_callbackq, ::testing::Values(false));

class test_callbackq_safe : public test_callbackq_base {

protected:

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
};

class test_callbackq_thread : public test_callbackq_safe, public ::testing::Test {
    UCS_TEST_BASE_IMPL;
};

UCS_MT_TEST_F(test_callbackq_thread, threads, 10) {

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

UCS_MT_TEST_F(test_callbackq_thread, remove_all, 10) {
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

class test_callbackq_async : public test_callbackq_safe,
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

    void add_timer(int *timer_id) {
        ucs_status_t status;
        status = ucs_async_add_timer(m_async.mode, ucs_time_from_msec(1),
                                     timer_callback, this, &m_async, timer_id);
        ASSERT_UCS_OK(status);
    }

    void remove_timer(int timer_id) {
        ucs_async_remove_handler(timer_id, 1);
    }

    void timer() {
        if (is_fast_path) {
            if ((m_add_count > 0) && ((rand() % 2) == 0)) {
                remove_safe(&m_cbctx);
                --m_add_count;
            } else {
                add_safe(&m_cbctx);
                ++m_add_count;
            }
        } else {
            callback_ctx *tmp = new callback_ctx;
            init_ctx(tmp);
            add(tmp);
            tmp->command = COMMAND_REMOVE_SELF;
            m_cbctxs.push_back(tmp);
        }
    }


    static void timer_callback(void *arg) {
        test_callbackq_async *self = reinterpret_cast<test_callbackq_async*>(arg);
        self->timer();
    }

protected:
    ucs_async_context_t            m_async;
    callback_ctx                   m_cbctx;
    volatile uint32_t              m_add_count;
    ucs::ptr_vector<callback_ctx>  m_cbctxs;

    UCS_TEST_BASE_IMPL
};


UCS_TEST_P(test_callbackq_async, test_fast) {
    int timer_id;

    is_fast_path = true;
    add_timer(&timer_id);

    ucs_time_t end_time   = ucs_get_time() + ucs_time_from_msec(300);
    while (ucs_get_time() < end_time) {
        dispatch(10);
        add(&m_cbctx);
        dispatch(10);
        remove(&m_cbctx);
    }

    remove_timer(timer_id);

    remove_all(&m_cbctx);
}

UCS_TEST_P(test_callbackq_async, test_slow) {
    int timer_id;

    is_fast_path = false;
    add_timer(&timer_id);

    ucs_time_t end_time   = ucs_get_time() + ucs_time_from_msec(100);
    while (ucs_get_time() < end_time) {
        dispatch(10);
    }

    remove_timer(timer_id);

    /* make sure every ctx element gets dispatched before to check count */
    dispatch();

    ucs::ptr_vector<callback_ctx>::const_iterator it;
    for (it = m_cbctxs.begin(); it != m_cbctxs.end(); ++it) {
        /* Count should be 1, because every context is created with
         * REMOVE_SELF command */
        EXPECT_EQ(1u, (*it)->count);
    }
}

INSTANTIATE_TEST_CASE_P(signal, test_callbackq_async, ::testing::Values(UCS_ASYNC_MODE_SIGNAL));
INSTANTIATE_TEST_CASE_P(thread, test_callbackq_async, ::testing::Values(UCS_ASYNC_MODE_THREAD));
