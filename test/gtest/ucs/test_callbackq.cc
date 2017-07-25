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

class test_callbackq :
    public ucs::test_base,
    public ::testing::TestWithParam<bool> {
protected:

    enum {
        COMMAND_NONE,
        COMMAND_REMOVE_SELF,
        COMMAND_ADD_ANOTHER
    };

    struct callback_ctx {
        test_callbackq            *test;
        int                       callback_id;
        uint32_t                  count;
        int                       command;
        callback_ctx              *to_add;
    };

    test_callbackq() {
        memset(&m_cbq, 0, sizeof(m_cbq)); /* Silence coverity */
    }

    virtual void init() {
        ucs_status_t status = ucs_callbackq_init(&m_cbq);
        ASSERT_UCS_OK(status);
    }

    virtual void cleanup() {
        ucs_callbackq_cleanup(&m_cbq);
        ucs::test_base::cleanup();
    }

    UCS_TEST_BASE_IMPL;

    static unsigned callback_proxy(void *arg)
    {
        callback_ctx *ctx = reinterpret_cast<callback_ctx*>(arg);
        return ctx->test->callback(ctx);
    }

    unsigned callback(callback_ctx *ctx)
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
        return 1;
    }

    void init_ctx(callback_ctx *ctx)
    {
        ctx->test        = this;
        ctx->count       = 0;
        ctx->command     = COMMAND_NONE;
        ctx->callback_id = UCS_CALLBACKQ_ID_NULL;
    }

    unsigned fast_path_flag() {
        return GetParam() ? UCS_CALLBACKQ_FLAG_FAST : 0;
    }

    void add(callback_ctx *ctx, unsigned flags = 0)
    {
        ctx->callback_id = ucs_callbackq_add(&m_cbq, callback_proxy,
                                             reinterpret_cast<void*>(ctx),
                                             fast_path_flag() | flags);
    }

    void remove(callback_ctx *ctx)
    {
        ucs_callbackq_remove(&m_cbq, ctx->callback_id);
    }

    void add_safe(callback_ctx *ctx, unsigned flags = 0)
    {
        ctx->callback_id = ucs_callbackq_add_safe(&m_cbq, callback_proxy,
                                                  reinterpret_cast<void*>(ctx),
                                                  fast_path_flag() | flags);
    }

    void remove_safe(callback_ctx *ctx)
    {
        ucs_callbackq_remove_safe(&m_cbq, ctx->callback_id);
    }

    unsigned dispatch(unsigned count = 1)
    {
        unsigned total = 0;
        for (unsigned i = 0; i < count; ++i) {
            total += ucs_callbackq_dispatch(&m_cbq);
        }
        return total;
    }

    ucs_callbackq_t     m_cbq;
};

UCS_TEST_P(test_callbackq, single) {
    callback_ctx ctx;

    init_ctx(&ctx);
    add(&ctx);
    dispatch();
    remove(&ctx);
    EXPECT_EQ(1u, ctx.count);
}

UCS_TEST_P(test_callbackq, count) {
    callback_ctx ctx;

    init_ctx(&ctx);
    add(&ctx);
    unsigned count = dispatch();
    remove(&ctx);
    EXPECT_EQ(1u, ctx.count);
    EXPECT_EQ(1u, count);
}

UCS_TEST_P(test_callbackq, multi) {
    for (unsigned count = 0; count < 20; ++count) {
        callback_ctx ctx[count];
        for (unsigned i = 0; i < count; ++i) {
            init_ctx(&ctx[i]);
            add(&ctx[i]);
        }

        dispatch(10);

        for (unsigned i = 0; i < count; ++i) {
            remove(&ctx[i]);
            EXPECT_EQ(10u, ctx[i].count);
        }
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

    unsigned count = ctx.count;

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

UCS_TEST_P(test_callbackq, oneshot) {
    callback_ctx ctx;

    if (GetParam()) {
        UCS_TEST_SKIP_R("oneshot is only for slow-path");
    }

    init_ctx(&ctx);
    ctx.command = COMMAND_NONE;

    add(&ctx, UCS_CALLBACKQ_FLAG_ONESHOT);
    dispatch(100);
    EXPECT_EQ(1u, ctx.count);
}


UCS_MT_TEST_P(test_callbackq, threads, 10) {

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
            barrier(); /*6*/ /* Next loop barrier*/
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
            barrier(); /*6*/ /* Next loop barrier*/
        }
    }
}

UCS_MT_TEST_P(test_callbackq, remove, 10) {
    static callback_ctx ctx1;

    init_ctx(&ctx1);

    if (barrier()) {
        add_safe(&ctx1);
    }

    sched_yield();

    if (barrier()) {
        callback_ctx ctx2;
        init_ctx(&ctx2);

        dispatch(1000);
        add(&ctx2);
        barrier();
        remove_safe(&ctx1);
        remove_safe(&ctx2);
        dispatch(1);

        /* this should remove all instances of 'ctx', including queued async
         * commands */
        uint32_t count1 = ctx1.count;
        dispatch(100);
        EXPECT_EQ(count1, ctx1.count);
    } else {
        dispatch(100);
        barrier();
    }
}

INSTANTIATE_TEST_CASE_P(fast_path, test_callbackq, ::testing::Values(true));
INSTANTIATE_TEST_CASE_P(slow_path, test_callbackq, ::testing::Values(false));

