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
    public ::testing::TestWithParam<int> {
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
        int                       key;
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

    void init_ctx(callback_ctx *ctx, int key = 0)
    {
        ctx->test        = this;
        ctx->count       = 0;
        ctx->command     = COMMAND_NONE;
        ctx->callback_id = UCS_CALLBACKQ_ID_NULL;
        ctx->key         = key;
    }

    virtual unsigned cb_flags() {
        return GetParam();
    }

    void add(callback_ctx *ctx, unsigned flags = 0)
    {
        ctx->callback_id = ucs_callbackq_add(&m_cbq, callback_proxy,
                                             reinterpret_cast<void*>(ctx),
                                             cb_flags() | flags);
    }

    void remove(callback_ctx *ctx)
    {
        ucs_callbackq_remove(&m_cbq, ctx->callback_id);
    }

    void add_safe(callback_ctx *ctx, unsigned flags = 0)
    {
        ctx->callback_id = ucs_callbackq_add_safe(&m_cbq, callback_proxy,
                                                  reinterpret_cast<void*>(ctx),
                                                  cb_flags() | flags);
    }

    void remove_safe(callback_ctx *ctx)
    {
        ucs_callbackq_remove_safe(&m_cbq, ctx->callback_id);
    }

    static int remove_if_pred(const ucs_callbackq_elem_t *elem, void *arg)
    {
        callback_ctx *ctx = reinterpret_cast<callback_ctx*>(elem->arg);
        int key = *reinterpret_cast<int*>(arg);

        /* remove callbacks with the given key */
        return (elem->cb == callback_proxy) && (ctx->key == key);
    }

    void remove_if(int key)
    {
        ucs_callbackq_remove_if(&m_cbq, remove_if_pred,
                                reinterpret_cast<void*>(&key));
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

    if (barrier()) /*1*/ {
        add_safe(&ctx1);
        dispatch(100);
        barrier(); /*2*/
        remove_safe(&ctx1);
        dispatch(1);

        uint32_t count1 = ctx1.count;
        dispatch(100);
        EXPECT_EQ(count1, ctx1.count);

        barrier();/*3*/
        dispatch(1); /* will remove ctx2 on other threads */
        barrier();/*4*/

        barrier();/*5*/
        dispatch(100);
        barrier();/*6*/
    } else {
        callback_ctx ctx2;
        init_ctx(&ctx2);
        add_safe(&ctx2);
        barrier(); /*2*/

        /* ask to ctx2 and wait until the main thread actually removes it */
        remove_safe(&ctx2);
        barrier(); /*3*/
        barrier(); /*4*/

        /* make sure ctx2 is not dispatched */
        uint32_t count1 = ctx2.count;
        barrier();/*5*/
        barrier();/*6*/
        EXPECT_EQ(count1, ctx2.count);
    }
}

INSTANTIATE_TEST_CASE_P(fast_path, test_callbackq, ::
                        testing::Values(static_cast<int>(UCS_CALLBACKQ_FLAG_FAST)));
INSTANTIATE_TEST_CASE_P(slow_path, test_callbackq, ::testing::Values(0));


class test_callbackq_noflags : public test_callbackq {
protected:
    virtual unsigned cb_flags() {
        return 0;
    }
};

UCS_TEST_F(test_callbackq_noflags, oneshot) {
    callback_ctx ctx;

    init_ctx(&ctx);
    ctx.command = COMMAND_NONE;

    add(&ctx, UCS_CALLBACKQ_FLAG_ONESHOT);
    dispatch(100);
    EXPECT_EQ(1u, ctx.count);
}

UCS_TEST_F(test_callbackq_noflags, remove_if) {
    const size_t count = 1000;
    const int num_keys = 10;
    std::vector<callback_ctx> ctx(count);
    size_t key_counts[num_keys] = {0};

    for (size_t i = 0; i < count; ++i) {
        init_ctx(&ctx[i], ucs::rand() % num_keys);
        add(&ctx[i], (i % 2) ? UCS_CALLBACKQ_FLAG_FAST : 0);
        ++key_counts[ctx[i].key];
    }

    /* calculate how many callbacks expected to remain after removing each of
     * the keys.
     */
    size_t exp_count[num_keys] = {0};
    for (int key = num_keys - 2; key >= 0; --key) {
        exp_count[key] = exp_count[key + 1] + key_counts[key + 1];
    }

    /* remove keys one after another and make sure the exact expected number
     * of callbacks is being called after every removal.
     */
    for (int key = 0; key < num_keys; ++key) {
        remove_if(key);

        /* count how many different callbacks were called */
        size_t num_cbs = 0;
        dispatch(1000);
        for (size_t i = 0; i < count; ++i) {
            num_cbs += !!ctx[i].count;
            ctx[i].count = 0; /* reset for next iteration */
        }

        EXPECT_EQ(exp_count[key], num_cbs) << "key=" << key;
    }
}

