/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <deque>

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
        COMMAND_REMOVE_SELF,
        COMMAND_ENQUEUE_USER_ID,
        COMMAND_ADD_ANOTHER,
        COMMAND_ADD_ANOTHER_ONESHOT,
        COMMAND_REMOVE_ANOTHER_ONESHOT,
        COMMAND_NONE
    };

    struct callback_ctx {
        test_callbackq            *test;
        int                       callback_id;
        uint32_t                  count;
        int                       command;
        callback_ctx              *to_add;
        void                      *to_remove;
        unsigned                  flags;
        void                      *key;
        int                       user_id;
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
        ucs_atomic_add32(&m_total_count, 1);

        switch (ctx->command) {
        case COMMAND_REMOVE_SELF:
            remove(ctx);
            break;
        case COMMAND_ADD_ANOTHER:
            add(ctx->to_add);
            break;
        case COMMAND_ADD_ANOTHER_ONESHOT:
            add_oneshot(ctx->to_add);
            break;
        case COMMAND_REMOVE_ANOTHER_ONESHOT:
            remove_oneshot(ctx->to_remove);
            break;
        case COMMAND_ENQUEUE_USER_ID:
            m_user_id_queue.push_back(ctx->user_id);
            break;
        case COMMAND_NONE:
        default:
            break;
        }
        return 1;
    }

    void init_ctx(callback_ctx *ctx, void *key = nullptr, int user_id = 0)
    {
        ctx->test        = this;
        ctx->callback_id = UCS_CALLBACKQ_ID_NULL;
        ctx->count       = 0;
        ctx->command     = COMMAND_NONE;
        ctx->to_add      = nullptr;
        ctx->to_remove   = nullptr;
        ctx->flags       = 0;
        ctx->key         = key;
        ctx->user_id     = user_id;
    }

    virtual unsigned cb_flags() {
        return GetParam();
    }

    void add(callback_ctx *ctx, unsigned flags = 0)
    {
        ctx->callback_id = ucs_callbackq_add(&m_cbq, callback_proxy,
                                             reinterpret_cast<void*>(ctx),
                                             ctx->flags | cb_flags() | flags);
    }

    void remove(int callback_id)
    {
        ucs_callbackq_remove(&m_cbq, callback_id);
    }

    void remove(callback_ctx *ctx)
    {
        remove(ctx->callback_id);
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

    static int remove_pred(const ucs_callbackq_elem_t *elem, void *arg)
    {
        callback_ctx *ctx = reinterpret_cast<callback_ctx*>(elem->arg);
        int user_id       = *reinterpret_cast<int*>(arg);

        /* remove callbacks with the given key */
        return (elem->cb == callback_proxy) && (ctx->user_id == user_id);
    }

    void remove_if(int user_id)
    {
        ucs_callbackq_remove_if(&m_cbq, remove_pred,
                                reinterpret_cast<void*>(&user_id));
    }

    void add_oneshot(callback_ctx *ctx)
    {
        ucs_callbackq_add_oneshot(&m_cbq, ctx->key, callback_proxy,
                                  reinterpret_cast<void*>(ctx));
    }

    void remove_oneshot(void *key = nullptr, int user_id = 0)
    {
        ucs_callbackq_remove_oneshot(&m_cbq, key, remove_pred,
                                     reinterpret_cast<void*>(&user_id));
    }

    unsigned dispatch(unsigned count = 1)
    {
        unsigned total = 0;
        for (unsigned i = 0; i < count; ++i) {
            total += ucs_callbackq_dispatch(&m_cbq);
        }
        return total;
    }

    ucs_callbackq_t  m_cbq;
    uint32_t         m_total_count = 0;
    std::vector<int> m_user_id_queue;
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
    if (cb_flags() & UCS_CALLBACKQ_FLAG_FAST) {
        count++; /* fast CBs are executed immediately after "add" */
    }

    dispatch();
    EXPECT_EQ(2u, ctx.count);
    EXPECT_EQ(count, ctx2.count);

    remove(&ctx);
    dispatch();
    EXPECT_EQ(2u, ctx.count);
    EXPECT_EQ(count + 1, ctx2.count);

    remove(&ctx2);
    dispatch();
    EXPECT_EQ(count + 1, ctx2.count);
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

INSTANTIATE_TEST_SUITE_P(fast_path, test_callbackq, ::
                        testing::Values(static_cast<int>(UCS_CALLBACKQ_FLAG_FAST)));
INSTANTIATE_TEST_SUITE_P(slow_path, test_callbackq, ::testing::Values(0));


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

UCS_TEST_F(test_callbackq_noflags, oneshot_recursive) {
    callback_ctx ctx;

    init_ctx(&ctx);
    ctx.command = COMMAND_ADD_ANOTHER;
    ctx.flags   = UCS_CALLBACKQ_FLAG_ONESHOT;
    ctx.to_add  = &ctx;

    add(&ctx);

    for (unsigned i = 0; i < 10; ++i) {
        dispatch(1);
        EXPECT_LE(i + 1, ctx.count);
    }

    remove(ctx.callback_id);
}

UCS_TEST_F(test_callbackq_noflags, remove_if) {
    const size_t count = 1000;
    const int num_keys = 10;
    std::vector<callback_ctx> ctx(count);
    size_t key_counts[num_keys] = {0};

    for (size_t i = 0; i < count; ++i) {
        init_ctx(&ctx[i], nullptr, ucs::rand() % num_keys);
        add(&ctx[i], (i % 2) ? UCS_CALLBACKQ_FLAG_FAST : 0);
        ++key_counts[ctx[i].user_id];
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

UCS_TEST_F(test_callbackq_noflags, ordering) {
    static const int UNUSED_CB_USER_ID = -1;
    static const int num_callbacks     = 100;
    std::vector<callback_ctx> ctxs(num_callbacks);
    std::deque<int> gc_list;
    std::vector<int> oneshot_callback_keys;

    for (int i = 0; i < num_callbacks; ++i) {
        callback_ctx &r_ctx = ctxs[i];

        // randomize: either permanent callback with key=i or oneshot callback
        // with key=-1
        init_ctx(&r_ctx);
        unsigned cb_flags = 0;
        if (ucs::rand() % 2) {
            // oneshot callback, which must stay in order
            r_ctx.user_id = i;
            r_ctx.command = COMMAND_ENQUEUE_USER_ID;
            cb_flags      = UCS_CALLBACKQ_FLAG_ONESHOT;
            oneshot_callback_keys.push_back(i);
        } else {
            // permanent
            r_ctx.user_id = UNUSED_CB_USER_ID;
            if (ucs::rand() % 2) {
                // do-nothing callback
                r_ctx.command = COMMAND_NONE;
            } else {
                // non-one-shot callback which removes itself - for more fun
                r_ctx.command = COMMAND_REMOVE_SELF;
            }
        }

        add(&r_ctx, cb_flags);

        if (r_ctx.command == COMMAND_NONE) {
            // we need to remove callbacks which don't remove themselves in the
            // end of the test
            gc_list.push_back(r_ctx.callback_id);
        }
    }

    dispatch(10);

    // make sure the ONESHOT callbacks were executed in order
    EXPECT_EQ(oneshot_callback_keys, m_user_id_queue);

    // remove remaining callbacks
    while (!gc_list.empty()) {
        remove(gc_list.front());
        gc_list.pop_front();
    }
}

UCS_MT_TEST_F(test_callbackq_noflags, oneshot_mt, 10)
{
    callback_ctx ctx;

    init_ctx(&ctx);
    ctx.command = COMMAND_NONE;

    add_oneshot(&ctx);
    dispatch(100);
    EXPECT_EQ(1u, ctx.count);
}

UCS_TEST_F(test_callbackq_noflags, oneshot_add_recursive) {
    callback_ctx ctx;

    init_ctx(&ctx);
    ctx.command = COMMAND_ADD_ANOTHER_ONESHOT;
    ctx.to_add  = &ctx;
    add_oneshot(&ctx);

    for (unsigned i = 0; i < 10; ++i) {
        dispatch(1);
        EXPECT_LE(i + 1, ctx.count);
    }

    remove_oneshot();
}

UCS_TEST_F(test_callbackq_noflags, oneshot_remove_self) {
    void *key = this;

    callback_ctx ctx1;
    init_ctx(&ctx1, key);
    ctx1.command   = COMMAND_REMOVE_ANOTHER_ONESHOT;
    ctx1.to_remove = key;
    add_oneshot(&ctx1);

    dispatch(100);
    EXPECT_EQ(1, ctx1.count);
}

UCS_TEST_F(test_callbackq_noflags, oneshot_remove_another) {
    int dummy[2];
    void *key1 = &dummy[0];
    void *key2 = &dummy[1];

    // Each callback removes the other one

    callback_ctx ctx1;
    init_ctx(&ctx1, key1);
    ctx1.command   = COMMAND_REMOVE_ANOTHER_ONESHOT;
    ctx1.to_remove = key2;
    add_oneshot(&ctx1);

    callback_ctx ctx2;
    init_ctx(&ctx2, key2);
    ctx2.command   = COMMAND_REMOVE_ANOTHER_ONESHOT;
    ctx2.to_remove = key1;
    add_oneshot(&ctx2);

    dispatch(100);

    // We don't know the order of the callbacks but only one of them should
    // be called
    EXPECT_EQ(1, ctx1.count + ctx2.count);
}

UCS_TEST_F(test_callbackq_noflags, oneshot_remove_by_key) {
    const size_t count = 1000;
    const int num_keys = 10;
    std::vector<callback_ctx> ctx(count);
    size_t key_counts[num_keys] = {0};
    int keys[num_keys];

    for (size_t i = 0; i < count; ++i) {
        /* The key is the address of key_idx in keys array */
        int key_idx = ucs::rand() % num_keys;
        init_ctx(&ctx[i], &keys[key_idx], i);
        add_oneshot(&ctx[i]);
        ++key_counts[key_idx];
    }

    unsigned remaining_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if ((ucs::rand() % 2) == 0) {
            remove_oneshot(ctx[i].key, i);
        } else {
            ++remaining_count;
        }
    }

    EXPECT_EQ(0, m_total_count);

    dispatch(1);
    EXPECT_EQ(remaining_count, m_total_count);

    dispatch(100);
    EXPECT_EQ(remaining_count, m_total_count);
}

UCS_TEST_F(test_callbackq_noflags, oneshot_ordering) {
    static const int count = 1000;
    void *key              = nullptr;
    std::vector<callback_ctx> ctx(count);

    for (size_t i = 0; i < count; ++i) {
        init_ctx(&ctx[i], key, i);
        ctx[i].command = COMMAND_ENQUEUE_USER_ID;
        add_oneshot(&ctx[i]);
    }

    std::vector<int> remaining_user_ids;
    for (size_t i = 0; i < count; ++i) {
        if ((ucs::rand() % 2) == 0) {
            remove_oneshot(key, i);
        } else {
            remaining_user_ids.push_back(i);
        }
    }

    dispatch(1);
    EXPECT_EQ(remaining_user_ids.size(), m_total_count);
    EXPECT_EQ(remaining_user_ids, m_user_id_queue);

    dispatch(100);
    EXPECT_EQ(remaining_user_ids.size(), m_total_count);
}
