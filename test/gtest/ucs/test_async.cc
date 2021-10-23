/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <common/test_helpers.h>

extern "C" {
#include <ucs/arch/atomic.h>
#include <ucs/async/async.h>
#include <ucs/async/pipe.h>
#include <ucs/sys/sys.h>
}

#include <sys/poll.h>


class base_async {
public:
    base_async(ucs_async_mode_t mode) : m_mode(mode), m_count(0), m_handler_set(0) {
    }

    virtual ~base_async() {
    }

    int count() const {
        return m_count;
    }

    void set_handler() {
        ASSERT_FALSE(m_handler_set);
        m_handler_set = 1;
    }

    void unset_handler(bool sync = true) {
        if (ucs_atomic_cswap32(&m_handler_set, 1, 0)) {
            ucs_status_t status = ucs_async_remove_handler(event_id(), sync);
            ucs_assert_always(status == UCS_OK);
        }
    }

private:
    base_async(const base_async& other);

protected:
    virtual void ack_event() = 0;
    virtual int event_id() = 0;

    static void cb(int id, ucs_event_set_types_t events, void *arg) {
        base_async *self = reinterpret_cast<base_async*>(arg);
        self->handler();
    }

    ucs_async_mode_t mode() const {
        return m_mode;
    }

    virtual void handler() {
        ++m_count;
        ack_event();
    }

    const ucs_async_mode_t m_mode;
    int                    m_count;
    uint32_t               m_handler_set;
};

class base_event : public base_async {
public:
    base_event(ucs_async_mode_t mode) : base_async(mode) {
        ucs_status_t status = ucs_async_pipe_create(&m_event_pipe);
        ASSERT_UCS_OK(status);
    }

    virtual ~base_event() {
        ucs_async_pipe_destroy(&m_event_pipe);
    }

    void set_handler(ucs_async_context_t *async) {
        ucs_status_t status =
            ucs_async_set_event_handler(mode(), event_fd(),
                                        UCS_EVENT_SET_EVREAD,
                                        cb, this, async);
        ASSERT_UCS_OK(status);
        base_async::set_handler();
    }

    virtual int event_id() {
        return event_fd();
    }

    void push_event() {
        ucs_async_pipe_push(&m_event_pipe);
    }

    void reset() {
        ucs_async_pipe_drain(&m_event_pipe);
    }

protected:
    virtual void ack_event() {
        reset();
    }

private:
    int event_fd() {
        return ucs_async_pipe_rfd(&m_event_pipe);
    }

    ucs_async_pipe_t m_event_pipe;
};

class base_timer : public base_async {
public:
    base_timer(ucs_async_mode_t mode) :
        base_async(mode), m_timer_id(-1)
    {
    }

    /*
     * Cannot call this from constructor - vptr not ready!
     */
    void set_timer(ucs_async_context_t *async, ucs_time_t interval) {
        ucs_assert(m_timer_id == -1);
        ucs_status_t status = ucs_async_add_timer(mode(), interval, cb,
                                                  this, async, &m_timer_id);
        ASSERT_UCS_OK(status);
        base_async::set_handler();
    }

    virtual int event_id() {
        return m_timer_id;
    }

protected:
    virtual void ack_event() {
    }

private:
    int          m_timer_id;
};


class async_poll {
public:
    virtual void poll() = 0;
    virtual ~async_poll() {
    }
};

class global : public async_poll {
public:
    virtual void poll() {
        ucs_async_poll(NULL);
    }

    virtual ~global() {
    }
};

class global_event : public global, public base_event {
public:
    global_event(ucs_async_mode_t mode) : base_event(mode) {
        set_handler(NULL);
    }

    ~global_event() {
        unset_handler();
    }
};

class global_timer : public global,  public base_timer {
public:
    global_timer(ucs_async_mode_t mode) : base_timer(mode) {
        set_timer(NULL, ucs_time_from_usec(1000));
    }

    ~global_timer() {
        unset_handler();
    }
};

class local : public async_poll {
public:
    local(ucs_async_mode_t mode) {
        ucs_status_t status = ucs_async_context_init(&m_async, mode);
        ASSERT_UCS_OK(status);
    }

    virtual ~local() {
        ucs_async_context_cleanup(&m_async);
    }

    void block() {
        UCS_ASYNC_BLOCK(&m_async);
    }

    void unblock() {
        UCS_ASYNC_UNBLOCK(&m_async);
    }

    bool is_blocked() const {
        return ucs_async_is_blocked(&m_async);
    }

    void check_miss() {
        ucs_async_check_miss(&m_async);
    }

    virtual void poll() {
        ucs_async_poll(&m_async);
    }

protected:
    ucs_async_context_t m_async;
};

class local_event : public local,
                    public base_event
{
public:
    local_event(ucs_async_mode_t mode) : local(mode), base_event(mode) {
        set_handler(&m_async);
    }

    ~local_event() {
        unset_handler();
    }
};

class local_timer : public local,
                    public base_timer
{
public:
    static const int TIMER_INTERVAL_USEC = 1000;

    local_timer(ucs_async_mode_t mode) : local(mode), base_timer(mode) {
        set_timer(&m_async, ucs_time_from_usec(TIMER_INTERVAL_USEC));
    }

    ~local_timer() {
        unset_handler();
    }
};

class test_async : public testing::TestWithParam<ucs_async_mode_t>,
public ucs::test_base {
public:
    UCS_TEST_BASE_IMPL;

protected:
    static const int      COUNT           = 40;
    static const unsigned SLEEP_USEC      = 1000;
    static const int      NUM_RETRIES     = 100;
    static const int      TIMER_EXP_COUNT = COUNT / 4;

    void suspend(double scale = 1.0) {
        ucs::safe_usleep(ucs_max(scale * SLEEP_USEC, 0) *
                         ucs::test_time_multiplier());
    }

    void suspend_and_poll(async_poll *p, double scale = 1.0) {
        if (GetParam() == UCS_ASYNC_MODE_POLL) {
            for (double t = 0; t < scale; t += 1.0) {
                suspend();
                p->poll();
            }
        } else {
            suspend(scale);
        }
    }

    void suspend_and_poll2(async_poll *p1, async_poll *p2, double scale = 1.0) {
        if (GetParam() == UCS_ASYNC_MODE_POLL) {
            for (double t = 0; t < scale; t += 1.0) {
                suspend();
                p1->poll();
                p2->poll();
            }
        } else {
            suspend(scale);
        }
    }

    template<typename E>
    void expect_count_GE(E& event, int value) {
        for (int retry = 0; retry < NUM_RETRIES; ++retry) {
             suspend_and_poll(&event, COUNT);
             if (event.count() >= value) {
                  return;
             }
             UCS_TEST_MESSAGE << "retry " << (retry + 1);
         }
         EXPECT_GE(event.count(), value) << "after " << int(NUM_RETRIES)
                                         << " retries";
    }

    void check_is_blocked(const local *le, bool expected)
    {
#if UCS_ENABLE_ASSERT
        EXPECT_EQ(expected, le->is_blocked());
#endif
    }
};

template<typename LOCAL>
class test_async_mt : public test_async {
protected:
    static const unsigned NUM_THREADS = 32;

    test_async_mt() {
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            m_ev[i] = NULL;
        }
    }

    virtual void init() {
        pthread_barrier_init(&m_barrier, NULL, NUM_THREADS + 1);
    }

    int thread_run(unsigned index) {
        LOCAL* le;
        m_ev[index] = le = new LOCAL(GetParam());
  
        check_is_blocked(le, false);

        barrier();

        while (!m_stop[index]) {
            le->block();
            check_is_blocked(le, true);
            unsigned before = le->count();
            suspend_and_poll(le, 1.0);
            unsigned after  = le->count();
            le->unblock();

            EXPECT_EQ(before, after); /* Should not handle while blocked */
            le->check_miss();
            suspend_and_poll(le, 1.0);
        }

        check_is_blocked(le, false);

        int result = le->count();
        delete le;
        m_ev[index] = NULL;
        return result;
    }

    void spawn() {
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            m_stop[i] = false;
            pthread_create(&m_threads[i], NULL, thread_func, (void*)this);
        }
        barrier();
    }

    void stop() {
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            m_stop[i] = true;
            void *result;
            pthread_join(m_threads[i], &result);
            m_thread_counts[i] = (int)(uintptr_t)result;
        }
    }

    LOCAL* event(unsigned thread) {
        return m_ev[thread];
    }

    int thread_count(unsigned thread) {
        return m_thread_counts[thread];
    }

    void is_blocked_test()
    {
        spawn();
        suspend();

        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            LOCAL *le = m_ev[i];

            EXPECT_FALSE(le->is_blocked());
            le->block();
            {
                EXPECT_TRUE(le->is_blocked());
            }
            le->unblock();
            EXPECT_FALSE(le->is_blocked());
        }

        stop();
    }

private:
    void barrier() {
        pthread_barrier_wait(&m_barrier);
    }

    static void *thread_func(void *arg)
    {
        test_async_mt *self = reinterpret_cast<test_async_mt*>(arg);

        for (unsigned index = 0; index < NUM_THREADS; ++index) {
            if (self->m_threads[index] == pthread_self()) {
                return (void*)(uintptr_t)self->thread_run(index);
            }
        }

        /* Not found */
        return (void*)-1;
    }

    pthread_t                      m_threads[NUM_THREADS];
    pthread_barrier_t              m_barrier;
    int                            m_thread_counts[NUM_THREADS];
    bool                           m_stop[NUM_THREADS];
    LOCAL*                         m_ev[NUM_THREADS];
};

UCS_TEST_P(test_async, global_event) {
    global_event ge(GetParam());
    ge.push_event();
    expect_count_GE(ge, 1);
}

UCS_TEST_P(test_async, global_timer) {
    global_timer gt(GetParam());
    expect_count_GE(gt, COUNT);
}

UCS_TEST_P(test_async, max_events, "ASYNC_MAX_EVENTS=4") {
    ucs_status_t status;
    ucs_async_context_t async;

    status = ucs_async_context_init(&async, GetParam());
    ASSERT_UCS_OK(status);

    /* 4 timers should be OK */
    std::vector<int> timers;
    for (unsigned count = 0; count < 4; ++count) {
        int timer_id;
        status = ucs_async_add_timer(GetParam(), ucs_time_from_sec(1.0),
                                     (ucs_async_event_cb_t)ucs_empty_function,
                                     NULL, &async, &timer_id);
        ASSERT_UCS_OK(status);
        timers.push_back(timer_id);
    }

    /* 5th timer should fail */
    int timer_id;
    status = ucs_async_add_timer(GetParam(), ucs_time_from_sec(1.0),
                                 (ucs_async_event_cb_t)ucs_empty_function,
                                 NULL, &async, &timer_id);
    EXPECT_EQ(UCS_ERR_EXCEEDS_LIMIT, status);

    if (status == UCS_OK) {
        timers.push_back(timer_id);
    }

    /* Release timers */
    for (std::vector<int>::iterator iter = timers.begin(); iter != timers.end(); ++iter) {
        status = ucs_async_remove_handler(*iter, 1);
        ASSERT_UCS_OK(status);
    }

    ucs_async_context_cleanup(&async);
}

UCS_TEST_P(test_async, many_timers) {
    const int max_iters  = ucs_max(200, 4010 / ucs::test_time_multiplier());
    const int max_timers = ucs_max(10, 250 / ucs::test_time_multiplier());

    std::vector<int> timers;
    timers.reserve(max_timers);

    for (int count = 0; count < max_iters; ++count) {
        for (int count2 = 0; count2 < max_timers; ++count2) {
            ucs_status_t status = ucs_async_add_timer(
                    GetParam(), ucs_time_from_sec(1.0),
                    reinterpret_cast<ucs_async_event_cb_t>(ucs_empty_function),
                    NULL, NULL, &timers[count2]);
            ASSERT_UCS_OK(status);
        }

        for (int count2 = 0; count2 < max_timers; ++count2) {
            ucs_async_remove_handler(timers[count2], 0);
        }
    }
}

UCS_TEST_P(test_async, ctx_event) {
    local_event le(GetParam());
    le.push_event();
    expect_count_GE(le, 1);
}

UCS_TEST_P(test_async, ctx_timer) {
    local_timer lt(GetParam());
    expect_count_GE(lt, TIMER_EXP_COUNT);
}

UCS_TEST_P(test_async, two_timers) {
    local_timer lt1(GetParam());
    local_timer lt2(GetParam());
    for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        suspend_and_poll2(&lt1, &lt2, COUNT * 4);
        if ((lt1.count() >= TIMER_EXP_COUNT) &&
            (lt2.count() >= TIMER_EXP_COUNT)) {
             break;
        }
        UCS_TEST_MESSAGE << "retry " << (retry + 1);
    }
    EXPECT_GE(lt1.count(), int(TIMER_EXP_COUNT));
    EXPECT_GE(lt2.count(), int(TIMER_EXP_COUNT));
}

UCS_TEST_P(test_async, ctx_event_block) {
    local_event le(GetParam());
    int count = 0;

    for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        le.block();
        count = le.count();
        le.push_event();
        suspend_and_poll(&le, COUNT);
        EXPECT_EQ(count, le.count());
        le.unblock();

        le.check_miss();
        if (le.count() > count) {
            break;
        }
        UCS_TEST_MESSAGE << "retry " << (retry + 1);
    }
    EXPECT_GT(le.count(), count);
}

UCS_TEST_P(test_async, ctx_event_block_two_miss) {
    local_event le(GetParam());

    /* Step 1: While async is blocked, generate two events */

    le.block();
    le.push_event();
    suspend_and_poll(&le, COUNT);

    le.push_event();
    suspend_and_poll(&le, COUNT);
    EXPECT_EQ(0, le.count());
    le.unblock();

    /* Step 2: When checking missed events, should get at least one event */

    le.check_miss();
    EXPECT_GT(le.count(), 0);
    int prev_count = le.count();

    /* Step 2: Block the async again and generate an event */

    le.block();
    le.push_event();
    suspend_and_poll(&le, COUNT);
    le.unblock();

    /* Step 2: Check missed events - another event should be found */

    le.check_miss();
    EXPECT_GT(le.count(), prev_count);
}

UCS_TEST_P(test_async, ctx_timer_block) {
    local_timer lt(GetParam());
    int count = 0;

    for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        lt.block();
        count = lt.count();
        suspend_and_poll(&lt, COUNT);
        EXPECT_EQ(count, lt.count());
        lt.unblock();

        lt.check_miss();
        if (lt.count() > count) {
            break;
        }
        UCS_TEST_MESSAGE << "retry " << (retry + 1);
    }
    EXPECT_GT(lt.count(), count); /* Timer could expire again after unblock */
}

UCS_TEST_P(test_async, modify_event) {
    local_event le(GetParam());
    int count;

    le.push_event();
    expect_count_GE(le, 1);

    ucs_async_modify_handler(le.event_id(), 0);
    sleep(1);
    count = le.count();
    le.push_event();
    suspend_and_poll(&le, COUNT);
    EXPECT_EQ(le.count(), count);
    le.reset();

    ucs_async_modify_handler(le.event_id(), UCS_EVENT_SET_EVREAD);
    count = le.count();
    le.push_event();
    expect_count_GE(le, count + 1);

    ucs_async_modify_handler(le.event_id(), 0);
    sleep(1);
    count = le.count();
    le.push_event();
    suspend_and_poll(&le, COUNT);
    EXPECT_EQ(le.count(), count);
}

UCS_TEST_P(test_async, warn_block) {
    {
        scoped_log_handler slh(hide_warns_logger);
        {
            local_event le(GetParam());
            le.block();
        }
    }

    int warn_count = m_warnings.size();
    for (int i = 0; i < warn_count; ++i) {
        UCS_TEST_MESSAGE << "< " << m_warnings[i] << " >";
    }

    if (GetParam() != UCS_ASYNC_MODE_POLL) {
        EXPECT_GE(warn_count, 1);
    }
}

UCS_TEST_P(test_async, check_blocks) {
    local_event le(GetParam());

    check_is_blocked(&le, false);

    le.block();
    {
        check_is_blocked(&le, true);
        le.block();
        {
            check_is_blocked(&le, true);
            le.block();
            {
                check_is_blocked(&le, true);
            }
            le.unblock();
            check_is_blocked(&le, true);
        }
        le.unblock();
        check_is_blocked(&le, true);
    }
    le.unblock();

    check_is_blocked(&le, false);
}

class local_timer_long_handler : public local_timer {
public:
    local_timer_long_handler(ucs_async_mode_t mode, int sleep_usec) :
        local_timer(mode), m_sleep_usec(sleep_usec) {
    }

    virtual void handler() {
        /* The handler would sleep long enough to increment the counter after
         * main thread already considers it removed - unless the main thread
         * waits for handler completion properly.
         * It sleeps only once to avoid timer overrun deadlock in signal mode.
         */
        ucs::safe_usleep(m_sleep_usec * 2);
        m_sleep_usec = 0;
        local_timer::handler();
    }

    int m_sleep_usec;
};

UCS_TEST_P(test_async, remove_sync) {

    /* create another handler so that removing the timer would not have to
     * completely cleanup the async context, and race condition could happen
     */
    local_timer le(GetParam());

    for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        local_timer_long_handler lt(GetParam(), SLEEP_USEC * 2);
        suspend_and_poll(&lt, 1);
        lt.unset_handler(true);
        int count = lt.count();
        suspend_and_poll(&lt, 1);
        ASSERT_EQ(count, lt.count());
    }
}

class local_timer_remove_handler : public local_timer {
public:
    local_timer_remove_handler(ucs_async_mode_t mode) : local_timer(mode) {
    }

protected:
    virtual void handler() {
         base_async::handler();
         unset_handler(false);
    }
};

UCS_TEST_P(test_async, timer_unset_from_handler) {
    local_timer_remove_handler lt(GetParam());

    expect_count_GE(lt, 1);
    suspend_and_poll(&lt, COUNT);
    EXPECT_LE(lt.count(), 5); /* timer could fire multiple times before we remove it */
    int count = lt.count();
    suspend_and_poll(&lt, COUNT);
    EXPECT_EQ(count, lt.count());
}

class local_event_remove_handler : public local_event {
public:
    local_event_remove_handler(ucs_async_mode_t mode, bool sync) :
        local_event(mode), m_sync(sync) {
    }

protected:
    virtual void handler() {
        base_async::handler();
        unset_handler(m_sync);
    }

private:
    bool m_sync;
};

class test_async_event_unset_from_handler : public test_async {
protected:
    void test_unset_from_handler(bool sync) {
        local_event_remove_handler le(GetParam(), sync);

        for (int iter = 0; iter < 5; ++iter) {
            le.push_event();
            expect_count_GE(le, 1);
            EXPECT_EQ(1, le.count());
        }
    }
};

UCS_TEST_P(test_async_event_unset_from_handler, sync) {
    test_unset_from_handler(true);
}

UCS_TEST_P(test_async_event_unset_from_handler, async) {
    test_unset_from_handler(false);
}

class local_event_add_handler : public local_event {
public:
    local_event_add_handler(ucs_async_mode_t mode) :
        local_event(mode), m_event_set(false)
    {
        int ret = pipe(m_pipefd);
        ucs_assertv_always(0 == ret, "%m");
    }

    ~local_event_add_handler() {
        close(m_pipefd[0]);
        close(m_pipefd[1]);
    }

    void unset_handler(int sync) {
        local_event::unset_handler(sync);
        if (m_event_set) {
            ucs_status_t status = ucs_async_remove_handler(m_pipefd[0], sync);
            ucs_assert_always(status == UCS_OK);
            m_event_set = false;
        }
    }

protected:
    static void dummy_cb(int id, ucs_event_set_types_t events, void *arg) {
    }

    virtual void handler() {
        base_async::handler();
        if (!m_event_set) {
            ucs_status_t status =
                    ucs_async_set_event_handler(mode(), m_pipefd[0],
                                                UCS_EVENT_SET_EVREAD,
                                                dummy_cb, this,
                                                &m_async);
            ASSERT_UCS_OK(status);
            m_event_set = true;
        }
    }

    int m_pipefd[2];
    bool m_event_set;
};

UCS_TEST_P(test_async, event_add_from_handler) {
    local_event_add_handler le(GetParam());

    le.push_event();
    sched_yield(); /* let the async handler run, to provoke the race */
    le.unset_handler(1);
}

typedef test_async_mt<local_event> test_async_event_mt;
typedef test_async_mt<local_timer> test_async_timer_mt;

/*
 * Run multiple threads which all process events independently.
 */
UCS_TEST_SKIP_COND_P(test_async_event_mt, multithread,
                     !(HAVE_DECL_F_SETOWN_EX)) {
    const int count         = ucs_max(4, COUNT / ucs::test_time_multiplier());
    const int exp_min_count = (int)(count * 0.5);
    int min_count           = 0;
    for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        spawn();
        for (int j = 0; j < count; ++j) {
            for (unsigned i = 0; i < NUM_THREADS; ++i) {
                event(i)->push_event();
                suspend();
            }
        }
        suspend();
        stop();

        min_count = std::numeric_limits<int>::max();
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            int count = thread_count(i);
            min_count = ucs_min(count, min_count);
        }
        if (min_count >= exp_min_count) {
            break;
        }

        UCS_TEST_MESSAGE << "retry " << (retry + 1);
    }
    EXPECT_GE(min_count, exp_min_count);
}

UCS_TEST_SKIP_COND_P(test_async_event_mt, check_blocks_multithread,
                     // This test blocks async in two threads simultaneously -
                     // poll_block and signal don't allow it
                     (GetParam() == UCS_ASYNC_MODE_POLL) ||
                     (GetParam() == UCS_ASYNC_MODE_SIGNAL))
{
    is_blocked_test();
}

UCS_TEST_P(test_async_timer_mt, multithread) {
    const int exp_min_count = (int)(COUNT * 0.10);
    int min_count = 0;
    for (int retry = 0; retry < NUM_RETRIES; ++retry) {
        spawn();
        suspend(2 * COUNT);
        stop();

        min_count = std::numeric_limits<int>::max();
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            int count = thread_count(i);
            min_count = ucs_min(count, min_count);
        }
        if (min_count >= exp_min_count) {
            break;
        }
    }
    EXPECT_GE(min_count, exp_min_count);
}

std::ostream& operator<<(std::ostream& os, ucs_async_mode_t mode)
{
    return os << ucs_async_mode_names[mode];
}

#define INSTANTIATE_ASYNC_TEST_CASES(_test_fixture) \
    INSTANTIATE_TEST_SUITE_P(signal,          _test_fixture, ::testing::Values(UCS_ASYNC_MODE_SIGNAL)); \
    INSTANTIATE_TEST_SUITE_P(thread_spinlock, _test_fixture, ::testing::Values(UCS_ASYNC_MODE_THREAD_SPINLOCK)); \
    INSTANTIATE_TEST_SUITE_P(thread_mutex,    _test_fixture, ::testing::Values(UCS_ASYNC_MODE_THREAD_MUTEX)); \
    INSTANTIATE_TEST_SUITE_P(poll,            _test_fixture, ::testing::Values(UCS_ASYNC_MODE_POLL));

INSTANTIATE_ASYNC_TEST_CASES(test_async);
INSTANTIATE_ASYNC_TEST_CASES(test_async_event_unset_from_handler);
INSTANTIATE_ASYNC_TEST_CASES(test_async_event_mt);
INSTANTIATE_ASYNC_TEST_CASES(test_async_timer_mt);
