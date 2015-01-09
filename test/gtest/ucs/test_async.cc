/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
 *
 * $COPYRIGHT$
 * $HEADER$
 */

#include <ucs/gtest/test.h>
#include <ucs/gtest/test_helpers.h>

extern "C" {
#include <ucs/async/async.h>
#include <ucs/async/pipe.h>
}

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <sys/poll.h>


class base {
public:
    base(ucs_async_mode_t mode) : m_mode(mode), m_count(0) {
    }

    virtual ~base() {
    }

    int count() const {
        return m_count;
    }
private:
    base(const base& other);

protected:
    virtual void ack_event() = 0;

    static void cb(void *arg) {
        base *self = reinterpret_cast<base*>(arg);
        self->handler();
    }

    ucs_async_mode_t mode() const {
        return m_mode;
    }

private:
    void handler() {
        ++m_count;
        ack_event();
    }

    const ucs_async_mode_t m_mode;
    int                    m_count;
};

class base_event : public base {
public:
    base_event(ucs_async_mode_t mode) : base(mode) {
        ucs_status_t status = ucs_async_pipe_create(&m_event_pipe);
        ASSERT_UCS_OK(status);
    }

    virtual ~base_event() {
        ucs_async_pipe_destroy(&m_event_pipe);
    }

    void set_handler(ucs_async_context_t *async) {
        ucs_status_t status = ucs_async_set_event_handler(mode(), event_fd(),
                                                          POLLIN, cb, this,
                                                          async);
        ASSERT_UCS_OK(status);
    }

    void unset_handler() {
        ucs_status_t status = ucs_async_unset_event_handler(event_fd());
        ASSERT_UCS_OK(status);
    }

    void push_event() {
        ucs_async_pipe_push(&m_event_pipe);
    }

protected:
    virtual void ack_event() {
        ucs_async_pipe_drain(&m_event_pipe);
    }

private:
    int event_fd() {
        return ucs_async_pipe_rfd(&m_event_pipe);
    }

    ucs_async_pipe_t m_event_pipe;
};

class base_timer : public base {
public:
    base_timer(ucs_async_mode_t mode) :
        base(mode), m_timer_id(-1)
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
    }

    void unset_timer() {
        if (m_timer_id != -1) {
            ucs_status_t status = ucs_async_remove_timer(m_timer_id);
            ASSERT_UCS_OK(status);
            m_timer_id = -1;
        }
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
};

class global : public async_poll {
public:
    virtual void poll() {
        ucs_async_poll(NULL);
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
    }

    ~global_timer() {
        unset_timer();
    }

    void start_timer(ucs_time_t interval) {
        set_timer(NULL, interval);
    }
};

class local : public async_poll {
public:
    local(ucs_async_mode_t mode) {
        ucs_status_t status = ucs_async_context_init(&m_async, mode);
        ASSERT_UCS_OK(status);
    }

    ~local() {
        ucs_async_context_cleanup(&m_async);
    }

    void block() {
        UCS_ASYNC_BLOCK(&m_async);
    }

    void unblock() {
        UCS_ASYNC_UNBLOCK(&m_async);
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
    local_timer(ucs_async_mode_t mode) : local(mode), base_timer(mode) {
    }

    ~local_timer() {
        unset_timer();
    }

    void start_timer(ucs_time_t interval) {
        set_timer(&m_async, interval);
    }

    void stop_timer() {
        unset_timer();
    }
};

class test_async : public testing::TestWithParam<ucs_async_mode_t>,
public ucs::test_base {
public:
    UCS_TEST_BASE_IMPL;

protected:
    static const unsigned COUNT       = 40;
    static const unsigned SLEEP_USEC  = 1000;

    void suspend(double scale = 1.0) {
        ucs::safe_usleep(scale * SLEEP_USEC * ucs::test_time_multiplier());
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
};

template<typename LOCAL>
class test_async_mt : public test_async {
protected:
    static const unsigned NUM_THREADS = 32;

    virtual void init() {
        pthread_barrier_init(&m_barrier, NULL, NUM_THREADS + 1);
    }

    int thread_run(unsigned index) {
        boost::shared_ptr<LOCAL> le;
        m_ev[index] = le = boost::make_shared<LOCAL>(GetParam());

        barrier();

        while (!m_stop[index]) {
            le->block();
            unsigned before = le->count();
            suspend_and_poll(le.get(), 0.5);
            unsigned after  = le->count();
            le->unblock();

            EXPECT_EQ(before, after); /* Should not handle while blocked */
            le->check_miss();
            suspend_and_poll(le.get(), 0.5);
        }

        return le->count();
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
        return m_ev[thread].get();
    }

    int ctx_count(unsigned thread) {
        return m_ev[thread]->count();
    }

    int thread_count(unsigned thread) {
        return m_thread_counts[thread];
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
    boost::shared_ptr<LOCAL>       m_ev[NUM_THREADS];
};


UCS_TEST_P(test_async, global_event) {
    global_event ge(GetParam());
    ge.push_event();
    suspend_and_poll(&ge);
    EXPECT_EQ(1, ge.count());
}

UCS_TEST_P(test_async, global_timer) {
    global_timer gt(GetParam());
    gt.start_timer(ucs_time_from_usec(SLEEP_USEC));
    suspend_and_poll(&gt, COUNT);
    EXPECT_GE(gt.count(), COUNT / 2);
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
                                     (ucs_notifier_chain_func_t)ucs_empty_function,
                                     NULL, &async, &timer_id);
        ASSERT_UCS_OK(status);
        timers.push_back(timer_id);
    }

    /* 5th timer should fail */
    int timer_id;
    status = ucs_async_add_timer(GetParam(), ucs_time_from_sec(1.0),
                                 (ucs_notifier_chain_func_t)ucs_empty_function,
                                 NULL, &async, &timer_id);
    EXPECT_EQ(UCS_ERR_EXCEEDS_LIMIT, status);

    if (status == UCS_OK) {
        timers.push_back(timer_id);
    }

    /* Release timers */
    for (std::vector<int>::iterator iter = timers.begin(); iter != timers.end(); ++iter) {
        status = ucs_async_remove_timer(*iter);
        ASSERT_UCS_OK(status);
    }

    ucs_async_context_cleanup(&async);
}

UCS_TEST_P(test_async, ctx_event) {
    local_event le(GetParam());
    le.push_event();
    suspend_and_poll(&le);
    EXPECT_EQ(1, le.count());
}

UCS_TEST_P(test_async, ctx_timer) {
    local_timer lt(GetParam());
    lt.start_timer(ucs_time_from_usec(SLEEP_USEC));
    suspend_and_poll(&lt, COUNT);
    EXPECT_GE(lt.count(), COUNT / 2);
}

UCS_TEST_P(test_async, two_timers) {
    local_timer lt1(GetParam());
    local_timer lt2(GetParam());
    lt1.start_timer(ucs_time_from_usec(SLEEP_USEC));
    lt2.start_timer(ucs_time_from_usec(SLEEP_USEC));
    suspend_and_poll2(&lt1, &lt2, COUNT);
    EXPECT_GE(lt1.count(), COUNT / 2);
    EXPECT_GE(lt2.count(), COUNT / 2);
}

UCS_TEST_P(test_async, ctx_event_block) {
    local_event le(GetParam());

    le.block();
    le.push_event();
    suspend_and_poll(&le);
    le.unblock();

    EXPECT_EQ(0, le.count());
    le.check_miss();
    EXPECT_EQ(1, le.count());
}

UCS_TEST_P(test_async, ctx_timer_block) {
    local_timer lt(GetParam());

    lt.block();
    lt.start_timer(ucs_time_from_usec(SLEEP_USEC));
    suspend_and_poll(&lt, COUNT);
    EXPECT_EQ(0, lt.count());
    lt.unblock();

    lt.check_miss();
    EXPECT_GE(lt.count(), 1); /* Timer could expire again after unblock */
    lt.stop_timer();
}

typedef test_async_mt<local_event> test_async_event_mt;
typedef test_async_mt<local_timer> test_async_timer_mt;

/*
 * Run multiple threads which all process events independently.
 */
UCS_TEST_P(test_async_event_mt, multithread) {
    spawn();

    for (unsigned j = 0; j < COUNT; ++j) {
        for (unsigned i = 0; i < NUM_THREADS; ++i) {
            event(i)->push_event();
            suspend();
        }
    }

    suspend();

    stop();

    for (unsigned i = 0; i < NUM_THREADS; ++i) {
        int count = thread_count(i);
        ASSERT_EQ(ctx_count(i), count);
        EXPECT_GE(count, (unsigned)(COUNT * 0.75));
    }
}
UCS_TEST_P(test_async_timer_mt, multithread) {
    spawn();

    for (unsigned i = 0; i < NUM_THREADS; ++i) {
        event(i)->start_timer(ucs_time_from_usec(SLEEP_USEC));
    }

    suspend(2 * COUNT);

    /* First stop the timers, to get exact counts */
    for (unsigned i = 0; i < NUM_THREADS; ++i) {
        event(i)->stop_timer();
    }

    stop();

    for (unsigned i = 0; i < NUM_THREADS; ++i) {
        int count = thread_count(i);
        ASSERT_EQ(ctx_count(i), count);
        EXPECT_GE(count, (unsigned)(COUNT * 0.5));
    }
}

INSTANTIATE_TEST_CASE_P(signal, test_async, ::testing::Values(UCS_ASYNC_MODE_SIGNAL));
INSTANTIATE_TEST_CASE_P(thread, test_async, ::testing::Values(UCS_ASYNC_MODE_THREAD));
INSTANTIATE_TEST_CASE_P(poll,   test_async, ::testing::Values(UCS_ASYNC_MODE_POLL));
INSTANTIATE_TEST_CASE_P(signal, test_async_event_mt, ::testing::Values(UCS_ASYNC_MODE_SIGNAL));
INSTANTIATE_TEST_CASE_P(thread, test_async_event_mt, ::testing::Values(UCS_ASYNC_MODE_THREAD));
INSTANTIATE_TEST_CASE_P(poll,   test_async_event_mt, ::testing::Values(UCS_ASYNC_MODE_POLL));
INSTANTIATE_TEST_CASE_P(signal, test_async_timer_mt, ::testing::Values(UCS_ASYNC_MODE_SIGNAL));
INSTANTIATE_TEST_CASE_P(thread, test_async_timer_mt, ::testing::Values(UCS_ASYNC_MODE_THREAD));
INSTANTIATE_TEST_CASE_P(poll,   test_async_timer_mt, ::testing::Values(UCS_ASYNC_MODE_POLL));
