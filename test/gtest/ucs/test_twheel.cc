/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/time/timer_wheel.h>
}

#include <time.h>

/**
 * note: the fast timer precision is dependent on context switch latency!!!
 * expected timer precision 2x wheel resolution plus some time for processing and
 * context switching
 */
class twheel : public ucs::test {
protected:

    struct hr_timer {
        ucs_wtimer_t timer;
        int          tid;
        ucs_time_t   start_time;
        ucs_time_t   end_time;
        ucs_time_t   d;
        ucs_time_t   total_time;
        twheel       *self;
    };

    ucs_twheel_t m_wheel;

    // @override
    virtual void init();

    // @override
    virtual void cleanup();

    static void timer_func(ucs_wtimer_t *self);
    void timer_expired(struct hr_timer *t);
    void add_timer(struct hr_timer *t);
    void init_timer(struct hr_timer *t, int id);
    void init_timerv(struct hr_timer *v, int n);
    void set_timer_delta(struct hr_timer *t, int how);
};

void twheel::init()
{
    ucs_twheel_init(&m_wheel, ucs_time_from_usec(32) * ucs::test_time_multiplier(),
                    ucs_get_time());
    ::srand(::time(NULL));
}

void twheel::cleanup()
{
    ucs_twheel_cleanup(&m_wheel);
}

void twheel::timer_func(ucs_wtimer_t *self)
{
    struct hr_timer *t = ucs_container_of(self, struct hr_timer, timer);
    t->self->timer_expired(t);
}

void twheel::timer_expired(struct hr_timer *t)
{
    t->total_time += (m_wheel.now - t->start_time);
    t->end_time   = m_wheel.now;
}

void twheel::add_timer(struct hr_timer *t)
{
    t->end_time = 0;
    ASSERT_EQ(ucs_wtimer_add(&m_wheel, &t->timer, t->d), UCS_OK);
    t->start_time = ucs_get_time();
}

void twheel::init_timer(struct hr_timer *t, int id)
{
    t->tid        = id;
    t->total_time = 0;
    t->self       = this;
    ucs_wtimer_init(&t->timer, timer_func);
}

void twheel::init_timerv(struct hr_timer *v, int n)
{
    for (int i = 0; i < n; i++) {
        init_timer(&v[i], i);
    }
}

void twheel::set_timer_delta(struct hr_timer *t, int how)
{
    int slot;

    switch (how) {
    case 0:
        /* first */
        slot = 1;
        break;
    case 1:
        /* last */
        slot = m_wheel.num_slots - 1;
        break;
    case 2:
        /* middle */
        slot = m_wheel.num_slots / 2;
        break;
    case -2:
        /* overflow */
        slot = m_wheel.num_slots + (ucs::rand() % 1000000);
        break;
    default:
        slot = 1 + ucs::rand() % (m_wheel.num_slots - 2);
        break;
    }

    if (how == -2) {
        t->d = m_wheel.res + m_wheel.res * (m_wheel.num_slots - 1) / 2;
    } else {
        t->d = m_wheel.res + m_wheel.res * slot / 2;
    }
}

#define N_LOOPS 20

UCS_TEST_F(twheel, precision_single) {
    UCS_TEST_SKIP; // Test is broken

#if 0
    struct hr_timer t;
    ucs_time_t now;
    int i, k;
    int fail_count;

    init_timer(&t, 0);
    for (k = 0; k < 10; k++ ) {
        set_timer_delta(&t, k);
        fail_count = 0;
        for (i = 0; i < N_LOOPS; i++) {
            t.total_time = 0;
            add_timer(&t);
            do {
                now = ucs_get_time();
                ucs_twheel_sweep(&m_wheel, now);
            } while (t.end_time == 0);

            if ((ucs_time_t)::abs(t.total_time - t.d) > 2 * m_wheel.res) {
                ++fail_count;
            }
        }
        EXPECT_LE(fail_count, N_LOOPS / 3);
    }
#endif
}

#define N_TIMERS 10000

UCS_TEST_F(twheel, precision_multi) {
    std::vector<struct hr_timer> t(N_TIMERS);

    UCS_TEST_SKIP; // Test is broken

#if 0
    ucs_time_t start, now, eps;
    init_timerv(&t[0], N_TIMERS);
    for (int i = 0; i < N_TIMERS; i++) {
        set_timer_delta(&t[i], i);
        add_timer(&t[i]);
    }

    start = ucs_get_time();
    /* all timers were delayed by at most eps */
    eps = start - m_wheel.now;
    do {
        now = ucs_get_time();
        ucs_twheel_sweep(&m_wheel, now);
    } while (now < start + m_wheel.res * m_wheel.num_slots);

    /* all timers should ve been triggered
     * correct delta
     */
    for (int i = 0; i < N_TIMERS; i++) {
        EXPECT_NE(t[i].end_time, (ucs_time_t)0);
        EXPECT_NEAR(t[i].total_time,  t[i].d, 2 * m_wheel.res + eps);
    }
#endif
}

UCS_TEST_F(twheel, add_twice) {
    struct hr_timer t;

    init_timer(&t, 0);

    set_timer_delta(&t, -1);
    add_timer(&t);

    set_timer_delta(&t, -1);
    EXPECT_EQ(ucs_wtimer_add(&m_wheel, &t.timer, t.d), UCS_ERR_BUSY);
    do {
        ucs_twheel_sweep(&m_wheel, ucs_get_time());
        /* coverity[loop_condition] */
    } while(t.end_time == 0);
}


UCS_TEST_F(twheel, add_overflow) {

    UCS_TEST_SKIP; // Test is broken

#if 0
    struct hr_timer t;
    init_timer(&t, 0);
    ::srand(::time(NULL));

    t.total_time = 0;
    set_timer_delta(&t, -2);
    for (int i = 0; i < N_LOOPS; i++) {
        add_timer(&t);
        do {
            ucs_twheel_sweep(&m_wheel, ucs_get_time());
        } while (t.end_time == 0);
    }
    EXPECT_NEAR(t.total_time , t.d * N_LOOPS, 4 * N_LOOPS * m_wheel.res);
#endif
}

UCS_TEST_F(twheel, delayed_sweep) {
    std::vector<struct hr_timer> t(N_TIMERS);

    init_timerv(&t[0], N_TIMERS);
    for (int i = 0; i < N_TIMERS; i++) {
        set_timer_delta(&t[i], i);
        add_timer(&t[i]);
    }

    sleep(1);

    ucs_twheel_sweep(&m_wheel, ucs_get_time());

    /* all timers should have been triggered */
    for (int i = 0; i < N_TIMERS; i++) {
        EXPECT_NE(t[i].end_time, (ucs_time_t)0);
    }
}

