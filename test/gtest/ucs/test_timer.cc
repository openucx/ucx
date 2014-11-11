/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
extern "C" {
#include <ucs/time/timerq.h>
}

#include <time.h>

class test_timer : public ucs::test {
protected:
    struct timer_with_counter {
        ucs_callback_t cb;
        unsigned       counter;
    };

    static void timer_func(ucs_callback_t *self)
    {
        struct timer_with_counter *ctx =
                        ucs_container_of(self, struct timer_with_counter, cb);
        ++ctx->counter;
    }
};

UCS_TEST_F(test_timer, time_calc) {
    double value = rand() % UCS_USEC_PER_SEC;

    EXPECT_NEAR(value * 1000, ucs_time_to_msec(ucs_time_from_sec (value)), 0.000001);
    EXPECT_NEAR(value * 1000, ucs_time_to_usec(ucs_time_from_msec(value)), 0.001);
    EXPECT_NEAR(value * 1000, ucs_time_to_nsec(ucs_time_from_usec(value)), 1.0);
}

UCS_TEST_F(test_timer, get_time) {
    if (ucs::test_time_multiplier() > 1) {
        UCS_TEST_SKIP;
    }

    ucs_time_t time1 = ucs_get_time();
    ucs_time_t time2 = ucs_get_time();
    EXPECT_GE(time2, time1);

    ucs_time_t start_time = ucs_get_time();
    ucs_time_t end_time = start_time + ucs_time_from_sec(1);
    ucs_time_t current_time;

    time_t system_start_time = time(NULL);

    uint64_t count = 0;
    do {
        current_time = ucs_get_time();
        ++count;
    } while (current_time <= end_time);

    /* Check the sleep interval is correct */
    ASSERT_NEAR(1.0, time(NULL) - system_start_time, 0.000001);

    double nsec = (ucs_time_to_nsec(current_time - start_time)) / count;
    EXPECT_LT(nsec, 20.0) << "ucs_get_time() performance is too bad";
}

UCS_TEST_F(test_timer, time_shift) {

    double accuracy_usec = ucs_time_to_usec(1ull << ucs_get_short_time_shift());

    UCS_TEST_MESSAGE << "Short time shift is " << ucs_get_short_time_shift() << " bit," <<
                    " accuracy is " << accuracy_usec << " usec";
    EXPECT_LT(accuracy_usec, 1000.0);
}

UCS_TEST_F(test_timer, timerq_basic) {
    ucs_timer_queue_t timerq;
    ucs_status_t status;

    ::srand(::time(NULL));
    for (unsigned test_count = 0; test_count < 500; ++test_count) {

        const ucs_time_t interval1 = (::rand() % 20) + 1;
        const ucs_time_t interval2 = (::rand() % 20) + 1;
        const ucs_time_t test_time = ::rand() % 10000;
        const ucs_time_t time_base = ::rand();

        struct timer_with_counter timerctx1;
        struct timer_with_counter timerctx2;

        status = ucs_timerq_init(&timerq);
        ASSERT_UCS_OK(status);

        timerctx1.cb.func = timer_func;
        timerctx2.cb.func = timer_func;

        ucs_time_t current_time = time_base;

        ucs_timer_add(&timerq, &timerctx1.cb, interval1);
        ucs_timer_add(&timerq, &timerctx2.cb, interval2);

        /*
         * Check that both timers are invoked
         */
        timerctx1.counter = 0;
        timerctx2.counter = 0;
        for (unsigned count = 0; count < test_time; ++count) {
            ++current_time;
            ucs_timerq_sweep(&timerq, current_time);
        }
        EXPECT_NEAR(test_time / interval1, timerctx1.counter, 1);
        EXPECT_NEAR(test_time / interval2, timerctx2.counter, 1);

        /*
         * Check that after canceling, only one timer is invoked
         */
        timerctx1.counter = 0;
        timerctx2.counter = 0;
        ucs_timer_remove(&timerq, &timerctx1.cb);
        for (unsigned count = 0; count < test_time; ++count) {
            ++current_time;
            ucs_timerq_sweep(&timerq, current_time);
        }
        EXPECT_EQ(0u, timerctx1.counter);
        EXPECT_NEAR(test_time / interval2, timerctx2.counter, 1);

        /*
         * Check that after rescheduling, both timers are invoked again
         */
        ucs_timer_add(&timerq, &timerctx1.cb, interval1);

        timerctx1.counter = 0;
        timerctx2.counter = 0;
        for (unsigned count = 0; count < test_time; ++count) {
            ++current_time;
            ucs_timerq_sweep(&timerq, current_time);
        }
        EXPECT_NEAR(test_time / interval1, timerctx1.counter, 1);
        EXPECT_NEAR(test_time / interval2, timerctx2.counter, 1);

        ucs_timer_remove(&timerq, &timerctx1.cb);
        ucs_timer_remove(&timerq, &timerctx2.cb);

        ucs_timerq_cleanup(&timerq);
    }
}


