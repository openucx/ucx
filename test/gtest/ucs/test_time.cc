/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/time/timerq.h>
}

#include <time.h>

class test_time : public ucs::test {
};

UCS_TEST_F(test_time, time_calc) {
    double value = ucs::rand() % UCS_USEC_PER_SEC;

    EXPECT_NEAR(value * 1000ull, ucs_time_to_msec(ucs_time_from_sec (value)), 0.000001);
    EXPECT_NEAR(value * 1000ull, ucs_time_to_usec(ucs_time_from_msec(value)), 0.02);
    EXPECT_NEAR(value * 1000ull, ucs_time_to_nsec(ucs_time_from_usec(value)), 20.0);
}

/* This test is only useful when used with high-precision timers */
#if HAVE_HW_TIMER
UCS_TEST_SKIP_COND_F(test_time, get_time,
                     (ucs::test_time_multiplier() > 1)) {
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
    if (ucs::perf_retry_count) {
        ASSERT_NEAR(1.0, time(NULL) - system_start_time, 1.00001);

        double nsec = (ucs_time_to_nsec(current_time - start_time)) / count;
        EXPECT_LT(nsec, 40.0) << "ucs_get_time() performance is too bad";
    }
}
#endif

UCS_TEST_F(test_time, timerq) {
    int timer_id1;
    int timer_id2;

    ucs_timer_queue_t timerq;
    ucs_status_t status;

    for (unsigned test_count = 0; test_count < 500; ++test_count) {

        const ucs_time_t interval1 = (ucs::rand() % 20) + 1;
        const ucs_time_t interval2 = (ucs::rand() % 20) + 1;
        const ucs_time_t test_time = ucs::rand() % 10000;
        const ucs_time_t time_base = ucs::rand();
        ucs_timer_t *timer;
        unsigned counter1, counter2;

        status = ucs_timerq_init(&timerq, "timer_test");
        ASSERT_UCS_OK(status);

        EXPECT_TRUE(ucs_timerq_is_empty(&timerq));
        EXPECT_EQ(UCS_TIME_INFINITY, ucs_timerq_min_interval(&timerq));

        ucs_time_t current_time = time_base;

        ucs_timerq_add(&timerq, interval1, &timer_id1);
        ucs_timerq_add(&timerq, interval2, &timer_id2);

        EXPECT_FALSE(ucs_timerq_is_empty(&timerq));
        EXPECT_EQ(std::min(interval1, interval2), ucs_timerq_min_interval(&timerq));

        /*
         * Check that both timers are invoked
         */
        counter1 = 0;
        counter2 = 0;
        for (unsigned count = 0; count < test_time; ++count) {
            ++current_time;
	    ucs_timerq_for_each_expired(timer, &timerq, current_time, {
                if (timer->id == timer_id1) ++counter1;
                if (timer->id == timer_id2) ++counter2;
            })
        }
        EXPECT_NEAR(test_time / interval1, counter1, 1);
        EXPECT_NEAR(test_time / interval2, counter2, 1);

        /*
         * Check that after canceling, only one timer is invoked
         */
        counter1 = 0;
        counter2 = 0;
        status = ucs_timerq_remove(&timerq, timer_id1);
        ASSERT_UCS_OK(status);
        for (unsigned count = 0; count < test_time; ++count) {
            ++current_time;
            ucs_timerq_for_each_expired(timer, &timerq, current_time, {
                if (timer->id == timer_id1) ++counter1;
                if (timer->id == timer_id2) ++counter2;
            })
        }
        EXPECT_EQ(0u, counter1);
        EXPECT_NEAR(test_time / interval2, counter2, 1);
        EXPECT_EQ(interval2, ucs_timerq_min_interval(&timerq));

        /*
         * Check that after rescheduling, both timers are invoked again
         */
        ucs_timerq_add(&timerq, interval1, &timer_id1);

        counter1 = 0;
        counter2 = 0;
        for (unsigned count = 0; count < test_time; ++count) {
            ++current_time;
            ucs_timerq_for_each_expired(timer, &timerq, current_time, {
                if (timer->id == timer_id1) ++counter1;
                if (timer->id == timer_id2) ++counter2;
            })
        }
        EXPECT_NEAR(test_time / interval1, counter1, 1);
        EXPECT_NEAR(test_time / interval2, counter2, 1);

        status = ucs_timerq_remove(&timerq, timer_id1);
        ASSERT_UCS_OK(status);
        status = ucs_timerq_remove(&timerq, timer_id2);
        ASSERT_UCS_OK(status);

        ucs_timerq_cleanup(&timerq);
    }
}


