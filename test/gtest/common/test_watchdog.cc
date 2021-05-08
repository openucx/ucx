/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <common/test.h>

class test_watchdog : public ucs::test {
public:
    void reset_to_default() {
        ucs::watchdog_signal();
        // all have to be set to their default values
        EXPECT_EQ(ucs::WATCHDOG_RUN, ucs::watchdog_get_state());
        EXPECT_EQ(ucs::watchdog_timeout, ucs::watchdog_get_timeout());
    }
};

UCS_TEST_F(test_watchdog, watchdog_set) {
    EXPECT_EQ(ucs::WATCHDOG_RUN, ucs::watchdog_get_state());
    EXPECT_EQ(ucs::watchdog_timeout, ucs::watchdog_get_timeout());
    EXPECT_EQ(SIGABRT, ucs::watchdog_get_kill_signal());

    ucs::watchdog_set(ucs::WATCHDOG_TEST);
    // when the test state is applied, the watchdog
    // changes state to WATCHDOG_DEFAULT_SET
    EXPECT_EQ(ucs::WATCHDOG_DEFAULT_SET, ucs::watchdog_get_state());
    EXPECT_EQ(ucs::watchdog_timeout, ucs::watchdog_get_timeout());
    EXPECT_EQ(SIGTERM, ucs::watchdog_get_kill_signal());

    reset_to_default();

    ucs::watchdog_set(500.);
    EXPECT_EQ(ucs::WATCHDOG_DEFAULT_SET, ucs::watchdog_get_state());
    EXPECT_EQ(500., ucs::watchdog_get_timeout());
    EXPECT_EQ(SIGABRT, ucs::watchdog_get_kill_signal());

    reset_to_default();

    ucs::watchdog_set(ucs::WATCHDOG_TEST, 100.);
    // when the test state and the timeout are applied,
    // the watchdog changes state to WATCHDOG_DEFAULT_SET
    EXPECT_EQ(ucs::WATCHDOG_DEFAULT_SET, ucs::watchdog_get_state());
    EXPECT_EQ(100., ucs::watchdog_get_timeout());
    EXPECT_EQ(SIGTERM, ucs::watchdog_get_kill_signal());

    reset_to_default();

    ucs::watchdog_set(ucs::WATCHDOG_DEFAULT_SET, 200.);
    // when the timeout and the timeout applied, the watchdog
    // changes state to WATCHDOG_DEFAULT_SET
    EXPECT_EQ(ucs::WATCHDOG_RUN, ucs::watchdog_get_state());
    EXPECT_EQ(ucs::watchdog_timeout, ucs::watchdog_get_timeout());
    EXPECT_EQ(SIGABRT, ucs::watchdog_get_kill_signal());

    ucs::watchdog_set(ucs::WATCHDOG_DEFAULT_SET);
    EXPECT_EQ(ucs::WATCHDOG_RUN, ucs::watchdog_get_state());
    EXPECT_EQ(ucs::watchdog_timeout, ucs::watchdog_get_timeout());
    EXPECT_EQ(SIGABRT, ucs::watchdog_get_kill_signal());
}

UCS_TEST_F(test_watchdog, watchdog_signal) {
    for (int i = 0; i < 10; i++) {
        ucs::watchdog_signal();
    }

    EXPECT_EQ(ucs::WATCHDOG_RUN, ucs::watchdog_get_state());
}

UCS_TEST_F(test_watchdog, watchdog_timeout) {
    double timeout, sleep_time;
    char *gtest_timeout, *gtest_sleep_time;

    /* This test can not be run with the other tests
     * because it terminates testing due to timeout
     */
    gtest_timeout = getenv("WATCHDOG_GTEST_TIMEOUT_");
    if (gtest_timeout == NULL) {
        UCS_TEST_SKIP_R("WATCHDOG_GTEST_TIMEOUT_ is not set");
    }
    timeout = atof(gtest_timeout);

    gtest_sleep_time = getenv("WATCHDOG_GTEST_SLEEP_TIME_");
    if (gtest_sleep_time == NULL) {
        UCS_TEST_SKIP_R("WATCHDOG_GTEST_SLEEP_TIME_ is not set");
    }
    sleep_time = atof(gtest_sleep_time);

    ucs::watchdog_set(ucs::WATCHDOG_TEST, timeout);

    sleep((int)ceil(sleep_time));

    // shouldn't reach this statement
    ASSERT_NE(timeout, timeout);
}
