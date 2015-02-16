/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#include <gtest/gtest.h>
extern "C" {
#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#include <ucs/sys/sys.h>
}
#include "test_helpers.h"

static int ucs_gtest_random_seed = -1;

class valgrind_errors_Test : public ::testing::Test {
private:
    virtual void TestBody() {
        long leaked, dubious, reachable, suppressed, errors;
        errors = VALGRIND_COUNT_ERRORS;
        VALGRIND_COUNT_LEAKS(leaked, dubious, reachable, suppressed);
        EXPECT_EQ(0, errors);
        EXPECT_EQ(0, leaked);
        (void)dubious;
        (void)reachable;
        (void)suppressed;
    }
};

void parse_test_opts(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "s:")) != -1) {
        switch (c) {
        case 's':
            ucs_gtest_random_seed = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: gtest [ -s rand-seed ]\n");
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    if (RUNNING_ON_VALGRIND) {
        ::testing::internal::MakeAndRegisterTestInfo(
                        "valgrind", "errors", "", "",
                        (::testing::internal::GetTestTypeId()),
                        ::testing::Test::SetUpTestCase,
                        ::testing::Test::TearDownTestCase,
                        new ::testing::internal::TestFactoryImpl<valgrind_errors_Test>);
    }
    parse_test_opts(argc, argv);
    if (ucs_gtest_random_seed == -1) {
        ucs_gtest_random_seed = time(NULL) % 32768;
    }
    UCS_TEST_MESSAGE << "Using random seed of " << ucs_gtest_random_seed;
    srand(ucs_gtest_random_seed);
    return RUN_ALL_TESTS();
}
