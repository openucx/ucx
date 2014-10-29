/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <gtest/gtest.h>
#include <boost/noncopyable.hpp>
extern "C" {
#include <ucs/sys/sys.h>
}

class valgrind_errors_Test : public ::testing::Test, boost::noncopyable {
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
            srand(atoi(optarg));
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
    return RUN_ALL_TESTS();
}
