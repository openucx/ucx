/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test.h"
#include "test_helpers.h"


class gtest_common : public ucs::test {
};


UCS_TEST_F(gtest_common, auto_ptr) {
    ucs::auto_ptr<int> p(new int);
}

