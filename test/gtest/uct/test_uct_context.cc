/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include <ucs/gtest/test.h>
extern "C" {
#include <uct/api/uct.h>
}


class test_uct : public ucs::test {
};


UCS_TEST_F(test_uct, init_cleanup) {
    ucs_status_t status;
    uct_context_h ucth;

    ucth = NULL;
    status = uct_init(&ucth);
    ASSERT_UCS_OK(status);
    ASSERT_TRUE(ucth != NULL);

    uct_cleanup(ucth);
}

