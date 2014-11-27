/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "uct_p2p_test.h"

class uct_p2p_rma_test : public uct_p2p_test {
};

UCS_TEST_P(uct_p2p_rma_test, put8) {
    uint64_t magic = 0xdeadbeed1ee7a880;
    ucs_status_t status;
    uct_rkey_bundle_t rkey;
    uct_lkey_t lkey;
    uint64_t val8;

    rkey = get_entity(1).mem_map(&val8, sizeof(val8), &lkey);

    val8 = 0;
    status = uct_ep_put_short(get_entity(0).ep(), &magic, sizeof(magic),
                              (uintptr_t)&val8, rkey.rkey);
    ASSERT_UCS_OK(status);

    get_entity(0).flush();

    EXPECT_EQ(magic, val8);

    get_entity(1).mem_unmap(lkey, rkey);
}

UCT_INSTANTIATE_TEST_CASE(uct_p2p_rma_test)
