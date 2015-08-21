/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_add_test : public uct_amo_test {
public:

    ucs_status_t add32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                       uint64_t *result, completion *comp) {
        return uct_ep_atomic_add32(ep, worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t add64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                       uint64_t *result, completion *comp) {
        return uct_ep_atomic_add64(ep, worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    template <typename T>
    void test_add(send_func_t send) {
        /*
         * Method: Add may random values from multiple workers running at the same
         * time. We expect the final result to be the sum of all these values.
         */

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        T value = rand64();
        *(T*)recvbuf.ptr() = value;

        T exp_result = value;
        std::vector<uint64_t> add_vec;
        for (unsigned i = 0; i < num_senders(); ++i) {
             value = rand64();
             add_vec.push_back(value);

             for (unsigned j = 0; j < count(); ++j) {
                 exp_result += value;
                 value = hash64(value);
             }
        }

        run_workers(send, recvbuf, add_vec, true);

        wait_for_remote();
        EXPECT_EQ(exp_result, *(T*)recvbuf.ptr());
    }
};


UCS_TEST_P(uct_amo_add_test, add32) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_ADD32);
    test_add<uint32_t>(static_cast<send_func_t>(&uct_amo_add_test::add32));
}

UCS_TEST_P(uct_amo_add_test, add64) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_ADD64);
    test_add<uint64_t>(static_cast<send_func_t>(&uct_amo_add_test::add64));
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_add_test)

