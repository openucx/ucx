/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_swap_test : public uct_amo_test {
public:

    ucs_status_t swap32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                        uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic_swap32(ep, worker.value, recvbuf.addr(), recvbuf.rkey(),
                                    (uint32_t*)result, &comp->uct);
    }

    ucs_status_t swap64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                        uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic_swap64(ep, worker.value, recvbuf.addr(), recvbuf.rkey(),
                                    result, &comp->uct);
    }

    template <typename T>
    void test_swap(send_func_t send) {
        /*
         * Method: Initialize the buffer to random value, and then have each
         * worker thread swap it to a series of generated values. We expect that
         * the replies will include the first value, and all other values, except
         * the final value of the buffer.
         */

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        /* Set ransom initial value */
        T value = rand64();
        *(T*)recvbuf.ptr() = value;

        std::vector<uint64_t> exp_replies;
        exp_replies.push_back(value);

        std::vector<uint64_t> swap_vec;
        for (unsigned i = 0; i < num_senders(); ++i) {
            value = rand64();
            swap_vec.push_back(value);

            for (unsigned j = 0; j < count(); ++j) {
                exp_replies.push_back(value);
                value = hash64(value);
            }
        }

        run_workers(send, recvbuf, swap_vec, true);

        wait_for_remote();
        add_reply_safe(*(T*)recvbuf.ptr());
        validate_replies(exp_replies);
    }
};


UCS_TEST_P(uct_amo_swap_test, swap32) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_SWAP32);
    test_swap<uint32_t>(static_cast<send_func_t>(&uct_amo_swap_test::swap32));
}

UCS_TEST_P(uct_amo_swap_test, swap64) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_SWAP64);
    test_swap<uint64_t>(static_cast<send_func_t>(&uct_amo_swap_test::swap64));
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_swap_test)

class uct_amo_swap_test_inlresp : public uct_amo_swap_test {};

UCS_TEST_P(uct_amo_swap_test_inlresp, swap32_inlresp0, "IB_TX_INLINE_RESP=0") {
    check_caps(UCT_IFACE_FLAG_ATOMIC_SWAP32);
    test_swap<uint32_t>(static_cast<send_func_t>(&uct_amo_swap_test::swap32));
}

UCS_TEST_P(uct_amo_swap_test_inlresp, swap32_inlresp32, "IB_TX_INLINE_RESP=32") {
    check_caps(UCT_IFACE_FLAG_ATOMIC_SWAP32);
    test_swap<uint32_t>(static_cast<send_func_t>(&uct_amo_swap_test::swap32));
}

UCS_TEST_P(uct_amo_swap_test_inlresp, swap32_inlresp64, "IB_TX_INLINE_RESP=64") {
    check_caps(UCT_IFACE_FLAG_ATOMIC_SWAP32);
    test_swap<uint32_t>(static_cast<send_func_t>(&uct_amo_swap_test::swap32));
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_amo_swap_test_inlresp)
