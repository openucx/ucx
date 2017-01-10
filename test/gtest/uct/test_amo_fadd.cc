/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_fadd_test : public uct_amo_test {
public:

    ucs_status_t fadd32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                        uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic_fadd32(ep, worker.value, recvbuf.addr(), recvbuf.rkey(),
                                    (uint32_t*)result, &comp->uct);
    }

    ucs_status_t fadd64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                        uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic_fadd64(ep, worker.value, recvbuf.addr(), recvbuf.rkey(),
                                    result, &comp->uct);
    }

    template <typename T>
    void test_fadd(send_func_t send) {
        /*
         * Method: Do concurrent atomic fetch-and-add of constant random value
         * to a single atomic variable. Check that every sender gets a unique reply
         * and the final value of atomic variable is the sum of all.
         */

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        T value = rand64();
        T add   = rand64();
        *(T*)recvbuf.ptr() = value;

        std::vector<uint64_t> exp_replies;
        for (unsigned i = 0; i < count() * num_senders(); ++i) {
            exp_replies.push_back(value);
            value += add;
        }

        run_workers(send, recvbuf, std::vector<uint64_t>(num_senders(), add), false);

        validate_replies(exp_replies);

        wait_for_remote();
        EXPECT_EQ(value, *(T*)recvbuf.ptr());
    }
};


UCS_TEST_P(uct_amo_fadd_test, fadd32) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD32);
    test_fadd<uint32_t>(static_cast<send_func_t>(&uct_amo_fadd_test::fadd32));
}

UCS_TEST_P(uct_amo_fadd_test, fadd64) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD64);
    test_fadd<uint64_t>(static_cast<send_func_t>(&uct_amo_fadd_test::fadd64));
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_fadd_test)

class uct_amo_fadd_test_inlresp : public uct_amo_fadd_test {};

UCS_TEST_P(uct_amo_fadd_test_inlresp, fadd64_inlresp0, "IB_TX_INLINE_RESP=0") {
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD64);
    test_fadd<uint64_t>(static_cast<send_func_t>(&uct_amo_fadd_test::fadd64));
}

UCS_TEST_P(uct_amo_fadd_test_inlresp, fadd64_inlresp32, "IB_TX_INLINE_RESP=32") {
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD64);
    test_fadd<uint64_t>(static_cast<send_func_t>(&uct_amo_fadd_test::fadd64));
}

UCS_TEST_P(uct_amo_fadd_test_inlresp, fadd64_inlresp64, "IB_TX_INLINE_RESP=64") {
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD64);
    test_fadd<uint64_t>(static_cast<send_func_t>(&uct_amo_fadd_test::fadd64));
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_amo_fadd_test_inlresp)

