/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_fand_for_test : public uct_amo_test {
public:

    ucs_status_t fand32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                        uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic32_fetch_nb(ep, UCT_ATOMIC_OP_AND, (uint32_t)worker.value,
                                        (uint32_t*)result, recvbuf.addr(), recvbuf.rkey(),
                                        &comp->uct);
    }

    ucs_status_t fand64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                        uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic64_fetch_nb(ep, UCT_ATOMIC_OP_AND, worker.value,
                                        (uint64_t*)result, recvbuf.addr(), recvbuf.rkey(),
                                        &comp->uct);
    }

    ucs_status_t for32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                       uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic32_fetch_nb(ep, UCT_ATOMIC_OP_OR, (uint32_t)worker.value,
                                        (uint32_t*)result, recvbuf.addr(), recvbuf.rkey(),
                                        &comp->uct);
    }

    ucs_status_t for64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                       uint64_t *result, completion *comp) {
        comp->self     = this;
        comp->uct.func = atomic_reply_cb;
        return uct_ep_atomic64_fetch_nb(ep, UCT_ATOMIC_OP_OR, worker.value,
                                        (uint64_t*)result, recvbuf.addr(), recvbuf.rkey(),
                                        &comp->uct);
    }

    template <typename T>
    void test_fop(send_func_t send, T (*op)(T, T)) {
        /*
         * Method: Do concurrent atomic fetch-and-and/or of constant random value
         * to a single atomic variable. Check that every sender gets a unique reply
         * and the final value of atomic variable is the and/or of all.
         */

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        T value = rand64();
        T add   = rand64();
        *(T*)recvbuf.ptr() = value;

        std::vector<uint64_t> exp_replies;
        for (unsigned i = 0; i < count() * num_senders(); ++i) {
            exp_replies.push_back(value);
            value = op(value, add);
        }

        run_workers(send, recvbuf, std::vector<uint64_t>(num_senders(), add), false);

        validate_replies(exp_replies);

        wait_for_remote();
        EXPECT_EQ(value, *(T*)recvbuf.ptr());
    }
};

template <typename T>
T and_op(T v1, T v2)
{
    return v1 & v2;
}

template <typename T>
T or_op(T v1, T v2)
{
    return v1 | v2;
}

UCS_TEST_P(uct_amo_fand_for_test, fand32) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), fop32);
    test_fop<uint32_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::fand32), and_op<uint32_t>);
}

UCS_TEST_P(uct_amo_fand_for_test, fand64) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::fand64), and_op<uint64_t>);
}

UCS_TEST_P(uct_amo_fand_for_test, for32) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), fop32);
    test_fop<uint32_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::for32), or_op<uint32_t>);
}

UCS_TEST_P(uct_amo_fand_for_test, for64) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::for64), or_op<uint64_t>);
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_fand_for_test)

class uct_amo_fand_for_test_inlresp : public uct_amo_fand_for_test {};

UCS_TEST_P(uct_amo_fand_for_test_inlresp, fand64_inlresp0, "IB_TX_INLINE_RESP=0") {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::fand64), and_op<uint64_t>);
}

UCS_TEST_P(uct_amo_fand_for_test_inlresp, fand64_inlresp32, "IB_TX_INLINE_RESP=32") {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::fand64), and_op<uint64_t>);
}

UCS_TEST_P(uct_amo_fand_for_test_inlresp, fand64_inlresp64, "IB_TX_INLINE_RESP=64") {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::fand64), and_op<uint64_t>);
}

UCS_TEST_P(uct_amo_fand_for_test_inlresp, for64_inlresp0, "IB_TX_INLINE_RESP=0") {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::for64), or_op<uint64_t>);
}

UCS_TEST_P(uct_amo_fand_for_test_inlresp, for64_inlresp32, "IB_TX_INLINE_RESP=32") {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::for64), or_op<uint64_t>);
}

UCS_TEST_P(uct_amo_fand_for_test_inlresp, for64_inlresp64, "IB_TX_INLINE_RESP=64") {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), fop64);
    test_fop<uint64_t>(static_cast<send_func_t>(&uct_amo_fand_for_test::for64), or_op<uint64_t>);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_amo_fand_for_test_inlresp)

