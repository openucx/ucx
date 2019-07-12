/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_fand_for_test : public uct_amo_test {
public:

    template <typename T, uct_atomic_op_t opcode>
    void test_fop(T (*op)(T, T)) {
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

        run_workers(static_cast<send_func_t>(&uct_amo_test::atomic_fop<T, opcode>),
                    recvbuf, std::vector<uint64_t>(num_senders(), add), false);

        validate_replies(exp_replies);

        wait_for_remote();
        EXPECT_EQ(value, *(T*)recvbuf.ptr());
    }
};

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test, fand32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP32)) {
    test_fop<uint32_t, UCT_ATOMIC_OP_AND>(and_op<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test, fand64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP64)) {
    test_fop<uint64_t, UCT_ATOMIC_OP_AND>(and_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test, for32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP32)) {
    test_fop<uint32_t, UCT_ATOMIC_OP_OR>(or_op<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test, for64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP64)) {
    test_fop<uint64_t, UCT_ATOMIC_OP_OR>(or_op<uint64_t>);
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_fand_for_test)

class uct_amo_fand_for_test_inlresp : public uct_amo_fand_for_test {};

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test_inlresp, fand64_inlresp0,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP64),
                     "IB_TX_INLINE_RESP=0") {
    test_fop<uint64_t, UCT_ATOMIC_OP_AND>(and_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test_inlresp, fand64_inlresp32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP64),
                     "IB_TX_INLINE_RESP=32") {
    test_fop<uint64_t, UCT_ATOMIC_OP_AND>(and_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test_inlresp, fand64_inlresp64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP64),
                     "IB_TX_INLINE_RESP=64") {
    test_fop<uint64_t, UCT_ATOMIC_OP_AND>(and_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test_inlresp, for64_inlresp0,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP64),
                     "IB_TX_INLINE_RESP=0") {
    test_fop<uint64_t, UCT_ATOMIC_OP_OR>(or_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test_inlresp, for64_inlresp32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP64),
                     "IB_TX_INLINE_RESP=32") {
    test_fop<uint64_t, UCT_ATOMIC_OP_OR>(or_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fand_for_test_inlresp, for64_inlresp64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP64),
                     "IB_TX_INLINE_RESP=64") {
    test_fop<uint64_t, UCT_ATOMIC_OP_OR>(or_op<uint64_t>);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_amo_fand_for_test_inlresp)

