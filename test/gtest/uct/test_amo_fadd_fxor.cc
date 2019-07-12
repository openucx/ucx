/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_fadd_fxor_test : public uct_amo_test {
public:

    template <typename T, uct_atomic_op_t OP>
    void test_fop(T (*op)(T, T)) {
        /*
         * Method: Do concurrent atomic fetch-and-add/xor of constant random value
         * to a single atomic variable. Check that every sender gets a unique reply
         * and the final value of atomic variable is the sum/xor of all.
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

        run_workers(static_cast<send_func_t>(&uct_amo_test::atomic_fop<T, OP>),
                    recvbuf, std::vector<uint64_t>(num_senders(), add), false);

        validate_replies(exp_replies);

        wait_for_remote();
        EXPECT_EQ(value, *(T*)recvbuf.ptr());
    }
};

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test, fadd32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP32)) {
    test_fop<uint32_t, UCT_ATOMIC_OP_ADD>(add_op<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test, fadd64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64)) {
    test_fop<uint64_t, UCT_ATOMIC_OP_ADD>(add_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test, fxor32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP32)) {
    test_fop<uint32_t, UCT_ATOMIC_OP_XOR>(xor_op<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test, fxor64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP64)) {
    test_fop<uint64_t, UCT_ATOMIC_OP_XOR>(xor_op<uint64_t>);
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_fadd_fxor_test)

class uct_amo_fadd_fxor_test_inlresp : public uct_amo_fadd_fxor_test {};

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test_inlresp, fadd64_inlresp0,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64),
                     "IB_TX_INLINE_RESP=0") {
    test_fop<uint64_t, UCT_ATOMIC_OP_ADD>(add_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test_inlresp, fadd64_inlresp32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64),
                     "IB_TX_INLINE_RESP=32") {
    test_fop<uint64_t, UCT_ATOMIC_OP_ADD>(add_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test_inlresp, fadd64_inlresp64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64),
                     "IB_TX_INLINE_RESP=64") {
    test_fop<uint64_t, UCT_ATOMIC_OP_ADD>(add_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test_inlresp, fxor64_inlresp0,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP64),
                     "IB_TX_INLINE_RESP=0") {
    test_fop<uint64_t, UCT_ATOMIC_OP_XOR>(xor_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test_inlresp, fxor64_inlresp32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP64),
                     "IB_TX_INLINE_RESP=32") {
    test_fop<uint64_t, UCT_ATOMIC_OP_XOR>(xor_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_fadd_fxor_test_inlresp, fxor64_inlresp64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP64),
                     "IB_TX_INLINE_RESP=64") {
    test_fop<uint64_t, UCT_ATOMIC_OP_XOR>(xor_op<uint64_t>);
}

UCT_INSTANTIATE_IB_TEST_CASE(uct_amo_fadd_fxor_test_inlresp)

