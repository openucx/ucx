/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_add_xor_test : public uct_amo_test {
public:

    template <typename T, uct_atomic_op_t opcode>
    void test_op(T (*op)(T, T)) {
        /*
         * Method: Add/xor may random values from multiple workers running at the same
         * time. We expect the final result to be the sum/xor of all these values.
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
                 exp_result = op(exp_result, value);
                 value = hash64(value);
             }
        }

        run_workers(static_cast<send_func_t>(&uct_amo_test::atomic_op<T, opcode>),
                    recvbuf, add_vec, true);

        wait_for_remote();
        EXPECT_EQ(exp_result, *(T*)recvbuf.ptr());
    }
};

UCS_TEST_SKIP_COND_P(uct_amo_add_xor_test, add32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP32)) {
    test_op<uint32_t, UCT_ATOMIC_OP_ADD>(add_op<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_add_xor_test, add64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP64)) {
    test_op<uint64_t, UCT_ATOMIC_OP_ADD>(add_op<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_add_xor_test, xor32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), OP32)) {
    test_op<uint32_t, UCT_ATOMIC_OP_XOR>(xor_op<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_add_xor_test, xor64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), OP64)) {
    test_op<uint64_t, UCT_ATOMIC_OP_XOR>(xor_op<uint64_t>);
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_add_xor_test)

