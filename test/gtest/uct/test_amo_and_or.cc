/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_and_or_test : public uct_amo_test {
public:

    template <typename T, uct_atomic_op_t opcode>
    void test_op(T (*op)(T, T), T (*val)(unsigned)) {
        /*
         * Method: Add may random values from multiple workers running at the same
         * time. We expect the final result to be the and/or of all these values.
         * This is simplified version of add/xor test: operated value is costant
         * for every worker to eliminate result to 0 or MAX_INT
         */

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        T value = 0x0ff0f00f;
        *(T*)recvbuf.ptr() = value;

        T exp_result = value;
        std::vector<uint64_t> op_vec;
        for (unsigned i = 0; i < num_senders(); ++i) {
             value = val(i);
             op_vec.push_back(value);

             for (unsigned j = 0; j < count(); ++j) {
                 exp_result = op(exp_result, value);
             }
        }

        run_workers(static_cast<send_func_t>(&uct_amo_test::atomic_op<T, opcode>),
                    recvbuf, op_vec, false);

        wait_for_remote();
        EXPECT_EQ(exp_result, *(T*)recvbuf.ptr());
    }
};

UCS_TEST_SKIP_COND_P(uct_amo_and_or_test, and32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), OP32)) {
    test_op<uint32_t, UCT_ATOMIC_OP_AND>(and_op<uint32_t>, and_val<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_and_or_test, add64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), OP64)) {
    test_op<uint64_t, UCT_ATOMIC_OP_AND>(and_op<uint64_t>, and_val<uint64_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_and_or_test, or32,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), OP32)) {
    test_op<uint32_t, UCT_ATOMIC_OP_OR>(or_op<uint32_t>, or_val<uint32_t>);
}

UCS_TEST_SKIP_COND_P(uct_amo_and_or_test, or64,
                     !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), OP64)) {
    test_op<uint64_t, UCT_ATOMIC_OP_OR>(or_op<uint64_t>, or_val<uint64_t>);
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_and_or_test)

