/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_amo.h"


class uct_amo_and_or_test : public uct_amo_test {
public:

    ucs_status_t and32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                       uint64_t *result, completion *comp) {
        return uct_ep_atomic32_post(ep, UCT_ATOMIC_OP_AND,
                                    worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t and64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                       uint64_t *result, completion *comp) {
        return uct_ep_atomic64_post(ep, UCT_ATOMIC_OP_AND,
                                    worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t or32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                      uint64_t *result, completion *comp) {
        return uct_ep_atomic32_post(ep, UCT_ATOMIC_OP_OR,
                                    worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t or64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf,
                      uint64_t *result, completion *comp) {
        return uct_ep_atomic64_post(ep, UCT_ATOMIC_OP_OR,
                                    worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    template <typename T>
    void test_and_or(send_func_t send, T (*op)(T, T), T (*val)(unsigned)) {
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

        run_workers(send, recvbuf, op_vec, false);

        wait_for_remote();
        EXPECT_EQ(exp_result, *(T*)recvbuf.ptr());
    }
};

template <typename T>
T and_op(T v1, T v2)
{
    return (v1 & v2);
}

template <typename T>
T or_op(T v1, T v2)
{
    return v1 | v2;
}

template <typename T>
T and_val(unsigned i)
{
    return ~(UCS_BIT(i * 2) | UCS_BIT(i + 16));
}

template <typename T>
T or_val(unsigned i)
{
    return UCS_BIT(i * 2) | UCS_BIT(i + 16);
}

UCS_TEST_P(uct_amo_and_or_test, and32) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), op32);
    test_and_or<uint32_t>(static_cast<send_func_t>(&uct_amo_and_or_test::and32),
                          and_op<uint32_t>, and_val<uint32_t>);
}

UCS_TEST_P(uct_amo_and_or_test, add64) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), op64);
    test_and_or<uint64_t>(static_cast<send_func_t>(&uct_amo_and_or_test::and64),
                          and_op<uint64_t>, and_val<uint64_t>);
}

UCS_TEST_P(uct_amo_and_or_test, or32) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), op32);
    test_and_or<uint32_t>(static_cast<send_func_t>(&uct_amo_and_or_test::or32),
                          or_op<uint32_t>, or_val<uint32_t>);
}

UCS_TEST_P(uct_amo_and_or_test, or64) {
    check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), op64);
    test_and_or<uint64_t>(static_cast<send_func_t>(&uct_amo_and_or_test::or64),
                          or_op<uint64_t>, or_val<uint64_t>);
}

UCT_INSTANTIATE_TEST_CASE(uct_amo_and_or_test)

