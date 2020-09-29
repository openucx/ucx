/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "uct_test.h"
#include <vector>
#include <functional>
#include <iostream>

using namespace std;

extern "C" {
#include <ucs/arch/atomic.h>
}

class uct_fence_test : public uct_test {
public:
    class worker;
    typedef ucs_status_t (uct_fence_test::* send_func_t)(uct_ep_h ep, worker& worker,
                                                         const mapped_buffer& recvbuf);
    typedef ucs_status_t (uct_fence_test::* recv_func_t)(uct_ep_h ep, worker& worker,
                                                         const mapped_buffer& recvbuf,
                                                         uct_completion_t *comp);
    static inline unsigned count() {
        return 1000 / ucs::test_time_multiplier();
    }

    virtual void init() {
        uct_test::init();

        entity *receiver = uct_test::create_entity(0);
        m_entities.push_back(receiver);

        check_skip_test();

        entity *sender = uct_test::create_entity(0);
        m_entities.push_back(sender);

        sender->connect(0, *receiver, 1);
        receiver->connect(1, *sender, 0);
    }

    virtual void cleanup() {
        uct_test::cleanup();
    }

    const entity& sender() {
        return m_entities.at(1);
    }

    const entity& receiver() {
        return m_entities.at(0);
    }

    class worker {
    public:
        worker(uct_fence_test* test, send_func_t send, recv_func_t recv,
               const mapped_buffer& recvbuf,
               const entity& entity, uct_atomic_op_t op, uint32_t* error) :
            test(test), value(0), result32(0), result64(0),
            error(error), running(true), op(op), m_send(send), m_recv(recv),
            m_recvbuf(recvbuf), m_entity(entity) {
            pthread_create(&m_thread, NULL, run, reinterpret_cast<void*>(this));
        }

        ~worker() {
            ucs_assert(!running);
        }

        static void *run(void *arg) {
            worker *self = reinterpret_cast<worker*>(arg);
            self->run();
            return NULL;
        }

        void join() {
            void *retval;
            pthread_join(m_thread, &retval);
            running = false;
        }

        uint64_t atomic_op_val(uct_atomic_op_t op, uint64_t v1, uint64_t v2)
        {
            switch (op) {
            case UCT_ATOMIC_OP_ADD:
                return v1 + v2;
            case UCT_ATOMIC_OP_AND:
                return v1 & v2;
            case UCT_ATOMIC_OP_OR:
                return v1 | v2;
            case UCT_ATOMIC_OP_XOR:
                return v1 ^ v2;
            default:
                return 0;
            }
        }

        uct_fence_test* const test;
        uint64_t value;
        uint32_t result32;
        uint64_t result64;
        uint32_t* error;
        bool running;
        uct_atomic_op_t op;

    private:
        void run() {
            uct_completion_t uct_comp;
            uct_comp.func = (uct_completion_callback_t)ucs_empty_function;
            for (unsigned i = 0; i < uct_fence_test::count(); i++) {
                uint64_t local_val  = ucs::rand();
                uint64_t remote_val = ucs::rand();
                uct_comp.count      = 1;
                uct_comp.status     = UCS_OK;

                if (m_recvbuf.length() == sizeof(uint32_t)) {
                    *(uint32_t*)m_recvbuf.ptr() = remote_val;
                } else {
                    *(uint64_t*)m_recvbuf.ptr() = remote_val;
                }
                value = local_val;

                (test->*m_send)(m_entity.ep(0), *this, m_recvbuf);
                uct_ep_fence(m_entity.ep(0), 0);
                (test->*m_recv)(m_entity.ep(0), *this,
                                m_recvbuf, &uct_comp);
                m_entity.flush();

                uint64_t result = (m_recvbuf.length() == sizeof(uint32_t)) ?
                                    result32 : result64;

                if (result != atomic_op_val(op, local_val, remote_val))
                    (*error)++;

                // reset for next loop
                result32 = 0;
                result64 = 0;
            }
        }

        send_func_t m_send;
        recv_func_t m_recv;
        const mapped_buffer& m_recvbuf;
        const entity& m_entity;
        pthread_t m_thread;
    };

    template <uct_atomic_op_t OP>
    void run_workers(send_func_t send, recv_func_t recv,
                     const mapped_buffer& recvbuf, uint32_t* error) {
        ucs::ptr_vector<worker> m_workers;
        m_workers.clear();
        m_workers.push_back(new worker(this, send, recv, recvbuf,
                                       sender(), OP, error));
        m_workers.at(0).join();
        m_workers.clear();
    }

    template <typename T, uct_atomic_op_t OP>
    ucs_status_t atomic_op(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf) {
        if (sizeof(T) == sizeof(uint32_t)) {
            return uct_ep_atomic32_post(ep, OP, worker.value, recvbuf.addr(), recvbuf.rkey());
        } else {
            return uct_ep_atomic64_post(ep, OP, worker.value, recvbuf.addr(), recvbuf.rkey());
        }
    }

    template <typename T, uct_atomic_op_t OP>
    ucs_status_t atomic_fop(uct_ep_h ep, worker& worker,
                            const mapped_buffer& recvbuf, uct_completion_t *comp) {
        if (sizeof(T) == sizeof(uint32_t)) {
            return uct_ep_atomic32_fetch(ep, OP, 0, &worker.result32,
                                         recvbuf.addr(), recvbuf.rkey(), comp);
        } else {
            return uct_ep_atomic64_fetch(ep, OP, 0, &worker.result64,
                                         recvbuf.addr(), recvbuf.rkey(), comp);
        }
    }

    template <typename T, uct_atomic_op_t OP>
    void test_fence() {

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        uint32_t error = 0;

        *(T*)recvbuf.ptr() = 0;

        run_workers<OP>(static_cast<send_func_t>(&uct_fence_test::atomic_op<T, OP>),
                        static_cast<recv_func_t>(&uct_fence_test::atomic_fop<T, OP>),
                        recvbuf, &error);

        EXPECT_EQ(error, (uint32_t)0);
    }
};

UCS_TEST_SKIP_COND_P(uct_fence_test, add32,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP32) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP32))) {
    test_fence<uint32_t, UCT_ATOMIC_OP_ADD>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, add64,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), OP64) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_ADD), FOP64))) {
    test_fence<uint64_t, UCT_ATOMIC_OP_ADD>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, and32,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), OP32) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP32))) {
    test_fence<uint32_t, UCT_ATOMIC_OP_AND>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, and64,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), OP64) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_AND), FOP64))) {
    test_fence<uint64_t, UCT_ATOMIC_OP_AND>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, or32,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), OP32) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP32))) {
    test_fence<uint32_t, UCT_ATOMIC_OP_OR>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, or64,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), OP64) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_OR), FOP64))) {
    test_fence<uint64_t, UCT_ATOMIC_OP_OR>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, xor32,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), OP32) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP32))) {
    test_fence<uint32_t, UCT_ATOMIC_OP_XOR>();
}

UCS_TEST_SKIP_COND_P(uct_fence_test, xor64,
                     (!check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), OP64) ||
                      !check_atomics(UCS_BIT(UCT_ATOMIC_OP_XOR), FOP64))) {
    test_fence<uint64_t, UCT_ATOMIC_OP_XOR>();
}

UCT_INSTANTIATE_TEST_CASE(uct_fence_test)
