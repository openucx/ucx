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

    static void completion_cb(uct_completion_t *self, ucs_status_t status) {}

    class worker {
    public:
        worker(uct_fence_test* test, send_func_t send, recv_func_t recv,
               const mapped_buffer& recvbuf,
               const entity& entity, uint64_t initial_value, uint32_t* error) :
            test(test), value(initial_value), result32(0), result64(0),
            error(error), running(true), m_send(send), m_recv(recv),
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

        uct_fence_test* const test;
        uint64_t value;
        uint32_t result32;
        uint64_t result64;
        uint32_t* error;
        bool running;

    private:
        void run() {
            uct_completion_t uct_comp;
            uct_comp.func = completion_cb;
            for (unsigned i = 0; i < uct_fence_test::count(); i++) {
                uct_comp.count = 1;
                (test->*m_send)(m_entity.ep(0), *this, m_recvbuf);
                uct_ep_fence(m_entity.ep(0), 0);
                (test->*m_recv)(m_entity.ep(0), *this,
                                m_recvbuf, &uct_comp);
                m_entity.flush();

                uint64_t result = (m_recvbuf.length() == sizeof(uint32_t)) ?
                                    result32 : result64;

                if (result != (uint64_t)(i+1))
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

    void run_workers(send_func_t send, recv_func_t recv,
                     const mapped_buffer& recvbuf,
                     uint64_t initial_value, uint32_t* error) {
        ucs::ptr_vector<worker> m_workers;
        m_workers.clear();
        m_workers.push_back(new worker(this, send, recv, recvbuf,
                                       sender(), initial_value, error));
        m_workers.at(0).join();
        m_workers.clear();
    }

    ucs_status_t add32(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf) {
        return uct_ep_atomic_add32(ep, worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t add64(uct_ep_h ep, worker& worker, const mapped_buffer& recvbuf) {
        return uct_ep_atomic_add64(ep, worker.value, recvbuf.addr(), recvbuf.rkey());
    }

    ucs_status_t fadd32(uct_ep_h ep, worker& worker,
                        const mapped_buffer& recvbuf, uct_completion_t *comp) {
        return uct_ep_atomic_fadd32(ep, 0, recvbuf.addr(), recvbuf.rkey(),
                                    &worker.result32, comp);
    }

    ucs_status_t fadd64(uct_ep_h ep, worker& worker,
                        const mapped_buffer& recvbuf, uct_completion_t *comp) {
        return uct_ep_atomic_fadd64(ep, 0, recvbuf.addr(), recvbuf.rkey(),
                                    &worker.result64, comp);
    }

    template <typename T>
    void test_fence(send_func_t send, recv_func_t recv) {

        mapped_buffer recvbuf(sizeof(T), 0, receiver());

        uint32_t error = 0;

        *(T*)recvbuf.ptr() = 0;

        run_workers(send, recv, recvbuf, 1, &error);

        EXPECT_EQ(error, (uint32_t)0);
    }
};

UCS_TEST_P(uct_fence_test, add32) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_ADD32);
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD32);
    test_fence<uint32_t>(static_cast<send_func_t>(&uct_fence_test::add32),
                         static_cast<recv_func_t>(&uct_fence_test::fadd32));
}

UCS_TEST_P(uct_fence_test, add64) {
    check_caps(UCT_IFACE_FLAG_ATOMIC_ADD64);
    check_caps(UCT_IFACE_FLAG_ATOMIC_FADD64);
    test_fence<uint64_t>(static_cast<send_func_t>(&uct_fence_test::add64),
                         static_cast<recv_func_t>(&uct_fence_test::fadd64));
}

UCT_INSTANTIATE_TEST_CASE(uct_fence_test)
