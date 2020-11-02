/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"


class test_ucp_fence : public ucp_test {
public:
    typedef void (test_ucp_fence::* send_func_t)(entity *e, uint64_t *initial_buf,
                                                 uint64_t *result_buf, void *memheap_addr,
                                                 ucp_rkey_h rkey);

    static void send_cb(void *request, ucs_status_t status)
    {
    }

    template <typename T>
    void blocking_add(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                      void *memheap_addr, ucp_rkey_h rkey) {
        ucs_status_t status = ucp_atomic_post(e->ep(), UCP_ATOMIC_POST_OP_ADD,
                                              *initial_buf, sizeof(T),
                                              (uintptr_t)memheap_addr, rkey);
        ASSERT_UCS_OK(status);
    }

    template <typename T>
    void blocking_fadd(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                       void *memheap_addr, ucp_rkey_h rkey)
    {
        void *request = ucp_atomic_fetch_nb(e->ep(), UCP_ATOMIC_FETCH_OP_FADD,
                                            *initial_buf, (T*)result_buf, sizeof(T),
                                            (uintptr_t)memheap_addr, rkey, send_cb);
        request_wait(request);
    }

    template <typename T, typename F>
    void test(F f1, F f2) {
        test_fence(static_cast<send_func_t>(f1),
                   static_cast<send_func_t>(f2), sizeof(T));
    }

    class worker {
    public:
        worker(test_ucp_fence* test, send_func_t send1, send_func_t send2,
               entity* entity, ucp_rkey_h rkey, void *memheap_ptr,
               uint64_t initial_value, uint32_t* error):
            test(test), value(initial_value), result(0), error(error),
            running(true), m_rkey(rkey), m_memheap(memheap_ptr),
            m_send_1(send1), m_send_2(send2), m_entity(entity) {
            pthread_create(&m_thread, NULL, run, reinterpret_cast<void*>(this));
        }

        ~worker() {
            assert(!running);
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

        test_ucp_fence* const test;
        uint64_t value, result;
        uint32_t* error;
        bool running;

    private:
        void run() {
            uint64_t zero = 0;

            for (int i = 0; i < 500 / ucs::test_time_multiplier(); i++) {
                (test->*m_send_1)(m_entity, &value, &result,
                                  m_memheap, m_rkey);

                m_entity->fence();

                (test->*m_send_2)(m_entity, &zero, &result,
                                  m_memheap, m_rkey);

                test->flush_worker(*m_entity);

                if (result != (uint64_t)(i+1))
                    (*error)++;

                result = 0; /* reset for the next loop */
            }
        }

        ucp_rkey_h m_rkey;
        void *m_memheap;
        send_func_t m_send_1, m_send_2;
        entity* m_entity;
        pthread_t m_thread;
    };

    void run_workers(send_func_t send1, send_func_t send2, entity* sender,
                     ucp_rkey_h rkey, void *memheap_ptr,
                     uint64_t initial_value, uint32_t* error) {
        ucs::ptr_vector<worker> m_workers;
        m_workers.clear();
        m_workers.push_back(new worker(this, send1, send2, sender, rkey,
                                       memheap_ptr, initial_value, error));
        m_workers.at(0).join();
        m_workers.clear();
    }

protected:
    void test_fence(send_func_t send1, send_func_t send2, size_t alignment) {
        static const size_t memheap_size = sizeof(uint64_t);
        uint32_t error = 0;

        sender().connect(&receiver(), get_ep_params());
        flush_worker(sender()); /* avoid deadlock for blocking amo */

        mapped_buffer buffer(memheap_size, receiver(), 0);

        EXPECT_LE(memheap_size, buffer.size());
        memset(buffer.ptr(), 0, memheap_size);

        run_workers(send1, send2, &sender(), buffer.rkey(sender()),
                    buffer.ptr(), 1, &error);

        EXPECT_EQ(error, (uint32_t)0);

        disconnect(sender());
        disconnect(receiver());
    }
};

class test_ucp_fence32 : public test_ucp_fence {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_AMO32);
    }
};

UCS_TEST_P(test_ucp_fence32, atomic_add_fadd) {
    test<uint32_t>(&test_ucp_fence32::blocking_add<uint32_t>,
                   &test_ucp_fence32::blocking_fadd<uint32_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence32)

class test_ucp_fence64 : public test_ucp_fence {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_AMO64);
    }
};

UCS_TEST_P(test_ucp_fence64, atomic_add_fadd) {
    test<uint64_t>(&test_ucp_fence64::blocking_add<uint64_t>,
                   &test_ucp_fence64::blocking_fadd<uint64_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence64)
