/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_atomic.h"
#include "common/gtest.h"

class test_ucp_fence : public test_ucp_atomic {
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
        wait(request);
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
        ucs_status_t status;

        ucp_mem_map_params_t params;
        ucp_mem_attr_t mem_attr;
        ucp_mem_h memh;
        void *memheap = NULL;

        void *rkey_buffer;
        size_t rkey_buffer_size;
        ucp_rkey_h rkey;

        uint32_t error = 0;

        sender().connect(&receiver(), get_ep_params());
        flush_worker(sender()); /* avoid deadlock for blocking amo */

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.length     = memheap_size;
        params.flags      = GetParam().variant;
        if (params.flags & UCP_MEM_MAP_FIXED) {
            params.address  = ucs::mmap_fixed_address();
            params.flags   |= UCP_MEM_MAP_ALLOCATE;
        } else {
            memheap = malloc(memheap_size);
            params.address = memheap;
            params.flags = params.flags & (~UCP_MEM_MAP_ALLOCATE);
        }

        status = ucp_mem_map(receiver().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);

        mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS |
                              UCP_MEM_ATTR_FIELD_LENGTH;
        status = ucp_mem_query(memh, &mem_attr);
        ASSERT_UCS_OK(status);
        EXPECT_LE(memheap_size, mem_attr.length);
        if (!memheap) {
            memheap = mem_attr.address;
        }
        memset(memheap, 0, memheap_size);

        status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
        ASSERT_UCS_OK(status);

        status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
        ASSERT_UCS_OK(status);

        ucp_rkey_buffer_release(rkey_buffer);

        run_workers(send1, send2, &sender(), rkey, memheap, 1, &error);

        EXPECT_EQ(error, (uint32_t)0);

        ucp_rkey_destroy(rkey);
        status = ucp_mem_unmap(receiver().ucph(), memh);
        ASSERT_UCS_OK(status);

        disconnect(sender());
        disconnect(receiver());

        if (!(GetParam().variant & UCP_MEM_MAP_FIXED)) {
            free(memheap);
        }
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_RMA;
        return params;
    }
};

class test_ucp_fence32 : public test_ucp_fence {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_fence::get_ctx_params();
        params.features |= UCP_FEATURE_AMO32;
        return params;
    }
};

UCS_TEST_P(test_ucp_fence32, atomic_add_fadd) {
    test<uint32_t>(&test_ucp_fence32::blocking_add<uint32_t>,
                   &test_ucp_fence32::blocking_fadd<uint32_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence32)

class test_ucp_fence64 : public test_ucp_fence {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_fence::get_ctx_params();
        params.features |= UCP_FEATURE_AMO64;
        return params;
    }
};

UCS_TEST_P(test_ucp_fence64, atomic_add_fadd) {
    test<uint64_t>(&test_ucp_fence64::blocking_add<uint64_t>,
                   &test_ucp_fence64::blocking_fadd<uint64_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence64)
