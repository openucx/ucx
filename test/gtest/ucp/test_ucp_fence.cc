/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_atomic.h"
extern "C" {
#include <ucp/core/ucp_context.h>
}

class test_ucp_fence : public test_ucp_atomic {
public:
    typedef void (test_ucp_fence::* send_func_t)(entity *e, uint64_t *initial_buf,
                                                 uint64_t *result_buf, void *memheap_addr,
                                                 ucp_rkey_h rkey);

    static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const ucp_worker_params_t& worker_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls) {
        std::vector<ucp_test_param> result;
        for (int variant = UCP_ATOMIC_MODE_CPU; variant < UCP_ATOMIC_MODE_LAST; variant++) {
            generate_test_params_variant(ctx_params, worker_params, name, test_case_name,
                                         tls, variant, result, true);
        }
        return result;
    }


    template <typename T>
    void blocking_add(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                      void *memheap_addr, ucp_rkey_h rkey) {
        ucs_status_t status;
        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_atomic_add32(e->ep(), (uint32_t)(*initial_buf),
                                      (uintptr_t)memheap_addr, rkey);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_atomic_add64(e->ep(), (uint64_t)(*initial_buf),
                                      (uintptr_t)memheap_addr, rkey);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);
    }

    template <typename T>
    void blocking_fadd(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                       void *memheap_addr, ucp_rkey_h rkey)
    {
        ucs_status_t status;
        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_atomic_fadd32(e->ep(), (uint32_t)(*initial_buf),
                                       (uintptr_t)memheap_addr,
                                       rkey, (uint32_t*)(void*)result_buf);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_atomic_fadd64(e->ep(), (uint64_t)(*initial_buf),
                                       (uintptr_t)memheap_addr,
                                       rkey, (uint64_t*)(void*)result_buf);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);
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
            run();
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

                m_entity->flush_worker();

                if (result != (uint64_t)(i+1))
                    (*error)++;

                result = 0; /* reset for the next loop */
            }
        }

        ucp_rkey_h m_rkey;
        void *m_memheap;
        send_func_t m_send_1, m_send_2;
        entity* m_entity;
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
        size_t memheap_size = sizeof(uint64_t) * mt_num_threads();
        ucs_status_t status;

        ucp_mem_map_params_t params;
        ucp_mem_attr_t mem_attr;
        ucp_mem_h memh;
        void *memheap = NULL;

        void *rkey_buffer;
        size_t rkey_buffer_size;

        sender().connect(&receiver());
        if (&sender() != &receiver()) {
            receiver().connect(&sender());
        }

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.length     = memheap_size;
        params.flags      = GetParam().variant;
        if (params.flags & UCP_MEM_MAP_FIXED) {
            params.address  = (void *)(uintptr_t)0xFF000000;
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

        std::vector<ucp_rkey_h> rkeys;
        sender().create_rkeys(rkey_buffer, &rkeys);

        ucp_rkey_buffer_release(rkey_buffer);

        UCS_OMP_PARALLEL_FOR(thread_id) {
            uint32_t error = 0;
            run_workers(send1, send2, &sender(), rkeys[sender().get_worker_index()],
                        (void *)((uintptr_t)memheap + thread_id * sizeof(uint64_t)),
                        1, &error);
            EXPECT_EQ(error, (uint32_t)0);
        }

        sender().destroy_rkeys(&rkeys);
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
