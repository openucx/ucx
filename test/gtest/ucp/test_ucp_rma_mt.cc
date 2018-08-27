/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>

#if _OPENMP
#include "omp.h"
#endif

using namespace ucs; /* For vector<char> serialization */

class test_ucp_rma_mt : public ucp_test {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features     = UCP_FEATURE_RMA;
        return params;
    }

    void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        for (int i = 0; i < sender().get_num_workers(); i++) {
            /* avoid deadlock for blocking rma */
            flush_worker(sender(), i);
        }
    }

    static void send_cb(void *req, ucs_status_t status)
    {
    }

    static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls)
    {
        std::vector<ucp_test_param> result;

        generate_test_params_variant(ctx_params, name, test_case_name, tls, 0,
                                     result, MULTI_THREAD_CONTEXT);
        generate_test_params_variant(ctx_params, name, test_case_name, tls, 0,
                                     result, MULTI_THREAD_WORKER);
        return result;
    }
};

UCS_TEST_P(test_ucp_rma_mt, put_get) {
    int i;
    ucs_status_t st;
    uint64_t orig_data[MT_TEST_NUM_THREADS] GTEST_ATTRIBUTE_UNUSED_;
    uint64_t target_data[MT_TEST_NUM_THREADS] GTEST_ATTRIBUTE_UNUSED_;

    ucp_mem_map_params_t params;
    ucp_mem_h memh;
    void *memheap = target_data;

    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = memheap;
    params.length     = sizeof(uint64_t) * MT_TEST_NUM_THREADS;
    params.flags      = GetParam().variant;

    st = ucp_mem_map(receiver().ucph(), &params, &memh);
    ASSERT_UCS_OK(st);

    void *rkey_buffer;
    size_t rkey_buffer_size;

    st = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(st);

    ucp_rkey_h *rkey;

    rkey = (ucp_rkey_h *)malloc(sizeof(ucp_rkey_h) * sender().get_num_workers());
    for (i = 0; i < sender().get_num_workers(); i++) {
        st = ucp_ep_rkey_unpack(sender().ep(i), rkey_buffer, &rkey[i]);
        ASSERT_UCS_OK(st);
    }

    ucp_rkey_buffer_release(rkey_buffer);

    /* test blocking PUT */

    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        orig_data[i] = 0xdeadbeefdeadbeef + 10 * i;
        target_data[i] = 0;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        int worker_index = 0;

        if (GetParam().thread_type == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        void* req = ucp_put_nb(sender().ep(worker_index), &orig_data[i],
                               sizeof(uint64_t), (uintptr_t)((uint64_t*)memheap + i),
                               rkey[worker_index], send_cb);
        wait(req, worker_index);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    /* test nonblocking PUT */

    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        orig_data[i] = 0xdeadbeefdeadbeef + 10 * i;
        target_data[i] = 0;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        ucs_status_t status;
        int worker_index = 0;

        if (GetParam().thread_type == MULTI_THREAD_CONTEXT)
            worker_index = i;

        status = ucp_put_nbi(sender().ep(worker_index), &orig_data[i], sizeof(uint64_t),
                             (uintptr_t)((uint64_t*)memheap + i), rkey[worker_index]);
        ASSERT_UCS_OK_OR_INPROGRESS(status);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    /* test blocking GET */

    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        orig_data[i] = 0;
        target_data[i] = 0xdeadbeefdeadbeef + 10 * i;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        int worker_index = 0;

        if (GetParam().thread_type == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        void *req = ucp_get_nb(sender().ep(worker_index), &orig_data[i],
                               sizeof(uint64_t), (uintptr_t)((uint64_t*)memheap + i),
                               rkey[worker_index], send_cb);
        wait(req, worker_index);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    /* test nonblocking GET */

    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        orig_data[i] = 0;
        target_data[i] = 0xdeadbeefdeadbeef + 10 * i;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (i = 0; i < MT_TEST_NUM_THREADS; i++) {
        ucs_status_t status;
        int worker_index = 0;

        if (GetParam().thread_type == MULTI_THREAD_CONTEXT)
            worker_index = i;

        status = ucp_get_nbi(sender().ep(worker_index), &orig_data[i], sizeof(uint64_t),
                             (uintptr_t)((uint64_t *)memheap + i), rkey[worker_index]);
        ASSERT_UCS_OK_OR_INPROGRESS(status);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    for (i = 0; i < sender().get_num_workers(); i++) {
        ucp_rkey_destroy(rkey[i]);
    }
    free(rkey);

    st = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(st);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_rma_mt)
