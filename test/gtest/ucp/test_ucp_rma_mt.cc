/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <common/test_helpers.h>

extern "C" {
#include <ucp/proto/proto_common.inl>
}

#if _OPENMP
#include "omp.h"
#endif

using namespace ucs; /* For vector<char> serialization */

class test_ucp_rma_mt : public ucp_test {
public:
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

    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant(variants, UCP_FEATURE_RMA, MULTI_THREAD_CONTEXT);
        add_variant(variants, UCP_FEATURE_RMA, MULTI_THREAD_WORKER);
    }

    ucp_mem_h mem_map(ucp_context_h context, void *data, size_t size)
    {
        ucp_mem_map_params_t params;
        ucp_mem_h memh;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = data;
        params.length     = size;
        params.flags      = get_variant_value();

        ASSERT_UCS_OK(ucp_mem_map(context, &params, &memh));
        return memh;
    }
};

UCS_TEST_P(test_ucp_rma_mt, put_get) {
    const unsigned num_threads = mt_num_threads();
    ucs_status_t st;
    uint64_t orig_data[num_threads] GTEST_ATTRIBUTE_UNUSED_;
    uint64_t target_data[num_threads] GTEST_ATTRIBUTE_UNUSED_;

    void *memheap = target_data;
    ucp_mem_h memh = mem_map(receiver().ucph(), memheap,
                             sizeof(uint64_t) * num_threads);

    void *rkey_buffer;
    size_t rkey_buffer_size;

    st = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer, &rkey_buffer_size);
    ASSERT_UCS_OK(st);

    std::vector<ucp_rkey_h> rkey;
    rkey.resize(num_threads);

    /* test parallel rkey unpack */
#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        int worker_index = 0;
        if (get_variant_thread_type() == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }
        ucs_status_t status = ucp_ep_rkey_unpack(sender().ep(worker_index),
                                                 rkey_buffer, &rkey[i]);
        ASSERT_UCS_OK(status);
    }
#endif

    ucp_rkey_buffer_release(rkey_buffer);

    /* test blocking PUT */

    for (int i = 0; i < num_threads; i++) {
        orig_data[i] = 0xdeadbeefdeadbeef + 10 * i;
        target_data[i] = 0;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        int worker_index = 0;

        if (get_variant_thread_type() == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        void* req = ucp_put_nb(sender().ep(worker_index), &orig_data[i],
                               sizeof(uint64_t), (uintptr_t)((uint64_t*)memheap + i),
                               rkey[i], send_cb);
        request_wait(req, {}, worker_index);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    /* test nonblocking PUT */

    for (int i = 0; i < num_threads; i++) {
        orig_data[i] = 0xdeadbeefdeadbeef + 10 * i;
        target_data[i] = 0;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ucs_status_t status;
        int worker_index = 0;

        if (get_variant_thread_type() == MULTI_THREAD_CONTEXT)
            worker_index = i;

        status = ucp_put_nbi(sender().ep(worker_index), &orig_data[i], sizeof(uint64_t),
                             (uintptr_t)((uint64_t*)memheap + i), rkey[i]);
        ASSERT_UCS_OK_OR_INPROGRESS(status);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    /* test blocking GET */

    for (int i = 0; i < num_threads; i++) {
        orig_data[i] = 0;
        target_data[i] = 0xdeadbeefdeadbeef + 10 * i;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        int worker_index = 0;

        if (get_variant_thread_type() == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        void *req = ucp_get_nb(sender().ep(worker_index), &orig_data[i],
                               sizeof(uint64_t), (uintptr_t)((uint64_t*)memheap + i),
                               rkey[i], send_cb);
        request_wait(req, {}, worker_index);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

    /* test nonblocking GET */

    for (int i = 0; i < num_threads; i++) {
        orig_data[i] = 0;
        target_data[i] = 0xdeadbeefdeadbeef + 10 * i;
    }

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ucs_status_t status;
        int worker_index = 0;

        if (get_variant_thread_type() == MULTI_THREAD_CONTEXT) {
            worker_index = i;
        }

        status = ucp_get_nbi(sender().ep(worker_index), &orig_data[i], sizeof(uint64_t),
                             (uintptr_t)((uint64_t *)memheap + i), rkey[i]);
        ASSERT_UCS_OK_OR_INPROGRESS(status);

        flush_worker(sender(), worker_index);

        EXPECT_EQ(orig_data[i], target_data[i]);
    }
#endif

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < num_threads; i++) {
        ucp_rkey_destroy(rkey[i]);
    }
#endif

    st = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(st);
}

UCS_TEST_P(test_ucp_rma_mt, rkey_pack) {
    uint8_t data[1024] GTEST_ATTRIBUTE_UNUSED_;
    ucp_context_h context = sender().ucph();
    ucp_mem_h memh        = mem_map(context, data, sizeof(data));

#if _OPENMP && ENABLE_MT
#pragma omp parallel for
    for (int i = 0; i < mt_num_threads(); i++) {
        if (i % 2 == 0) {
            void *rkey;
            size_t rkey_size;
            ASSERT_UCS_OK(ucp_rkey_pack(context, memh, &rkey, &rkey_size));
            ucp_rkey_buffer_release(rkey);
        } else {
            ucs_sys_dev_distance_t sys_dev            = {};
            ucp_request req                           = {};
            req.send.ep                               = sender().ep();
            req.send.state.dt_iter.type.contig.memh   = memh;
            req.send.state.dt_iter.type.contig.buffer = data;
            req.send.state.dt_iter.length             = sizeof(data);

            uint8_t rkey[1024];
            ucp_proto_request_pack_rkey(&req, memh->md_map, 0, &sys_dev, rkey);
        }
    }
#endif

    ASSERT_UCS_OK(ucp_mem_unmap(context, memh));
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_rma_mt)
