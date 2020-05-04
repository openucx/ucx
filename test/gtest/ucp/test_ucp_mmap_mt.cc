/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
extern "C" {
#include <ucp/core/ucp_mm.h>
}

#if _OPENMP
#include "omp.h"
#endif

class test_ucp_mmap_mt : public test_ucp_memheap {
    public:
        static ucp_params_t get_ctx_params() {
            ucp_params_t params = ucp_test::get_ctx_params();
            params.features |= UCP_FEATURE_RMA;
            return params;
        }

        static int rand_flags() {
            if ((ucs::rand() % 2) == 0) {
                return 0;
            } else {
                return UCP_MEM_MAP_NONBLOCK;
            }
        }

        static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls)
        {
            std::vector<ucp_test_param> result;

            generate_test_params_variant(ctx_params, name, test_case_name, tls, 0,
                                        result, MULTI_THREAD_WORKER);
            return result;
        }

    protected:
        void test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy);
};

void test_ucp_mmap_mt::test_rkey_management(entity *e, ucp_mem_h memh, bool is_dummy)
{
    size_t rkey_size;
    void *rkey_buffer;
    ucs_status_t status;

    /* Some transports don't support memory registration, so the memory
     * can be inaccessible remotely. But it should always be possible
     * to pack/unpack a key, even if empty. */
    status = ucp_rkey_pack(e->ucph(), memh, &rkey_buffer, &rkey_size);
    if (status == UCS_ERR_UNSUPPORTED && !is_dummy) {
        return;
    }
    ASSERT_UCS_OK(status);

    EXPECT_EQ(ucp_rkey_packed_size(e->ucph(), memh->md_map), rkey_size);

    /* Unpack remote key buffer */
    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(e->ep(), rkey_buffer, &rkey);
    if (status == UCS_ERR_UNREACHABLE && !is_dummy) {
        ucp_rkey_buffer_release(rkey_buffer);
        ucp_rkey_destroy(rkey);
        return;
    }
    ASSERT_UCS_OK(status);

     /* Test obtaining direct-access pointer */
    void *ptr;
    status = ucp_rkey_ptr(rkey, (uint64_t)memh->address, &ptr);
    if (status == UCS_OK) {
        EXPECT_EQ(0, memcmp(memh->address, ptr, memh->length));
    } else {
        EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
    }

    ucp_rkey_destroy(rkey);
    ucp_rkey_buffer_release(rkey_buffer);
}

UCS_TEST_P(test_ucp_mmap_mt, alloc_mt) {
    sender().connect(&sender(), get_ep_params());
    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {

        #if _OPENMP && ENABLE_MT
        #pragma omp parallel for
        for (int i = 0; i < MT_TEST_NUM_THREADS; i++) {
            size_t size = ucs::rand() % (UCS_MBYTE);
            const bool is_dummy = (size == 0);

            ucp_mem_h memh;
            ucp_mem_map_params_t params;

            params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                                UCP_MEM_MAP_PARAM_FIELD_FLAGS;
            params.address    = NULL;
            params.length     = size;
            params.flags      = rand_flags() | UCP_MEM_MAP_ALLOCATE;

            ucs_status_t status = ucp_mem_map(sender().ucph(), &params, &memh);
            ASSERT_UCS_OK(status);

            test_rkey_management(&sender(), memh, is_dummy);

            status = ucp_mem_unmap(sender().ucph(), memh);
            ASSERT_UCS_OK(status);
        }
        #endif
    }
}

UCS_TEST_P(test_ucp_mmap_mt, reg_mt) {
    sender().connect(&sender(), get_ep_params());
    for (int i = 0; i < 1000 / ucs::test_time_multiplier(); ++i) {

        #if _OPENMP && ENABLE_MT
        #pragma omp parallel for
        for (int i = 0; i < MT_TEST_NUM_THREADS; i++) {
            ucs_status_t status;

            size_t size = ucs::rand() % (UCS_MBYTE);
            const bool is_dummy = (size == 0);

            void *ptr = malloc(size);
            ucs::fill_random(ptr, size);

            ucp_mem_h memh;
            ucp_mem_map_params_t params;

            params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                                UCP_MEM_MAP_PARAM_FIELD_FLAGS;
            params.address    = ptr;
            params.length     = size;
            params.flags      = rand_flags();

            status = ucp_mem_map(sender().ucph(), &params, &memh);
            ASSERT_UCS_OK(status);

            test_rkey_management(&sender(), memh, is_dummy);

            status = ucp_mem_unmap(sender().ucph(), memh);
            ASSERT_UCS_OK(status);

            free(ptr);
        }
        #endif
    }
}

UCS_TEST_P(test_ucp_mmap_mt, reg_advise_mt) {
    sender().connect(&sender(), get_ep_params());
    #if _OPENMP && ENABLE_MT
    #pragma omp parallel for
    for (int i = 0; i < MT_TEST_NUM_THREADS; i++) {
        ucs_status_t status;

        size_t size = 128 * UCS_MBYTE;
        const bool is_dummy = (size == 0);

        void *ptr = malloc(size);
        ucs::fill_random(ptr, size);

        ucp_mem_h               memh;
        ucp_mem_map_params_t    params;
        ucp_mem_attr_t          mem_attr;
        ucp_mem_advise_params_t advise_params;

        params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
        params.address    = ptr;
        params.length     = size;
        params.flags      = UCP_MEM_MAP_NONBLOCK;

        status = ucp_mem_map(sender().ucph(), &params, &memh);
        ASSERT_UCS_OK(status);

        mem_attr.field_mask = UCP_MEM_ATTR_FIELD_ADDRESS;
        status = ucp_mem_query(memh, &mem_attr);
        ASSERT_UCS_OK(status);

        advise_params.field_mask = UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS |
                                   UCP_MEM_ADVISE_PARAM_FIELD_LENGTH |
                                   UCP_MEM_ADVISE_PARAM_FIELD_ADVICE;
        advise_params.address    = mem_attr.address;
        advise_params.length     = size;
        advise_params.advice     = UCP_MADV_WILLNEED;
        status = ucp_mem_advise(sender().ucph(), memh, &advise_params);
        ASSERT_UCS_OK(status);
        test_rkey_management(&sender(), memh, is_dummy);

        status = ucp_mem_unmap(sender().ucph(), memh);
        ASSERT_UCS_OK(status);

        free(ptr);
    }
    #endif
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_mmap_mt)
