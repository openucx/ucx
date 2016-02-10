/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

class test_ucp_atomic : public test_ucp_memheap {
public:
    template <typename T>
    void blocking_add(entity *e,  size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T add, prev;

        prev = *(T*)memheap_addr;
        add  = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_atomic_add32(e->ep(), add, (uintptr_t)memheap_addr, rkey);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_atomic_add64(e->ep(), add, (uintptr_t)memheap_addr, rkey);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        expected_data.resize(sizeof(T));
        *(T*)&expected_data[0] = add + prev;
    }

    template<uint32_t> void blocking_add(entity *e,  size_t max_size, void *memheap_addr,
                                         ucp_rkey_h rkey, std::string& expected_data);

    void unaligned_blocking_add64(entity *e,  size_t max_size, void *memheap_addr,
                             ucp_rkey_h rkey, std::string& expected_data)
    {
        /* Test that unaligned addresses generate error */
        ucs_status_t status;
        status = ucp_atomic_add64(e->ep(), 0, (uintptr_t)memheap_addr + 1, rkey);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
        expected_data.clear();
    }

    template <typename T>
    void blocking_fadd(entity *e,  size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T add, prev, result;

        prev = *(T*)memheap_addr;
        add  = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_atomic_fadd32(e->ep(), add, (uintptr_t)memheap_addr, rkey,
                                       (uint32_t*)(void*)&result);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_atomic_fadd64(e->ep(), add, (uintptr_t)memheap_addr, rkey,
                                       (uint64_t*)(void*)&result);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        EXPECT_EQ(prev, result);

        expected_data.resize(sizeof(T));
        *(T*)&expected_data[0] = add + prev;
    }

    template <typename T>
    void blocking_swap(entity *e,  size_t max_size, void *memheap_addr,
                       ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T swap, prev, result;

        prev = *(T*)memheap_addr;
        swap = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_atomic_swap32(e->ep(), swap, (uintptr_t)memheap_addr,
                                       rkey, (uint32_t*)(void*)&result);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_atomic_swap64(e->ep(), swap, (uintptr_t)memheap_addr,
                                       rkey, (uint64_t*)(void*)&result);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        EXPECT_EQ(prev, result);

        expected_data.resize(sizeof(T));
        *(T*)&expected_data[0] = swap;
    }

    template <typename T>
    void blocking_cswap(entity *e,  size_t max_size, void *memheap_addr,
                        ucp_rkey_h rkey, std::string& expected_data)
    {
        ucs_status_t status;
        T compare, swap, prev, result;

        prev = *(T*)memheap_addr;
        if ((rand() % 2) == 0) {
            compare = prev; /* success mode */
        } else {
            compare = ~prev; /* fail mode */
        }
        swap = (T)rand() * (T)rand();

        if (sizeof(T) == sizeof(uint32_t)) {
            status = ucp_atomic_cswap32(e->ep(), compare, swap,
                                        (uintptr_t)memheap_addr, rkey,
                                        (uint32_t*)(void*)&result);
        } else if (sizeof(T) == sizeof(uint64_t)) {
            status = ucp_atomic_cswap64(e->ep(), compare, swap,
                                        (uintptr_t)memheap_addr, rkey,
                                        (uint64_t*)(void*)&result);
        } else {
            status = UCS_ERR_UNSUPPORTED;
        }
        ASSERT_UCS_OK(status);

        EXPECT_EQ(prev, result);

        expected_data.resize(sizeof(T));
        if (compare == prev) {
            *(T*)&expected_data[0] = swap;
        } else {
            *(T*)&expected_data[0] = prev;
        }
    }

    template <typename T, typename F>
    void test(F f) {
        test_blocking_xfer(static_cast<blocking_send_func_t>(f), sizeof(T));
    }

};

class test_ucp_atomic32 : public test_ucp_atomic {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_memheap::get_ctx_params();
        params.features |= UCP_FEATURE_AMO32;
        return params;
    }
};

UCS_TEST_P(test_ucp_atomic32, atomic_add) {
    test<uint32_t>(&test_ucp_atomic32::blocking_add<uint32_t>);
}

UCS_TEST_P(test_ucp_atomic32, atomic_fadd) {
    test<uint32_t>(&test_ucp_atomic32::blocking_fadd<uint32_t>);
}

UCS_TEST_P(test_ucp_atomic32, atomic_swap) {
    test<uint32_t>(&test_ucp_atomic32::blocking_swap<uint32_t>);
}

UCS_TEST_P(test_ucp_atomic32, atomic_cswap) {
    test<uint32_t>(&test_ucp_atomic32::blocking_cswap<uint32_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic32)

class test_ucp_atomic64 : public test_ucp_atomic {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = test_ucp_memheap::get_ctx_params();
        params.features |= UCP_FEATURE_AMO64;
        return params;
    }
};

UCS_TEST_P(test_ucp_atomic64, atomic_add) {
    test<uint64_t>(&test_ucp_atomic64::blocking_add<uint64_t>);
}

UCS_TEST_P(test_ucp_atomic64, atomic_fadd) {
    test<uint64_t>(&test_ucp_atomic64::blocking_fadd<uint64_t>);
}

UCS_TEST_P(test_ucp_atomic64, atomic_swap) {
    test<uint64_t>(&test_ucp_atomic64::blocking_swap<uint64_t>);
}

UCS_TEST_P(test_ucp_atomic64, atomic_cswap) {
    test<uint64_t>(&test_ucp_atomic64::blocking_cswap<uint64_t>);
}

UCS_TEST_P(test_ucp_atomic64, unaligned_atomic_add) {
    test<uint64_t>(&test_ucp_atomic::unaligned_blocking_add64);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic64)

