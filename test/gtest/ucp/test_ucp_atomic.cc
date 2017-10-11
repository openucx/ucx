/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_atomic.h"
extern "C" {
#include <ucp/core/ucp_context.h>
}

std::vector<ucp_test_param>
test_ucp_atomic::enum_test_params(const ucp_params_t& ctx_params,
                                  const std::string& name,
                                  const std::string& test_case_name,
                                  const std::string& tls)
{
    std::vector<ucp_test_param> result;
    generate_test_params_variant(ctx_params, name,
                                 test_case_name, tls, UCP_ATOMIC_MODE_CPU, result);
    generate_test_params_variant(ctx_params, name,
                                 test_case_name, tls, UCP_ATOMIC_MODE_DEVICE, result);
    generate_test_params_variant(ctx_params, name,
                                 test_case_name, tls, UCP_ATOMIC_MODE_GUESS, result);
    return result;
}

void test_ucp_atomic::init() {
    const char *atomic_mode =
                    (GetParam().variant == UCP_ATOMIC_MODE_CPU)    ? "cpu" :
                    (GetParam().variant == UCP_ATOMIC_MODE_DEVICE) ? "device" :
                    (GetParam().variant == UCP_ATOMIC_MODE_GUESS)  ? "guess" :
                    "";
    modify_config("ATOMIC_MODE", atomic_mode);
    test_ucp_memheap::init();
}

template <typename T>
void test_ucp_atomic::blocking_add(entity *e,  size_t max_size, void *memheap_addr,
                  ucp_rkey_h rkey, std::string& expected_data)
{
    ucs_status_t status;
    T add, prev;

    prev = *(T*)memheap_addr;
    add  = (T)ucs::rand() * (T)ucs::rand();

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

void test_ucp_atomic::unaligned_blocking_add64(entity *e,  size_t max_size,
                                               void *memheap_addr, ucp_rkey_h rkey,
                                               std::string& expected_data)
{
    /* Test that unaligned addresses generate error */
    ucs_status_t status;
    status = ucp_atomic_add64(e->ep(), 0, (uintptr_t)memheap_addr + 1, rkey);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    expected_data.clear();
}

template <typename T>
void test_ucp_atomic::blocking_fadd(entity *e,  size_t max_size,
                                    void *memheap_addr, ucp_rkey_h rkey,
                                    std::string& expected_data)
{
    ucs_status_t status;
    T add, prev, result;

    prev = *(T*)memheap_addr;
    add  = (T)ucs::rand() * (T)ucs::rand();

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
void test_ucp_atomic::blocking_swap(entity *e,  size_t max_size, void *memheap_addr,
                                    ucp_rkey_h rkey, std::string& expected_data)
{
    ucs_status_t status;
    T swap, prev, result;

    prev = *(T*)memheap_addr;
    swap = (T)ucs::rand() * (T)ucs::rand();

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
void test_ucp_atomic::blocking_cswap(entity *e,  size_t max_size, void *memheap_addr,
                    ucp_rkey_h rkey, std::string& expected_data)
{
    ucs_status_t status;
    T compare, swap, prev, result;

    prev = *(T*)memheap_addr;
    if ((ucs::rand() % 2) == 0) {
        compare = prev; /* success mode */
    } else {
        compare = ~prev; /* fail mode */
    }
    swap = (T)ucs::rand() * (T)ucs::rand();

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

template <typename T>
ucs_status_t test_ucp_atomic::ucp_atomic_post_nbi(ucp_ep_h ep, ucp_atomic_post_op_t opcode,
                                              T value, void *remote_addr,
                                              ucp_rkey_h rkey)
{
    return ucp_atomic_post(ep, opcode, value, sizeof(T), (uintptr_t)remote_addr, rkey);
}

template <typename T>
void test_ucp_atomic::nb_add(entity *e,  size_t max_size, void *memheap_addr,
                  ucp_rkey_h rkey, std::string& expected_data)
{
    ucs_status_t status;
    T add, prev;

    prev = *(T*)memheap_addr;
    add  = (T)ucs::rand() * (T)ucs::rand();

    status = test_ucp_atomic::ucp_atomic_post_nbi<T>(e->ep(), UCP_ATOMIC_POST_OP_ADD, add,
                                                 memheap_addr, rkey);

    if (status == UCS_INPROGRESS) {
        flush_worker(*e);
    } else {
        ASSERT_UCS_OK(status);
    }
    expected_data.resize(sizeof(T));
    *(T*)&expected_data[0] = add + prev;
}

void test_ucp_atomic::unaligned_nb_add64(entity *e,  size_t max_size,
                                         void *memheap_addr, ucp_rkey_h rkey,
                                         std::string& expected_data)
{
    /* Test that unaligned addresses generate error */
    ucs_status_t status;
    
    status = test_ucp_atomic::ucp_atomic_post_nbi<uint64_t>(e->ep(),
                                                        UCP_ATOMIC_POST_OP_ADD, 0,
                                                        (void *)((uintptr_t)memheap_addr + 1),
                                                        rkey);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    expected_data.clear();
}

template <typename T>
ucs_status_ptr_t test_ucp_atomic::ucp_atomic_fetch(ucp_ep_h ep, 
                                                   ucp_atomic_fetch_op_t opcode,
                                                   T value, T *result,
                                                   void *remote_addr, ucp_rkey_h rkey)
{
    return ucp_atomic_fetch_nb(ep, opcode, value, result, sizeof(T),
                               (uintptr_t)remote_addr, rkey, send_completion);
}

template <typename T>
void test_ucp_atomic::nb_fadd(entity *e,  size_t max_size,
                              void *memheap_addr, ucp_rkey_h rkey,
                              std::string& expected_data)
{
    void *amo_req;
    T add, prev, result;

    prev = *(T*)memheap_addr;
    add  = (T)ucs::rand() * (T)ucs::rand();

    amo_req = test_ucp_atomic::ucp_atomic_fetch<T>(e->ep(), UCP_ATOMIC_FETCH_OP_FADD,
                                                   add, &result, memheap_addr, rkey);
    if(UCS_PTR_IS_PTR(amo_req)){
        wait(amo_req);
    }

    EXPECT_EQ(prev, result);

    expected_data.resize(sizeof(T));
    *(T*)&expected_data[0] = add + prev;
}

template <typename T>
void test_ucp_atomic::nb_swap(entity *e,  size_t max_size, void *memheap_addr,
                              ucp_rkey_h rkey, std::string& expected_data)
{
    T swap, prev, result;
    void *amo_req;

    prev = *(T*)memheap_addr;
    swap = (T)ucs::rand() * (T)ucs::rand();

    amo_req = test_ucp_atomic::ucp_atomic_fetch<T>(e->ep(), UCP_ATOMIC_FETCH_OP_SWAP,
                                                   swap, &result, memheap_addr, rkey);
    if(UCS_PTR_IS_PTR(amo_req)){
        wait(amo_req);
    }

    EXPECT_EQ(prev, result);

    expected_data.resize(sizeof(T));
    *(T*)&expected_data[0] = swap;
}

template <typename T>
void test_ucp_atomic::nb_cswap(entity *e,  size_t max_size, void *memheap_addr,
                    ucp_rkey_h rkey, std::string& expected_data)
{
    T compare, swap, prev, result;
    void *amo_req;

    prev = *(T*)memheap_addr;
    if ((ucs::rand() % 2) == 0) {
        compare = prev; /* success mode */
    } else {
        compare = ~prev; /* fail mode */
    }
    swap = result = (T)ucs::rand() * (T)ucs::rand();

    amo_req = test_ucp_atomic::ucp_atomic_fetch<T>(e->ep(), UCP_ATOMIC_FETCH_OP_CSWAP,
                                                   compare, &result,
                                                   memheap_addr, rkey);
    if(UCS_PTR_IS_PTR(amo_req)){
        wait(amo_req);
    }

    EXPECT_EQ(prev, result);

    expected_data.resize(sizeof(T));
    if (compare == prev) {
        *(T*)&expected_data[0] = swap;
    } else {
        *(T*)&expected_data[0] = prev;
    }
}

template <typename T, typename F>
void test_ucp_atomic::test(F f, bool malloc_allocate) {
    test_blocking_xfer(static_cast<blocking_send_func_t>(f), 
                       DEFAULT_SIZE, DEFAULT_ITERS,
                       sizeof(T),
                       malloc_allocate, false);
}


class test_ucp_atomic32 : public test_ucp_atomic {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_AMO32;
        return params;
    }
};

UCS_TEST_P(test_ucp_atomic32, atomic_add) {
    test<uint32_t>(&test_ucp_atomic32::blocking_add<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::blocking_add<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_add_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_add<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_add<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_fadd) {
    test<uint32_t>(&test_ucp_atomic32::blocking_fadd<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::blocking_fadd<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_fadd_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_fadd<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_fadd<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_swap) {
    test<uint32_t>(&test_ucp_atomic32::blocking_swap<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::blocking_swap<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_swap_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_swap<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_swap<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_cswap) {
    test<uint32_t>(&test_ucp_atomic32::blocking_cswap<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::blocking_cswap<uint32_t>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_cswap_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_cswap<uint32_t>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_cswap<uint32_t>, true);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic32)

class test_ucp_atomic64 : public test_ucp_atomic {
public:
    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        params.features |= UCP_FEATURE_AMO64;
        return params;
    }
};

UCS_TEST_P(test_ucp_atomic64, atomic_add) {
    test<uint64_t>(&test_ucp_atomic64::blocking_add<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::blocking_add<uint64_t>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_add_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_add<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_add<uint64_t>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_fadd) {
    test<uint64_t>(&test_ucp_atomic64::blocking_fadd<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::blocking_fadd<uint64_t>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_fadd_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_fadd<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_fadd<uint64_t>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_swap) {
    test<uint64_t>(&test_ucp_atomic64::blocking_swap<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::blocking_swap<uint64_t>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_swap_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_swap<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_swap<uint64_t>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_cswap) {
    test<uint64_t>(&test_ucp_atomic64::blocking_cswap<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::blocking_cswap<uint64_t>, true);
}


UCS_TEST_P(test_ucp_atomic64, atomic_cswap_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_cswap<uint64_t>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_cswap<uint64_t>, true);
}

#if ENABLE_PARAMS_CHECK
UCS_TEST_P(test_ucp_atomic64, unaligned_atomic_add) {
    test<uint64_t>(&test_ucp_atomic::unaligned_blocking_add64, false);
    test<uint64_t>(&test_ucp_atomic::unaligned_blocking_add64, true);
}

UCS_TEST_P(test_ucp_atomic64, unaligned_atomic_add_nb) {
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_add64, false);
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_add64, true);
}
#endif

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic64)
