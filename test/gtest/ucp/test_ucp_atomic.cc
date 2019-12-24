/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_atomic.h"
extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.inl>
}

template <typename T>
class test_ucp_atomic_check_mem_type :
    public test_ucp_memheap_check_mem_type {
public:
    void init() {
        // do nothing
    }

    void cleanup() {
        // do nothing
    }

    static ucp_params_t get_ctx_params() {
        ucp_params_t params = ucp_test::get_ctx_params();
        if (sizeof(T) == sizeof(uint32_t)) {
            params.features |= UCP_FEATURE_AMO32;
        } else if (sizeof(T) == sizeof(uint64_t)) {
            params.features |= UCP_FEATURE_AMO64;
        } else {
            UCS_TEST_ABORT("Unsupported AMO operation type - " <<
                           sizeof(T));
        }
        return params;
    }

    static void fetch_nb_completion(void *request, ucs_status_t status) {}

    size_t get_data_size() const {
        return sizeof(T);
    }

    bool check_gpu_direct_support(ucs_memory_type_t mem_type) {
        return ((m_remote_mem_buf_rkey->cache.amo_lane != UCP_NULL_LANE) &&
                !strcmp(m_remote_mem_buf_rkey->cache.amo_proto->name,
                        "basic_amo") &&
                (ucp_ep_md_attr(sender().ep(),
                                m_remote_mem_buf_rkey->cache.amo_lane)->
                 cap.reg_mem_types & UCS_BIT(mem_type)));
    }

    void test_mem_type() {
        for (int amo_mode = UCP_ATOMIC_MODE_CPU;
             amo_mode < UCP_ATOMIC_MODE_LAST; ++amo_mode) {
            modify_config("ATOMIC_MODE", ucp_atomic_modes[amo_mode]);
            test_ucp_memheap_check_mem_type::init();

            // Test AMO post
            {
                err_exp_str = get_err_exp_str("AMO", false, true);
                scoped_log_handler log_handler(error_handler);

                ucs_status_t status = ucp_atomic_post(sender().ep(), UCP_ATOMIC_POST_OP_ADD,
                                                      1lu, sizeof(T),
                                                      (uintptr_t)m_remote_mem_buf->ptr(),
                                                      m_remote_mem_buf_rkey);
                check_mem_type_op_status(status, false);
                flush_ep(sender());
            }

            // Test AMO non-blocking fetch
            {
                err_exp_str = get_err_exp_str("AMO");
                scoped_log_handler log_handler(error_handler);

                ucs_status_ptr_t status_ptr = ucp_atomic_fetch_nb(sender().ep(),
                                                                  UCP_ATOMIC_FETCH_OP_FADD,
                                                                  1lu, m_local_mem_buf->ptr(),
                                                                  sizeof(T),
                                                                  (uintptr_t)m_remote_mem_buf->ptr(),
                                                                  m_remote_mem_buf_rkey,
                                                                  fetch_nb_completion);
                if (UCS_PTR_IS_PTR(status_ptr)){
                    wait(status_ptr);
                } else {
                    check_mem_type_op_status(UCS_PTR_STATUS(status_ptr), true, true, false, true);
                }
                flush_ep(sender());
            }

            test_ucp_memheap_check_mem_type::cleanup();
        }
    }
};

class test_ucp_atomic32_check_mem_type :
    public test_ucp_atomic_check_mem_type<uint32_t> {
};

UCS_TEST_P(test_ucp_atomic32_check_mem_type, basic) {
    test_mem_type();
}

UCP_INSTANTIATE_TEST_CASE_CUDA_AWARE(test_ucp_atomic32_check_mem_type)

class test_ucp_atomic64_check_mem_type :
    public test_ucp_atomic_check_mem_type<uint32_t> {
};

UCS_TEST_P(test_ucp_atomic64_check_mem_type, basic) {
    test_mem_type();
}

UCP_INSTANTIATE_TEST_CASE_CUDA_AWARE(test_ucp_atomic64_check_mem_type)


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
    modify_config("ATOMIC_MODE", ucp_atomic_modes[GetParam().variant]);
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
    ucs_status_t status;
    {
        /* Test that unaligned addresses generate error */
        scoped_log_handler slh(hide_errors_logger);
        status = ucp_atomic_add64(e->ep(), 0, (uintptr_t)memheap_addr + 1, rkey);
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
    expected_data.clear();
}

template <typename T>
ucs_status_t test_ucp_atomic::ucp_atomic_post_nbi(ucp_ep_h ep, ucp_atomic_post_op_t opcode,
                                                  T value, void *remote_addr,
                                                  ucp_rkey_h rkey)
{
    return ucp_atomic_post(ep, opcode, value, sizeof(T), (uintptr_t)remote_addr, rkey);
}

template <typename T, ucp_atomic_post_op_t OP>
void test_ucp_atomic::nb_post(entity *e,  size_t max_size, void *memheap_addr,
                              ucp_rkey_h rkey, std::string& expected_data)
{
    ucs_status_t status;
    T val, prev;

    prev   = *(T*)memheap_addr;
    val    = (T)ucs::rand() * (T)ucs::rand();

    status = test_ucp_atomic::ucp_atomic_post_nbi<T>(e->ep(), OP, val, memheap_addr, rkey);

    if (status == UCS_INPROGRESS) {
        flush_worker(*e);
    } else {
        ASSERT_UCS_OK(status);
    }
    expected_data.resize(sizeof(T));
    *(T*)&expected_data[0] = atomic_op_val<T, OP>(val, prev);
}

template <ucp_atomic_post_op_t OP>
void test_ucp_atomic::unaligned_nb_post(entity *e,  size_t max_size,
                                        void *memheap_addr, ucp_rkey_h rkey,
                                        std::string& expected_data)
{
    ucs_status_t status;
    {
        /* Test that unaligned addresses generate error */
        scoped_log_handler slh(hide_errors_logger);
        status = test_ucp_atomic::ucp_atomic_post_nbi<uint64_t>
                (e->ep(), OP, 0, (void *)((uintptr_t)memheap_addr + 1), rkey);
    }
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

template <typename T, ucp_atomic_fetch_op_t FOP>
void test_ucp_atomic::nb_fetch(entity *e,  size_t max_size,
                               void *memheap_addr, ucp_rkey_h rkey,
                               std::string& expected_data)
{
    void *amo_req;
    T val, prev, result;

    prev    = *(T*)memheap_addr;
    val     = (T)ucs::rand() * (T)ucs::rand();

    amo_req = test_ucp_atomic::ucp_atomic_fetch<T>(e->ep(), FOP,
                                                   val, &result, memheap_addr, rkey);
    if(UCS_PTR_IS_PTR(amo_req)){
        wait(amo_req);
    }

    EXPECT_EQ(prev, result);

    expected_data.resize(sizeof(T));
    *(T*)&expected_data[0] = atomic_fop_val<T, FOP>(val, prev);
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
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_ADD>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_ADD>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_and_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_AND>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_AND>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_or_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_OR>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_OR>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_xor_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_XOR>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_post<uint32_t, UCP_ATOMIC_POST_OP_XOR>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_fadd_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FADD>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FADD>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_fand_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FAND>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FAND>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_for_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FOR>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FOR>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_fxor_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FXOR>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_FXOR>, true);
}

UCS_TEST_P(test_ucp_atomic32, atomic_swap_nb) {
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_SWAP>, false);
    test<uint32_t>(&test_ucp_atomic32::nb_fetch<uint32_t, UCP_ATOMIC_FETCH_OP_SWAP>, true);
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
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_ADD>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_ADD>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_and_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_AND>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_AND>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_or_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_OR>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_OR>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_xor_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_XOR>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_post<uint64_t, UCP_ATOMIC_POST_OP_XOR>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_fadd_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FADD>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FADD>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_fand_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FAND>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FAND>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_for_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FOR>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FOR>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_fxor_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FXOR>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_FXOR>, true);
}

UCS_TEST_P(test_ucp_atomic64, atomic_swap_nb) {
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_SWAP>, false);
    test<uint64_t>(&test_ucp_atomic64::nb_fetch<uint64_t, UCP_ATOMIC_FETCH_OP_SWAP>, true);
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
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_ADD>, false);
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_ADD>, true);
}

UCS_TEST_P(test_ucp_atomic64, unaligned_atomic_and_nb) {
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_AND>, false);
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_AND>, true);
}

UCS_TEST_P(test_ucp_atomic64, unaligned_atomic_or_nb) {
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_OR>, false);
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_OR>, true);
}

UCS_TEST_P(test_ucp_atomic64, unaligned_atomic_xor_nb) {
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_XOR>, false);
    test<uint64_t>(&test_ucp_atomic::unaligned_nb_post<UCP_ATOMIC_POST_OP_XOR>, true);
}
#endif

UCP_INSTANTIATE_TEST_CASE(test_ucp_atomic64)
