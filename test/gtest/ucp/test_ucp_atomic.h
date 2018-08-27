/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2016-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_UCP_ATOMIC_H_
#define TEST_UCP_ATOMIC_H_

#include "test_ucp_memheap.h"


class test_ucp_atomic : public test_ucp_memheap {
public:
    static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls);

    virtual void init();

    template <typename T>
    void blocking_add(entity *e,  size_t max_size, void *memheap_addr,
                      ucp_rkey_h rkey, std::string& expected_data);

    template<uint32_t> void blocking_add(entity *e,  size_t max_size, void *memheap_addr,
                                         ucp_rkey_h rkey, std::string& expected_data);

    void unaligned_blocking_add64(entity *e,  size_t max_size, void *memheap_addr,
                                  ucp_rkey_h rkey, std::string& expected_data);

    template <ucp_atomic_post_op_t OP>
    void unaligned_nb_post(entity *e,  size_t max_size, void *memheap_addr,
                           ucp_rkey_h rkey, std::string& expected_data);

    template <typename T>
    void nb_cswap(entity *e,  size_t max_size, void *memheap_addr,
                  ucp_rkey_h rkey, std::string& expected_data);
    
    template <typename T, typename F>
    void test(F f, bool malloc_allocate);

    template <typename T, ucp_atomic_post_op_t OP>
    void nb_post(entity *e,  size_t max_size, void *memheap_addr,
                 ucp_rkey_h rkey, std::string& expected_data);

    template <typename T, ucp_atomic_fetch_op_t FOP>
    void nb_fetch(entity *e,  size_t max_size, void *memheap_addr,
                  ucp_rkey_h rkey, std::string& expected_data);

    template <typename T, ucp_atomic_post_op_t OP>
    T atomic_op_val(T v1, T v2)
    {
        /* coverity[switch_selector_expr_is_constant] */
        switch (OP) {
        case UCP_ATOMIC_POST_OP_ADD:
            return v1 + v2;
        case UCP_ATOMIC_POST_OP_AND:
            return v1 & v2;
        case UCP_ATOMIC_POST_OP_OR:
            return v1 | v2;
        case UCP_ATOMIC_POST_OP_XOR:
            return v1 ^ v2;
        default:
            return 0;
        }
    }

    template <typename T, ucp_atomic_fetch_op_t OP>
    T atomic_fop_val(T v1, T v2)
    {
        /* coverity[switch_selector_expr_is_constant] */
        switch (OP) {
        case UCP_ATOMIC_FETCH_OP_FADD:
            return v1 + v2;
        case UCP_ATOMIC_FETCH_OP_FAND:
            return v1 & v2;
        case UCP_ATOMIC_FETCH_OP_FOR:
            return v1 | v2;
        case UCP_ATOMIC_FETCH_OP_FXOR:
            return v1 ^ v2;
        case UCP_ATOMIC_FETCH_OP_SWAP:
            return v1;
        default:
            return 0;
        }
    }

private:
    static void send_completion(void *request, ucs_status_t status){}
    template <typename T>
    ucs_status_t ucp_atomic_post_nbi(ucp_ep_h ep, ucp_atomic_post_op_t opcode,
                                 T value, void *remote_addr,
                                 ucp_rkey_h rkey);
    template <typename T>
    ucs_status_ptr_t ucp_atomic_fetch(ucp_ep_h ep, ucp_atomic_fetch_op_t opcode,
                                      T value, T *result,
                                      void *remote_addr, ucp_rkey_h rkey);
};

#endif
