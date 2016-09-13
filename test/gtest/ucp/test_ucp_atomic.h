/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
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

    template <typename T>
    void blocking_fadd(entity *e,  size_t max_size, void *memheap_addr,
                       ucp_rkey_h rkey, std::string& expected_data);

    template <typename T>
    void blocking_swap(entity *e,  size_t max_size, void *memheap_addr,
                       ucp_rkey_h rkey, std::string& expected_data);

    template <typename T>
    void blocking_cswap(entity *e,  size_t max_size, void *memheap_addr,
                        ucp_rkey_h rkey, std::string& expected_data);

    template <typename T, typename F>
    void test(F f, bool malloc_allocate);
};

#endif
