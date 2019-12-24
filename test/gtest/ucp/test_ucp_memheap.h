/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_MEMHEAP_H
#define TEST_UCP_MEMHEAP_H

#include "ucp_test.h"
extern "C" {
#include <ucp/core/ucp_mm.h>
#include <ucp/rma/rma.h>
}

class test_ucp_memheap : public ucp_test {
public:
    /*
     * @param [in]  max_size       Maximal size of data to send.
     * @param [in]  memheap_addr   VA to perform the RMA operation to,
     * @param [in]  rkey           Memheap remote key.
     * @param [out] expected_data  What should the memheap contain at the given
     *                             address after the operation (also can be used
     *                             as a source/destination data).
     */
    typedef void (test_ucp_memheap::* blocking_send_func_t)(entity *e,
                                                            size_t max_size,
                                                            void *memheap_addr,
                                                            ucp_rkey_h rkey,
                                                            std::string& expected_data);

    /*
     * @param [in]  max_size       Maximal size of data to send.
     * @param [in]  memheap_addr   VA to perform the RMA operation to,
     * @param [in]  rkey           Memheap remote key.
     * @param [out] expected_data  What should the memheap contain at the given
     *                             address after the operation (also can be used
     *                             as a source/destination data).
     */
    typedef void (test_ucp_memheap::* nonblocking_send_func_t)(entity *e,
                                                               size_t max_size,
                                                               void *memheap_addr,
                                                               ucp_rkey_h rkey,
                                                               std::string& expected_data);

    static std::vector<ucp_test_param> enum_test_params(const ucp_params_t& ctx_params,
                                                        const std::string& name,
                                                        const std::string& test_case_name,
                                                        const std::string& tls);


protected:
    const static size_t DEFAULT_SIZE  = 0;
    const static int    DEFAULT_ITERS = 0;

    void mem_map_and_rkey_exchange(ucp_test_base::entity &receiver,
                                   ucp_test_base::entity &sender,
                                   const ucp_mem_map_params_t &params,
                                   ucp_mem_h &receiver_memh,
                                   ucp_rkey_h &sender_rkey,
                                   void **memheap_addr = NULL);

    void test_blocking_xfer(blocking_send_func_t send, size_t len, int max_iters,
                            size_t alignment, bool malloc_allocate, bool is_ep_flush);

    void test_nonblocking_implicit_stream_xfer(nonblocking_send_func_t send,
                                               size_t len, int max_iters,
                                               size_t alignment, bool malloc_allocate,
                                               bool is_ep_flush);
};

class test_ucp_memheap_check_mem_type : public test_ucp_memheap {
public:
    void init();

    void cleanup();

    static std::vector<ucp_test_param>
    enum_test_params(const ucp_params_t& ctx_params,
                     const std::string& name,
                     const std::string& test_case_name,
                     const std::string& tls);

    static ucs_log_func_rc_t
    error_handler(const char *file, unsigned line, const char *function,
                  ucs_log_level_t level, const char *message, va_list ap);

    std::string get_err_exp_str(const std::string &op_type,
                                bool check_local  = true,
                                bool check_remote = true);

    virtual size_t get_data_size() const = 0;

    virtual bool check_gpu_direct_support(ucs_memory_type_t mem_type) = 0;

    void check_mem_type_op_status(ucs_status_t status,
                                  bool check_local = true,
                                  bool check_remote = true,
                                  bool allow_gpu_direct_local = true,
                                  bool allow_gpu_direct_remote = true);

protected:
    mem_buffer        *m_remote_mem_buf;
    mem_buffer        *m_local_mem_buf;
    ucp_mem_h         m_remote_mem_buf_memh;
    ucp_rkey_h        m_remote_mem_buf_rkey;
    ucs_memory_type_t m_local_mem_type;
    ucs_memory_type_t m_remote_mem_type;

public:
    static std::vector<std::vector<ucs_memory_type_t> > mem_type_pairs;
    static std::string                                  err_exp_str;
};


#endif
