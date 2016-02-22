/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_MEMHEAP_H
#define TEST_UCP_MEMHEAP_H

#include "ucp_test.h"


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
protected:
    void test_blocking_xfer(blocking_send_func_t send, size_t alignment);
    void test_nonblocking_implicit_stream_xfer(nonblocking_send_func_t send, size_t alignment);

    static ucp_params_t get_ctx_params();
};


#endif
