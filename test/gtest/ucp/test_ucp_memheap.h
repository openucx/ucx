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
     * Function type for memheap send operation
     *
     * @param [in]  size            Size of data to send
     * @param [in]  target_ptr      VA to perform the RMA operation to
     * @param [in]  rkey            RMA remote key
     * @param [in]  expected_data   Buffer to fill with expected data at 'target_ptr'
     * @param [in]  arg             User-defined argument
     *
     * @note The expected data buffer memory type is 'send_mem_type' as passed
     *       to @ref test_xfer
     * @note The expected_data buffer does not have to be filled during the
     *       function itself, however it must be there after endpoint/worker flush.
     */
    typedef void
    (test_ucp_memheap::* send_func_t)(size_t size, void *target_ptr,
                                      ucp_rkey_h rkey, void *expected_data,
                                      void *arg);

protected:
    virtual void init();

    void test_xfer(send_func_t send_func, size_t size, unsigned num_iters,
                   size_t alignment, ucs_memory_type_t send_mem_type,
                   ucs_memory_type_t target_mem_type, unsigned mem_map_flags,
                   bool is_ep_flush, void *arg);
};

#endif
