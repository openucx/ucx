/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef TEST_UCP_MEMHEAP_H
#define TEST_UCP_MEMHEAP_H

#include "ucp_test.h"

#include <common/mem_buffer.h>


class test_ucp_memheap : public ucp_test {
public:
    /*
     * Function type for memheap send operation
     *
     * @param [in]  size            Size of data to send
     * @param [in]  expected_data   Buffer to fill with expected data at 'target_ptr'
     * @param [in]  memh            Memory handle of local buffer
     * @param [in]  target_ptr      VA to perform the RMA operation to
     * @param [in]  rkey            RMA remote key
     * @param [in]  arg             User-defined argument
     *
     * @note The expected data buffer memory type is 'send_mem_type' as passed
     *       to @ref test_xfer
     * @note The expected_data buffer does not have to be filled during the
     *       function itself, however it must be there after endpoint/worker flush.
     */
    typedef void
    (test_ucp_memheap::* send_func_t)(size_t size, void *expected_data,
                                      ucp_mem_h memh, void *target_ptr,
                                      ucp_rkey_h rkey, void *arg);

protected:
    virtual void init();

    /* Factory for the buffers used by @ref test_xfer. Subclasses can override
     * it to change how the buffers are allocated (e.g. asynchronously). */
    virtual mem_buffer *create_mem_buffer(size_t size,
                                          ucs_memory_type_t mem_type)
    {
        return new mem_buffer(size, mem_type);
    }

    void test_xfer(send_func_t send_func, size_t size, unsigned num_iters,
                   size_t alignment, ucs_memory_type_t send_mem_type,
                   ucs_memory_type_t target_mem_type, unsigned mem_map_flags,
                   bool is_ep_flush, bool user_memh, void *arg,
                   size_t reg_offset = 0);
};

#endif
