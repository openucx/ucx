/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

#include <common/mem_buffer.h>
#include <common/test_helpers.h>
#include <ucs/sys/sys.h>


void test_ucp_memheap::init()
{
    ucp_test::init();
    sender().connect(&receiver(), get_ep_params());
}

void test_ucp_memheap::test_xfer(send_func_t send_func, size_t size,
                                 unsigned num_iters, size_t alignment,
                                 ucs_memory_type_t send_mem_type,
                                 ucs_memory_type_t target_mem_type,
                                 unsigned mem_map_flags,
                                 bool is_ep_flush, void *arg)
{
    ucp_mem_map_params_t params;
    ucs_status_t status;
    ptrdiff_t padding;
    ucp_mem_h memh;

    ucs_assert(!(mem_map_flags & (UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_FIXED)));

    mem_buffer memheap(num_iters * size + alignment, target_mem_type);
    padding = UCS_PTR_BYTE_DIFF(memheap.ptr(),
                                ucs_align_up_pow2_ptr(memheap.ptr(), alignment));
    ucs_assert(padding >= 0);
    ucs_assert(padding < alignment);

    /* Allocate heap */
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = memheap.ptr();
    params.length     = memheap.size();
    params.flags      = mem_map_flags;

    status = ucp_mem_map(receiver().ucph(), &params, &memh);
    ASSERT_UCS_OK(status);

    mem_buffer::pattern_fill(memheap.ptr(), memheap.size(), ucs::rand(),
                             memheap.mem_type());

    /* Unpack remote key */
    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver().ucph(), memh, &rkey_buffer,
                           &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);

    mem_buffer expected_data(memheap.size(), send_mem_type);

    /* Perform data sends */
    for (unsigned i = 0; i < num_iters; ++i) {
        ptrdiff_t offset = padding + (i * size);
        ucs_assert(offset + size <= memheap.size());

        (this->*send_func)(size,
                           UCS_PTR_BYTE_OFFSET(memheap.ptr(), offset),
                           rkey,
                           UCS_PTR_BYTE_OFFSET(expected_data.ptr(), offset),
                           arg);
        if (num_errors() > 0) {
            break;
        }
    }

    /* Flush to make sure memheap is synchronized */
    if (is_ep_flush) {
        flush_ep(sender());
    } else {
        flush_worker(sender());
    }

    /* Validate data */
    if (!mem_buffer::compare(UCS_PTR_BYTE_OFFSET(expected_data.ptr(), padding),
                             UCS_PTR_BYTE_OFFSET(memheap.ptr(), padding),
                             size * num_iters, send_mem_type, target_mem_type)) {
        ADD_FAILURE() << "data validation failed";
    }

    ucp_rkey_destroy(rkey);

    status = ucp_mem_unmap(receiver().ucph(), memh);
    ASSERT_UCS_OK(status);
}
