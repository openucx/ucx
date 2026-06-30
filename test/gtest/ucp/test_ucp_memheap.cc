/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

#include <common/mem_buffer.h>
#include <common/test_helpers.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/ptr_arith.h>


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
                                 bool is_ep_flush, bool user_memh, void *arg,
                                 size_t reg_offset)
{
    ucp_mem_map_params_t params;
    ucs_status_t status;
    ptrdiff_t padding;
    ucp_mem_h memheap_memh, send_memh = NULL;
    void *mapped_ptr;

    ucs_assert(!(mem_map_flags & (UCP_MEM_MAP_ALLOCATE | UCP_MEM_MAP_FIXED)));

    ucs::auto_ptr<mem_buffer> memheap_buf(
            create_mem_buffer(reg_offset + num_iters * size + alignment,
                              target_mem_type));
    mem_buffer &memheap = *memheap_buf;
    mapped_ptr          = UCS_PTR_BYTE_OFFSET(memheap.ptr(), reg_offset);
    padding    = UCS_PTR_BYTE_DIFF(mapped_ptr,
                                   ucs_align_up_pow2_ptr(mapped_ptr,
                                                         alignment));
    ucs_assert(padding >= 0);
    ucs_assert(padding < alignment);

    /* Allocate heap */
    params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                        UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                        UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    params.address    = mapped_ptr;
    params.length     = memheap.size() - reg_offset;
    params.flags      = mem_map_flags;

    status = ucp_mem_map(receiver().ucph(), &params, &memheap_memh);
    ASSERT_UCS_OK(status);

    mem_buffer::pattern_fill(memheap.ptr(), memheap.size(), ucs::rand(),
                             memheap.mem_type());

    /* Unpack remote key */
    void *rkey_buffer;
    size_t rkey_buffer_size;
    status = ucp_rkey_pack(receiver().ucph(), memheap_memh, &rkey_buffer,
                           &rkey_buffer_size);
    ASSERT_UCS_OK(status);

    ucp_rkey_h rkey;
    status = ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &rkey);
    ASSERT_UCS_OK(status);

    ucp_rkey_buffer_release(rkey_buffer);

    ucs::auto_ptr<mem_buffer> expected_data_buf(
            create_mem_buffer(memheap.size(), send_mem_type));
    mem_buffer &expected_data = *expected_data_buf;

    if (user_memh) {
        params.address = expected_data.ptr();
        params.length  = expected_data.size();
        status         = ucp_mem_map(sender().ucph(), &params, &send_memh);
        ASSERT_UCS_OK(status);
    }

    /* Perform data sends */
    for (unsigned i = 0; i < num_iters; ++i) {
        ptrdiff_t offset = reg_offset + padding + (i * size);
        ucs_assert(offset + size <= memheap.size());

        (this->*send_func)(size,
                           UCS_PTR_BYTE_OFFSET(expected_data.ptr(), offset),
                           send_memh,
                           UCS_PTR_BYTE_OFFSET(memheap.ptr(), offset),
                           rkey, arg);
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
    if (!mem_buffer::compare(UCS_PTR_BYTE_OFFSET(expected_data.ptr(),
                                                reg_offset + padding),
                             UCS_PTR_BYTE_OFFSET(memheap.ptr(),
                                                reg_offset + padding),
                             size * num_iters, send_mem_type, target_mem_type)) {
        ADD_FAILURE() << "data validation failed";
    }

    ucp_rkey_destroy(rkey);

    status = ucp_mem_unmap(receiver().ucph(), memheap_memh);
    ASSERT_UCS_OK(status);

    if (user_memh) {
        status = ucp_mem_unmap(sender().ucph(), send_memh);
        ASSERT_UCS_OK(status);
    }
}
