/**
 * See file LICENSE for terms.
 */

#include "rocm_vmm_mem_buffer.h"

#include <uct/test_md.h>


class test_rocm_copy_vmm_md : public test_md {
};

UCS_TEST_P(test_rocm_copy_vmm_md, mem_type) {
    rocm_vmm_mem_buffer buffer(1, 2);
    const size_t length = 64;
    uct_md_mem_attr_t mem_attr;
    ucs_memory_type_t mem_type;
    uct_mem_h memh;

    /* The start of the range, an interior address and a second mapping */
    for (void *address : {buffer.ptr(), UCS_PTR_BYTE_OFFSET(buffer.ptr(), 128),
                          buffer.chunk(1)}) {
        ASSERT_UCS_OK(
                uct_md_detect_memory_type(md(), address, length, &mem_type));
        EXPECT_EQ(UCS_MEMORY_TYPE_ROCM, mem_type) << address;

        mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_MEM_TYPE;
        ASSERT_UCS_OK(uct_md_mem_query(md(), address, length, &mem_attr));
        EXPECT_EQ(UCS_MEMORY_TYPE_ROCM, mem_attr.mem_type) << address;
    }

    ASSERT_UCS_OK(reg_mem(UCT_MD_MEM_ACCESS_ALL, buffer.ptr(),
                          buffer.chunk_size(), &memh));
    EXPECT_UCS_OK(uct_md_mem_dereg(md(), memh));
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_rocm_copy_vmm_md, rocm_cpy)
