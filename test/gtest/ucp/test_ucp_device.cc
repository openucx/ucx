/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/ucp_test.h>

#include <ucp/api/device/ucp_device_types.h>

#include "cuda/test_kernels.h"

class test_ucp_device : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_RMA | UCP_FEATURE_AMO64);
    }

    virtual void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }
    }

    class mem_list {
        using mapped_buffer_ptr = std::unique_ptr<mapped_buffer>;

        struct params {
            std::unique_ptr<ucp_mem_list_elem_t[]> elems;
            std::vector<ucs::handle<ucp_rkey_h>>   rkeys;
            ucp_mem_list_params_t                  params;
        };

        entity&                                  m_sender;
        entity&                                  m_receiver;

        std::vector<mapped_buffer_ptr>           m_src;
        std::vector<mapped_buffer_ptr>           m_dst;

        params                                   m_params;

    public:
        mem_list(entity &sender, entity &receiver) :
            m_sender(sender), m_receiver(receiver)
        {
        }

        void add(size_t size, unsigned count = 1,
                 ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_CUDA)
        {
            for (auto i = 0; i < count; ++i) {
                m_src.emplace_back(
                        new mapped_buffer(size, m_sender, 0, mem_type));
                m_dst.emplace_back(
                        new mapped_buffer(size, m_receiver, 0, mem_type));
            }
        }

        const ucp_mem_list_params_t&
        params(size_t elem_size = sizeof(ucp_mem_list_elem_t))
        {
            m_params.rkeys.clear();
            m_params.elems = std::unique_ptr<ucp_mem_list_elem_t[]>(
                    new ucp_mem_list_elem_t[m_src.size()]);

            for (auto i = 0; i < m_src.size(); ++i) {
                m_params.elems[i].field_mask = UCP_MEM_LIST_ELEM_FIELD_MEMH |
                                               UCP_MEM_LIST_ELEM_FIELD_RKEY;
                m_params.elems[i].memh = m_src[i]->memh();
                m_params.rkeys.push_back(m_dst[i]->rkey(m_sender));
                m_params.elems[i].rkey = m_params.rkeys.back();
            }

            auto& params        = m_params.params;
            params.field_mask   = UCP_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                  UCP_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                                  UCP_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
            params.element_size = elem_size;
            params.num_elements = m_src.size();
            params.elements     = m_params.elems.get();
            return params;
        }
    };
};

UCS_TEST_P(test_ucp_device, mapped_buffer_kernel_memcmp)
{
    size_t size = 100 * UCS_MBYTE;

    mapped_buffer dst(size, receiver(), 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer src(size, sender(), 0, UCS_MEMORY_TYPE_CUDA);

    src.pattern_fill(0x1234, size);
    src.pattern_check(0x1234, size);

    ASSERT_EQ(cudaSuccess, cudaMemset(src.ptr(), 0x11, size));
    ASSERT_EQ(cudaSuccess, cudaMemset(dst.ptr(), 0xde, size));

    ASSERT_EQ(1, cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
    ASSERT_EQ(cudaSuccess, cudaMemset(dst.ptr(), 0x11, size));
    ASSERT_EQ(0, cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
    ASSERT_EQ(cudaSuccess,
              cudaMemset(UCS_PTR_BYTE_OFFSET(dst.ptr(), size / 10), 0xfa, 10));
    ASSERT_EQ(1, cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
}

UCS_TEST_P(test_ucp_device, put_single)
{
    const size_t size = 16 * UCS_KBYTE;

    mapped_buffer src(size, sender(), 0, UCS_MEMORY_TYPE_CUDA);
    mapped_buffer dst(size, receiver(), 0, UCS_MEMORY_TYPE_CUDA);

    src.pattern_fill(0x1234, size);
    src.pattern_check(0x1234, size);

    // TODO create mem list
    ucp_device_mem_list_handle_h mem_list = nullptr;

    ucs_status_t status = cuda::launch_ucp_put_single(mem_list, src.ptr(),
                                                      (uint64_t)dst.ptr(),
                                                      size);
    EXPECT_EQ(UCS_ERR_NOT_IMPLEMENTED, status);
}

UCS_TEST_P(test_ucp_device, create_fail)
{
    ucp_device_mem_list_handle_h handle = nullptr;
    ucp_mem_list_params_t empty_params  = {};

    ASSERT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_mem_list_create(sender().ep(), NULL, &handle));

    empty_params.field_mask = UCP_MEM_LIST_PARAMS_FIELD_ELEMENTS;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_mem_list_create(sender().ep(), &empty_params, &handle));

    mem_list list(sender(), receiver());
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_mem_list_create(sender().ep(), &list.params(), &handle));
    list.add(1 * UCS_MBYTE, 10);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_mem_list_create(sender().ep(), &list.params(31), &handle));
    list.add(1 * UCS_MBYTE, 10, UCS_MEMORY_TYPE_HOST);
    EXPECT_EQ(UCS_ERR_UNSUPPORTED,
              ucp_mem_list_create(sender().ep(), &list.params(), &handle));
}

UCS_TEST_P(test_ucp_device, create_success)
{
    mem_list list(sender(), receiver());
    ucp_device_mem_list_handle_h handle = nullptr;

    list.add(4 * UCS_MBYTE, 1);
    ASSERT_EQ(UCS_OK,
              ucp_mem_list_create(sender().ep(), &list.params(), &handle));
    EXPECT_NE(nullptr, handle);
    ucp_mem_list_release(sender().ep(), handle);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device, gdaki, "rc,gdaki")
