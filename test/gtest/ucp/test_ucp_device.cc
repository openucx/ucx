/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/ucp_test.h>

#include <ucp/api/device/ucp_device_types.h>

#include "cuda/test_kernels.h"


template <typename T>
class dev_buffer : public mem_buffer {
public:
    using mem_buffer::mem_buffer;

    operator T*() {
        return reinterpret_cast<T*>(ptr());
    }
};

template <typename T>
static dev_buffer<T> make_dev_buffer(const std::vector<T>& src)
{
    return dev_buffer<T>(src.size() * sizeof(T), UCS_MEMORY_TYPE_CUDA,
                         src.data());
}

class test_ucp_device : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants,
                    UCP_FEATURE_RMA | UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64 | UCP_FEATURE_DEVICE);
    }

    virtual void init()
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        if (!is_loopback()) {
            receiver().connect(&sender(), get_ep_params());
        }

        ucp_device_mem_list_handle_h handle;
        while (ucp_device_mem_list_create(sender().ep(), NULL, &handle) ==
               UCS_ERR_NOT_CONNECTED) {
            progress();
        }
    }

    template<typename T>
    static std::vector<T> extract_vector(std::vector<T>& input,
                                         std::vector<unsigned>& indices)
    {
        std::vector<T> output;

        if (indices.size() == 0) {
            output = input;
        } else {
            for (auto i : indices) {
                output.push_back(input[i]);
            }
        }

        return output;
    }

    class mem_list {
        using mapped_buffer_ptr = std::unique_ptr<mapped_buffer>;

        std::vector<ucp_device_mem_list_elem_t>       m_elems;
        std::vector<ucs::handle<ucp_rkey_h>>          m_rkeys;
        ucp_device_mem_list_params_t                  m_params;

        entity&                        m_sender;
        entity&                        m_receiver;
        std::vector<mapped_buffer_ptr> m_src;
        std::vector<mapped_buffer_ptr> m_dst;

        std::vector<void*>             m_addresses;
        std::vector<uint64_t>          m_remote_addresses;
        std::vector<size_t>            m_lengths;


    public:
        mem_list(entity &sender, entity &receiver) :
            m_params({}), m_sender(sender), m_receiver(receiver)
        {
        }

        std::vector<mapped_buffer_ptr> &src()
        {
            return m_src;
        }

        std::vector<mapped_buffer_ptr> &dst()
        {
            return m_dst;
        }

        std::vector<void*> &addresses()
        {
            return m_addresses;
        }

        std::vector<uint64_t> &remote_addresses()
        {
            return m_remote_addresses;
        }

        std::vector<size_t> &lengths()
        {
            return m_lengths;
        }

        const ucp_device_mem_list_params_t &
        make(unsigned count = 1, size_t size = 0,
             ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_CUDA,
             size_t element_size = sizeof(ucp_device_mem_list_elem_t))
        {
            m_rkeys.clear();
            m_src.clear();
            m_dst.clear();
            m_elems.resize(size);
            m_remote_addresses.clear();
            m_addresses.clear();
            m_lengths.clear();

            for (auto i = 0; i < count; ++i) {
                m_src.emplace_back(
                        new mapped_buffer(size, m_sender, 0, mem_type));
                m_dst.emplace_back(
                        new mapped_buffer(size, m_receiver, 0, mem_type));

                m_src.back()->pattern_fill(0x1234, size);
                m_dst.back()->pattern_fill(0x4321, size);
                m_addresses.push_back(m_src.back()->ptr());
                m_remote_addresses.push_back(m_dst.back()->addr());
                m_lengths.push_back(size);
            }

            for (auto i = 0; i < count; ++i) {
                m_elems[i].field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                                        UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
                if ((i + 1) == count) {
                    m_elems[i].memh = NULL;
                } else {
                    m_elems[i].memh = m_src[i]->memh();
                }

                m_rkeys.push_back(m_dst[i]->rkey(m_sender));
                m_elems[i].rkey = m_rkeys.back();
            }

            m_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                    UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE |
                                    UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS;
            m_params.element_size = element_size;
            m_params.num_elements = m_src.size();
            m_params.elements     = m_elems.size() ? m_elems.data() : nullptr;
            return m_params;
        }
    };

    void put_multi(std::vector<unsigned> indices)
    {
        // Create memory list
        mem_list list(sender(), receiver());
        const size_t size = 4 * UCS_KBYTE;
        unsigned count    = 11;
        auto &params      = list.make(count, size);

        // Create range arrays from list of indices
        std::set<unsigned> rcvd(indices.begin(), indices.end());

        auto h_addresses        = extract_vector(list.addresses(), indices);
        auto h_remote_addresses = extract_vector(list.remote_addresses(), indices);
        auto h_lengths          = extract_vector(list.lengths(), indices);

        auto mem_list_indices   = make_dev_buffer(indices);
        auto addresses          = make_dev_buffer(h_addresses);
        auto remote_addresses   = make_dev_buffer(h_remote_addresses);
        auto lengths            = make_dev_buffer(h_lengths);

        // Create memory list handle
        ucp_device_mem_list_handle_h handle;
        ASSERT_EQ(UCS_OK,
                  ucp_device_mem_list_create(sender().ep(), &params, &handle));

        // Prepare destination counter
        const uint64_t counter_remote_address = list.remote_addresses().back();
        const uint64_t counter_inc_value      = 7;
        uint64_t counter_base                 = 0x1122334455667788;
        list.dst().back()->pattern_fill(counter_base, sizeof(counter_base));

        for (auto iter = 1000; iter > 0; --iter) {
            // Trigger copy
            ucs_status_t status;

            if (indices.size() > 0) {
                status = ucx_cuda::launch_ucp_put_multi_partial(
                    handle, mem_list_indices, indices.size(), addresses,
                    remote_addresses, lengths, counter_inc_value, counter_remote_address);
            } else {
                status = ucx_cuda::launch_ucp_put_multi(
                    handle, addresses, remote_addresses, lengths,
                    counter_inc_value, counter_remote_address);
            }

            EXPECT_EQ(UCS_OK, status);

            // Wait for counter
            counter_base += counter_inc_value;
            while (!mem_buffer::compare(&counter_base, list.dst().back()->ptr(),
                                        sizeof(counter_base), UCS_MEMORY_TYPE_CUDA));

            // Check data and clean buffer
            for (unsigned i = 0; i < count - 1; ++i) {
                uint64_t pattern = ((indices.size() == 0) ||
                                    (rcvd.find(i) != rcvd.end()) ?
                                    0x1234 : 0x4321);
                list.dst()[i]->pattern_check(pattern, size);
                list.dst()[i]->pattern_fill(0x4321, size);
            }
        }

        ucp_device_mem_list_release(handle);
    }
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

    ASSERT_EQ(1, ucx_cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
    ASSERT_EQ(cudaSuccess, cudaMemset(dst.ptr(), 0x11, size));
    ASSERT_EQ(0, ucx_cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
    ASSERT_EQ(cudaSuccess,
              cudaMemset(UCS_PTR_BYTE_OFFSET(dst.ptr(), size / 10), 0xfa, 10));
    ASSERT_EQ(1, ucx_cuda::launch_memcmp(src.ptr(), dst.ptr(), size));
}

UCS_TEST_P(test_ucp_device, create_fail)
{
    ucp_device_mem_list_handle_h handle       = nullptr;
    ucp_device_mem_list_params_t empty_params = {};

    ASSERT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(sender().ep(), NULL, &handle));

    empty_params.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(sender().ep(), &empty_params,
                                         &handle));

    mem_list list(sender(), receiver());
    // Empty list
    auto params = list.make(0);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(sender().ep(), &params, &handle));
    // Wrong element size
    auto params1 = list.make(1, 1 * UCS_MBYTE, UCS_MEMORY_TYPE_CUDA, 0);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(sender().ep(), &params1, &handle));
}

UCS_TEST_P(test_ucp_device, create_success)
{
    scoped_log_handler wrap_err(wrap_errors_logger);
    mem_list list(sender(), receiver());
    ucp_device_mem_list_handle_h handle = nullptr;

    auto &params = list.make(4, 4 * UCS_MBYTE);
    ASSERT_EQ(UCS_OK,
              ucp_device_mem_list_create(sender().ep(), &params, &handle));
    EXPECT_NE(nullptr, handle);
    ucp_device_mem_list_release(handle);
}

UCS_TEST_P(test_ucp_device, counter_inc)
{
    mem_list list(sender(), receiver());

    auto count   = 6;
    auto &params = list.make(count, UCS_KBYTE);

    // Create memory list handle
    ucp_device_mem_list_handle_h handle;
    ASSERT_EQ(UCS_OK,
              ucp_device_mem_list_create(sender().ep(), &params, &handle));

    const uint64_t inc_value = 9;
    uint64_t counter_base    = 0x1122334455667788;
    list.dst().back()->pattern_fill(counter_base, sizeof(counter_base));

    for (auto iter = 1000; iter > 0; --iter) {
        auto status = ucx_cuda::launch_ucp_counter_inc(handle,
                                                       count - 1,
                                                       inc_value,
                                                       list.remote_addresses().back());
        EXPECT_EQ(UCS_OK, status);

        counter_base += inc_value;
        while (!mem_buffer::compare(&counter_base, list.dst().back()->ptr(),
                                    sizeof(counter_base), UCS_MEMORY_TYPE_CUDA));
    }

    ucp_device_mem_list_release(handle);
}

UCS_TEST_P(test_ucp_device, put_single)
{
    mem_list list(sender(), receiver());
    const size_t size = 32 * UCS_KBYTE;
    auto &params      = list.make(6, size);

    // Create memory list
    ucp_device_mem_list_handle_h handle;
    ASSERT_EQ(UCS_OK,
              ucp_device_mem_list_create(sender().ep(), &params, &handle));

    // Target specific memory index
    unsigned mem_list_index = 3;
    auto src_ptr            = list.src()[mem_list_index]->ptr();
    auto dst_ptr            = list.dst()[mem_list_index]->addr();

    // Perform the transfer
    auto status = ucx_cuda::launch_ucp_put_single(handle,
                                                  mem_list_index,
                                                  src_ptr, dst_ptr,
                                                  size);
    EXPECT_EQ(UCS_OK, status);

    // Check proper index received data
    list.dst()[mem_list_index - 1]->pattern_check(0x4321, size);
    list.dst()[mem_list_index]->pattern_check(0x1234, size);
    list.dst()[mem_list_index + 1]->pattern_check(0x4321, size);

    // Perform another transfer
    mem_list_index = 2;
    src_ptr        = list.src()[mem_list_index]->ptr();
    dst_ptr        = list.dst()[mem_list_index]->addr();
    status = ucx_cuda::launch_ucp_put_single(handle,
                                             mem_list_index,
                                             src_ptr, dst_ptr,
                                             size);
    EXPECT_EQ(UCS_OK, status);
    list.dst()[mem_list_index]->pattern_check(0x1234, size);

    ucp_device_mem_list_release(handle);
}

UCS_TEST_P(test_ucp_device, put_multi_partial)
{
    put_multi({1, 3, 5, 7, 9});
}

UCS_TEST_P(test_ucp_device, put_multi)
{
    put_multi({});
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device, gdaki, "rc,rc_gda")
