/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <cuda_runtime.h>
#include <ucp/ucp_test.h>

#include <ucp/api/device/ucp_device_types.h>

#include <common/cuda.h>
#include "cuda/test_kernels.h"

class test_ucp_device : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants);

    virtual void init() override;

protected:
    class mem_list {
    public:
        static constexpr uint64_t SEED_SRC = 0x1234;
        static constexpr uint64_t SEED_DST = 0x4321;

        mem_list(entity &sender, entity &receiver, size_t size, unsigned count,
                 ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_CUDA);
        ~mem_list();

        void *src_ptr(unsigned index) const;

        uint64_t dst_ptr(unsigned index) const;

        std::vector<void*> src_ptrs() const;

        std::vector<uint64_t> dst_ptrs() const;

        void dst_counter_init(unsigned index);

        uint64_t dst_counter_read(unsigned index) const;

        ucp_device_mem_list_handle_h handle() const;

        void dst_pattern_check(unsigned index, uint64_t seed) const;

    private:
        entity                                      &m_receiver;
        std::vector<std::unique_ptr<mapped_buffer>> m_src, m_dst;
        std::vector<ucs::handle<ucp_rkey_h>>        m_rkeys;
        ucp_device_mem_list_handle_h                m_mem_list_h;

        ucp_device_counter_params_t dst_counter_params(unsigned index) const;
    };

    size_t counter_size();
};


void test_ucp_device::get_test_variants(std::vector<ucp_test_variant> &variants)
{
    add_variant(variants,
                UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_DEVICE);
}

void test_ucp_device::init()
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
test_ucp_device::mem_list::mem_list(entity &sender, entity &receiver,
                                    size_t size, unsigned count,
                                    ucs_memory_type_t mem_type) :
    m_receiver(receiver)
{
    // Prepare src and dst buffers
    for (auto i = 0; i < count; ++i) {
        m_src.emplace_back(new mapped_buffer(size, sender, 0, mem_type));
        m_dst.emplace_back(new mapped_buffer(size, receiver, 0, mem_type));
        m_rkeys.push_back(m_dst.back()->rkey(sender));
        m_src.back()->pattern_fill(SEED_SRC, size);
        m_dst.back()->pattern_fill(SEED_DST, size);
    }

    // Initialize elements
    std::vector<ucp_device_mem_list_elem_t> elems(count);
    for (auto i = 0; i < count; ++i) {
        auto &elem      = elems[i];
        elem.field_mask = UCP_DEVICE_MEM_LIST_ELEM_FIELD_MEMH |
                          UCP_DEVICE_MEM_LIST_ELEM_FIELD_RKEY;
        elem.memh       = m_src[i]->memh();
        elem.rkey       = m_rkeys[i];
    }

    // Initialize parameters
    ucp_device_mem_list_params_t params;
    params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                          UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                          UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE;
    params.element_size = sizeof(elems[0]);
    params.num_elements = count;
    params.elements     = elems.data();

    // Create memory list
    ASSERT_UCS_OK(
            ucp_device_mem_list_create(sender.ep(), &params, &m_mem_list_h));
}

test_ucp_device::mem_list::~mem_list()
{
    ucp_device_mem_list_release(m_mem_list_h);
}

void *test_ucp_device::mem_list::src_ptr(unsigned index) const
{
    return m_src[index]->ptr();
}

uint64_t test_ucp_device::mem_list::dst_ptr(unsigned index) const
{
    return reinterpret_cast<uint64_t>(m_dst[index]->ptr());
}

std::vector<void*> test_ucp_device::mem_list::src_ptrs() const
{
    std::vector<void*> result;
    for (const auto &src : m_src) {
        result.push_back(src->ptr());
    }
    return result;
}

std::vector<uint64_t> test_ucp_device::mem_list::dst_ptrs() const
{
    std::vector<uint64_t> result;
    for (const auto &dst : m_dst) {
        result.push_back(reinterpret_cast<uint64_t>(dst->ptr()));
    }
    return result;
}

ucp_device_counter_params_t
test_ucp_device::mem_list::dst_counter_params(unsigned index) const
{
    ucp_device_counter_params_t params;
    params.field_mask = UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH;
    params.memh       = m_dst[index]->memh();
    return params;
}

void test_ucp_device::mem_list::dst_counter_init(unsigned index)
{
    ucp_device_counter_params_t params = dst_counter_params(index);
    ASSERT_UCS_OK(ucp_device_counter_init(m_receiver.worker(), &params,
                                          m_dst[index]->ptr()));
}

uint64_t test_ucp_device::mem_list::dst_counter_read(unsigned index) const
{
    ucp_device_counter_params_t params = dst_counter_params(index);
    return ucp_device_counter_read(m_receiver.worker(), &params,
                                   m_dst[index]->ptr());
}

void test_ucp_device::mem_list::dst_pattern_check(unsigned index,
                                                  uint64_t seed) const
{
    m_dst[index]->pattern_check(seed, m_dst[index]->size());
}

ucp_device_mem_list_handle_h test_ucp_device::mem_list::handle() const
{
    return m_mem_list_h;
}

size_t test_ucp_device::counter_size()
{
    ucp_context_attr_t attr;
    attr.field_mask = UCP_ATTR_FIELD_DEVICE_COUNTER_SIZE;
    ASSERT_UCS_OK(ucp_context_query(receiver().ucph(), &attr));
    return attr.device_counter_size;
}

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

UCS_TEST_P(test_ucp_device, create_success)
{
    mem_list list(sender(), receiver(), 4 * UCS_MBYTE, 4);
    EXPECT_NE(nullptr, list.handle());
}

UCS_TEST_P(test_ucp_device, put_single)
{
    static constexpr size_t size = 32 * UCS_KBYTE;
    mem_list list(sender(), receiver(), size, 6);

    // Perform the transfer
    static constexpr unsigned mem_list_index = 3;
    ucx_cuda::kernel_params params;
    params.mem_list              = list.handle();
    params.single.mem_list_index = mem_list_index;
    params.single.address        = list.src_ptr(mem_list_index);
    params.single.remote_address = list.dst_ptr(mem_list_index);
    params.single.length         = size;
    ucs_status_t status          = ucx_cuda::launch_ucp_put_single(params);
    ASSERT_EQ(UCS_OK, status);

    // Check proper index received data
    list.dst_pattern_check(mem_list_index - 1, mem_list::SEED_DST);
    list.dst_pattern_check(mem_list_index, mem_list::SEED_SRC);
    list.dst_pattern_check(mem_list_index + 1, mem_list::SEED_DST);
}

UCS_TEST_P(test_ucp_device, put_multi)
{
    static constexpr size_t size    = 32 * UCS_KBYTE;
    static constexpr unsigned count = 32;
    mem_list list(sender(), receiver(), size, count + 1);

    const unsigned counter_index = count;
    list.dst_counter_init(counter_index);

    auto addresses        = ucx_cuda::make_device_vector(list.src_ptrs());
    auto remote_addresses = ucx_cuda::make_device_vector(list.dst_ptrs());
    auto lengths          = ucx_cuda::make_device_vector(std::vector<size_t>(count, size));

    // Perform the transfer
    ucx_cuda::kernel_params params;
    params.mem_list                     = list.handle();
    params.multi.addresses              = addresses.ptr();
    params.multi.remote_addresses       = remote_addresses.ptr();
    params.multi.lengths                = lengths.ptr();
    params.multi.counter_remote_address = list.dst_ptr(counter_index);
    params.multi.counter_inc_value      = 1;
    ucs_status_t status = ucx_cuda::launch_ucp_put_multi(params);
    ASSERT_EQ(UCS_OK, status);

    // Check received data
    for (unsigned i = 0; i < count; ++i) {
        list.dst_pattern_check(i, mem_list::SEED_SRC);
    }

    // Check counter
    EXPECT_EQ(1, list.dst_counter_read(counter_index));
}

UCS_TEST_P(test_ucp_device, put_multi_partial)
{
    static constexpr size_t size          = 32 * UCS_KBYTE;
    static constexpr unsigned total_count = 32;
    mem_list list(sender(), receiver(), size, total_count + 1);

    const unsigned counter_index = total_count;
    list.dst_counter_init(counter_index);

    // Random list of indexes
    std::vector<unsigned> indexes_vec;
    for (unsigned i = 0; i < total_count; ++i) {
        if (ucs::rand() % 2) {
            indexes_vec.push_back(i);
        }
    }

    std::vector<void*> addresses_vec;
    std::vector<uint64_t> remote_addresses_vec;
    for (auto index : indexes_vec) {
        addresses_vec.push_back(list.src_ptr(index));
        remote_addresses_vec.push_back(list.dst_ptr(index));
    }

    auto indexes          = ucx_cuda::make_device_vector(indexes_vec);
    auto addresses        = ucx_cuda::make_device_vector(addresses_vec);
    auto remote_addresses = ucx_cuda::make_device_vector(remote_addresses_vec);
    auto lengths          = ucx_cuda::make_device_vector(
            std::vector<size_t>(indexes_vec.size(), size));

    // Perform the transfer
    ucx_cuda::kernel_params params;
    params.mem_list                       = list.handle();
    params.partial.addresses              = addresses.ptr();
    params.partial.remote_addresses       = remote_addresses.ptr();
    params.partial.lengths                = lengths.ptr();
    params.partial.mem_list_indices       = indexes.ptr();
    params.partial.mem_list_count         = indexes_vec.size();
    params.partial.counter_index          = counter_index;
    params.partial.counter_remote_address = list.dst_ptr(counter_index);
    params.partial.counter_inc_value      = 1;
    ucs_status_t status = ucx_cuda::launch_ucp_put_multi_partial(params);
    ASSERT_EQ(UCS_OK, status);

    // Check received data
    std::set<unsigned> indexes_set(indexes_vec.begin(), indexes_vec.end());
    for (auto index : indexes_vec) {
        uint64_t seed = (indexes_set.find(index) == indexes_set.end()) ?
                                mem_list::SEED_DST :
                                mem_list::SEED_SRC;
        list.dst_pattern_check(index, seed);
    }

    // Check counter
    EXPECT_EQ(1, list.dst_counter_read(counter_index));
}

UCS_TEST_P(test_ucp_device, counter)
{
    const size_t size = counter_size();
    mem_list list(sender(), receiver(), size, 1);

    static constexpr unsigned mem_list_index = 0;
    list.dst_counter_init(mem_list_index);

    // Perform the transfer
    ucx_cuda::kernel_params params;
    params.mem_list               = list.handle();
    params.counter.mem_list_index = mem_list_index;
    params.counter.inc_value      = 1;
    params.counter.remote_address = list.dst_ptr(mem_list_index);
    ucs_status_t status           = ucx_cuda::launch_ucp_counter_inc(params);
    ASSERT_EQ(UCS_OK, status);

    // Check destination
    EXPECT_EQ(1, list.dst_counter_read(mem_list_index));
}

UCS_TEST_P(test_ucp_device, create_fail)
{
    ucp_device_mem_list_handle_h handle = nullptr;
    auto ep                             = sender().ep();

    scoped_log_handler wrap_err(wrap_errors_logger);

    // Null params
    ASSERT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(ep, NULL, &handle));

    // Empty params
    ucp_device_mem_list_params_t empty_params = {};
    empty_params.field_mask = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(ep, &empty_params, &handle));
    EXPECT_EQ(nullptr, handle);

    // Empty mem list
    ucp_device_mem_list_params_t invalid_params = {};
    invalid_params.field_mask   = UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENTS |
                                  UCP_DEVICE_MEM_LIST_PARAMS_FIELD_NUM_ELEMENTS |
                                  UCP_DEVICE_MEM_LIST_PARAMS_FIELD_ELEMENT_SIZE;
    invalid_params.elements     = NULL;
    invalid_params.num_elements = 0;
    invalid_params.element_size = sizeof(ucp_device_mem_list_elem_t);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(ep, &invalid_params, &handle));
    EXPECT_EQ(nullptr, handle);

    // Zero element size
    ucp_device_mem_list_elem_t dummy_elem = {};
    invalid_params.elements               = &dummy_elem;
    invalid_params.num_elements           = 1;
    invalid_params.element_size           = 0;
    EXPECT_EQ(UCS_ERR_INVALID_PARAM,
              ucp_device_mem_list_create(ep, &invalid_params, &handle));
    EXPECT_EQ(nullptr, handle);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device, gdaki, "rc,rc_gda")
