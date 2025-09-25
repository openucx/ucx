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
    };

    size_t counter_size();

    static void counter_init(const mapped_buffer &buffer);

    static uint64_t counter_read(const mapped_buffer &buffer);

    void launch_kernel(const test_ucp_device_kernel_params_t &params);
};


void test_ucp_device::get_test_variants(std::vector<ucp_test_variant> &variants)
{
    add_variant(variants,
                UCP_FEATURE_RMA | UCP_FEATURE_AMO64 | UCP_FEATURE_DEVICE);
}

void test_ucp_device::init()
{
    m_env.push_back(new ucs::scoped_setenv("UCX_CUDA_IPC_ENABLE_SAME_PROCESS", "y"));
    m_env.push_back(new ucs::scoped_setenv("UCX_IB_GPU_IB_DISTANCE_LATENCY_THRESH", "1000ns"));
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

void test_ucp_device::mem_list::dst_counter_init(unsigned index)
{
    test_ucp_device::counter_init(*m_dst[index]);
}

uint64_t test_ucp_device::mem_list::dst_counter_read(unsigned index) const
{
    return test_ucp_device::counter_read(*m_dst[index]);
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

void test_ucp_device::counter_init(const mapped_buffer &buffer)
{
    ucp_device_counter_params_t params;
    params.field_mask = UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH;
    params.memh       = buffer.memh();
    ASSERT_UCS_OK(
            ucp_device_counter_init(buffer.worker(), &params, buffer.ptr()));
}

uint64_t test_ucp_device::counter_read(const mapped_buffer &buffer)
{
    ucp_device_counter_params_t params;
    params.field_mask = UCP_DEVICE_COUNTER_PARAMS_FIELD_MEMH;
    params.memh       = buffer.memh();
    return ucp_device_counter_read(buffer.worker(), &params, buffer.ptr());
}

UCS_TEST_P(test_ucp_device, create_success)
{
    mem_list list(sender(), receiver(), 4 * UCS_MBYTE, 4);
    EXPECT_NE(nullptr, list.handle());
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

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device, rc_gda, "rc,rc_gda")


class test_ucp_device_kernel : public test_ucp_device {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        // TODO move to UCS
        static const char *ucs_device_level_names[] = {"thread", "warp",
                                                       "block", "grid"};
        add_variant_values(variants, test_ucp_device::get_test_variants,
                           UCS_BIT(UCS_DEVICE_LEVEL_THREAD) |
                                   UCS_BIT(UCS_DEVICE_LEVEL_WARP),
                           ucs_device_level_names);
    }

protected:
    ucs_device_level_t get_device_level() const
    {
        return static_cast<ucs_device_level_t>(get_variant_value(0));
    }

    test_ucp_device_kernel_params_t init_params(unsigned num_iters = 1)
    {
        test_ucp_device_kernel_params_t params = {};
        params.num_threads                     = get_num_threads();
        params.num_blocks                      = 1;
        params.level                           = get_device_level();
        params.num_iters                       = num_iters;
        return params;
    }

    virtual unsigned get_num_threads() const
    {
        switch (get_device_level()) {
        case UCS_DEVICE_LEVEL_THREAD:
            return 1;
        case UCS_DEVICE_LEVEL_WARP:
            return UCS_DEVICE_NUM_THREADS_IN_WARP;
        default:
            return 1;
        }
    }

    void launch_kernel(const test_ucp_device_kernel_params_t &params)
    {
        ucs_status_t status = launch_test_ucp_device_kernel(params);
        ASSERT_UCS_OK(status);
    }
};

UCS_TEST_P(test_ucp_device_kernel, local_counter)
{
    mapped_buffer counter_buffer(counter_size(), receiver(), 0,
                                 UCS_MEMORY_TYPE_CUDA);
    const uint64_t value = 1764;

    // Perform the write
    auto params                  = init_params(1);
    params.operation             = TEST_UCP_DEVICE_KERNEL_COUNTER_WRITE;
    params.local_counter.address = counter_buffer.ptr();
    params.local_counter.value   = value;
    launch_kernel(params);

    EXPECT_TRUE(mem_buffer::compare(&value, counter_buffer.ptr(), sizeof(value),
                                    counter_buffer.mem_type()));

    // Check counter value using device API
    params.operation = TEST_UCP_DEVICE_KERNEL_COUNTER_READ;
    launch_kernel(params);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device_kernel, rc_gda,
                                        "rc,rc_gda")


class test_ucp_device_xfer : public test_ucp_device_kernel {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_values(variants, test_ucp_device_kernel::get_test_variants,
                           NODELAY_WITH_REQ, "nodelay_with_req");
        add_variant_values(variants, test_ucp_device_kernel::get_test_variants,
                           NODELAY_WITHOUT_REQ, "nodelay_without_req");
        add_variant_values(variants, test_ucp_device_kernel::get_test_variants,
                           LAZY_WITHOUT_REQ, "lazy_without_req");
    }

protected:
    typedef enum {
        NODELAY_WITH_REQ,
        NODELAY_WITHOUT_REQ,
        LAZY_WITHOUT_REQ,
    } send_mode_t;

    test_ucp_device_kernel_params_t init_params()
    {
        // TODO: Test different sizes and alignment
        test_ucp_device_kernel_params_t params;
        params.num_threads = get_num_threads();
        params.num_blocks  = 1;
        params.level       = get_device_level();
        params.num_iters   = get_num_iters();
        switch (get_send_mode()) {
        case NODELAY_WITH_REQ:
            params.with_no_delay = true;
            params.with_request  = true;
            break;
        case NODELAY_WITHOUT_REQ:
            params.with_no_delay = true;
            params.with_request  = false;
            break;
        case LAZY_WITHOUT_REQ:
            params.with_no_delay = false;
            params.with_request  = false;
            break;
        default:
            UCS_TEST_ABORT("Invalid send mode");
        }
        return params;
    }

    send_mode_t get_send_mode() const
    {
        return static_cast<send_mode_t>(get_variant_value(1));
    }

    virtual unsigned get_num_threads() const override
    {
        switch (get_device_level()) {
        case UCS_DEVICE_LEVEL_THREAD:
            /* When using thread-level use less threads to shorten the test */
            return 8;
        case UCS_DEVICE_LEVEL_WARP:
            return 128;
        default:
            return 1;
        }
    }

    unsigned get_multi_elem_count() const
    {
        /* When using thread-level use less threads to shorten the test */
        switch (get_device_level()) {
        case UCS_DEVICE_LEVEL_THREAD:
            /* Thread level uses less elements to not overflow the QP */
            return 16;
        case UCS_DEVICE_LEVEL_WARP:
            return 32;
        default:
            return 1;
        }
    }

    size_t get_num_iters() const
    {
        return 10;
    }

    size_t get_num_ops_multiplier() const
    {
        switch (get_device_level()) {
        case UCS_DEVICE_LEVEL_THREAD:
            return get_num_threads();
        case UCS_DEVICE_LEVEL_WARP:
            return get_num_threads() / 32;
        default:
            return 1;
        }
    }

    void wait_for_counter(const mem_list &list, unsigned counter_index)
    {
        const size_t multiplier = get_num_ops_multiplier();
        uint64_t target_value   = get_num_iters() * multiplier;

        wait_for_cond(
                [&list, counter_index, target_value]() {
                    return list.dst_counter_read(counter_index) == target_value;
                },
                [] {});
        EXPECT_EQ(get_num_iters() * multiplier,
                  list.dst_counter_read(counter_index))
                << "multiplier: " << multiplier;
    }
};

UCS_TEST_P(test_ucp_device_xfer, put_single)
{
    static constexpr size_t size = 32 * UCS_KBYTE;
    mem_list list(sender(), receiver(), size, 6);

    // Perform the transfer
    static constexpr unsigned mem_list_index = 3;
    auto params = init_params();
    params.operation             = TEST_UCP_DEVICE_KERNEL_PUT_SINGLE;
    params.mem_list              = list.handle();
    params.single.mem_list_index = mem_list_index;
    params.single.address        = list.src_ptr(mem_list_index);
    params.single.remote_address = list.dst_ptr(mem_list_index);
    params.single.length         = size;
    launch_kernel(params);

    // Check proper index received data
    list.dst_pattern_check(mem_list_index - 1, mem_list::SEED_DST);
    list.dst_pattern_check(mem_list_index, mem_list::SEED_SRC);
    list.dst_pattern_check(mem_list_index + 1, mem_list::SEED_DST);
}

UCS_TEST_P(test_ucp_device_xfer, put_multi)
{
    static constexpr size_t size = 32 * UCS_KBYTE;
    unsigned count               = get_multi_elem_count();
    mem_list list(sender(), receiver(), size, count + 1);

    const unsigned counter_index = count;
    list.dst_counter_init(counter_index);

    auto addresses        = ucx_cuda::make_device_vector(list.src_ptrs());
    auto remote_addresses = ucx_cuda::make_device_vector(list.dst_ptrs());
    auto lengths          = ucx_cuda::make_device_vector(std::vector<size_t>(count, size));
    auto params           = init_params();
    params.operation      = TEST_UCP_DEVICE_KERNEL_PUT_MULTI;

    params.mem_list                     = list.handle();
    params.multi.addresses              = addresses.ptr();
    params.multi.remote_addresses       = remote_addresses.ptr();
    params.multi.lengths                = lengths.ptr();
    params.multi.counter_remote_address = list.dst_ptr(counter_index);
    params.multi.counter_inc_value      = 1;
    launch_kernel(params);

    // Check received data
    for (unsigned i = 0; i < count; ++i) {
        list.dst_pattern_check(i, mem_list::SEED_SRC);
    }

    wait_for_counter(list, counter_index);
}

UCS_TEST_P(test_ucp_device_xfer, put_multi_partial)
{
    static constexpr size_t size = 32 * UCS_KBYTE;
    unsigned total_count         = get_multi_elem_count() * 2;
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
    auto params           = init_params();
    params.operation      = TEST_UCP_DEVICE_KERNEL_PUT_MULTI_PARTIAL;

    params.mem_list                       = list.handle();
    params.partial.addresses              = addresses.ptr();
    params.partial.remote_addresses       = remote_addresses.ptr();
    params.partial.lengths                = lengths.ptr();
    params.partial.mem_list_indices       = indexes.ptr();
    params.partial.mem_list_count         = indexes_vec.size();
    params.partial.counter_index          = counter_index;
    params.partial.counter_remote_address = list.dst_ptr(counter_index);
    params.partial.counter_inc_value      = 1;
    launch_kernel(params);

    // Check received data
    std::set<unsigned> indexes_set(indexes_vec.begin(), indexes_vec.end());
    for (auto index : indexes_vec) {
        uint64_t seed = (indexes_set.find(index) == indexes_set.end()) ?
                                mem_list::SEED_DST :
                                mem_list::SEED_SRC;
        list.dst_pattern_check(index, seed);
    }

    wait_for_counter(list, counter_index);
}

UCS_TEST_P(test_ucp_device_xfer, counter)
{
    const size_t size = counter_size();
    mem_list list(sender(), receiver(), size, 1);

    static constexpr unsigned mem_list_index = 0;
    list.dst_counter_init(mem_list_index);

    auto params                       = init_params();
    params.operation                  = TEST_UCP_DEVICE_KERNEL_COUNTER_INC;
    params.mem_list                   = list.handle();
    params.counter_inc.mem_list_index = mem_list_index;
    params.counter_inc.inc_value      = 1;
    params.counter_inc.remote_address = list.dst_ptr(mem_list_index);
    launch_kernel(params);

    // Check destination
    wait_for_counter(list, mem_list_index);
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_device_xfer, rc_gda,
                                        "rc,rc_gda")
