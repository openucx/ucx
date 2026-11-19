/**
 * See file LICENSE for terms.
 */

#include "rocm_vmm_mem_buffer.h"

#include <uct/uct_test.h>


class test_rocm_copy_vmm_ep : public uct_test {
protected:
    void init() override
    {
        uct_test::init();

        m_entity = uct_test::create_entity(0);
        m_entities.push_back(m_entity);
        m_entity->connect(0, *m_entity, 0);
    }

    /* Data at the device address, read with HIP */
    std::vector<uint8_t> device_data(const void *address, size_t length)
    {
        std::vector<uint8_t> data(length);

        EXPECT_EQ(hipSuccess, hipMemcpy(data.data(), address, length,
                                        hipMemcpyDeviceToHost));
        return data;
    }

    entity *m_entity;
};

/* Copies below the d2h/h2d thresholds, which the CPU does for hipMalloc
 * memory, must not touch memory mapped with hipMemMap from the CPU */
UCS_TEST_P(test_rocm_copy_vmm_ep, small_copies) {
    const size_t length = 64;
    rocm_vmm_mem_buffer buffer(length);
    const uint64_t remote_addr = (uintptr_t)buffer.ptr();
    std::vector<uint8_t> data(length), result(length);
    uct_iov_t iov;

    ucs::fill_random(data);
    ASSERT_UCS_OK(uct_ep_put_short(m_entity->ep(0), data.data(), length,
                                   remote_addr, UCT_INVALID_RKEY));
    EXPECT_EQ(data, device_data(buffer.ptr(), length));

    ASSERT_UCS_OK(uct_ep_get_short(m_entity->ep(0), result.data(), length,
                                   remote_addr, UCT_INVALID_RKEY));
    EXPECT_EQ(data, result);

    ucs::fill_random(data);
    iov.buffer = data.data();
    iov.length = length;
    iov.memh   = UCT_MEM_HANDLE_NULL;
    iov.stride = 0;
    iov.count  = 1;
    ASSERT_UCS_OK(uct_ep_put_zcopy(m_entity->ep(0), &iov, 1, remote_addr,
                                   UCT_INVALID_RKEY, NULL));
    EXPECT_EQ(data, device_data(buffer.ptr(), length));

    iov.buffer = result.data();
    ASSERT_UCS_OK(uct_ep_get_zcopy(m_entity->ep(0), &iov, 1, remote_addr,
                                   UCT_INVALID_RKEY, NULL));
    EXPECT_EQ(data, result);
}

_UCT_INSTANTIATE_TEST_CASE(test_rocm_copy_vmm_ep, rocm_copy)
