/**
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rocm_vmm_mem_buffer.h"

#include <uct/uct_test.h>

extern "C" {
#include <uct/rocm/ipc/rocm_ipc_cache.h>
}


class test_rocm_ipc_vmm : public uct_test {
protected:
    void init() override
    {
        uct_test::init();

        m_entity = uct_test::create_entity(0);
        m_entities.push_back(m_entity);
        m_entity->connect(0, *m_entity, 0);
    }

    void zcopy(bool is_put, void *buffer, size_t length, void *remote_address,
               uct_rkey_t rkey)
    {
        uct_iov_t iov;
        ucs_status_t status;

        iov.buffer = buffer;
        iov.length = length;
        iov.memh   = UCT_MEM_HANDLE_NULL;
        iov.stride = 0;
        iov.count  = 1;
        if (is_put) {
            status = uct_ep_put_zcopy(m_entity->ep(0), &iov, 1,
                                      (uintptr_t)remote_address, rkey, NULL);
        } else {
            status = uct_ep_get_zcopy(m_entity->ep(0), &iov, 1,
                                      (uintptr_t)remote_address, rkey, NULL);
        }

        ASSERT_UCS_OK_OR_INPROGRESS(status);
        flush();
    }

    entity *m_entity;
};

UCS_TEST_P(test_rocm_ipc_vmm, share) {
    /* A buffer within an exportable allocation which is mapped after another
     * one, so that the shared allocation does not start at the range start */
    const size_t length = 4096;
    rocm_vmm_mem_buffer buffer(2 * length, 2,
                               hipMemHandleTypePosixFileDescriptor);
    void *address = UCS_PTR_BYTE_OFFSET(buffer.chunk(1), length);
    std::vector<uint8_t> rkey_buffer(m_entity->md_attr().rkey_packed_size);
    uct_md_mkey_pack_params_t pack_params  = {};
    uct_rkey_unpack_params_t unpack_params = {};
    uct_md_mem_reg_params_t reg_params     = {};
    uct_component_h component              = m_entity->md()->component;
    uct_rkey_bundle_t rkey_bundle;
    ucs_status_t status;
    uct_mem_h memh;

    ASSERT_UCS_OK(uct_md_mem_reg_v2(m_entity->md(), address, length,
                                    &reg_params, &memh));
    ASSERT_UCS_OK(uct_md_mkey_pack_v2(m_entity->md(), memh, address, length,
                                      &pack_params, rkey_buffer.data()));

    status = uct_rkey_unpack_v2(component, rkey_buffer.data(), &unpack_params,
                                &rkey_bundle);
#if UCT_ROCM_IPC_HAVE_VMEM_IPC
    if (uct_rocm_ipc_vmem_ipc_supported()) {
        std::vector<uint8_t> data(length), result(length);
        void *local;

        ASSERT_UCS_OK(status);
        ASSERT_EQ(hipSuccess, hipMalloc(&local, length));

        /* The memory is mapped again, at another address, to access it */
        ucs::fill_random(data);
        ASSERT_EQ(hipSuccess, hipMemcpy(local, data.data(), length,
                                        hipMemcpyHostToDevice));
        zcopy(true, local, length, address, rkey_bundle.rkey);
        ASSERT_EQ(hipSuccess, hipMemcpy(result.data(), address, length,
                                        hipMemcpyDeviceToHost));
        EXPECT_EQ(data, result);

        ucs::fill_random(data);
        ASSERT_EQ(hipSuccess, hipMemcpy(address, data.data(), length,
                                        hipMemcpyHostToDevice));
        zcopy(false, local, length, address, rkey_bundle.rkey);
        ASSERT_EQ(hipSuccess, hipMemcpy(result.data(), local, length,
                                        hipMemcpyDeviceToHost));
        EXPECT_EQ(data, result);

        EXPECT_EQ(hipSuccess, hipFree(local));
        uct_rkey_release(component, &rkey_bundle);
    } else
#endif
    {
        /* Registered without an IPC handle, so peers use another transport */
        EXPECT_EQ(UCS_ERR_UNREACHABLE, status);
    }

    EXPECT_UCS_OK(uct_md_mem_dereg(m_entity->md(), memh));
}

_UCT_INSTANTIATE_TEST_CASE(test_rocm_ipc_vmm, rocm_ipc)
