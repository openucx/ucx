/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/test_md.h>
#include <uct/test_p2p_rma.h>

extern "C" {
#include <ucs/sys/ptr_arith.h>
}

#include <cuda.h>
#include <cuda_runtime.h>

#include <thread>

class test_switch_cuda_device : public test_md {
public:
    template<class T> void detect_mem_type(ucs_memory_type_t mem_type) const;
};

template<class T>
void test_switch_cuda_device::detect_mem_type(ucs_memory_type_t mem_type) const
{
    int num_devices;
    ASSERT_EQ(cudaGetDeviceCount(&num_devices), cudaSuccess);

    if (num_devices < 2) {
        UCS_TEST_SKIP_R("less than two cuda devices available");
    }

    int current_device;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((current_device + 1) % num_devices), cudaSuccess);

    const size_t size = 16;
    T buffer(size, mem_type);

    EXPECT_EQ(cudaSetDevice(current_device), cudaSuccess);

    ucs_memory_type_t detected_mem_type;
    ASSERT_EQ(uct_md_detect_memory_type(m_md, buffer.ptr(), size,
                                        &detected_mem_type),
              UCS_OK);
    EXPECT_EQ(detected_mem_type, mem_type);
}

class cuda_vmm_mem_buffer {
public:
    cuda_vmm_mem_buffer() = default;
    cuda_vmm_mem_buffer(size_t size, ucs_memory_type_t mem_type);
    virtual ~cuda_vmm_mem_buffer();
    void *ptr() const;

protected:
    void init(size_t size, unsigned handle_type);

private:
    size_t m_size                               = 0;
    CUmemGenericAllocationHandle m_alloc_handle = 0;
    CUdeviceptr m_ptr                           = 0;
};

cuda_vmm_mem_buffer::cuda_vmm_mem_buffer(size_t size,
                                         ucs_memory_type_t mem_type)
{
    init(size, 0);
}

cuda_vmm_mem_buffer::~cuda_vmm_mem_buffer()
{
    cuMemUnmap(m_ptr, m_size);
    cuMemAddressFree(m_ptr, m_size);
    cuMemRelease(m_alloc_handle);
}

void *cuda_vmm_mem_buffer::ptr() const
{
    return (void*)m_ptr;
}

void cuda_vmm_mem_buffer::init(size_t size, unsigned handle_type)
{
    size_t granularity          = 0;
    CUmemAllocationProp prop    = {};
    CUmemAccessDesc access_desc = {};
    CUdevice device;
    if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
        UCS_TEST_ABORT("failed to get the device handle for the current "
                       "context");
    }

    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = device;
    if (handle_type != 0) {
        prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_type;
    }
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
        CUDA_SUCCESS) {
        goto err;
    }

    m_size = ucs_align_up(size, granularity);
    if (cuMemCreate(&m_alloc_handle, m_size, &prop, 0) != CUDA_SUCCESS) {
        goto err;
    }

    if (cuMemAddressReserve(&m_ptr, m_size, 0, 0, 0) != CUDA_SUCCESS) {
        goto err_mem_release;
    }

    if (cuMemMap(m_ptr, m_size, 0, m_alloc_handle, 0) != CUDA_SUCCESS) {
        goto err_address_free;
    }

    access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access_desc.location.id   = device;
    access_desc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (cuMemSetAccess(m_ptr, m_size, &access_desc, 1) != CUDA_SUCCESS) {
        goto err_mem_unmap;
    }

    return;

err_mem_unmap:
    cuMemUnmap(m_ptr, m_size);
err_address_free:
    cuMemAddressFree(m_ptr, m_size);
err_mem_release:
    cuMemRelease(m_alloc_handle);
err:
    UCS_TEST_SKIP_R("failed to allocate CUDA fabric memory");
}

#if HAVE_CUDA_FABRIC
class cuda_fabric_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_fabric_mem_buffer(size_t size, ucs_memory_type_t mem_type);
};

cuda_fabric_mem_buffer::cuda_fabric_mem_buffer(size_t size,
                                               ucs_memory_type_t mem_type)
{
    init(size, CU_MEM_HANDLE_TYPE_FABRIC);
}
#endif

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_managed)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA_MANAGED);
}

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_vmm)
{
    detect_mem_type<cuda_vmm_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

#if HAVE_CUDA_FABRIC
UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_fabric)
{
    detect_mem_type<cuda_fabric_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}
#endif

_UCT_MD_INSTANTIATE_TEST_CASE(test_switch_cuda_device, cuda_cpy);

class test_p2p_create_destroy_ctx : public uct_p2p_rma_test {
public:
    void test_xfer(send_func_t send, size_t length, unsigned flags,
                   ucs_memory_type_t mem_type) override;
};

void test_p2p_create_destroy_ctx::test_xfer(send_func_t send, size_t length,
                                            unsigned flags,
                                            ucs_memory_type_t mem_type)
{
    int num_devices;
    ASSERT_EQ(cuDeviceGetCount(&num_devices), CUDA_SUCCESS);

    if (num_devices < 1) {
        UCS_TEST_SKIP_R("no cuda devices available");
    }

    CUdevice device;
    ASSERT_EQ(cuDeviceGet(&device, 0), CUDA_SUCCESS);

    CUcontext ctx;
    ASSERT_EQ(cuCtxCreate(&ctx, 0, device), CUDA_SUCCESS);
    uct_p2p_rma_test::test_xfer(send, length, flags, mem_type);
    EXPECT_EQ(cuCtxDestroy(ctx), CUDA_SUCCESS);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, put_short)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_short), 1,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, get_short)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_short), 1,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, put_zcopy)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::put_zcopy),
              sender().iface_attr().cap.put.min_zcopy + 1,
              TEST_UCT_FLAG_SEND_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_p2p_create_destroy_ctx, get_zcopy)
{
    test_xfer(static_cast<send_func_t>(&uct_p2p_rma_test::get_zcopy),
              sender().iface_attr().cap.get.min_zcopy + 1,
              TEST_UCT_FLAG_RECV_ZCOPY, UCS_MEMORY_TYPE_CUDA);
}

_UCT_INSTANTIATE_TEST_CASE(test_p2p_create_destroy_ctx, cuda_copy)

class test_another_thread : public test_md {
public:
    template<class T> void detect_mem_type(ucs_memory_type_t mem_type) const;
};

template<class T>
void test_another_thread::detect_mem_type(ucs_memory_type_t mem_type) const
{
    const size_t size = 16;
    T buffer(size, mem_type);

    ucs_memory_type_t detected_mem_type;
    ucs_status_t status;
    std::thread([&]() {
        status = uct_md_detect_memory_type(this->m_md, buffer.ptr(), size,
                                           &detected_mem_type);
    }).join();

    ASSERT_EQ(status, UCS_OK);
    EXPECT_EQ(detected_mem_type, mem_type);
}

UCS_TEST_P(test_another_thread, detect_mem_type_cuda)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_another_thread, detect_mem_type_cuda_managed)
{
    detect_mem_type<mem_buffer>(UCS_MEMORY_TYPE_CUDA_MANAGED);
}

UCS_TEST_P(test_another_thread, detect_mem_type_cuda_vmm)
{
    detect_mem_type<cuda_vmm_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}

#if HAVE_CUDA_FABRIC
UCS_TEST_P(test_another_thread, detect_mem_type_cuda_fabric)
{
    detect_mem_type<cuda_fabric_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
}
#endif

_UCT_MD_INSTANTIATE_TEST_CASE(test_another_thread, cuda_cpy);
