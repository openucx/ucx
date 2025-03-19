/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <uct/test_md.h>

extern "C" {
#include <ucs/sys/ptr_arith.h>
}

#include <cuda.h>
#include <cuda_runtime.h>
#include <thread>

class test_switch_cuda_device : public test_md {
protected:
    template<class T> void detect_mem_type(ucs_memory_type_t mem_type) const;

    void init() override {
        test_md::init();
        ASSERT_EQ(cudaGetDeviceCount(&m_num_devices), cudaSuccess);
    }

    int m_num_devices;
};

template<class T>
void test_switch_cuda_device::detect_mem_type(ucs_memory_type_t mem_type) const
{
    if (m_num_devices < 2) {
        UCS_TEST_SKIP_R("less than two cuda devices available");
    }

    int current_device;
    ASSERT_EQ(cudaGetDevice(&current_device), cudaSuccess);
    ASSERT_EQ(cudaSetDevice((current_device + 1) % m_num_devices), cudaSuccess);

    const size_t size = 16;
    T buffer(size, mem_type);

    EXPECT_EQ(cudaSetDevice(current_device), cudaSuccess);

    ucs_memory_type_t detected_mem_type;
    ASSERT_EQ(uct_md_detect_memory_type(m_md, buffer.ptr(), size,
                                        &detected_mem_type),
              UCS_OK);
    EXPECT_EQ(detected_mem_type, mem_type);
}

#if HAVE_CUDA_FABRIC
class cuda_fabric_mem_buffer {
public:
    cuda_fabric_mem_buffer(size_t size, ucs_memory_type_t mem_type);
    virtual ~cuda_fabric_mem_buffer();
    void *ptr() const;

private:
    size_t m_size;
    CUmemGenericAllocationHandle m_alloc_handle;
    CUdeviceptr m_ptr;
};

cuda_fabric_mem_buffer::cuda_fabric_mem_buffer(size_t size,
                                               ucs_memory_type_t mem_type) :
    m_size(size)
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
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
                                      CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
        CUDA_SUCCESS) {
        goto err;
    }

    m_size = ucs_align_up(m_size, granularity);
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

cuda_fabric_mem_buffer::~cuda_fabric_mem_buffer()
{
    cuMemUnmap(m_ptr, m_size);
    cuMemAddressFree(m_ptr, m_size);
    cuMemRelease(m_alloc_handle);
}

void *cuda_fabric_mem_buffer::ptr() const
{
    return (void*)m_ptr;
}

UCS_TEST_P(test_switch_cuda_device, detect_mem_type_cuda_fabric)
{
    detect_mem_type<cuda_fabric_mem_buffer>(UCS_MEMORY_TYPE_CUDA);
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

_UCT_MD_INSTANTIATE_TEST_CASE(test_switch_cuda_device, cuda_cpy);

class test_mem_alloc_device : public test_switch_cuda_device {
protected:
    void init() override {
        test_switch_cuda_device::init();

        sys_dev.clear();
        for (auto device = 0; device < m_num_devices; ++device) {
            uct_md_mem_attr_t attr = {
                .field_mask = UCT_MD_MEM_ATTR_FIELD_SYS_DEV
            };
            size_t size            = 4096;
            void *ptr;

            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
            ASSERT_EQ(cudaSuccess, cudaMalloc(&ptr, size));
            EXPECT_UCS_OK(uct_md_mem_query(md(), ptr, size, &attr));
            ASSERT_EQ(cudaSuccess, cudaFree(ptr));

            UCS_TEST_MESSAGE << "CUDA device " << device << " sys_dev "
                             << (int)attr.sys_dev;
            sys_dev.push_back(attr.sys_dev);
        }

        ASSERT_EQ(m_num_devices, sys_dev.size());
    }

    ucs_status_t
    allocate(ucs_memory_type_t mem_type, int device = -1, size_t size = 1024)
    {
        uct_alloc_method_t method = UCT_ALLOC_METHOD_MD;
        uct_md_h md_p             = md();
        uct_mem_alloc_params_t params;

        params.field_mask = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS |
                            UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                            UCT_MEM_ALLOC_PARAM_FIELD_MDS;
        params.flags      = UCT_MD_MEM_ACCESS_ALL;
        params.mem_type   = mem_type;
        params.mds.mds    = &md_p;
        params.mds.count  = 1;
        if (device > -1) {
            params.field_mask |= UCT_MEM_ALLOC_PARAM_FIELD_SYS_DEVICE;
            params.sys_device  = sys_dev[device];
        }

        return uct_mem_alloc(size, &method, 1, &params, &mem);
    }

    ucs_sys_device_t query_sys_dev(uct_allocated_memory_t &mem)
    {
        uct_md_mem_attr_t attr = {
            .field_mask = UCT_MD_MEM_ATTR_FIELD_SYS_DEV
        };

        ucs_status_t status = uct_md_mem_query(md(), mem.address, mem.length,
                                               &attr);
        return (status == UCS_OK) ? attr.sys_dev : UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    void test_per_device_alloc(ucs_memory_type_t mem_type)
    {
        CUdevice current;

        for (auto device = 0; device < m_num_devices; ++device) {
            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        }

        for (auto device = 0; device < m_num_devices; ++device) {
            ASSERT_UCS_OK(allocate(mem_type, device));
            EXPECT_NE(CU_DEVICE_INVALID, sys_dev[device]);
            EXPECT_EQ(sys_dev[device], query_sys_dev(mem));
            EXPECT_EQ(cudaGetDevice(&current), cudaSuccess);
            EXPECT_EQ(m_num_devices - 1, current);
            ASSERT_UCS_OK(uct_mem_free(&mem));
        }
    }

    void test_same_device_alloc(ucs_memory_type_t mem_type, int set_sys_dev = 1)
    {
        for (auto device = 0; device < m_num_devices; ++device) {
            ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
            ASSERT_UCS_OK(allocate(mem_type, set_sys_dev ? device : -1));
            EXPECT_EQ(sys_dev[device], query_sys_dev(mem));
            ASSERT_UCS_OK(uct_mem_free(&mem));
        }
    }

    void skip_if_no_fabric(ucs_memory_type_t mem_type)
    {
#if HAVE_CUDA_FABRIC
        cuda_fabric_mem_buffer test_fabric_support(1024, mem_type);
#else
        UCS_TEST_SKIP_R("build without fabric support");
#endif
    }

    std::vector<ucs_sys_device_t> sys_dev;

public:
    uct_allocated_memory_t mem;

};

UCS_TEST_P(test_mem_alloc_device, different_device_cuda)
{
    test_per_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, different_device_cuda_fabric,
           "CUDA_COPY_ENABLE_FABRIC=y")
{
    skip_if_no_fabric(UCS_MEMORY_TYPE_CUDA);
    test_per_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda)
{
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda_fabric,
           "CUDA_COPY_ENABLE_FABRIC=y")
{
    skip_if_no_fabric(UCS_MEMORY_TYPE_CUDA);
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda_implicit)
{
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA, 0);
}

UCS_TEST_P(test_mem_alloc_device, same_device_cuda_fabric_implicit,
           "CUDA_COPY_ENABLE_FABRIC=y")
{
    skip_if_no_fabric(UCS_MEMORY_TYPE_CUDA);
    test_same_device_alloc(UCS_MEMORY_TYPE_CUDA, 0);
}

_UCT_MD_INSTANTIATE_TEST_CASE(test_mem_alloc_device, cuda_cpy);
