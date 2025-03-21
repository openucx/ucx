/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "test_cuda_custom_buffer.h"
#include <common/test_helpers.h>

extern "C" {
#include <ucs/sys/ptr_arith.h>
}

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
cuda_fabric_mem_buffer::cuda_fabric_mem_buffer(size_t size,
                                               ucs_memory_type_t mem_type)
{
    init(size, CU_MEM_HANDLE_TYPE_FABRIC);
}

cuda_mem_pool::cuda_mem_pool(size_t size, ucs_memory_type_t mem_type)
{
    alloc_mempool(&m_ptr, &m_mpool, &m_stream, size);
}

cuda_mem_pool::~cuda_mem_pool()
{
    free_mempool(&m_ptr, &m_mpool, &m_stream);
}

void *cuda_mem_pool::ptr() const
{
    return (void*)m_ptr;
}

void cuda_mem_pool::alloc_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool,
                                  CUstream *cu_stream, size_t size)
{
    CUmemPoolProps pool_props = {};
    CUmemAccessDesc map_desc;
    CUdevice cu_device;

    EXPECT_EQ(CUDA_SUCCESS, cuCtxGetDevice(&cu_device));

    pool_props.allocType     = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_props.location.id   = (int)cu_device;
    pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pool_props.handleTypes   = CU_MEM_HANDLE_TYPE_FABRIC;
    pool_props.maxSize       = size;
    map_desc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    map_desc.location        = pool_props.location;

    EXPECT_EQ(CUDA_SUCCESS,
              cuStreamCreate(cu_stream, CU_STREAM_NON_BLOCKING));
    EXPECT_EQ(CUDA_SUCCESS, cuMemPoolCreate(mpool, &pool_props));
    EXPECT_EQ(CUDA_SUCCESS, cuMemPoolSetAccess(*mpool, &map_desc, 1));
    EXPECT_EQ(CUDA_SUCCESS,
              cuMemAllocFromPoolAsync(ptr, size, *mpool, *cu_stream));
    EXPECT_EQ(CUDA_SUCCESS, cuStreamSynchronize(*cu_stream));
}

void cuda_mem_pool::free_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool,
                                 CUstream *cu_stream)
{
    EXPECT_EQ(CUDA_SUCCESS, cuMemFree(*ptr));
    EXPECT_EQ(CUDA_SUCCESS, cuMemPoolDestroy(*mpool));
    EXPECT_EQ(CUDA_SUCCESS, cuStreamDestroy(*cu_stream));
}
#endif
