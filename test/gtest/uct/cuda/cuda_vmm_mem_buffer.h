/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_TEST_CUDA_VMM_MEM_BUFFER_H
#define UCT_TEST_CUDA_VMM_MEM_BUFFER_H

#include <common/test_helpers.h>
#include <cuda.h>

extern "C" {
#include <ucs/sys/ptr_arith.h>
}

class cuda_vmm_mem_buffer {
public:
    cuda_vmm_mem_buffer() = default;

    cuda_vmm_mem_buffer(size_t size, ucs_memory_type_t mem_type)
    {
        init(size, 0);
    }

    virtual ~cuda_vmm_mem_buffer()
    {
        cuMemUnmap(m_ptr, m_size);
        cuMemAddressFree(m_ptr, m_size);
        cuMemRelease(m_alloc_handle);
    }

    void *ptr() const
    {
        return (void*)m_ptr;
    }

    size_t size() const
    {
        return m_size;
    }

protected:
    void init(size_t size, unsigned handle_type)
    {
        size_t granularity          = 0;
        CUmemAllocationProp prop    = {};
        CUmemAccessDesc access_desc = {};
        CUdevice device;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
            UCS_TEST_ABORT("failed to get the device handle for the current "
                           "context");
        }

        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = device;
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
        UCS_TEST_SKIP_R("failed to allocate CUDA VMM memory");
    }

private:
    size_t m_size                               = 0;
    CUmemGenericAllocationHandle m_alloc_handle = 0;
    CUdeviceptr m_ptr                           = 0;
};

#if HAVE_CUDA_FABRIC
class cuda_fabric_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_fabric_mem_buffer(size_t size, ucs_memory_type_t mem_type)
    {
        init(size, CU_MEM_HANDLE_TYPE_FABRIC);
    }
};

class cuda_posix_fd_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_posix_fd_mem_buffer(size_t size, ucs_memory_type_t mem_type)
    {
        init(size, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }
};
#endif

#endif
