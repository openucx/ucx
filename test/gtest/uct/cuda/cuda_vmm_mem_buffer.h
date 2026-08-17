/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_TEST_CUDA_VMM_MEM_BUFFER_H
#define UCT_TEST_CUDA_VMM_MEM_BUFFER_H

#include <common/test_helpers.h>
#include <cuda.h>
#include <vector>

extern "C" {
#include <ucs/sys/ptr_arith.h>
}

static inline CUresult uct_test_cuda_ctx_create_compat(CUcontext *ctx,
                                                       unsigned int flags,
                                                       CUdevice dev)
{
#if CUDA_VERSION >= 13000
    CUctxCreateParams ctx_create_params = {};
    return cuCtxCreate(ctx, &ctx_create_params, flags, dev);
#else
    return cuCtxCreate(ctx, flags, dev);
#endif
}

class cuda_vmm_mem_buffer {
public:
    cuda_vmm_mem_buffer() = default;

    /* Reserve one virtual range and back it by @a num_chunks separate physical
     * allocations, each of the granularity-aligned chunk size */
    cuda_vmm_mem_buffer(size_t size, ucs_memory_type_t mem_type,
                        size_t num_chunks = 1)
    {
        init(size, 0, CU_MEM_LOCATION_TYPE_DEVICE, num_chunks);
    }

    virtual ~cuda_vmm_mem_buffer()
    {
        cleanup();
    }

    void *ptr() const
    {
        return (void*)m_ptr;
    }

    /* total mapped length, which is the chunk size times the chunk count */
    size_t size() const
    {
        return m_size;
    }

protected:
    void init(size_t size, unsigned handle_type,
              CUmemLocationType location_type = CU_MEM_LOCATION_TYPE_DEVICE,
              size_t num_chunks               = 1)
    {
        size_t granularity             = 0;
        CUmemAllocationProp prop       = {};
        CUmemAccessDesc access_desc[2] = {};
        unsigned num_access            = 1;
        bool host_located = (location_type != CU_MEM_LOCATION_TYPE_DEVICE);
        CUdevice device;
        if (cuCtxGetDevice(&device) != CUDA_SUCCESS) {
            UCS_TEST_ABORT("failed to get the device handle for the current "
                           "context");
        }

        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = location_type;
        prop.location.id   = host_located ? 0 : device;
        if (handle_type != 0) {
            prop.requestedHandleTypes = (CUmemAllocationHandleType)handle_type;
        }
        if (cuMemGetAllocationGranularity(&granularity, &prop,
                                          CU_MEM_ALLOC_GRANULARITY_MINIMUM) !=
            CUDA_SUCCESS) {
            goto err;
        }

        m_chunk_size = ucs_align_up(size, granularity);
        m_size       = m_chunk_size * num_chunks;

        if (cuMemAddressReserve(&m_ptr, m_size, 0, 0, 0) != CUDA_SUCCESS) {
            m_ptr = 0;
            goto err;
        }

        for (size_t i = 0; i < num_chunks; ++i) {
            CUmemGenericAllocationHandle alloc_handle;

            if (cuMemCreate(&alloc_handle, m_chunk_size, &prop, 0) !=
                CUDA_SUCCESS) {
                goto err;
            }

            m_alloc_handles.push_back(alloc_handle);

            if (cuMemMap(m_ptr + (i * m_chunk_size), m_chunk_size, 0,
                         alloc_handle, 0) != CUDA_SUCCESS) {
                goto err;
            }

            ++m_num_mapped;
        }

        access_desc[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_desc[0].location.id   = device;
        access_desc[0].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (host_located) {
            access_desc[1].location.type = location_type;
            access_desc[1].location.id   = 0;
            access_desc[1].flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            num_access                   = 2;
        }
        if (cuMemSetAccess(m_ptr, m_size, access_desc, num_access) !=
            CUDA_SUCCESS) {
            goto err;
        }

        return;

    err:
        cleanup();
        UCS_TEST_SKIP_R("failed to allocate CUDA VMM memory");
    }

private:
    /* Unmap only the chunks which were actually mapped, so this is also the
     * unwind path of a partially completed init() */
    void cleanup()
    {
        for (size_t i = 0; i < m_num_mapped; ++i) {
            cuMemUnmap(m_ptr + (i * m_chunk_size), m_chunk_size);
        }

        for (auto alloc_handle : m_alloc_handles) {
            cuMemRelease(alloc_handle);
        }

        if (m_ptr != 0) {
            cuMemAddressFree(m_ptr, m_size);
        }

        m_alloc_handles.clear();
        m_num_mapped = 0;
        m_ptr        = 0;
    }

    size_t m_chunk_size = 0;
    size_t m_size       = 0;
    size_t m_num_mapped = 0;
    std::vector<CUmemGenericAllocationHandle> m_alloc_handles;
    CUdeviceptr m_ptr = 0;
};

#if HAVE_CUDA_FABRIC
class cuda_fabric_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_fabric_mem_buffer(size_t size, ucs_memory_type_t mem_type)
    {
        init(size, CU_MEM_HANDLE_TYPE_FABRIC);
    }
};
#endif

class cuda_posix_fd_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_posix_fd_mem_buffer(size_t size, ucs_memory_type_t mem_type)
    {
        init(size, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR);
    }
};

#if CUDA_VERSION >= 12020
class cuda_host_vmm_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_host_vmm_mem_buffer(size_t size, ucs_memory_type_t mem_type)
    {
        init(size, 0, CU_MEM_LOCATION_TYPE_HOST_NUMA);
    }
};
#endif

#endif
