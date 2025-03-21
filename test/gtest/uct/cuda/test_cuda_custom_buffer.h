/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/memory/memory_type.h>

#include <cuda.h>


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


#if HAVE_CUDA_FABRIC
class cuda_fabric_mem_buffer : public cuda_vmm_mem_buffer {
public:
    cuda_fabric_mem_buffer(size_t size, ucs_memory_type_t mem_type);
};

class cuda_mem_pool {
public:
    cuda_mem_pool(size_t size, ucs_memory_type_t mem_type);
    virtual ~cuda_mem_pool();
    void *ptr() const;
    static void alloc_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool,
                              CUstream *cu_stream, size_t size);
    static void
    free_mempool(CUdeviceptr *ptr, CUmemoryPool *mpool, CUstream *cu_stream);

private:
    CUdeviceptr m_ptr;
    CUmemoryPool m_mpool;
    CUstream m_stream;
};
#endif
