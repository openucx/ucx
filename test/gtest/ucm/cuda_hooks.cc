/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <ucm/api/ucm.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/ptr_arith.h>
#include <common/cuda_context.h>
#include <common/test.h>
#include <cuda.h>
#include <cuda_runtime.h>


class cuda_hooks : public ucs::test {
protected:
    virtual void init() {
        ucs_status_t result;
        ucs::test::init();

        /* Avoid memory allocation in event callbacks */
        m_alloc_events.reserve(1000);
        m_free_events.reserve(1000);

        result = ucm_set_event_handler(UCM_EVENT_VM_MAPPED, 0,
                                       cuda_vm_mapped_callback, this);
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_VM_UNMAPPED, 0,
                                       cuda_vm_unmapped_callback, this);
        ASSERT_UCS_OK(result);

        /* Install memory hooks */
        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0,
                                       cuda_mem_alloc_callback, this);
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0,
                                       cuda_mem_free_callback, this);
        ASSERT_UCS_OK(result);
    }

    virtual void cleanup()
    {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, cuda_mem_free_callback,
                                this);
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC,
                                cuda_mem_alloc_callback, this);
        ucm_unset_event_handler(UCM_EVENT_VM_UNMAPPED,
                                cuda_vm_unmapped_callback, this);
        ucm_unset_event_handler(UCM_EVENT_VM_MAPPED, cuda_vm_mapped_callback,
                                this);


        ucs::test::cleanup();
    }

    void check_mem_alloc_events(
            void *ptr, size_t size,
            ucs_memory_type_t expect_mem_type = UCS_MEMORY_TYPE_CUDA) const
    {
        check_event_present(m_alloc_events, "alloc", ptr, size,
                            expect_mem_type);
    }

    void check_mem_free_events(void *ptr, size_t size) const
    {
        check_event_present(m_free_events, "free", ptr, size);
    }

    CUdevice device() const
    {
        return cuda_ctx.cuda_device();
    }

private:
    struct mem_event {
        void              *address;
        size_t            size;
        ucs_memory_type_t mem_type;
    };

    using mem_event_vec_t = std::vector<mem_event>;

    void check_event_present(
            const mem_event_vec_t &events, const std::string &name, void *ptr,
            size_t size,
            ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_UNKNOWN) const
    {
        for (const auto e : events) {
            if (/* Start address match */
                (ptr >= e.address) &&
                /* End address match */
                (UCS_PTR_BYTE_OFFSET(ptr, size) <=
                 UCS_PTR_BYTE_OFFSET(e.address, e.size)) &&
                /* Memory type match */
                ((e.mem_type == mem_type) ||
                 (e.mem_type == UCS_MEMORY_TYPE_UNKNOWN))) {
                return;
            }
        }

        FAIL() << "Could not find memory " << name << " event for " << ptr
               << ".." << UCS_PTR_BYTE_OFFSET(ptr, size) << " type "
               << ucs_memory_type_names[mem_type];
    }

    static void push_event(mem_event_vec_t &events, const mem_event &e)
    {
        ucs_assertv(events.size() < events.capacity(), "size=%zu capacity=%zu",
                    events.size(), events.capacity());
        events.push_back(e);
    }

    void mem_alloc_event(void *address, size_t size, ucs_memory_type_t mem_type)
    {
        push_event(m_alloc_events, {address, size, mem_type});
    }

    void mem_free_event(void *address, size_t size)
    {
        push_event(m_free_events, {address, size, UCS_MEMORY_TYPE_UNKNOWN});
    }

    static void cuda_vm_mapped_callback(ucm_event_type_t event_type,
                                        ucm_event_t *event, void *arg)
    {
        auto self = reinterpret_cast<cuda_hooks*>(arg);
        self->mem_alloc_event(event->vm_mapped.address, event->vm_mapped.size,
                              UCS_MEMORY_TYPE_HOST);
    }

    static void cuda_vm_unmapped_callback(ucm_event_type_t event_type,
                                          ucm_event_t *event, void *arg)
    {
        auto self = reinterpret_cast<cuda_hooks*>(arg);
        self->mem_free_event(event->vm_unmapped.address,
                             event->vm_unmapped.size);
    }

    static void cuda_mem_alloc_callback(ucm_event_type_t event_type,
                                        ucm_event_t *event, void *arg)
    {
        auto self = reinterpret_cast<cuda_hooks*>(arg);
        self->mem_alloc_event(event->mem_type.address, event->mem_type.size,
                              event->mem_type.mem_type);
    }

    static void cuda_mem_free_callback(ucm_event_type_t event_type,
                                       ucm_event_t *event, void *arg)
    {
        auto self = reinterpret_cast<cuda_hooks*>(arg);
        self->mem_free_event(event->mem_type.address, event->mem_type.size);
    }

    cuda_context    cuda_ctx;
    mem_event_vec_t m_alloc_events;
    mem_event_vec_t m_free_events;
};

UCS_TEST_F(cuda_hooks, test_cuMem_Alloc_Free) {
    CUresult ret;
    CUdeviceptr dptr, dptr1;

    /* small allocation */
    ret = cuMemAlloc(&dptr, 64);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, 64);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, 64);

    /* large allocation */
    ret = cuMemAlloc(&dptr, 256 * UCS_MBYTE);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 256 * UCS_MBYTE);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, 256 * UCS_MBYTE);

    /* multiple allocations, cudafree in reverse order */
    ret = cuMemAlloc(&dptr, UCS_MBYTE);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, UCS_MBYTE);

    ret = cuMemAlloc(&dptr1, UCS_MBYTE);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr1, UCS_MBYTE);

    ret = cuMemFree(dptr1);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr1, UCS_MBYTE);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, UCS_MBYTE);
}

UCS_TEST_F(cuda_hooks, test_cuMemAllocHost) {
    CUresult ret;
    void *ptr;

    ret = cuMemAllocHost(&ptr, 64);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_HOST);

    ret = cuMemFreeHost(ptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events(ptr, 64);
}

UCS_TEST_F(cuda_hooks, test_cuMemAllocManaged) {
    CUresult ret;
    CUdeviceptr dptr;

    ret = cuMemAllocManaged(&dptr, 64, CU_MEM_ATTACH_GLOBAL);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 64, UCS_MEMORY_TYPE_CUDA_MANAGED);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, 64);
}

UCS_TEST_F(cuda_hooks, test_cuMemAllocPitch) {
    const size_t width          = 4;
    const size_t height         = 8;
    const unsigned element_size = 4;
    CUresult ret;
    CUdeviceptr dptr;
    size_t pitch;

    ret = cuMemAllocPitch(&dptr, &pitch, width, height, element_size);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, width * height);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, width * height);
}

UCS_TEST_F(cuda_hooks, test_cuMemMapUnmap) {
    CUmemAllocationProp prop = {};
    CUmemGenericAllocationHandle handle;
    size_t size, granularity;
    CUdeviceptr ptr;
    CUresult ret;

    ret = cuMemGetAllocationGranularity(&granularity, &prop,
                                        CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    size = ucs_align_up(256 * UCS_KBYTE, granularity);

    prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id   = device();
    ret                = cuMemCreate(&handle, size, &prop, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);

    ret = cuMemAddressReserve(&ptr, size, 0, 0, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);

    ret = cuMemMap(ptr, size, 0, handle, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)ptr, size, UCS_MEMORY_TYPE_CUDA);

    ret = cuMemUnmap(ptr, size);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)ptr, size);

    ret = cuMemAddressFree(ptr, size);
    ASSERT_EQ(ret, CUDA_SUCCESS);

    // Free the memory resources
    ret = cuMemRelease(handle);
    ASSERT_EQ(ret, CUDA_SUCCESS);
}

#if CUDA_VERSION >= 11020
UCS_TEST_F(cuda_hooks, test_cuMemAllocAsync) {
    CUresult ret;
    CUdeviceptr dptr;

    /* release with cuMemFree */
    ret = cuMemAllocAsync(&dptr, 64, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 64, UCS_MEMORY_TYPE_CUDA);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, 64);

    /* release with cuMemFreeAsync */
    ret = cuMemAllocAsync(&dptr, 64, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 64);

    ret = cuMemFreeAsync(dptr, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_free_events((void*)dptr, 64);
}
#endif


UCS_TEST_F(cuda_hooks, test_cuda_Malloc_Free) {
    cudaError_t ret;
    void *ptr, *ptr1;

    /* small allocation */
    ret = cudaMalloc(&ptr, 64);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, 64);

    /* large allocation */
    ret = cudaMalloc(&ptr, 256 * UCS_MBYTE);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 256 * UCS_MBYTE);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, 256 * UCS_MBYTE);

    /* multiple allocations, cudafree in reverse order */
    ret = cudaMalloc(&ptr, UCS_MBYTE);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, UCS_MBYTE);

    ret = cudaMalloc(&ptr1, UCS_MBYTE);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr1, UCS_MBYTE);

    ret = cudaFree(ptr1);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr1, UCS_MBYTE);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, UCS_MBYTE);

    /* cudaFree with NULL */
    ret = cudaFree(NULL);
    ASSERT_EQ(ret, cudaSuccess);
}

UCS_TEST_F(cuda_hooks, test_cudaMallocManaged) {
    cudaError_t ret;
    void *ptr;

    ret = cudaMallocManaged(&ptr, 64, cudaMemAttachGlobal);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_CUDA_MANAGED);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, 64);
}

UCS_TEST_F(cuda_hooks, test_cudaMallocPitch) {
    cudaError_t ret;
    void *devPtr;
    size_t pitch;

    ret = cudaMallocPitch(&devPtr, &pitch, 4, 8);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(devPtr, (4 * 8));

    ret = cudaFree(devPtr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(devPtr, (4 * 8));
}

#if CUDA_VERSION >= 11020
UCS_TEST_F(cuda_hooks, test_cudaMallocAsync) {
    cudaError_t ret;
    void *ptr;

    /* release with cudaFree */
    ret = cudaMallocAsync(&ptr, 64, 0);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_CUDA);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, 64);

    /* release with cudaFreeAsync */
    ret = cudaMallocAsync(&ptr, 64, 0);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64);

    ret = cudaFreeAsync(ptr, 0);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, 64);
}
#endif
