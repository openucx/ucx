/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <ucm/api/ucm.h>
#include <common/test.h>
#include <cuda.h>
#include <cuda_runtime.h>

static struct {
    void              *address;
    size_t            size;
    ucs_memory_type_t mem_type;
} alloc_event, free_event;

static void cuda_vm_mapped_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    alloc_event.address  = event->vm_mapped.address;
    alloc_event.size     = event->vm_mapped.size;
    alloc_event.mem_type = UCS_MEMORY_TYPE_HOST;
}

static void cuda_vm_unmapped_callback(ucm_event_type_t event_type,
                                      ucm_event_t *event, void *arg)
{
    free_event.address  = event->vm_unmapped.address;
    free_event.size     = event->vm_unmapped.size;
    free_event.mem_type = UCS_MEMORY_TYPE_UNKNOWN;
}

static void cuda_mem_alloc_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    alloc_event.address  = event->mem_type.address;
    alloc_event.size     = event->mem_type.size;
    alloc_event.mem_type = event->mem_type.mem_type;
}

static void cuda_mem_free_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event, void *arg)
{
    free_event.address  = event->mem_type.address;
    free_event.size     = event->mem_type.size;
    free_event.mem_type = UCS_MEMORY_TYPE_UNKNOWN;
}

class cuda_hooks : public ucs::test {
protected:

    virtual void init() {
        ucs_status_t result;
        CUresult ret;
        ucs::test::init();

        /* intialize device context */
        if (cudaSetDevice(0) != cudaSuccess) {
            UCS_TEST_SKIP_R("can't set cuda device");
        }

        ret = cuInit(0);
        if (ret != CUDA_SUCCESS) {
            UCS_TEST_SKIP_R("can't init cuda device");
        }

        ret = cuDeviceGet(&device, 0);
        if (ret != CUDA_SUCCESS) {
            UCS_TEST_SKIP_R("can't get cuda device");
        }

        ret = cuCtxCreate(&context, 0, device);
        if (ret != CUDA_SUCCESS) {
            UCS_TEST_SKIP_R("can't create cuda context");
        }

        memset(&alloc_event, 0, sizeof(alloc_event));
        memset(&free_event, 0, sizeof(free_event));

        result = ucm_set_event_handler(UCM_EVENT_VM_MAPPED, 0,
                                       cuda_vm_mapped_callback, NULL);
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_VM_UNMAPPED, 0,
                                       cuda_vm_unmapped_callback, NULL);
        ASSERT_UCS_OK(result);

        /* install memory hooks */
        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0,
                                       cuda_mem_alloc_callback, NULL);
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0,
                                       cuda_mem_free_callback, NULL);
        ASSERT_UCS_OK(result);
    }

    virtual void cleanup()
    {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, cuda_mem_free_callback,
                                NULL);
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC,
                                cuda_mem_alloc_callback, NULL);
        ucm_unset_event_handler(UCM_EVENT_VM_UNMAPPED,
                                cuda_vm_unmapped_callback, NULL);
        ucm_unset_event_handler(UCM_EVENT_VM_MAPPED, cuda_vm_mapped_callback,
                                NULL);

        CUresult ret = cuCtxDestroy(context);
        EXPECT_EQ(ret, CUDA_SUCCESS);

        ucs::test::cleanup();
    }

    void check_mem_alloc_events(
            void *ptr, size_t size,
            ucs_memory_type_t expect_mem_type = UCS_MEMORY_TYPE_CUDA)
    {
        ASSERT_EQ(ptr, alloc_event.address);
        ASSERT_GE(alloc_event.size, size);
        EXPECT_TRUE((alloc_event.mem_type == expect_mem_type) ||
                    (alloc_event.mem_type == UCS_MEMORY_TYPE_UNKNOWN));
    }

    CUdevice   device;
    CUcontext  context;
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
    EXPECT_EQ((void*)dptr, free_event.address);

    /* large allocation */
    ret = cuMemAlloc(&dptr, (256 * 1024 *1024));
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, (256 * 1024 *1024));

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr, free_event.address);

    /* multiple allocations, cudafree in reverse order */
    ret = cuMemAlloc(&dptr, (1 * 1024 *1024));
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, (1 * 1024 *1024));

    ret = cuMemAlloc(&dptr1, (1 * 1024 *1024));
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void *)dptr1, (1 * 1024 *1024));

    ret = cuMemFree(dptr1);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr1, free_event.address);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr, free_event.address);
}

UCS_TEST_F(cuda_hooks, test_cuMemAllocHost) {
    CUresult ret;
    void *ptr;

    ret = cuMemAllocHost(&ptr, 64);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_HOST);

    ret = cuMemFreeHost(ptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ(ptr, free_event.address);
}

UCS_TEST_F(cuda_hooks, test_cuMemAllocManaged) {
    CUresult ret;
    CUdeviceptr dptr;

    ret = cuMemAllocManaged(&dptr, 64, CU_MEM_ATTACH_GLOBAL);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 64, UCS_MEMORY_TYPE_CUDA_MANAGED);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr, free_event.address);
}

UCS_TEST_F(cuda_hooks, test_cuMemAllocPitch) {
    CUresult ret;
    CUdeviceptr dptr;
    size_t pitch;

    ret = cuMemAllocPitch(&dptr, &pitch, 4, 8, 4);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void *)dptr, (4 * 8));

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr, free_event.address);
}

#if CUDA_VERSION >= 11020
UCS_TEST_F(cuda_hooks, test_cuMemAllocAsync) {
    CUresult ret;
    CUdeviceptr dptr;

    /* release with cuMemFree */
    ret = cuMemAllocAsync(&dptr, 64, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 64, UCS_MEMORY_TYPE_CUDA_MANAGED);

    ret = cuMemFree(dptr);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr, free_event.address);

    /* release with cuMemFreeAsync */
    ret = cuMemAllocAsync(&dptr, 64, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    check_mem_alloc_events((void*)dptr, 64);

    ret = cuMemFreeAsync(dptr, 0);
    ASSERT_EQ(ret, CUDA_SUCCESS);
    EXPECT_EQ((void*)dptr, free_event.address);
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
    EXPECT_EQ(ptr, free_event.address);

    /* large allocation */
    ret = cudaMalloc(&ptr, (256 * 1024 *1024));
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, (256 * 1024 *1024));

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr, free_event.address);

    /* multiple allocations, cudafree in reverse order */
    ret = cudaMalloc(&ptr, (1 * 1024 *1024));
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, (1 * 1024 *1024));

    ret = cudaMalloc(&ptr1, (1 * 1024 *1024));
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr1, (1 * 1024 *1024));

    ret = cudaFree(ptr1);
    ASSERT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr1, free_event.address);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr, free_event.address);

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
    EXPECT_EQ(ptr, free_event.address);
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
    EXPECT_EQ(devPtr, free_event.address);
}

#if CUDA_VERSION >= 11020
UCS_TEST_F(cuda_hooks, test_cudaMallocAsync) {
    cudaError_t ret;
    void *ptr;

    /* release with cudaFree */
    ret = cudaMallocAsync(&ptr, 64, 0);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_CUDA_MANAGED);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr, free_event.address);

    /* release with cudaFreeAsync */
    ret = cudaMallocAsync(&ptr, 64, 0);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64);

    ret = cudaFreeAsync(ptr, 0);
    ASSERT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr, free_event.address);
}
#endif
