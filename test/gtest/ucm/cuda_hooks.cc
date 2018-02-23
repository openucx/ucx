/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <ucm/api/ucm.h>
#include <common/test.h>
#include <cuda.h>
#include <cuda_runtime.h>

static void  *free_ptr;
static ucm_event_t alloc_event, free_event;

static void cuda_mem_event_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    free_ptr = event->vm_unmapped.address;
}

static void cuda_mem_alloc_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    alloc_event.mem_type.address  = event->mem_type.address;
    alloc_event.mem_type.size     = event->mem_type.size;
    alloc_event.mem_type.mem_type = event->mem_type.mem_type;
}

static void cuda_mem_free_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event, void *arg)
{
    free_event.mem_type.address  = event->mem_type.address;
    free_event.mem_type.size     = event->mem_type.size;
    free_event.mem_type.mem_type = event->mem_type.mem_type;
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

        /* install memory hooks */
        result = ucm_set_event_handler(UCM_EVENT_VM_UNMAPPED, 0, cuda_mem_event_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0, cuda_mem_alloc_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0, cuda_mem_free_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);
    }

    virtual void cleanup() {
        CUresult ret;

        ucm_unset_event_handler(UCM_EVENT_VM_UNMAPPED, cuda_mem_event_callback,
                                reinterpret_cast<void*>(this));
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, cuda_mem_alloc_callback,
                                reinterpret_cast<void*>(this));
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, cuda_mem_free_callback,
                                reinterpret_cast<void*>(this));

        ret = cuCtxDestroy(context);
        EXPECT_EQ(ret, CUDA_SUCCESS);

        ucs::test::cleanup();
    }


    void check_mem_alloc_events(void *ptr, size_t size) {
        ASSERT_EQ(ptr, alloc_event.mem_type.address);
        ASSERT_EQ(size, alloc_event.mem_type.size);
        ASSERT_EQ(UCM_MEM_TYPE_CUDA, alloc_event.mem_type.mem_type);
    }

    void check_mem_free_events(void *ptr) {
        ASSERT_EQ(ptr, free_ptr);
        ASSERT_EQ(ptr, free_event.mem_type.address);
        ASSERT_EQ(UCM_MEM_TYPE_CUDA, free_event.mem_type.mem_type);
    }

    CUdevice   device;
    CUcontext  context;
};

UCS_TEST_F(cuda_hooks, test_cuda_Malloc_Free) {
    cudaError_t ret;
    void *ptr, *ptr1;

    /* small allocation */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, 64);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, 64);

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr);

    /* large allocation */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, (256 * 1024 *1024));
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, (256 * 1024 *1024));

    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    ASSERT_EQ(ptr, free_ptr);
    check_mem_free_events(ptr);

    /* multiple allocations, cudafree in reverse order */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, (1 * 1024 *1024));
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr, (1 * 1024 *1024));

    ret = cudaMalloc(&ptr1, (1 * 1024 *1024));
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(ptr1, (1 * 1024 *1024));

    ret = cudaFree(ptr1);
    ASSERT_EQ(ret, cudaSuccess);
    ASSERT_EQ(ptr1, free_ptr);
    check_mem_free_events(ptr1);

    free_ptr = NULL;
    ret = cudaFree(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    ASSERT_EQ(ptr, free_ptr);
    check_mem_free_events(ptr);
}

UCS_TEST_F(cuda_hooks, test_cudaFreeHost) {
    cudaError_t ret;
    void **pDevice;
    void *ptr;

    ret = cudaMallocHost(&ptr, 4096);
    ASSERT_EQ(ret, cudaSuccess);

    ret = cudaHostGetDevicePointer(&pDevice, ptr, 0);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_alloc_events(pDevice, 4096);

    ret = cudaFreeHost(ptr);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events(pDevice);
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
    ASSERT_EQ(devPtr, free_ptr);
    check_mem_free_events(devPtr);
}

UCS_TEST_F(cuda_hooks, test_cudaHostUnregister) {
    cudaError_t ret;
    void *dptr;
    void *p;

    p = malloc(65536);
    EXPECT_TRUE(p != NULL);

    ret = cudaHostRegister(p, 65536, cudaHostRegisterMapped);
    ASSERT_EQ(ret, cudaSuccess);
    ret = cudaHostGetDevicePointer(&dptr, p, 0);
    ASSERT_EQ(ret, cudaSuccess);
    ret = cudaHostUnregister(p);
    ASSERT_EQ(ret, cudaSuccess);
    check_mem_free_events((void *)dptr);

    free(p);
}
