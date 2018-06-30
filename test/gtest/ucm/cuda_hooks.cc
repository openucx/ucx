/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <ucm/api/ucm.h>
#include <common/test.h>
#include <cuda.h>
#include <cuda_runtime.h>

class cuda_hook : public ucs::test {
};

static void  *free_ptr;
static size_t free_size;

static void cuda_mem_event_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    free_ptr  = event->vm_unmapped.address;
    free_size = event->vm_unmapped.size;
}

static void check_mem_free_events(void *ptr, size_t size)
{
    ASSERT_EQ(ptr, free_ptr);
    ASSERT_TRUE(!size || (size == free_size));
}

UCS_TEST_F(cuda_hook, cudafree) {
    ucs_status_t result;
    cudaError_t ret;
    void *ptr, *ptr1;

    /* set cuda device */
    if (cudaSetDevice(0) != cudaSuccess) {
        UCS_TEST_SKIP_R("can't set cuda device");
    }

    /* Install memory hooks */
    result = ucm_set_event_handler(UCM_EVENT_VM_UNMAPPED, 0, cuda_mem_event_callback,
                                   reinterpret_cast<void*>(this));
    ASSERT_UCS_OK(result);

    /* small allocation */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, 64);
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaFree(ptr);
    EXPECT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, 64);

    /* large allocation */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, (256 * 1024 *1024));
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaFree(ptr);
    EXPECT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, (256 * 1024 *1024));

    /* multiple allocations, cudafree in reverse order */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, (1 * 1024 *1024));
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaMalloc(&ptr1, (1 * 1024 *1024));
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaFree(ptr1);
    EXPECT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr1, (1 * 1024 *1024));

    free_ptr = NULL;
    ret = cudaFree(ptr);
    EXPECT_EQ(ret, cudaSuccess);
    check_mem_free_events(ptr, (1 * 1024 *1024));

    ucm_unset_event_handler(UCM_EVENT_VM_UNMAPPED, cuda_mem_event_callback,
                            reinterpret_cast<void*>(this));
}
