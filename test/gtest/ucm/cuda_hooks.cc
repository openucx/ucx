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
static void cuda_mem_event_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    free_ptr = event->vm_unmapped.address;
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
    EXPECT_EQ(ptr, free_ptr);

    /* large allocation */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, (256 * 1024 *1024));
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaFree(ptr);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr, free_ptr);

    /* multiple allocations, cudafree in reverse order */
    free_ptr = NULL;
    ret = cudaMalloc(&ptr, (1 * 1024 *1024));
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaMalloc(&ptr1, (1 * 1024 *1024));
    EXPECT_EQ(ret, cudaSuccess);

    ret = cudaFree(ptr1);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr1, free_ptr);

    free_ptr = NULL;
    ret = cudaFree(ptr);
    EXPECT_EQ(ret, cudaSuccess);
    EXPECT_EQ(ptr, free_ptr);

    ucm_unset_event_handler(UCM_EVENT_VM_UNMAPPED, cuda_mem_event_callback,
                            reinterpret_cast<void*>(this));
}
