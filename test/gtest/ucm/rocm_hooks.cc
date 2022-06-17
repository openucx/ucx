/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
* Copyright (C) Advanced Micro Devices, Inc. 2019.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/
#include <ucm/api/ucm.h>
#include <common/test.h>
#include <hip_runtime.h>

static ucm_event_t alloc_event, free_event;

static void rocm_mem_alloc_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    alloc_event.mem_type.address  = event->mem_type.address;
    alloc_event.mem_type.size     = event->mem_type.size;
    alloc_event.mem_type.mem_type = event->mem_type.mem_type;
}

static void rocm_mem_free_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event, void *arg)
{
    free_event.mem_type.address  = event->mem_type.address;
    free_event.mem_type.size     = event->mem_type.size;
    free_event.mem_type.mem_type = event->mem_type.mem_type;
}

class rocm_hooks : public ucs::test {
protected:

    virtual void init() {
        int dev_count;
        ucs_status_t result;
        hipError_t ret;
        ucs::test::init();

        ret = hipGetDeviceCount(&dev_count);
        if ((ret != hipSuccess) || (dev_count < 1)) {
            UCS_TEST_SKIP_R("no ROCm device detected");
        }

        if (hipSetDevice(0) != hipSuccess) {
            UCS_TEST_SKIP_R("can't set ROCm device");
        }

        /* install memory hooks */
        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0,
                                       rocm_mem_alloc_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);

        result = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0,
                                       rocm_mem_free_callback,
                                       reinterpret_cast<void*>(this));
        ASSERT_UCS_OK(result);
    }

    virtual void cleanup() {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC,
                                rocm_mem_alloc_callback,
                                reinterpret_cast<void*>(this));
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE,
                                rocm_mem_free_callback,
                                reinterpret_cast<void*>(this));
        ucs::test::cleanup();
    }

    void check_mem_alloc_events(void *ptr, size_t size,
                                int expect_mem_type = UCS_MEMORY_TYPE_ROCM)  {
        ASSERT_EQ(ptr, alloc_event.mem_type.address);
        ASSERT_EQ(size, alloc_event.mem_type.size);
        EXPECT_TRUE((alloc_event.mem_type.mem_type == expect_mem_type) ||
                    (alloc_event.mem_type.mem_type == UCS_MEMORY_TYPE_UNKNOWN));
    }

    void check_mem_free_events(void *ptr, size_t size,
                               int expect_mem_type = UCS_MEMORY_TYPE_ROCM) {
        ASSERT_EQ(ptr, free_event.mem_type.address);
        ASSERT_EQ(expect_mem_type, free_event.mem_type.mem_type);
    }

};

UCS_TEST_F(rocm_hooks, test_hipMem_Alloc_Free) {
    hipError_t ret;
    void *dptr, *dptr1;

    /* small allocation */
    ret = hipMalloc(&dptr, 64);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_alloc_events((void *)dptr, 64);

    ret = hipFree(dptr);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_free_events((void *)dptr, 64);

    /* large allocation */
    ret = hipMalloc(&dptr, (256 * UCS_MBYTE));
    ASSERT_EQ(ret, hipSuccess);
    check_mem_alloc_events((void *)dptr, (256 * UCS_MBYTE));

    ret = hipFree(dptr);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_free_events((void *)dptr, (256 * UCS_MBYTE));

    /* multiple allocations, hipfree in reverse order */
    ret = hipMalloc(&dptr, (1 * UCS_MBYTE));
    ASSERT_EQ(ret, hipSuccess);
    check_mem_alloc_events((void *)dptr, (1 * UCS_MBYTE));

    ret = hipMalloc(&dptr1, (1 * UCS_MBYTE));
    ASSERT_EQ(ret, hipSuccess);
    check_mem_alloc_events((void *)dptr1, (1 * UCS_MBYTE));

    ret = hipFree(dptr1);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_free_events((void *)dptr1, (1 * UCS_MBYTE));

    ret = hipFree(dptr);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_free_events((void *)dptr, (1 * UCS_MBYTE));
}

UCS_TEST_F(rocm_hooks, test_hipMallocManaged) {
    hipError_t ret;
    void * dptr;

    if (mem_buffer::is_rocm_managed_supported()) {
        ret = hipMallocManaged(&dptr, 64);
        ASSERT_EQ(ret, hipSuccess);
        check_mem_alloc_events((void *)dptr, 64);

        ret = hipFree(dptr);
        ASSERT_EQ(ret, hipSuccess);
        check_mem_free_events((void *)dptr, 0, UCS_MEMORY_TYPE_ROCM_MANAGED);
    }
}

UCS_TEST_F(rocm_hooks, test_hipMallocPitch) {
    hipError_t ret;
    void * dptr;
    size_t pitch;

    ret = hipMallocPitch(&dptr, &pitch, 4, 8);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_alloc_events((void *)dptr, (pitch * 8));

    ret = hipFree(dptr);
    ASSERT_EQ(ret, hipSuccess);
    check_mem_free_events((void *)dptr, 0);
}
