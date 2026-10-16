/*
 * Copyright (C) MThreads 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <ucm/api/ucm.h>
#include <common/test.h>
#include <musa_runtime_api.h>

static ucm_event_t musa_alloc_event;
static ucm_event_t musa_free_event;

static void musa_mem_alloc_callback(ucm_event_type_t event_type,
                                    ucm_event_t *event, void *arg)
{
    musa_alloc_event.mem_type = event->mem_type;
}

static void musa_mem_free_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event, void *arg)
{
    musa_free_event.mem_type = event->mem_type;
}

class musa_hooks : public ucs::test {
protected:
    void init() override
    {
        int dev_count;

        ucs::test::init();
        if ((musaGetDeviceCount(&dev_count) != musaSuccess) ||
            (dev_count == 0)) {
            UCS_TEST_SKIP_R("no MUSA device detected");
        }

        ASSERT_EQ(musaSuccess, musaSetDevice(0));
        ASSERT_UCS_OK(ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0,
                                            musa_mem_alloc_callback, this));
        ASSERT_UCS_OK(ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0,
                                            musa_mem_free_callback, this));
    }

    void cleanup() override
    {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC,
                                musa_mem_alloc_callback, this);
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE,
                                musa_mem_free_callback, this);
        ucs::test::cleanup();
    }
};

UCS_TEST_F(musa_hooks, test_musaMalloc_musaFree)
{
    void *ptr;

    ASSERT_EQ(musaSuccess, musaMalloc(&ptr, 64));
    EXPECT_EQ(ptr, musa_alloc_event.mem_type.address);
    EXPECT_EQ(64ul, musa_alloc_event.mem_type.size);
    EXPECT_EQ(UCS_MEMORY_TYPE_MUSA, musa_alloc_event.mem_type.mem_type);
    EXPECT_EQ(UCS_MEM_FLAG_REGISTRABLE, musa_alloc_event.mem_type.mem_flags);

    ASSERT_EQ(musaSuccess, musaFree(ptr));
    EXPECT_EQ(ptr, musa_free_event.mem_type.address);
    EXPECT_EQ(UCS_MEMORY_TYPE_MUSA, musa_free_event.mem_type.mem_type);
}
