/**
 * Copyright (C) Intel Corporation, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <ucm/api/ucm.h>
#include <common/test.h>
#include <level_zero/ze_api.h>

class ze_hooks : public ucs::test {
protected:
    virtual void init()
    {
        ucs_status_t status;

        ucs::test::init();

        init_ze_context();

        m_alloc_events.reserve(1000);
        m_free_events.reserve(1000);

        status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0,
                                       ze_mem_alloc_callback, this);
        ASSERT_UCS_OK(status);

        status = ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0,
                                       ze_mem_free_callback, this);
        ASSERT_UCS_OK(status);
    }

    virtual void cleanup()
    {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, ze_mem_free_callback,
                                this);
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC,
                                ze_mem_alloc_callback, this);
        cleanup_ze_context();
        ucs::test::cleanup();
    }

    void check_mem_alloc_events(void *ptr, size_t size,
                                ucs_memory_type_t expect_mem_type) const
    {
        check_event_present(m_alloc_events, "alloc", ptr, size,
                            expect_mem_type);
    }

    void check_mem_free_events(void *ptr, size_t size,
                               ucs_memory_type_t expect_mem_type) const
    {
        check_event_present(m_free_events, "free", ptr, size,
                            expect_mem_type);
    }

    size_t free_event_count() const
    {
        return m_free_events.size();
    }

    ze_context_handle_t ze_context() const
    {
        return m_ze_context;
    }

    ze_device_handle_t ze_device() const
    {
        return m_ze_device;
    }

private:
    struct mem_event {
        void              *address;
        size_t            size;
        ucs_memory_type_t mem_type;
    };

    using mem_event_vec_t = std::vector<mem_event>;

    void init_ze_context()
    {
        ze_context_desc_t context_desc = {};
        ze_result_t ret;
        uint32_t driver_count;
        uint32_t device_count;

        ret = zeInit(ZE_INIT_FLAG_GPU_ONLY);
        if (ret != ZE_RESULT_SUCCESS) {
            UCS_TEST_SKIP_R("zeInit failed or no ZE GPU is available");
        }

        driver_count = 1;
        ret          = zeDriverGet(&driver_count, &m_ze_driver);
        if ((ret != ZE_RESULT_SUCCESS) || (driver_count == 0)) {
            UCS_TEST_SKIP_R("no ZE driver detected");
        }

        device_count = 1;
        ret          = zeDeviceGet(m_ze_driver, &device_count, &m_ze_device);
        if ((ret != ZE_RESULT_SUCCESS) || (device_count == 0)) {
            UCS_TEST_SKIP_R("no ZE device detected");
        }

        ret = zeContextCreate(m_ze_driver, &context_desc, &m_ze_context);
        if (ret != ZE_RESULT_SUCCESS) {
            UCS_TEST_SKIP_R("failed to create ZE context");
        }
    }

    void cleanup_ze_context()
    {
        if (m_ze_context != NULL) {
            EXPECT_EQ(ZE_RESULT_SUCCESS, zeContextDestroy(m_ze_context));
            m_ze_context = NULL;
        }
    }

    void check_event_present(const mem_event_vec_t &events,
                             const std::string &name, void *ptr, size_t size,
                             ucs_memory_type_t mem_type) const
    {
        for (const auto &event : events) {
            if ((event.address == ptr) && (event.size >= size) &&
                ((event.mem_type == mem_type) ||
                 (event.mem_type == UCS_MEMORY_TYPE_UNKNOWN))) {
                return;
            }
        }

        FAIL() << "Could not find memory " << name << " event for " << ptr
               << " size " << size << " type "
               << ucs_memory_type_names[mem_type];
    }

    static void ze_mem_alloc_callback(ucm_event_type_t event_type,
                                      ucm_event_t *event, void *arg)
    {
        ze_hooks *self = reinterpret_cast<ze_hooks*>(arg);

        static_cast<void>(event_type);

        self->m_alloc_events.push_back({event->mem_type.address,
                                        event->mem_type.size,
                                        event->mem_type.mem_type});
    }

    static void ze_mem_free_callback(ucm_event_type_t event_type,
                                     ucm_event_t *event, void *arg)
    {
        ze_hooks *self = reinterpret_cast<ze_hooks*>(arg);

        static_cast<void>(event_type);

        self->m_free_events.push_back({event->mem_type.address,
                                       event->mem_type.size,
                                       event->mem_type.mem_type});
    }

    ze_driver_handle_t m_ze_driver  = NULL;
    ze_context_handle_t m_ze_context = NULL;
    ze_device_handle_t m_ze_device  = NULL;
    mem_event_vec_t m_alloc_events;
    mem_event_vec_t m_free_events;
};

UCS_TEST_F(ze_hooks, test_zeMemAllocHost)
{
    ze_host_mem_alloc_desc_t host_desc = {};
    void *ptr                          = NULL;

    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocHost(ze_context(), &host_desc, 64, 1, &ptr));
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_ZE_HOST);

    ASSERT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_context(), ptr));
    check_mem_free_events(ptr, 64, UCS_MEMORY_TYPE_ZE_HOST);
}

UCS_TEST_F(ze_hooks, test_zeMemAllocDevice)
{
    ze_device_mem_alloc_desc_t device_desc = {};
    void *ptr                              = NULL;
    void *ptr1                             = NULL;

    /* small allocation */
    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_context(), &device_desc, 64, 1,
                               ze_device(), &ptr));
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_context(), ptr));
    check_mem_free_events(ptr, 64, UCS_MEMORY_TYPE_ZE_DEVICE);

    /* large allocation */
    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_context(), &device_desc, 256 * UCS_MBYTE, 1,
                               ze_device(), &ptr));
    check_mem_alloc_events(ptr, 256 * UCS_MBYTE, UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_context(), ptr));
    check_mem_free_events(ptr, 256 * UCS_MBYTE, UCS_MEMORY_TYPE_ZE_DEVICE);

    /* multiple allocations, free in reverse order */
    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_context(), &device_desc, UCS_MBYTE, 1,
                               ze_device(), &ptr));
    check_mem_alloc_events(ptr, UCS_MBYTE, UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocDevice(ze_context(), &device_desc, UCS_MBYTE, 1,
                               ze_device(), &ptr1));
    check_mem_alloc_events(ptr1, UCS_MBYTE, UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_context(), ptr1));
    check_mem_free_events(ptr1, UCS_MBYTE, UCS_MEMORY_TYPE_ZE_DEVICE);

    ASSERT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_context(), ptr));
    check_mem_free_events(ptr, UCS_MBYTE, UCS_MEMORY_TYPE_ZE_DEVICE);
}

UCS_TEST_F(ze_hooks, test_zeMemAllocShared)
{
    ze_device_mem_alloc_desc_t device_desc = {};
    ze_host_mem_alloc_desc_t host_desc     = {};
    void *ptr                              = NULL;

    ASSERT_EQ(ZE_RESULT_SUCCESS,
              zeMemAllocShared(ze_context(), &device_desc, &host_desc, 64, 1,
                               ze_device(), &ptr));
    check_mem_alloc_events(ptr, 64, UCS_MEMORY_TYPE_ZE_MANAGED);

    ASSERT_EQ(ZE_RESULT_SUCCESS, zeMemFree(ze_context(), ptr));
    check_mem_free_events(ptr, 64, UCS_MEMORY_TYPE_ZE_MANAGED);
}

UCS_TEST_F(ze_hooks, test_zeMemFreeNull)
{
    const size_t free_events_before = free_event_count();
    ze_result_t ret;

    /* Unlike cudaFree(NULL), zeMemFree(NULL) is expected to return an error. */
    ret = zeMemFree(ze_context(), NULL);
    ASSERT_NE(ZE_RESULT_SUCCESS, ret);
    EXPECT_EQ(free_events_before, free_event_count());
}
