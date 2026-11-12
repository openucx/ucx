/**
 * Copyright (C) SambaNova Systems, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <ucm/api/ucm.h>
#include <common/mem_buffer.h>
#include <common/test.h>

#include <vector>


class rdu_hooks : public ucs::test {
protected:
    struct mem_event {
        void              *address;
        size_t            size;
        ucs_memory_type_t mem_type;
    };

    using mem_event_vec_t = std::vector<mem_event>;

    virtual void init()
    {
        ucs::test::init();

        if (!mem_buffer::is_rdu_supported()) {
            UCS_TEST_SKIP_R("RDU memory is not supported");
        }

        ASSERT_UCS_OK(ucm_set_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, 0,
                                            mem_event_callback,
                                            &m_alloc_events));
        ASSERT_UCS_OK(ucm_set_event_handler(UCM_EVENT_MEM_TYPE_FREE, 0,
                                            mem_event_callback,
                                            &m_free_events));
    }

    virtual void cleanup()
    {
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_FREE, mem_event_callback,
                                &m_free_events);
        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, mem_event_callback,
                                &m_alloc_events);
        ucs::test::cleanup();
    }

    /* Find the event of a buffer. Events of allocations which were not made
     * through the hook report the size from the runtime, which may be rounded
     * up, so for them the size is only a lower bound. */
    static void check_event(const mem_event_vec_t &events,
                            const std::string &name, void *ptr, size_t size,
                            bool exact_size)
    {
        for (const auto &event : events) {
            if ((event.address == ptr) &&
                (exact_size ? (event.size == size) : (event.size >= size))) {
                EXPECT_EQ(UCS_MEMORY_TYPE_UNKNOWN, event.mem_type);
                return;
            }
        }

        ADD_FAILURE() << "could not find memory " << name << " event for "
                      << ptr << " size " << size;
    }

    void test_alloc_free(size_t size, mem_buffer::rdu_mem_kind kind)
    {
        void *ptr = mem_buffer::allocate_rdu(size, kind);

        check_event(m_alloc_events, "alloc", ptr, size, true);
        EXPECT_TRUE(m_free_events.empty());

        mem_buffer::release(ptr, UCS_MEMORY_TYPE_RDU);
        check_event(m_free_events, "free", ptr, size, false);
    }

    void test_existing_alloc(size_t size, mem_buffer::rdu_mem_kind kind)
    {
        const int events_mask = UCM_EVENT_MEM_TYPE_ALLOC |
                                UCM_EVENT_FLAG_EXISTING_ALLOC;
        void *ptr             = mem_buffer::allocate_rdu(size, kind);
        mem_event_vec_t events;

        /* The handler receives allocations made before it was registered */
        ASSERT_UCS_OK(ucm_set_event_handler(events_mask, 0, mem_event_callback,
                                            &events));
        check_event(events, "existing alloc", ptr, size, false);

        ucm_unset_event_handler(UCM_EVENT_MEM_TYPE_ALLOC, mem_event_callback,
                                &events);
        mem_buffer::release(ptr, UCS_MEMORY_TYPE_RDU);
    }

    static void mem_event_callback(ucm_event_type_t event_type,
                                   ucm_event_t *event, void *arg)
    {
        mem_event_vec_t *events = reinterpret_cast<mem_event_vec_t*>(arg);

        events->push_back({event->mem_type.address, event->mem_type.size,
                           event->mem_type.mem_type});
    }

    mem_event_vec_t m_alloc_events;
    mem_event_vec_t m_free_events;
};

UCS_TEST_F(rdu_hooks, alloc_free_hbm) {
    test_alloc_free(UCS_MBYTE, mem_buffer::rdu_mem_kind::HBM);
}

UCS_TEST_F(rdu_hooks, alloc_free_ddr) {
    test_alloc_free(UCS_MBYTE, mem_buffer::rdu_mem_kind::DDR);
}

UCS_TEST_F(rdu_hooks, alloc_free_small) {
    test_alloc_free(64, mem_buffer::rdu_mem_kind::HBM);
}

UCS_TEST_F(rdu_hooks, existing_alloc_hbm) {
    test_existing_alloc(UCS_MBYTE, mem_buffer::rdu_mem_kind::HBM);
}

UCS_TEST_F(rdu_hooks, existing_alloc_ddr) {
    test_existing_alloc(UCS_MBYTE, mem_buffer::rdu_mem_kind::DDR);
}
