/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_request.inl>
#include <ucp/rma/rma.h>
}


class test_ucp_fence_lane_state : public ucs::test {
};

UCS_TEST_F(test_ucp_fence_lane_state, replace_unstarted_lane)
{
    ucp_lane_map_t all_lanes     = UCS_BIT(0) | UCS_BIT(1);
    ucp_lane_map_t lane_mask     = UCS_BIT(1);
    ucp_lane_map_t started_lanes = 0;
    int            count_diff;

    count_diff = ucp_ep_flush_lane_state_update(
            UCS_BIT(0) | UCS_BIT(2), 0, &started_lanes, &all_lanes,
            &lane_mask);

    EXPECT_EQ(0, count_diff);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(2), all_lanes);
    EXPECT_EQ(UCS_BIT(1) | UCS_BIT(2), lane_mask);
}

UCS_TEST_F(test_ucp_fence_lane_state, replace_started_lane)
{
    ucp_lane_map_t all_lanes     = UCS_BIT(0) | UCS_BIT(1);
    ucp_lane_map_t lane_mask     = UCS_BIT(1);
    ucp_lane_map_t started_lanes = UCS_BIT(1);
    int            count_diff;

    count_diff = ucp_ep_flush_lane_state_update(
            UCS_BIT(0) | UCS_BIT(2), 0, &started_lanes, &all_lanes,
            &lane_mask);

    EXPECT_EQ(1, count_diff);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(2), all_lanes);
    EXPECT_EQ(UCS_BIT(1) | UCS_BIT(2), lane_mask);
}

UCS_TEST_F(test_ucp_fence_lane_state, replace_same_index_lane)
{
    ucp_lane_map_t all_lanes     = UCS_BIT(0) | UCS_BIT(1);
    ucp_lane_map_t lane_mask     = UCS_BIT(0);
    ucp_lane_map_t started_lanes = UCS_BIT(0);
    int            count_diff;

    count_diff = ucp_ep_flush_lane_state_update(
            UCS_BIT(0) | UCS_BIT(1), 1, &started_lanes, &all_lanes,
            &lane_mask);

    /* Keep the old started completion, remove one old unstarted completion,
     * and reserve completions for both current lanes. */
    EXPECT_EQ(1, count_diff);
    EXPECT_EQ(0, started_lanes);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(1), all_lanes);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(1), lane_mask);
}

UCS_TEST_F(test_ucp_fence_lane_state, normalize_stale_lane_map)
{
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(2),
              ucp_ep_fence_lane_map_update(UCS_BIT(0) | UCS_BIT(1),
                                           UCS_BIT(0) | UCS_BIT(2)));
}

UCS_TEST_F(test_ucp_fence_lane_state, preserve_current_lane_subset)
{
    EXPECT_EQ(UCS_BIT(1),
              ucp_ep_fence_lane_map_update(UCS_BIT(1),
                                           UCS_BIT(0) | UCS_BIT(1) |
                                           UCS_BIT(2)));
}

UCS_TEST_F(test_ucp_fence_lane_state, dirty_normalize_expands_lane_subset)
{
    ucp_lane_map_t live_lanes = UCS_BIT(0) | UCS_BIT(1);

    EXPECT_EQ(live_lanes,
              ucp_ep_fence_lane_map_normalize(UCS_BIT(0), live_lanes, 1));
    EXPECT_EQ(UCS_BIT(0),
              ucp_ep_fence_lane_map_normalize(UCS_BIT(0), live_lanes, 0));
}


UCS_TEST_F(test_ucp_fence_lane_state, destroyed_started_lane_is_not_unstarted)
{
    EXPECT_FALSE(ucp_ep_flush_has_unstarted_lanes(
            UCS_BIT(0) | UCS_BIT(2),
            UCS_BIT(0) | UCS_BIT(1) | UCS_BIT(2)));
}

UCS_TEST_F(test_ucp_fence_lane_state, live_unstarted_lane)
{
    EXPECT_TRUE(ucp_ep_flush_has_unstarted_lanes(
            UCS_BIT(0) | UCS_BIT(2),
            UCS_BIT(0) | UCS_BIT(1)));
}

static ucp_ep_h test_fence_flush_mutation_ep;
static int test_fence_saw_clean_before_mutation;

static ucs_status_t
test_fence_flush_mutate_and_fail(uct_ep_h uct_ep, unsigned,
                                 uct_completion_t *)
{
    test_fence_saw_clean_before_mutation =
            !test_fence_flush_mutation_ep->ext->fence_lanes_dirty;
    ucp_ep_set_lane(test_fence_flush_mutation_ep, 0, uct_ep);
    return UCS_ERR_IO_ERROR;
}



class test_ucp_fence : public ucp_test {
public:
    virtual void init() {
        if (get_variant_value() & EP_BASED_FENCE) {
            if (!is_proto_enabled()) {
                UCS_TEST_SKIP_R("Proto v2 is disabled");
            }
            modify_config("FENCE_MODE", "ep_based");
        }

        ucp_test::init();
    }

    typedef void (test_ucp_fence::* send_func_t)(entity *e, uint64_t *initial_buf,
                                                 uint64_t *result_buf, void *memheap_addr,
                                                 ucp_rkey_h rkey);

    static void send_cb(void *request, ucs_status_t status)
    {
    }

    static ucs_status_t defer_once(uct_pending_req_t *self)
    {
        ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

        if (req->send.state.completed_size++ == 0) {
            req->flags |= UCP_REQUEST_FLAG_FENCE_BLOCKED;
            return UCP_STATUS_FENCE_DEFER;
        }

        return UCS_OK;
    }

    template <typename T>
    void blocking_add(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                      void *memheap_addr, ucp_rkey_h rkey) {
        ucp_request_param_t param;

        param.op_attr_mask  = UCP_OP_ATTR_FIELD_DATATYPE;
        param.datatype      = ucp_dt_make_contig(sizeof(T));
        void *request       = ucp_atomic_op_nbx(e->ep(), UCP_ATOMIC_OP_ADD,
                                                initial_buf, 1,
                                                (uintptr_t)memheap_addr, rkey,
                                                &param);
        ucs_status_t status = request_wait(request, {e});
        ASSERT_UCS_OK(status);
    }

    template <typename T>
    void blocking_fadd(entity *e, uint64_t *initial_buf, uint64_t *result_buf,
                       void *memheap_addr, ucp_rkey_h rkey)
    {
        void *request = ucp_atomic_fetch_nb(e->ep(), UCP_ATOMIC_FETCH_OP_FADD,
                                            *initial_buf, (T*)result_buf, sizeof(T),
                                            (uintptr_t)memheap_addr, rkey, send_cb);
        request_wait(request, {e});
    }

    template <typename T, typename F>
    void test(F f1, F f2) {
        test_fence(static_cast<send_func_t>(f1),
                   static_cast<send_func_t>(f2), sizeof(T));
    }

    class worker {
    public:
        worker(test_ucp_fence* test, send_func_t send1, send_func_t send2,
               entity* entity, ucp_rkey_h rkey, void *memheap_ptr,
               uint64_t initial_value, uint32_t* error):
            test(test), value(initial_value), result(0), error(error),
            running(true), m_rkey(rkey), m_memheap(memheap_ptr),
            m_send_1(send1), m_send_2(send2), m_entity(entity) {
            pthread_create(&m_thread, NULL, run, reinterpret_cast<void*>(this));
        }

        ~worker() {
            assert(!running);
            assert(m_thread == pthread_self());
        }

        static void *run(void *arg) {
            worker *self = reinterpret_cast<worker*>(arg);
            self->run();
            self->running = false;
            return NULL;
        }

        void join() {
            void *retval;
            pthread_join(m_thread, &retval);
            m_thread = pthread_self();
        }

        test_ucp_fence* const test;
        uint64_t value, result;
        uint32_t* error;
        bool running;

    private:
        void run() {
            uint64_t zero = 0;

            for (int i = 0; i < 500 / ucs::test_time_multiplier(); i++) {
                (test->*m_send_1)(m_entity, &value, &result,
                                  m_memheap, m_rkey);

                m_entity->fence();

                (test->*m_send_2)(m_entity, &zero, &result,
                                  m_memheap, m_rkey);

                test->flush_worker(*m_entity, 0, {m_entity});

                if (result != (uint64_t)(i+1))
                    (*error)++;

                result = 0; /* reset for the next loop */
            }
        }

        ucp_rkey_h m_rkey;
        void *m_memheap;
        send_func_t m_send_1, m_send_2;
        entity* m_entity;
        pthread_t m_thread;
    };

    void run_workers(send_func_t send1, send_func_t send2, entity* sender,
                     ucp_rkey_h rkey, void *memheap_ptr,
                     uint64_t initial_value, uint32_t* error) {
        ucs::ptr_vector<worker> m_workers;
        m_workers.clear();
        m_workers.push_back(new worker(this, send1, send2, sender, rkey,
                                       memheap_ptr, initial_value, error));
        if (!is_loopback()) {
            /* allow receiver to progress incoming ops */
            while (m_workers.front()->running) {
                progress({&receiver()});
            }
        }
        m_workers.at(0).join();
        m_workers.clear();
    }

protected:
    void test_fence(send_func_t send1, send_func_t send2, size_t alignment) {
        static const size_t memheap_size = sizeof(uint64_t);
        uint32_t error = 0;

        sender().connect(&receiver(), get_ep_params());
        flush_worker(sender()); /* avoid deadlock for blocking amo */

        mapped_buffer buffer(memheap_size, receiver(), 0);

        EXPECT_LE(memheap_size, buffer.size());
        memset(buffer.ptr(), 0, memheap_size);

        run_workers(send1, send2, &sender(), buffer.rkey(sender()),
                    buffer.ptr(), 1, &error);

        EXPECT_EQ(error, (uint32_t)0);

        disconnect(sender());
        disconnect(receiver());
    }

    enum {
        EP_BASED_FENCE = UCS_BIT(0)
    };
};

class test_ucp_fence32 : public test_ucp_fence {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_AMO32, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_AMO32, EP_BASED_FENCE,
                               "ep_based");
    }
};

UCS_TEST_P(test_ucp_fence32, atomic_add_fadd) {
    test<uint32_t>(&test_ucp_fence32::blocking_add<uint32_t>,
                   &test_ucp_fence32::blocking_fadd<uint32_t>);
}

UCS_TEST_P(test_ucp_fence32, lane_topology_dirty_lifecycle)
{
    uint64_t lane_generation;
    ucp_ep_h ep;
    uct_ep_h lane;

    uct_ep_flush_func_t flush_func;
    uct_iface_h iface;
    ucs_status_t status;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Direct lane-state test requires self transport");
    }

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();

    ep->ext->fence_lanes_dirty = 0;
    ep->ext->unflushed_lanes   = UCS_BIT(0);
    lane_generation            = ep->ext->lane_generation;
    lane                       = ucp_ep_get_lane(ep, 0);

    /* Replacing a lane at the same index must still invalidate fence
     * tracking. */
    ucp_ep_set_lane(ep, 0, lane);
    EXPECT_EQ(lane_generation + 1, ep->ext->lane_generation);
    EXPECT_TRUE(ep->ext->fence_lanes_dirty);

    EXPECT_UCS_OK(ucp_ep_fence_strong(ep));
    EXPECT_EQ(0, ep->ext->unflushed_lanes);
    EXPECT_FALSE(ep->ext->fence_lanes_dirty);

    ep->ext->unflushed_lanes = UCS_BIT(0);
    ucp_ep_set_lane(ep, 0, lane);
    EXPECT_TRUE(ep->ext->fence_lanes_dirty);

    iface                                = lane->iface;
    flush_func                           = iface->ops.ep_flush;
    test_fence_flush_mutation_ep         = ep;
    test_fence_saw_clean_before_mutation = 0;
    iface->ops.ep_flush                  = test_fence_flush_mutate_and_fail;

    {
        scoped_log_handler slh(hide_errors_logger);
        scoped_log_handler warn_slh(hide_warns_logger);
        status = ucp_ep_fence_strong(ep);
    }

    iface->ops.ep_flush          = flush_func;
    test_fence_flush_mutation_ep = NULL;

    EXPECT_EQ(UCS_ERR_IO_ERROR, status);
    EXPECT_TRUE(test_fence_saw_clean_before_mutation);
    EXPECT_TRUE(ep->ext->fence_lanes_dirty);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, fresh_defer_without_uct_lane)
{
    if (!is_self()) {
        UCS_TEST_SKIP_R("Synthetic fence-queue test uses a stack request");
    }

    ucp_request_t req = {};
    ucp_ep_h ep;
    void *flush_req;

    sender().connect(&receiver(), get_ep_params());
    ep                              = sender().ep();
    req.send.ep                     = ep;
    req.send.lane                   = UCP_NULL_LANE;
    req.send.pending_lane           = UCP_NULL_LANE;
    req.send.fenced_req.fence_seq   = ep->ext->fence_seq;
    req.send.uct.func               = defer_once;

    ucp_request_send(&req);

    EXPECT_FALSE(ucs_queue_is_empty(&ep->ext->fence_pending_q));
    flush_req = ucp_worker_flush_nbx(sender().worker(),
                                     &ucp_request_null_param);
    EXPECT_TRUE(UCS_PTR_IS_PTR(flush_req));

    while (!ucs_queue_is_empty(&ep->ext->fence_pending_q)) {
        progress({&sender()});
    }
    EXPECT_FALSE(req.flags & UCP_REQUEST_FLAG_FENCE_BLOCKED);
    if (UCS_PTR_IS_PTR(flush_req)) {
        EXPECT_UCS_OK(request_wait(flush_req, {&sender()}));
    }

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, pending_queue_orders_fence_epochs)
{
    if (!is_self()) {
        UCS_TEST_SKIP_R("Synthetic fence-queue test uses stack requests");
    }

    ucp_request_t first_epoch = {};
    ucp_request_t second_epoch_first = {};
    ucp_request_t second_epoch_second = {};
    ucp_ep_h ep;

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();

    first_epoch.send.ep                   = ep;
    first_epoch.send.fenced_req.fence_seq = 1;
    first_epoch.send.uct.func             = defer_once;
    first_epoch.send.state.completed_size = 1;
    first_epoch.flags                     = UCP_REQUEST_FLAG_FENCE_BLOCKED;
    second_epoch_first.send.ep            = ep;
    second_epoch_first.send.fenced_req.fence_seq = 2;
    second_epoch_first.send.uct.func      = defer_once;
    second_epoch_first.send.state.completed_size = 1;
    second_epoch_first.flags              = UCP_REQUEST_FLAG_FENCE_BLOCKED;
    second_epoch_second.send.ep           = ep;
    second_epoch_second.send.fenced_req.fence_seq = 2;
    second_epoch_second.send.uct.func     = defer_once;
    second_epoch_second.send.state.completed_size = 1;
    second_epoch_second.flags             = UCP_REQUEST_FLAG_FENCE_BLOCKED;

    ucp_ep_fence_pending_add(ep, &second_epoch_first.send.uct);
    ucp_ep_fence_pending_add(ep, &first_epoch.send.uct);
    ucp_ep_fence_pending_add(ep, &second_epoch_second.send.uct);

    EXPECT_EQ(&first_epoch.send.fenced_req.fence_pending_elem,
              ep->ext->fence_pending_q.head);
    EXPECT_EQ(&second_epoch_first.send.fenced_req.fence_pending_elem,
              ep->ext->fence_pending_q.head->next);
    EXPECT_EQ(&second_epoch_second.send.fenced_req.fence_pending_elem,
              ep->ext->fence_pending_q.head->next->next);

    while (!ucs_queue_is_empty(&ep->ext->fence_pending_q)) {
        progress({&sender()});
    }

    disconnect(sender());
    disconnect(receiver());
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence32)

class test_ucp_fence64 : public test_ucp_fence {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_AMO64, 0, "");
        add_variant_with_value(variants, UCP_FEATURE_AMO64, EP_BASED_FENCE,
                               "ep_based");
    }
};

UCS_TEST_P(test_ucp_fence64, atomic_add_fadd) {
    test<uint64_t>(&test_ucp_fence64::blocking_add<uint64_t>,
                   &test_ucp_fence64::blocking_fadd<uint64_t>);
}

UCS_TEST_P(test_ucp_fence64, atomic_add_fadd_strong, "FENCE_MODE=strong") {
    test<uint64_t>(&test_ucp_fence64::blocking_add<uint64_t>,
                   &test_ucp_fence64::blocking_fadd<uint64_t>);
}

UCP_INSTANTIATE_TEST_CASE(test_ucp_fence64)
