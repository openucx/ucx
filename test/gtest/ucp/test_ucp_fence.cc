/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_request.inl>
#include <ucp/rma/rma.h>
#include <uct/base/uct_iface.h>
}

class test_ucp_flush_lane_state : public ucs::test {
};

UCS_TEST_F(test_ucp_flush_lane_state, replace_unstarted_lane)
{
    ucp_lane_map_t all_lanes     = UCS_BIT(0) | UCS_BIT(1);
    ucp_lane_map_t lane_mask     = UCS_BIT(1);
    ucp_lane_map_t started_lanes = 0;
    int count_diff;

    count_diff = ucp_ep_flush_lane_state_update(
            UCS_BIT(0) | UCS_BIT(2), 0, &started_lanes, &all_lanes,
            &lane_mask);

    EXPECT_EQ(0, count_diff);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(2), all_lanes);
    EXPECT_EQ(UCS_BIT(1) | UCS_BIT(2), lane_mask);
}

UCS_TEST_F(test_ucp_flush_lane_state, replace_started_lane)
{
    ucp_lane_map_t all_lanes     = UCS_BIT(0) | UCS_BIT(1);
    ucp_lane_map_t lane_mask     = UCS_BIT(1);
    ucp_lane_map_t started_lanes = UCS_BIT(1);
    int count_diff;

    count_diff = ucp_ep_flush_lane_state_update(
            UCS_BIT(0) | UCS_BIT(2), 0, &started_lanes, &all_lanes,
            &lane_mask);

    EXPECT_EQ(1, count_diff);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(2), all_lanes);
    EXPECT_EQ(UCS_BIT(1) | UCS_BIT(2), lane_mask);
}

UCS_TEST_F(test_ucp_flush_lane_state, replace_same_index_lane)
{
    ucp_lane_map_t all_lanes     = UCS_BIT(0) | UCS_BIT(1);
    ucp_lane_map_t lane_mask     = UCS_BIT(0);
    ucp_lane_map_t started_lanes = UCS_BIT(0);
    int count_diff;

    count_diff = ucp_ep_flush_lane_state_update(
            UCS_BIT(0) | UCS_BIT(1), 1, &started_lanes, &all_lanes,
            &lane_mask);

    EXPECT_EQ(1, count_diff);
    EXPECT_EQ(0, started_lanes);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(1), all_lanes);
    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(1), lane_mask);
}

UCS_TEST_F(test_ucp_flush_lane_state, destroyed_started_lane_is_not_unstarted)
{
    EXPECT_FALSE(ucp_ep_flush_has_unstarted_lanes(
            UCS_BIT(0) | UCS_BIT(2),
            UCS_BIT(0) | UCS_BIT(1) | UCS_BIT(2)));
}

UCS_TEST_F(test_ucp_flush_lane_state, live_unstarted_lane)
{
    EXPECT_TRUE(ucp_ep_flush_has_unstarted_lanes(
            UCS_BIT(0) | UCS_BIT(2), UCS_BIT(0) | UCS_BIT(1)));
}

static unsigned test_flush_call_count;

static ucs_status_t
test_flush_count_calls(uct_ep_h, unsigned, uct_completion_t *)
{
    ++test_flush_call_count;
    return UCS_ERR_IO_ERROR;
}

static uct_completion_t *test_flush_comp;
static uct_ep_h test_flush_eps[2];

static ucs_status_t
test_flush_inprogress(uct_ep_h ep, unsigned, uct_completion_t *comp)
{
    if (test_flush_call_count < ucs_static_array_size(test_flush_eps)) {
        test_flush_eps[test_flush_call_count] = ep;
    }

    ++test_flush_call_count;
    test_flush_comp = comp;
    return UCS_INPROGRESS;
}

static void test_flush_completion(ucp_request_t *req)
{
    ucp_request_complete_send(req, req->status);
}

static int test_fence_count_oneshot(const ucs_callbackq_elem_t *, void *arg)
{
    ++*static_cast<unsigned*>(arg);
    return 0;
}

static ucs_status_t test_fence_defer_once(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    if (req->send.state.completed_size++ == 0) {
        return UCP_STATUS_FENCE_DEFER;
    }

    return UCS_OK;
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

UCS_TEST_P(test_ucp_fence32, empty_lane_mask_skips_transport_flush)
{
    ucp_request_param_t param = {};
    ucp_ep_h ep;
    uct_iface_h iface;
    uct_ep_flush_func_t flush_func;
    ucs_status_ptr_t request;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Direct flush interception requires self transport");
    }

    sender().connect(&receiver(), get_ep_params());
    ep         = sender().ep();
    iface      = ucp_ep_get_lane(ep, 0)->iface;
    flush_func = iface->ops.ep_flush;

    test_flush_call_count = 0;
    iface->ops.ep_flush   = test_flush_count_calls;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
    request = ucp_ep_flush_lanes_internal(
            ep, 0, &param, NULL, test_flush_completion, "flush_lane_mask_test",
            UCT_FLUSH_FLAG_LOCAL, 0);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    iface->ops.ep_flush = flush_func;

    EXPECT_EQ(0, test_flush_call_count);
    EXPECT_EQ(NULL, request);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, replace_lane_during_selective_flush)
{
    ucp_request_param_t param = {};
    uct_ep_t replacement_lane = {};
    ucp_ep_h ep;
    uct_ep_h original_lane;
    uct_iface_h iface;
    uct_ep_flush_func_t flush_func;
    ucs_status_ptr_t request;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Direct flush interception requires self transport");
    }

    sender().connect(&receiver(), get_ep_params());
    ep                       = sender().ep();
    original_lane            = ucp_ep_get_lane(ep, 0);
    iface                    = original_lane->iface;
    replacement_lane.iface   = iface;
    flush_func               = iface->ops.ep_flush;
    test_flush_call_count    = 0;
    test_flush_comp          = NULL;
    test_flush_eps[0]        = NULL;
    test_flush_eps[1]        = NULL;
    iface->ops.ep_flush      = test_flush_inprogress;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
    request = ucp_ep_flush_lanes_internal(
            ep, 0, &param, NULL, test_flush_completion,
            "replace_lane_during_flush", UCT_FLUSH_FLAG_LOCAL, UCS_BIT(0));
    EXPECT_TRUE(UCS_PTR_IS_PTR(request));
    EXPECT_EQ(1u, test_flush_call_count);
    EXPECT_EQ(original_lane, test_flush_eps[0]);

    ucp_ep_set_lane(ep, 0, &replacement_lane);
    uct_invoke_completion(test_flush_comp, UCS_OK);

    EXPECT_EQ(2u, test_flush_call_count);
    EXPECT_EQ(&replacement_lane, test_flush_eps[1]);

    uct_invoke_completion(test_flush_comp, UCS_OK);
    ucp_ep_set_lane(ep, 0, original_lane);
    iface->ops.ep_flush = flush_func;
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    EXPECT_UCS_OK(ucp_request_check_status(request));
    ucp_request_release(request);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, lane_generation_updates_on_pointer_change)
{
    uint64_t lane_generation;
    ucp_ep_h ep;
    uct_ep_h lane;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Direct lane-state test requires self transport");
    }

    sender().connect(&receiver(), get_ep_params());
    ep              = sender().ep();
    lane_generation = ep->ext->lane_generation;
    lane            = ucp_ep_get_lane(ep, 0);

    ucp_ep_set_lane(ep, 0, lane);
    EXPECT_EQ(lane_generation, ep->ext->lane_generation);

    ucp_ep_set_lane(ep, 0, NULL);
    EXPECT_EQ(lane_generation + 1, ep->ext->lane_generation);

    ucp_ep_set_lane(ep, 0, lane);
    EXPECT_EQ(lane_generation + 2, ep->ext->lane_generation);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, recycled_ep_lane_storage_initialization)
{
    ucp_worker_h worker = sender().worker();
    ucp_ep_h ep;
    ucp_ep_h recycled_ep;
    ucp_lane_index_t lane;

    UCS_ASYNC_BLOCK(&worker->async);
    ASSERT_UCS_OK(ucp_ep_create_base(worker, UCP_EP_INIT_FLAG_INTERNAL,
                                     "lane-init", "lane-init", &ep));

    for (lane = 0; lane < UCP_MAX_FAST_PATH_LANES; ++lane) {
        ep->uct_eps[lane] = reinterpret_cast<uct_ep_h>(
                static_cast<uintptr_t>(lane + 1));
    }

    ucp_ep_delete(ep);

    ASSERT_UCS_OK(ucp_ep_create_base(worker, UCP_EP_INIT_FLAG_INTERNAL,
                                     "lane-recycle", "lane-recycle",
                                     &recycled_ep));

    EXPECT_EQ(ep, recycled_ep);
    EXPECT_EQ(0, recycled_ep->ext->lane_generation);
    for (lane = 0; lane < UCP_MAX_FAST_PATH_LANES; ++lane) {
        EXPECT_EQ(NULL, recycled_ep->uct_eps[lane]);
    }

    ucp_ep_delete(recycled_ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
}

UCS_TEST_P(test_ucp_fence32, slow_lane_storage_initialization)
{
    const unsigned num_lanes = UCP_MAX_FAST_PATH_LANES + 2;
    ucp_worker_h worker       = sender().worker();
    ucp_ep_h ep;
    ucp_lane_index_t lane;

    UCS_ASYNC_BLOCK(&worker->async);
    ASSERT_UCS_OK(ucp_ep_create_base(worker, UCP_EP_INIT_FLAG_INTERNAL,
                                     "lane-init", "lane-init", &ep));
    ASSERT_UCS_OK(ucp_ep_realloc_lanes(ep, num_lanes));

    for (lane = UCP_MAX_FAST_PATH_LANES; lane < num_lanes; ++lane) {
        ep->ext->uct_eps[lane - UCP_MAX_FAST_PATH_LANES] =
                reinterpret_cast<uct_ep_h>(static_cast<uintptr_t>(lane + 1));
    }

    ep->ext->lane_generation = 0;
    ASSERT_UCS_OK(ucp_ep_realloc_lanes(ep, num_lanes));

    EXPECT_EQ(0, ep->ext->lane_generation);
    for (lane = 0; lane < num_lanes; ++lane) {
        EXPECT_EQ(NULL, ucp_ep_get_lane(ep, lane));
    }

    ucp_ep_delete(ep);
    UCS_ASYNC_UNBLOCK(&worker->async);
}

UCS_TEST_P(test_ucp_fence32, async_fence_tracks_inflight_flush)
{
    ucp_request_t blocked_req = {};
    unsigned ep_refcount;
    uint64_t prev_fence_seq;
    uint64_t fence_seq;
    uct_iface_h iface;
    uct_ep_flush_func_t flush_func;
    ucp_ep_h ep;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Direct flush interception requires self transport");
    }

    sender().connect(&receiver(), get_ep_params());
    ep                    = sender().ep();
    iface                 = ucp_ep_get_lane(ep, 0)->iface;
    flush_func            = iface->ops.ep_flush;
    fence_seq             = ep->ext->fence_seq + 1;
    ep->ext->unflushed_lanes = UCS_BIT(0);
    test_flush_call_count = 0;
    test_flush_comp       = NULL;
    iface->ops.ep_flush   = test_flush_inprogress;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
    EXPECT_UCS_OK(ucp_ep_fence_strong_nb(ep, fence_seq));
    EXPECT_EQ(1u, test_flush_call_count);
    EXPECT_TRUE(ep->ext->fence_inflight_req != NULL);
    EXPECT_EQ(UCS_INPROGRESS, ep->ext->fence_status);
    EXPECT_NE(fence_seq, ep->ext->fence_seq);

    ASSERT_NE(static_cast<uct_completion_t*>(NULL), test_flush_comp);
    uct_invoke_completion(test_flush_comp, UCS_OK);

    EXPECT_TRUE(ep->ext->fence_inflight_req == NULL);
    EXPECT_EQ(UCS_OK, ep->ext->fence_status);
    EXPECT_EQ(fence_seq, ep->ext->fence_seq);
    EXPECT_EQ(0, ep->ext->unflushed_lanes);

    prev_fence_seq                    = ep->ext->fence_seq;
    fence_seq                         = prev_fence_seq + 1;
    blocked_req.id                    = UCS_PTR_MAP_KEY_INVALID;
    blocked_req.send.ep               = ep;
    blocked_req.send.fence_seq        = fence_seq;
    blocked_req.send.uct.func         = test_fence_defer_once;
    blocked_req.flags                 = UCP_REQUEST_FLAG_FENCE_BLOCKED;
    ep->ext->unflushed_lanes          = UCS_BIT(0);
    test_flush_comp                   = NULL;
    ep_refcount                       = ep->refcount;
    EXPECT_UCS_OK(ucp_ep_fence_strong_nb(ep, fence_seq));
    EXPECT_EQ(ep_refcount + 1, ep->refcount);
    ucp_ep_fence_pending_add(ep, &blocked_req.send.uct);
    ASSERT_NE(static_cast<uct_completion_t*>(NULL), test_flush_comp);
    uct_invoke_completion(test_flush_comp, UCS_ERR_IO_ERROR);
    EXPECT_EQ(ep_refcount, ep->refcount);
    EXPECT_EQ(NULL, ep->ext->fence_inflight_req);
    EXPECT_EQ(UCS_ERR_IO_ERROR, ep->ext->fence_status);
    EXPECT_EQ(prev_fence_seq, ep->ext->fence_seq);
    EXPECT_TRUE(ucs_queue_is_empty(&ep->ext->fence_pending_q));
    EXPECT_TRUE(blocked_req.flags & UCP_REQUEST_FLAG_COMPLETED);
    EXPECT_EQ(UCS_ERR_IO_ERROR, blocked_req.status);

    iface->ops.ep_flush = flush_func;
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, pending_purge_preserves_scheduled_progress)
{
    ucp_request_t purged_req  = {};
    ucp_request_t readded_req = {};
    unsigned num_oneshots;
    ucp_ep_h ep;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Synthetic fence-queue test uses stack requests");
    }

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();

    purged_req.id                        = UCS_PTR_MAP_KEY_INVALID;
    purged_req.send.ep                   = ep;
    purged_req.send.fence_seq            = ep->ext->fence_seq;
    purged_req.send.uct.func             = test_fence_defer_once;
    purged_req.flags                     = UCP_REQUEST_FLAG_FENCE_BLOCKED;
    readded_req.id                       = UCS_PTR_MAP_KEY_INVALID;
    readded_req.send.ep                  = ep;
    readded_req.send.fence_seq           = ep->ext->fence_seq;
    readded_req.send.uct.func            = test_fence_defer_once;
    readded_req.send.state.completed_size = 1;
    readded_req.flags                    = UCP_REQUEST_FLAG_FENCE_BLOCKED;

    ucp_ep_fence_pending_add(ep, &purged_req.send.uct);
    ASSERT_TRUE(ep->ext->fence_pending_scheduled);

    num_oneshots = 0;
    ucs_callbackq_remove_oneshot(&ep->worker->uct->progress_q, ep,
                                 test_fence_count_oneshot, &num_oneshots);
    EXPECT_EQ(1u, num_oneshots);

    ucp_ep_fence_pending_purge(ep, UCS_ERR_IO_ERROR);
    EXPECT_TRUE(ucs_queue_is_empty(&ep->ext->fence_pending_q));
    EXPECT_TRUE(ep->ext->fence_pending_scheduled);
    EXPECT_FALSE(purged_req.flags & UCP_REQUEST_FLAG_FENCE_BLOCKED);
    EXPECT_TRUE(purged_req.flags & UCP_REQUEST_FLAG_COMPLETED);
    EXPECT_EQ(UCS_ERR_IO_ERROR, purged_req.status);

    ucp_ep_fence_pending_add(ep, &readded_req.send.uct);
    progress({&sender()});

    EXPECT_TRUE(ucs_queue_is_empty(&ep->ext->fence_pending_q));
    EXPECT_FALSE(ep->ext->fence_pending_scheduled);
    EXPECT_FALSE(readded_req.flags & UCP_REQUEST_FLAG_FENCE_BLOCKED);
    EXPECT_EQ(2, readded_req.send.state.completed_size);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_fence32, pending_queue_orders_fence_epochs)
{
    ucp_request_t first_epoch         = {};
    ucp_request_t second_epoch_first  = {};
    ucp_request_t second_epoch_second = {};
    ucp_ep_h ep;

    if (!is_self()) {
        UCS_TEST_SKIP_R("Synthetic fence-queue test uses stack requests");
    }

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();

    first_epoch.send.ep                   = ep;
    first_epoch.send.fence_seq            = 1;
    first_epoch.send.uct.func             = test_fence_defer_once;
    first_epoch.send.state.completed_size = 1;
    first_epoch.flags                     = UCP_REQUEST_FLAG_FENCE_BLOCKED;
    second_epoch_first.send.ep            = ep;
    second_epoch_first.send.fence_seq     = 2;
    second_epoch_first.send.uct.func      = test_fence_defer_once;
    second_epoch_first.send.state.completed_size = 1;
    second_epoch_first.flags                     =
            UCP_REQUEST_FLAG_FENCE_BLOCKED;
    second_epoch_second.send.ep                  = ep;
    second_epoch_second.send.fence_seq            = 2;
    second_epoch_second.send.uct.func              = test_fence_defer_once;
    second_epoch_second.send.state.completed_size  = 1;
    second_epoch_second.flags                      =
            UCP_REQUEST_FLAG_FENCE_BLOCKED;

    ucp_ep_fence_pending_add(ep, &second_epoch_first.send.uct);
    ucp_ep_fence_pending_add(ep, &first_epoch.send.uct);
    ucp_ep_fence_pending_add(ep, &second_epoch_second.send.uct);

    EXPECT_EQ(&first_epoch.send.fence_pending_elem,
              ep->ext->fence_pending_q.head);
    EXPECT_EQ(&second_epoch_first.send.fence_pending_elem,
              ep->ext->fence_pending_q.head->next);
    EXPECT_EQ(&second_epoch_second.send.fence_pending_elem,
              ep->ext->fence_pending_q.head->next->next);

    while (!ucs_queue_is_empty(&ep->ext->fence_pending_q)) {
        progress({&sender()});
    }

    EXPECT_FALSE(first_epoch.flags & UCP_REQUEST_FLAG_FENCE_BLOCKED);
    EXPECT_FALSE(second_epoch_first.flags & UCP_REQUEST_FLAG_FENCE_BLOCKED);
    EXPECT_FALSE(second_epoch_second.flags & UCP_REQUEST_FLAG_FENCE_BLOCKED);

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
