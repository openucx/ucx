/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <common/test_helpers.h>

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>
#include <uct/base/uct_iface.h>
}

static unsigned test_flush_call_count;

static ucs_status_t
test_flush_count_calls(uct_ep_h, unsigned, uct_completion_t *)
{
    ++test_flush_call_count;
    return UCS_ERR_IO_ERROR;
}

static uct_completion_t *test_flush_comp;
static uct_ep_h test_flush_eps[4];
static uct_ep_h test_flush_no_resource_ep;

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

static ucs_status_t
test_flush_inprogress_except_ep(uct_ep_h ep, unsigned flags,
                                uct_completion_t *comp)
{
    if (ep == test_flush_no_resource_ep) {
        ++test_flush_call_count;
        return UCS_ERR_NO_RESOURCE;
    }

    return test_flush_inprogress(ep, flags, comp);
}

static ucs_status_t
test_flush_no_resource(uct_ep_h, unsigned, uct_completion_t *)
{
    return UCS_ERR_NO_RESOURCE;
}

static ucs_status_t
test_flush_pending_add(uct_ep_h, uct_pending_req_t *, unsigned)
{
    return UCS_OK;
}

static void test_flush_completion(ucp_request_t *req)
{
    ucp_request_complete_send(req, req->status);
}

class test_flush_worker_cs_guard {
public:
    explicit test_flush_worker_cs_guard(ucp_worker_h worker) : m_worker(worker)
    {
        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(m_worker);
    }

    ~test_flush_worker_cs_guard()
    {
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(m_worker);
    }

private:
    ucp_worker_h m_worker;
};

class test_ucp_flush : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_TAG);
    }
};

static void test_flush_err_handler(void*, ucp_ep_h, ucs_status_t)
{
}

class test_ucp_flush_failover : public test_ucp_flush {
public:
    ucp_ep_params_t get_ep_params() override
    {
        ucp_ep_params_t params = test_ucp_flush::get_ep_params();

        params.field_mask    |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode        = UCP_ERR_HANDLING_MODE_FAILOVER;
        params.err_handler.cb  = test_flush_err_handler;
        params.err_handler.arg = NULL;
        return params;
    }
};

UCS_TEST_P(test_ucp_flush, empty_lane_mask_skips_transport_flush)
{
    ucp_request_param_t param = {};
    ucp_ep_h ep;
    uct_iface_h iface;
    ucs_status_ptr_t request;

    sender().connect(&receiver(), get_ep_params());
    ep    = sender().ep();
    iface = ucp_ep_get_lane(ep, 0)->iface;

    test_flush_call_count = 0;
    {
        ucs::mock mock;
        mock.setup(&iface->ops.ep_flush, test_flush_count_calls);

        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
        request = ucp_ep_flush_lanes_internal(
                ep, 0, &param, NULL, test_flush_completion,
                "flush_lane_mask_test", UCT_FLUSH_FLAG_LOCAL, 0);
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    }

    EXPECT_EQ(0, test_flush_call_count);
    EXPECT_EQ(NULL, request);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_flush, replace_lane_during_selective_flush)
{
    ucp_request_param_t param = {};
    uct_ep_t replacement_lane = {};
    ucp_ep_h ep;
    uct_ep_h original_lane;
    ucs_status_ptr_t request;
    unsigned i;
    unsigned num_lanes;

    sender().connect(&receiver(), get_ep_params());
    ep                       = sender().ep();
    num_lanes                = ucp_ep_num_lanes(ep);
    original_lane            = ucp_ep_get_lane(ep, 0);
    replacement_lane.iface   = original_lane->iface;
    test_flush_call_count    = 0;
    test_flush_comp          = NULL;
    test_flush_eps[0]        = NULL;
    test_flush_eps[1]        = NULL;
    {
        ucs::mock mock;
        mock_ep_flush(ep, mock, test_flush_inprogress);

        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
        request = ucp_ep_flush_lanes_internal(
                ep, 0, &param, NULL, test_flush_completion,
                "replace_lane_during_flush", UCT_FLUSH_FLAG_LOCAL, UCS_BIT(0));
        EXPECT_TRUE(UCS_PTR_IS_PTR(request));
        EXPECT_EQ(1u, test_flush_call_count);
        EXPECT_EQ(original_lane, test_flush_eps[0]);

        ucp_ep_set_lane(ep, 0, &replacement_lane);
        EXPECT_TRUE(test_flush_comp != NULL);
        if (test_flush_comp != NULL) {
            uct_invoke_completion(test_flush_comp, UCS_OK);
        }

        EXPECT_EQ(num_lanes + 1, test_flush_call_count);
        EXPECT_EQ(&replacement_lane, test_flush_eps[1]);

        EXPECT_TRUE(test_flush_comp != NULL);
        if (test_flush_comp != NULL) {
            for (i = 0; i < num_lanes; ++i) {
                uct_invoke_completion(test_flush_comp, UCS_OK);
            }
        }
        ucp_ep_set_lane(ep, 0, original_lane);
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    }

    EXPECT_UCS_OK(ucp_request_check_status(request));
    ucp_request_release(request);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_flush, replace_other_lane_during_inprogress_flush)
{
    ucp_ep_h ep;
    uct_ep_h original_lane;
    uct_ep_h original_second_lane;
    ucs_status_ptr_t request;
    unsigned i;
    unsigned num_lanes;

    ucp_request_param_t param = {};
    uct_ep_t replacement_second_lane = {};

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();
    if (ucp_ep_num_lanes(ep) < 2) {
        disconnect(sender());
        disconnect(receiver());
        UCS_TEST_SKIP_R("requires two endpoint lanes");
    }

    original_lane                 = ucp_ep_get_lane(ep, 0);
    num_lanes                     = ucp_ep_num_lanes(ep);
    original_second_lane          = ucp_ep_get_lane(ep, 1);
    replacement_second_lane.iface = original_second_lane->iface;
    test_flush_call_count         = 0;
    test_flush_comp               = NULL;
    test_flush_eps[0]             = NULL;
    test_flush_eps[1]             = NULL;
    test_flush_eps[2]             = NULL;
    test_flush_eps[3]             = NULL;
    {
        ucs::mock mock;
        mock_ep_flush(ep, mock, test_flush_inprogress);

        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
        request = ucp_ep_flush_lanes_internal(
                ep, 0, &param, NULL, test_flush_completion,
                "replace_other_lane_during_flush", UCT_FLUSH_FLAG_LOCAL,
                UCS_BIT(0) | UCS_BIT(1));
        EXPECT_TRUE(UCS_PTR_IS_PTR(request));
        EXPECT_EQ(2u, test_flush_call_count);
        EXPECT_EQ(original_lane, test_flush_eps[0]);
        EXPECT_EQ(original_second_lane, test_flush_eps[1]);

        ucp_ep_set_lane(ep, 1, &replacement_second_lane);
        EXPECT_TRUE(test_flush_comp != NULL);
        if (test_flush_comp != NULL) {
            uct_invoke_completion(test_flush_comp, UCS_OK);
            uct_invoke_completion(test_flush_comp, UCS_OK);
        }

        EXPECT_EQ(num_lanes + 2, test_flush_call_count);
        EXPECT_EQ(original_lane, test_flush_eps[2]);
        EXPECT_EQ(&replacement_second_lane, test_flush_eps[3]);

        EXPECT_TRUE(test_flush_comp != NULL);
        if (test_flush_comp != NULL) {
            for (i = 0; i < num_lanes; ++i) {
                uct_invoke_completion(test_flush_comp, UCS_OK);
            }
        }
        ucp_ep_set_lane(ep, 1, original_second_lane);
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    }

    EXPECT_UCS_OK(ucp_request_check_status(request));
    ucp_request_release(request);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_flush, partial_mask_pending_reschedule, "MAX_EAGER_LANES=2")
{
    ucp_request_param_t param = {};
    ucp_request_t *req = NULL;
    ucp_ep_h ep;
    uct_iface_h iface;
    ucs_status_ptr_t request = NULL;

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();
    if (ucp_ep_num_lanes(ep) < 2) {
        disconnect(sender());
        disconnect(receiver());
        UCS_TEST_SKIP_R("requires two endpoint lanes");
    }

    iface = ucp_ep_get_lane(ep, 1)->iface;
    {
        ucs::mock mock;
        mock.setup(&iface->ops.ep_flush, test_flush_no_resource);
        mock.setup(&iface->ops.ep_pending_add, test_flush_pending_add);

        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
        request = ucp_ep_flush_lanes_internal(
                ep, 0, &param, NULL, test_flush_completion,
                "partial_mask_pending_reschedule", UCT_FLUSH_FLAG_LOCAL,
                UCS_BIT(1));
        EXPECT_TRUE(UCS_PTR_IS_PTR(request));
        if (UCS_PTR_IS_PTR(request)) {
            req = static_cast<ucp_request_t*>(request) - 1;

            EXPECT_EQ(0, req->send.flush.started_lanes);
            ep->flags |= UCP_EP_FLAG_BLOCK_FLUSH;
            EXPECT_EQ(UCS_OK, ucp_ep_flush_progress_pending(&req->send.uct));
            ep->flags &= ~UCP_EP_FLAG_BLOCK_FLUSH;
            ucp_ep_flush_request_ff(req, UCS_ERR_CANCELED);
        }
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
    }

    if (UCS_PTR_IS_PTR(request)) {
        EXPECT_EQ(UCS_ERR_CANCELED, ucp_request_check_status(request));
        ucp_request_release(request);
    }

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_flush_failover, restart_pending_skips_stale_resume,
           "MAX_EAGER_LANES=2")
{
    ucp_ep_h ep;
    unsigned initial_flush_count;

    ucp_request_param_t param = {};
    ucp_request_t *req        = NULL;
    ucs_status_ptr_t request  = NULL;

    sender().connect(&receiver(), get_ep_params());
    ep = sender().ep();
    if (ucp_ep_num_lanes(ep) < 2) {
        disconnect(sender());
        disconnect(receiver());
        UCS_TEST_SKIP_R("requires two endpoint lanes");
    }

    test_flush_call_count     = 0;
    test_flush_comp           = NULL;
    test_flush_no_resource_ep = ucp_ep_get_lane(ep, 1);
    {
        ucs::mock mock;
        mock_ep_flush(ep, mock, test_flush_inprogress_except_ep);
        mock.setup(&test_flush_no_resource_ep->iface->ops.ep_pending_add,
                   test_flush_pending_add);

        {
            test_flush_worker_cs_guard cs_guard(ep->worker);

            request = ucp_ep_flush_lanes_internal(
                    ep, 0, &param, NULL, test_flush_completion,
                    "restart_pending_skips_stale_resume", UCT_FLUSH_FLAG_LOCAL,
                    UCS_BIT(1));
            ASSERT_TRUE(UCS_PTR_IS_PTR(request));
            req = static_cast<ucp_request_t*>(request) - 1;
            initial_flush_count = test_flush_call_count;
            ASSERT_GT(initial_flush_count, 0u);

            ep->flags |= UCP_EP_FLAG_BLOCK_FLUSH;
            ASSERT_UCS_OK(ucp_ep_flush_progress_pending(&req->send.uct));
            ep->flags &= ~UCP_EP_FLAG_BLOCK_FLUSH;
            req->send.flush.sw_state = UCP_EP_FLUSH_SW_STATE_RESTART_PENDING;
        }

        sender().progress();
        EXPECT_EQ(initial_flush_count, test_flush_call_count);

        {
            test_flush_worker_cs_guard cs_guard(ep->worker);

            req->send.flush.sw_state = UCP_EP_FLUSH_SW_STATE_NOT_STARTED;
            req->send.flush.uct_flags_orig |= UCT_FLUSH_FLAG_CANCEL;
            req->send.lane = UCP_NULL_LANE;
            ucp_ep_flush_request_ff(req, UCS_ERR_CANCELED);
        }
    }

    EXPECT_EQ(UCS_ERR_CANCELED, ucp_request_check_status(request));
    ucp_request_release(request);

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_flush_failover, completion_error_restarts_flush)
{
    ucp_request_param_t param = {};
    ucp_request_t *req = NULL;
    ucp_ep_h ep;
    uct_iface_h iface;
    ucs_status_ptr_t request = NULL;
    unsigned i;

    sender().connect(&receiver(), get_ep_params());
    flush_ep(sender());
    ep                    = sender().ep();
    iface                 = ucp_ep_get_lane(ep, 0)->iface;
    test_flush_call_count = 0;
    test_flush_comp       = NULL;
    {
        ucs::mock mock;
        mock.setup(&iface->ops.ep_flush, test_flush_inprogress);

        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
        request = ucp_ep_flush_lanes_internal(
                ep, 0, &param, NULL, test_flush_completion,
                "completion_error_restarts_flush", UCT_FLUSH_FLAG_LOCAL,
                UCS_BIT(0));
        EXPECT_TRUE(UCS_PTR_IS_PTR(request));
        if (UCS_PTR_IS_PTR(request)) {
            req = static_cast<ucp_request_t*>(request) - 1;
            EXPECT_TRUE(test_flush_comp != NULL);
            if (test_flush_comp != NULL) {
                uct_invoke_completion(test_flush_comp, UCS_ERR_ENDPOINT_TIMEOUT);
                EXPECT_EQ(UCP_EP_FLUSH_SW_STATE_RESTART_PENDING,
                          req->send.flush.sw_state);
            }
        }
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

        if (req != NULL) {
            for (i = 0; i < 100; ++i) {
                sender().progress();
                if (test_flush_call_count >= 2) {
                    break;
                }
            }

            EXPECT_GE(test_flush_call_count, 2u);
            EXPECT_TRUE(test_flush_comp != NULL);
            if (test_flush_comp != NULL) {
                UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
                for (i = 0; (i < 100) &&
                            (req->send.state.uct_comp.count > 0); ++i) {
                    uct_invoke_completion(test_flush_comp, UCS_OK);
                }
                EXPECT_EQ(0, req->send.state.uct_comp.count);
                ucp_ep_flush_remote_completed(req);
                UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);
            }
        }
    }

    if (UCS_PTR_IS_PTR(request)) {
        EXPECT_UCS_OK(ucp_request_check_status(request));
        ucp_request_release(request);
    }

    disconnect(sender());
    disconnect(receiver());
}

UCS_TEST_P(test_ucp_flush_failover,
           restart_pending_removes_remote_completion_request)
{
    ucp_request_param_t param = {};
    ucp_ep_flush_state_t *flush_state;
    ucp_request_t *req        = NULL;
    ucp_ep_h ep;
    ucs_status_ptr_t request  = NULL;
    uint32_t send_sn;

    sender().connect(&receiver(), get_ep_params());
    flush_ep(sender());
    ep          = sender().ep();
    flush_state = ucp_ep_flush_state(ep);
    send_sn     = flush_state->send_sn;
    ++flush_state->send_sn;

    test_flush_call_count = 0;
    test_flush_comp       = NULL;
    {
        ucs::mock mock;
        mock_ep_flush(ep, mock, test_flush_inprogress);

        {
            test_flush_worker_cs_guard cs_guard(ep->worker);

            request = ucp_ep_flush_lanes_internal(
                    ep, 0, &param, NULL, test_flush_completion,
                    "restart_pending_removes_remote_completion_request",
                    0, UCS_BIT(0));
            ASSERT_TRUE(UCS_PTR_IS_PTR(request));
            req = static_cast<ucp_request_t*>(request) - 1;
            ASSERT_TRUE(test_flush_comp != NULL);

            uct_invoke_completion(test_flush_comp, UCS_OK);
            ASSERT_EQ(UCP_EP_FLUSH_SW_STATE_STARTED,
                      req->send.flush.sw_state);
            ASSERT_EQ(0, req->send.flush.sw_done);
            ASSERT_EQ(1ul, ucs_hlist_length(&flush_state->reqs));

            /* Model a reissued lane flush which fails after the software
             * completion stage has queued this request. */
            req->send.state.uct_comp.count = 1;
            uct_invoke_completion(test_flush_comp, UCS_ERR_ENDPOINT_TIMEOUT);
            ASSERT_EQ(UCP_EP_FLUSH_SW_STATE_RESTART_PENDING,
                      req->send.flush.sw_state);
        }

        sender().progress();
        EXPECT_TRUE(ucs_hlist_is_empty(&flush_state->reqs));

        {
            test_flush_worker_cs_guard cs_guard(ep->worker);

            /* The mock does not register the reissued UCT flush, so complete
             * the request explicitly after verifying the queue invariant. */
            req->status = UCS_ERR_CANCELED;
            test_flush_completion(req);
        }
    }

    EXPECT_EQ(UCS_ERR_CANCELED, ucp_request_check_status(request));
    ucp_request_release(request);

    flush_state->send_sn = send_sn;
    disconnect(sender());
    disconnect(receiver());
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_flush, self, "self")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_flush, shm_ib, "shm,ib")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_flush_failover, shm_ib, "shm,ib")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_flush_failover, shm_tcp, "shm,tcp")
