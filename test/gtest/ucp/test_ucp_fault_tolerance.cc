/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
#include <algorithm>
#include <memory>
#include <random>

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_context.h>
}

/**
 * Test class for fault tolerance with injected failures
 */
class test_ucp_fault_tolerance : public test_ucp_memheap {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        add_variant_with_value(variants, UCP_FEATURE_RMA, 0, "rma");
    }

    test_ucp_fault_tolerance() {
        configure_peer_failure_settings();
    }

protected:
    enum {
        GOOD_EP_INDEX = 0,      /* Index for good endpoint */
        INJECTED_EP_INDEX = 1   /* Index for failure-injected endpoint */
    };

    enum failure_side_t {
        FAILURE_SIDE_INITIATOR, /* Inject failure on sender (initiator) side */
        FAILURE_SIDE_TARGET     /* Inject failure on receiver (target) side */
    };

    enum test_op_t {
        TEST_OP_PUT,
        TEST_OP_GET
    };

    void init() override {
        ucp_test::init();

        ucp_ep_params_t ep_params = get_ep_params();
        sender().connect(&receiver(), ep_params, GOOD_EP_INDEX);
        sender().connect(&receiver(), ep_params, INJECTED_EP_INDEX);
        receiver().connect(&sender(), ep_params, GOOD_EP_INDEX);
        receiver().connect(&sender(), ep_params, INJECTED_EP_INDEX);
    }

    /**
     * Get endpoint parameters with optional failure injection flag
     */
    ucp_ep_params_t get_ep_params() override {
        ucp_ep_params_t params = test_ucp_memheap::get_ep_params();

        params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode        = UCP_ERR_HANDLING_MODE_FAILOVER;
        params.err_handler.cb  = err_cb;
        params.err_handler.arg = reinterpret_cast<void*>(this);

        return params;
    }

    /**
     * Error callback for endpoint failures
     */
    static void err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
        test_ucp_fault_tolerance *self =
            reinterpret_cast<test_ucp_fault_tolerance*>(arg);

        UCS_TEST_MESSAGE << "Error callback invoked: " << ucs_status_string(status);

        EXPECT_TRUE((UCS_ERR_CONNECTION_RESET == status) ||
                    (UCS_ERR_ENDPOINT_TIMEOUT == status) ||
                    (UCS_ERR_CANCELED == status));

        self->m_err_status = status;
        ++self->m_err_count;
    }

    /**
     * Check if we have at least 2 RMA lanes, skip test if not
     */
    void skip_if_insufficient_rma_lanes(ucp_ep_h ep, ucp_lane_index_t failure_lane) {
        ucp_lane_index_t num_lanes = ucp_ep_num_lanes(ep);

        if (num_lanes <= failure_lane) {
            UCS_TEST_SKIP_R("Only " + std::to_string(int(num_lanes)) + " / " + std::to_string(int(failure_lane + 1)) + "lanes available");
        } else {
            UCS_TEST_MESSAGE << "Endpoint has " << int(num_lanes) << " lanes, failure lane is " << int(failure_lane);
        }
    }

    /**
     * Common helper function to test RMA operation with injected failure
     */
    void test_rma_with_injected_failure(failure_side_t failure_side, test_op_t op) {
        const size_t size   = 1 * UCS_GBYTE;
        const char *op_name = (op == TEST_OP_PUT) ? "PUT" : "GET";

        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();

        std::vector<ucp_lane_index_t> rma_bw_lanes;
        ucp_lane_index_t *rma_bw_lane_idx;
        ucs_carray_for_each(rma_bw_lane_idx,
                            ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.rma_bw_lanes,
                            UCP_MAX_LANES) {
            if (*rma_bw_lane_idx != UCP_NULL_LANE) {
                rma_bw_lanes.push_back(*rma_bw_lane_idx);
            }
        }

        if (rma_bw_lanes.size() < 2) {
            UCS_TEST_SKIP_R("At least 2 RMA BW lanes are required, but only " +
                            std::to_string(rma_bw_lanes.size()) + " available");
        }

        { // allocate randomizer on heap to avoid exceeding stack frame size limits
            std::unique_ptr<std::random_device> rnd_device(new std::random_device);
            std::unique_ptr<std::mt19937> rng(new std::mt19937((*rnd_device)()));
            std::shuffle(rma_bw_lanes.begin(), rma_bw_lanes.end(), *rng);
        }

        for (ucp_lane_index_t lane : rma_bw_lanes) {
            UCS_TEST_MESSAGE << "RMA BW lane: " << size_t(lane) << "/" << rma_bw_lanes.size();
        }

        mem_buffer lbuf(size, UCS_MEMORY_TYPE_HOST);
        mapped_buffer rbuf(size, receiver());
        ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());

        if (op == TEST_OP_PUT) {
            lbuf.pattern_fill(m_seed);
        } else if (op == TEST_OP_GET) {
            rbuf.pattern_fill(m_seed);
        } else {
            UCS_TEST_ABORT("Invalid operation type");
        }

        UCS_TEST_MESSAGE << "Attempting " << op_name << " operation before failure injection...";
        ucs_status_t status = do_rma_and_wait(sender().ep(0, INJECTED_EP_INDEX), op, lbuf, rbuf,
                                              rkey.get(), size);
        EXPECT_EQ(UCS_OK, status) << op_name << " operation returned status: "
                                  << ucs_status_string(status);

        ucp_ep_h injected_ucp_ep = (failure_side == FAILURE_SIDE_INITIATOR) ?
                                   sender().ep(0, INJECTED_EP_INDEX) :
                                   receiver().ep(0, INJECTED_EP_INDEX);
        for (size_t lane_idx = 0; lane_idx < rma_bw_lanes.size() - 1; ++lane_idx) {
            ucp_lane_index_t lane = rma_bw_lanes[lane_idx];
            uct_ep_h injected_uct_ep = ucp_ep_get_lane(injected_ucp_ep, lane);
            status = uct_ep_invalidate(injected_uct_ep, 0);
            if (status == UCS_ERR_UNSUPPORTED) {
                UCS_TEST_SKIP_R("uct_ep_invalidate is not supported");
            }

            EXPECT_EQ(UCS_OK, status) << "uct_ep_invalidate returned status: "
                                    << ucs_status_string(status);

            UCS_TEST_MESSAGE << "Attempting " << op_name << " operation after failure injection on lane "
                             << size_t(lane) << '/' << rma_bw_lanes.size() << "...";
            status = do_rma_and_wait(sender().ep(0, INJECTED_EP_INDEX),
                                     op, lbuf, rbuf, rkey.get(), size);
            EXPECT_EQ(UCS_OK, status) << op_name << " operation returned status: "
                                    << ucs_status_string(status);
        }

        short_progress_loop();
        ASSERT_EQ(0, m_err_count) << "Error callback invoked " << m_err_count << " times";
        UCS_TEST_MESSAGE << "Success";
    }

private:
    ucs_status_t do_put_and_wait(ucp_ep_h ep, mem_buffer &lbuf, mapped_buffer &rbuf,
                                 ucp_rkey_h rkey, size_t size) {
        ucp_request_param_t param;
        param.op_attr_mask = 0;

        rbuf.memset(0);
        ucs_status_ptr_t status_ptr = ucp_put_nbx(ep, lbuf.ptr(), size, uintptr_t(rbuf.ptr()), rkey,
                                                  &param);
        ucs_status_t status         = request_wait(status_ptr);
        if (status == UCS_OK) {
            rbuf.pattern_check(m_seed, size);
        }

        return status;
    }

    ucs_status_t do_get_and_wait(ucp_ep_h ep, mem_buffer &lbuf, mapped_buffer &rbuf,
                                 ucp_rkey_h rkey, size_t size) {
        ucp_request_param_t param;
        param.op_attr_mask = 0;

        lbuf.memset(0);
        ucs_status_ptr_t status_ptr = ucp_get_nbx(ep, lbuf.ptr(), size, uintptr_t(rbuf.ptr()), rkey,
                                                  &param);
        ucs_status_t status         = request_wait(status_ptr);
        if (status == UCS_OK) {
            lbuf.pattern_check(m_seed, size);
        }

        return status;
    }

    ucs_status_t do_rma_and_wait(ucp_ep_h ep, test_op_t op, mem_buffer &lbuf, mapped_buffer &rbuf,
                                 ucp_rkey_h rkey, size_t size) {
        switch (op) {
            case TEST_OP_PUT:
                return do_put_and_wait(ep, lbuf, rbuf, rkey, size);
            case TEST_OP_GET:
                return do_get_and_wait(ep, lbuf, rbuf, rkey, size);
            default:
                UCS_TEST_ABORT("Invalid operation type");
                return UCS_ERR_INVALID_PARAM;
        }
    }

    size_t       m_err_count       = 0;
    ucs_status_t m_err_status      = UCS_OK;
    static constexpr uint64_t m_seed = 0x12345678;
};

UCP_INSTANTIATE_TEST_CASE(test_ucp_fault_tolerance)

UCS_TEST_P(test_ucp_fault_tolerance, put_with_initiator_failure)
{
    test_rma_with_injected_failure(FAILURE_SIDE_INITIATOR, TEST_OP_PUT);
}

UCS_TEST_P(test_ucp_fault_tolerance, put_with_target_failure)
{
    test_rma_with_injected_failure(FAILURE_SIDE_TARGET, TEST_OP_PUT);
}

UCS_TEST_P(test_ucp_fault_tolerance, get_with_initiator_failure)
{
    test_rma_with_injected_failure(FAILURE_SIDE_INITIATOR, TEST_OP_GET);
}

UCS_TEST_P(test_ucp_fault_tolerance, get_with_target_failure)
{
    test_rma_with_injected_failure(FAILURE_SIDE_TARGET, TEST_OP_GET);
}
