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
        params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
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
     * Perform a PUT operation and wait for completion
     */
    ucs_status_t do_put_and_wait(ucp_ep_h ep, mem_buffer &src_buf, 
                                 mapped_buffer &dst_buf, ucp_rkey_h rkey) {
        ucp_request_param_t param;
        param.op_attr_mask = 0;

        /* Issue PUT operation */
        ucs_status_ptr_t status_ptr = ucp_put_nbx(ep, src_buf.ptr(), src_buf.size(),
                                                  (uintptr_t)dst_buf.ptr(),
                                                  rkey, &param);
        return request_wait(status_ptr);
    }


    /**
     * Common helper function to test PUT operation with injected failure
     */
    void test_put_with_injected_failure(failure_side_t failure_side) {
        const uint64_t seed   = 0x12345678;
        const size_t size     = 1 * UCS_GBYTE;

        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();

        std::vector<ucp_lane_index_t> rma_bw_lanes;
        for (ucp_lane_index_t i = 0; i < UCP_MAX_LANES; ++i) {
            if (ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.rma_bw_lanes[i] != UCP_NULL_LANE) {
                rma_bw_lanes.push_back(ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.rma_bw_lanes[i]);
            }
        }

        if (rma_bw_lanes.size() < 2) {
            UCS_TEST_SKIP_R("At least 2 RMA BW lanes are required, but only " + std::to_string(rma_bw_lanes.size()) + " available");
        }

        {
            std::unique_ptr<std::mt19937> rng(new std::mt19937(std::random_device()()));
            std::shuffle(rma_bw_lanes.begin(), rma_bw_lanes.end(), *rng);
        }

        for (ucp_lane_index_t lane : rma_bw_lanes) {
            UCS_TEST_MESSAGE << "RMA BW lane: " << size_t(lane) << "/" << rma_bw_lanes.size();
        }

        /* Setup RMA buffers using mapped_buffer */
        mem_buffer src_buf(size, UCS_MEMORY_TYPE_HOST);
        mapped_buffer dst_buf(size, receiver());
        ucs::handle<ucp_rkey_h> rkey = dst_buf.rkey(sender());

        /* Fill source with pattern, clear destination */
        src_buf.pattern_fill(seed);
        dst_buf.memset(0);

        UCS_TEST_MESSAGE << "Attempting PUT operation before failure injection...";
        ucs_status_t status = do_put_and_wait(sender().ep(0, INJECTED_EP_INDEX), src_buf, dst_buf, rkey.get());
        EXPECT_EQ(UCS_OK, status) << "PUT operation returned status: " 
                                  << ucs_status_string(status);
        UCS_TEST_MESSAGE << "Success";

        ucp_ep_h injected_ucp_ep = (failure_side == FAILURE_SIDE_INITIATOR) ? sender().ep(0, INJECTED_EP_INDEX) :
                                   receiver().ep(0, INJECTED_EP_INDEX);
        for (size_t lane_idx = 0; lane_idx < rma_bw_lanes.size() - 1; ++lane_idx) {
            ucp_lane_index_t lane = rma_bw_lanes[lane_idx];
            uct_ep_h injected_uct_ep = ucp_ep_get_lane(injected_ucp_ep, lane);
            status = uct_ep_invalidate(injected_uct_ep, 0);
            EXPECT_EQ(UCS_OK, status) << "uct_ep_invalidate returned status: "
                                    << ucs_status_string(status);

            dst_buf.memset(0);
            UCS_TEST_MESSAGE << "Attempting PUT operation after failure injection on lane("
                             << lane_idx << '/' << size_t(lane) << ")/" << rma_bw_lanes.size() << "...";
            status = do_put_and_wait(sender().ep(0, INJECTED_EP_INDEX), src_buf, dst_buf, rkey.get());
            EXPECT_EQ(UCS_OK, status) << "PUT operation returned status: "
                                    << ucs_status_string(status);
            UCS_TEST_MESSAGE << "Success";
            dst_buf.pattern_check(seed, size);
            UCS_TEST_MESSAGE << "Data integrity check passed";
        }

        short_progress_loop();
        ASSERT_EQ(0, m_err_count) << "Error callback invoked " << m_err_count << " times";
    }

    size_t       m_err_count = 0;
    ucs_status_t m_err_status = UCS_OK;
};

UCP_INSTANTIATE_TEST_CASE(test_ucp_fault_tolerance)

/**
 * Test fault tolerance: PUT operation with initiator-side failure injection
 * The sender's endpoint fails, testing recovery when the initiator side fails
 */
UCS_TEST_P(test_ucp_fault_tolerance, put_with_initiator_failure, "MAX_RMA_RAILS=32")
{
    test_put_with_injected_failure(FAILURE_SIDE_INITIATOR);
}

/**
 * Test fault tolerance: PUT operation with target-side failure injection
 * The receiver's endpoint fails, testing recovery when the target side fails
 */
UCS_TEST_P(test_ucp_fault_tolerance, put_with_target_failure, "MAX_RMA_RAILS=32")
{
    test_put_with_injected_failure(FAILURE_SIDE_TARGET);
}

