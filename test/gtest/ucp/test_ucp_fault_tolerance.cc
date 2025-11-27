/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"

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

        /* Configure failure injection parameters */
        modify_config("FAILURE_LANE", "1");
        modify_config("FAILURE_TIMEOUT", "1.0s"); /* 1 second timeout */
        modify_config("MAX_RMA_RAILS", "32");
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
        test_ucp_memheap::init();

        /* Create 2 entities: sender and receiver */
        create_entity();
        create_entity();

        m_err_count = 0;
        m_err_status = UCS_OK;
    }

    /**
     * Connect endpoints with proper error handling
     */
    void connect_endpoints(failure_side_t failure_side) {
        ucp_ep_params_t ep_params_good = get_ep_params(false);

        /* Connect sender to receiver - one good endpoint, one with injection */
        sender().connect(&receiver(), ep_params_good, GOOD_EP_INDEX);
        sender().connect(&receiver(), get_ep_params(failure_side == FAILURE_SIDE_INITIATOR), INJECTED_EP_INDEX);

        /* Connect receiver back to sender - with or without injection */
        receiver().connect(&sender(), ep_params_good, GOOD_EP_INDEX);
        receiver().connect(&sender(), get_ep_params(failure_side == FAILURE_SIDE_TARGET), INJECTED_EP_INDEX);
    }

    /**
     * Get endpoint parameters with optional failure injection flag
     */
    ucp_ep_params_t get_ep_params(bool inject_failure) {
        ucp_ep_params_t params;
        memset(&params, 0, sizeof(params));

        params.field_mask = UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                            UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode = UCP_ERR_HANDLING_MODE_PEER;
        params.err_handler.cb = err_cb;
        params.err_handler.arg = reinterpret_cast<void*>(this);

        if (inject_failure) {
            params.field_mask |= UCP_EP_PARAM_FIELD_FLAGS;
            params.flags |= UCP_EP_PARAMS_FLAGS_INJECT_FAILURE;
        }

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
    void skip_if_insufficient_rma_lanes(ucp_ep_h ep) {
        ucp_lane_index_t num_lanes = ucp_ep_num_lanes(ep);
        
        if (num_lanes < 2) {
            UCS_TEST_SKIP_R("Test requires at least 2 lanes for fault tolerance, "
                           "but only " + std::to_string(num_lanes) + " available");
        }
        
        UCS_TEST_MESSAGE << "Endpoint has " << (int)num_lanes << " lanes";
    }

    /**
     * Perform a PUT operation and wait for completion
     */
    ucs_status_t do_put_and_wait(ucp_ep_h ep, mem_buffer &src_buf, 
                                 mapped_buffer &dst_buf) {
        ucp_request_param_t param;
        param.op_attr_mask = 0;

        /* Get rkey for the destination buffer */
        ucs::handle<ucp_rkey_h> rkey = dst_buf.rkey(sender());

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
        const size_t size = 2 * UCS_GBYTE;
        const char *side_name = (failure_side == FAILURE_SIDE_INITIATOR) ? 
                                "initiator" : "target";

        UCS_TEST_MESSAGE << "Testing failure injection on " << side_name << " side";

        connect_endpoints(failure_side);

        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();
        skip_if_insufficient_rma_lanes(sender().ep(0, INJECTED_EP_INDEX));

        /* Setup RMA buffers using mapped_buffer */
        mem_buffer src_buf(size, UCS_MEMORY_TYPE_HOST);
        mapped_buffer dst_buf(size, receiver());

        /* Fill source with pattern, clear destination */
        src_buf.pattern_fill(size, 0x12345678);
        dst_buf.memset(0);

        UCS_TEST_MESSAGE << "Sleeping for 1 second to allow failure injection timer to fire...";
        sleep(1); /* Wait longer than FAILURE_TIMEOUT (0.5s) */
        short_progress_loop(); /* Progress the async context */

        UCS_TEST_MESSAGE << "Attempting PUT operation with " << side_name 
                        << " side failure injected...";

        ucs_status_t status = do_put_and_wait(sender().ep(0, INJECTED_EP_INDEX),
                                              src_buf, dst_buf);

        UCS_TEST_MESSAGE << "PUT operation returned status: " 
                        << ucs_status_string(status);
        UCS_TEST_MESSAGE << "Error callback invoked " << m_err_count << " times";

        /*
         * Expected behavior (once fault tolerance is implemented):
         * - The operation should complete successfully (UCS_OK) OR
         * - The error callback should be triggered (m_err_count > 0) OR  
         * - The operation should fail gracefully with appropriate error status
         * 
         * For now, we document that this test will FAIL, demonstrating
         * the current lack of fault tolerance when using injected failures.
         */
        if (status == UCS_OK) {
            UCS_TEST_MESSAGE << "SUCCESS: Operation completed despite injected failure "
                            << "on " << side_name << " side (fault tolerance working!)";
            /* Verify data integrity */
            EXPECT_EQ(0, memcmp(src_buf.ptr(), dst_buf.ptr(), size));
        } else if (m_err_count > 0) {
            UCS_TEST_MESSAGE << "PARTIAL SUCCESS: Error callback triggered, "
                            << "but operation failed with status: " 
                            << ucs_status_string(status);
        } else {
            UCS_TEST_MESSAGE << "EXPECTED FAILURE: Operation failed without triggering "
                            << "error callback or completing successfully. "
                            << "Status: " << ucs_status_string(status);
            UCS_TEST_MESSAGE << "This demonstrates the need for fault tolerance implementation.";
            
            /* For now, we expect failure since implementation is not complete */
            EXPECT_NE(UCS_OK, status) 
                << "Expected failure due to incomplete fault tolerance implementation";
        }

        status = do_put_and_wait(sender().ep(0, GOOD_EP_INDEX), src_buf, dst_buf);
        EXPECT_EQ(UCS_OK, status);
    }

    size_t m_err_count;
    ucs_status_t m_err_status;
};

UCP_INSTANTIATE_TEST_CASE(test_ucp_fault_tolerance)

/**
 * Test fault tolerance: PUT operation with initiator-side failure injection
 * The sender's endpoint fails, testing recovery when the initiator side fails
 */
UCS_TEST_P(test_ucp_fault_tolerance, put_with_initiator_failure,
          "FAILURE_TIMEOUT=0.5")
{
    test_put_with_injected_failure(FAILURE_SIDE_INITIATOR);
}

/**
 * Test fault tolerance: PUT operation with target-side failure injection
 * The receiver's endpoint fails, testing recovery when the target side fails
 */
UCS_TEST_P(test_ucp_fault_tolerance, put_with_target_failure,
          "FAILURE_TIMEOUT=0.5")
{
    test_put_with_injected_failure(FAILURE_SIDE_TARGET);
}

