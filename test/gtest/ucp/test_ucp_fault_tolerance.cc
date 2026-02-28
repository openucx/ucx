/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
#include <algorithm>
#include <cstring> // for std::memcpy
#include <memory>
#include <random>
#include <string>

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
        add_variant_with_value(variants, UCP_FEATURE_RMA|UCP_FEATURE_AM, 0, "rma|am");
    }

    test_ucp_fault_tolerance() {
        configure_peer_failure_settings();
    }

protected:
    static constexpr uint16_t AM_ID = 0;

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
        TEST_OP_GET,
        TEST_OP_FLUSH
    };

    void init() override {
        ucp_test::init();

        ucp_ep_params_t ep_params = get_ep_params();
        sender().connect(&receiver(), ep_params, GOOD_EP_INDEX);
        sender().connect(&receiver(), ep_params, INJECTED_EP_INDEX);
        receiver().connect(&sender(), ep_params, GOOD_EP_INDEX);
        receiver().connect(&sender(), ep_params, INJECTED_EP_INDEX);

        set_am_handler();
    }

    void set_am_handler() {
        ucp_am_handler_param_t param;
        param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                           UCP_AM_HANDLER_PARAM_FIELD_CB |
                           UCP_AM_HANDLER_PARAM_FIELD_ARG;
        param.id         = AM_ID;
        param.cb         = am_recv_cb;
        param.arg        = reinterpret_cast<void*>(this);

        ucs_status_t status = ucp_worker_set_am_recv_handler(receiver().worker(),
                                                            &param);
        ASSERT_UCS_OK(status);
    }

    static ucs_status_t am_recv_cb(void *arg, const void *header,
                                   size_t header_length, void *data,
                                   size_t length,
                                   const ucp_am_recv_param_t *param) {
        test_ucp_fault_tolerance *self =
            reinterpret_cast<test_ucp_fault_tolerance*>(arg);

        if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
            self->m_am_rbuf.resize(length);
            std::memcpy(self->m_am_rbuf.data(), data, length);
            self->m_am_received = true;
        }

        EXPECT_FALSE(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) <<
                "RNDV is not covered yet";

        return UCS_OK;
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

    static void shuffle_lanes(std::vector<ucp_lane_index_t> &lanes, const std::string &lane_type) {
        if (lanes.size() < 2) {
            UCS_TEST_SKIP_R("At least 2 " + lane_type + "s are required, but only " +
                            std::to_string(lanes.size()) + " " + lane_type + "s available");
        }

        /* Allocate randomizer on heap to avoid exceeding stack frame size limits. */
        std::unique_ptr<std::random_device> rnd_device(new std::random_device);
        std::unique_ptr<std::mt19937> rng(new std::mt19937((*rnd_device)()));
        std::shuffle(lanes.begin(), lanes.end(), *rng);

        for (ucp_lane_index_t lane : lanes) {
            UCS_TEST_MESSAGE << lane_type << ": " << size_t(lane) << "/" << lanes.size();
        }
    }

    ucp_ep_h get_ucp_ep_for_err_injection(failure_side_t failure_side) {
        return (failure_side == FAILURE_SIDE_INITIATOR) ? sender().ep(0, INJECTED_EP_INDEX) :
               receiver().ep(0, INJECTED_EP_INDEX);
    }

    /**
     * Common helper function to test AM send 1KB with injected failure
     */
    void test_am_with_injected_failure(failure_side_t failure_side, bool flush_after = false) {
        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();

        std::vector<ucp_lane_index_t> am_bw_lanes;
        const ucp_lane_index_t *am_bw_lane_idx;
        const ucp_lane_index_t *am_bw_lanes_key_p =
                ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.am_bw_lanes;

        ucs_carray_for_each(am_bw_lane_idx, am_bw_lanes_key_p, UCP_MAX_LANES) {
            if (*am_bw_lane_idx != UCP_NULL_LANE) {
                am_bw_lanes.push_back(*am_bw_lane_idx);
            }
        }

        shuffle_lanes(am_bw_lanes, "AM BW lane");

        UCS_TEST_MESSAGE << "Attempting AM send before failure injection...";
        ucs_status_t status = do_am_send_and_wait(sender().ep(0, INJECTED_EP_INDEX), am_msg_size(),
                                                  flush_after);
        EXPECT_EQ(UCS_OK, status) << "AM send returned status: " << ucs_status_string(status);

        ucp_ep_h ucp_ep_for_injection = get_ucp_ep_for_err_injection(failure_side);
        for (size_t lane_idx = 0; lane_idx < am_bw_lanes.size() - 1; ++lane_idx) {
            ucp_lane_index_t lane = am_bw_lanes[lane_idx];
            uct_ep_h uct_ep_for_injection = ucp_ep_get_lane(ucp_ep_for_injection, lane);
            status = uct_ep_invalidate(uct_ep_for_injection, 0);
            if (status == UCS_ERR_UNSUPPORTED) {
                UCS_TEST_SKIP_R("uct_ep_invalidate is not supported");
            }

            EXPECT_EQ(UCS_OK, status) << "uct_ep_invalidate returned status: "
                                      << ucs_status_string(status);

            UCS_TEST_MESSAGE << "Attempting AM send after failure injection on lane "
                             << size_t(lane) << '/' << am_bw_lanes.size() << "...";
            status = do_am_send_and_wait(sender().ep(0, INJECTED_EP_INDEX), am_msg_size(), flush_after);
            EXPECT_EQ(UCS_OK, status) << "AM send returned status: " << ucs_status_string(status);
        }

        short_progress_loop();
        ASSERT_EQ(0, m_err_count) << "Error callback invoked " << m_err_count << " times";
        UCS_TEST_MESSAGE << "Success";
    }

    /**
     * Common helper function to test RMA operation with injected failure
     */
    void test_rma_with_injected_failure(failure_side_t failure_side, test_op_t op) {
        const size_t size = rma_msg_size();
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

        shuffle_lanes(rma_bw_lanes, "RMA BW lane");

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

        ucp_ep_h ucp_ep_for_injection = get_ucp_ep_for_err_injection(failure_side);
        for (size_t lane_idx = 0; lane_idx < rma_bw_lanes.size() - 1; ++lane_idx) {
            ucp_lane_index_t lane = rma_bw_lanes[lane_idx];
            uct_ep_h uct_ep_for_injection = ucp_ep_get_lane(ucp_ep_for_injection, lane);
            status = uct_ep_invalidate(uct_ep_for_injection, 0);
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
    static size_t rma_msg_size() {
        return ucs::limit_buffer_size((100 * UCS_MBYTE) /
                                    ucs::test_time_multiplier());
    }

    static size_t am_msg_size() {
        return ucs::limit_buffer_size(UCS_KBYTE);
    }

    static std::string op_name(unsigned op_mask)
    {
        std::string name;

        if (op_mask & TEST_OP_PUT) {
            name += "PUT|";
        }

        if (op_mask & TEST_OP_GET) {
            name += "GET|";
        }

        if (op_mask & TEST_OP_FLUSH) {
            name += "FLUSH|";
        }

        if (!name.empty()) {
            name.pop_back();
        }

        return name;
    }

    ucs_status_t do_am_send_and_wait(ucp_ep_h ep, size_t size, bool flush_after) {
        m_am_received = false;

        mem_buffer sbuf(size, UCS_MEMORY_TYPE_HOST);
        sbuf.pattern_fill(m_seed, size);

        ucp_request_param_t param;
        param.op_attr_mask = 0;

        ucs_status_ptr_t sptr = ucp_am_send_nbx(ep, AM_ID, NULL, 0, sbuf.ptr(),
                                                size, &param);
        // TODO: enable flush_after when PR #11210 is merged
        if (false && flush_after) {
            ucs_status_t status = request_wait(ucp_ep_flush_nbx(ep, &param));
            if (status != UCS_OK) {
                return status;
            }
        }

        ucs_status_t status = request_wait(sptr);
        if (status != UCS_OK) {
            return status;
        }

        while (!m_am_received) {
            short_progress_loop();
        }

        mem_buffer::pattern_check(m_am_rbuf.data(), size, m_seed);
        return UCS_OK;
    }

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

protected:
    static constexpr uint64_t m_seed = 0x12345678;

    std::vector<uint8_t> m_am_rbuf = std::vector<uint8_t>(am_msg_size());
    volatile bool m_am_received    = false;

private:
    size_t       m_err_count  = 0;
    ucs_status_t m_err_status = UCS_OK;
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

UCS_TEST_P(test_ucp_fault_tolerance, am_send_with_initiator_failure, "MAX_EAGER_LANES=8", "ZCOPY_THRESH=0")
{
    test_am_with_injected_failure(FAILURE_SIDE_INITIATOR);
}

UCS_TEST_P(test_ucp_fault_tolerance, am_send_with_target_failure, "MAX_EAGER_LANES=8", "ZCOPY_THRESH=0")
{
    test_am_with_injected_failure(FAILURE_SIDE_TARGET);
}

UCS_TEST_P(test_ucp_fault_tolerance, am_send_flush_with_target_failure, "MAX_EAGER_LANES=8", "ZCOPY_THRESH=0")
{
    test_am_with_injected_failure(FAILURE_SIDE_TARGET, true);
}
