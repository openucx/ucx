/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
#include <algorithm>
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
        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_PUT,
                               op_name(TEST_OP_PUT));
        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_PUT | TEST_OP_FLUSH,
                               op_name(TEST_OP_PUT | TEST_OP_FLUSH));
        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_GET,
                               op_name(TEST_OP_GET));
        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_GET | TEST_OP_FLUSH,
                               op_name(TEST_OP_GET | TEST_OP_FLUSH));
        add_variant_with_value(variants, UCP_FEATURE_AM,  TEST_OP_AM,
                               op_name(TEST_OP_AM));
        add_variant_with_value(variants, UCP_FEATURE_AM,  TEST_OP_AM | TEST_OP_FLUSH,
                               op_name(TEST_OP_AM | TEST_OP_FLUSH));

        add_variant_with_value(variants, UCP_FEATURE_AM | UCP_FEATURE_RMA,
                               TEST_OP_PUT | TEST_OP_AM | TEST_OP_FLUSH,
                               op_name(TEST_OP_PUT |TEST_OP_AM | TEST_OP_FLUSH));
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
        TEST_OP_PUT   = UCS_BIT(0),
        TEST_OP_GET   = UCS_BIT(1),
        TEST_OP_AM    = UCS_BIT(2),
        TEST_OP_FLUSH = UCS_BIT(3),
    };

    void init() override {
        ucp_test::init();

        ucp_ep_params_t ep_params = get_ep_params();
        sender().connect(&receiver(), ep_params, GOOD_EP_INDEX);
        sender().connect(&receiver(), ep_params, INJECTED_EP_INDEX);
        receiver().connect(&sender(), ep_params, GOOD_EP_INDEX);
        receiver().connect(&sender(), ep_params, INJECTED_EP_INDEX);

        if (get_variant_value() & TEST_OP_AM) {
            set_am_handler();
        }
    }

    void set_am_handler() {
        ucp_am_handler_param_t param;
        param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                           UCP_AM_HANDLER_PARAM_FIELD_CB |
                           UCP_AM_HANDLER_PARAM_FIELD_ARG;
        param.id         = AM_ID;
        param.cb         = am_recv_cb;
        param.arg        = reinterpret_cast<void*>(this);

        ucs_status_t status = ucp_worker_set_am_recv_handler(receiver().worker(), &param);
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
            memcpy(self->m_am_rbuf.data(), data, length);
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
                            std::to_string(lanes.size()) + " available");
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

    std::vector<ucp_lane_index_t> get_lanes(unsigned op_mask) {
        std::set<ucp_lane_index_t> tmp_lanes;
        const ucp_lane_index_t *lane_idx;
        const ucp_lane_index_t *lanes_key_p;
        
        if (op_mask & (TEST_OP_PUT | TEST_OP_GET)) {
            lanes_key_p = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.rma_bw_lanes;

            ucs_carray_for_each(lane_idx, lanes_key_p, UCP_MAX_LANES) {
                if (*lane_idx != UCP_NULL_LANE) {
                    tmp_lanes.insert(*lane_idx);
                }
            }
        }

        if (op_mask & TEST_OP_AM) {
            lanes_key_p = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.am_bw_lanes;

            ucs_carray_for_each(lane_idx, lanes_key_p, UCP_MAX_LANES) {
                if (*lane_idx != UCP_NULL_LANE) {
                    tmp_lanes.insert(*lane_idx);
                }
            }   
        }

        std::vector<ucp_lane_index_t> lanes(tmp_lanes.begin(), tmp_lanes.end());
        shuffle_lanes(lanes, op_name(op_mask) + " lanes");
        return lanes;
    }

    /**
     * Common helper function to test PUT, AM and FLUSH operations with injected failure
     */
    void test_put_am_flush_with_injected_failure(failure_side_t failure_side, unsigned op_mask) {
        const std::string op_str = op_name(op_mask);

        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();

        std::vector<ucp_lane_index_t> lanes = get_lanes(op_mask);

        size_t size = rma_msg_size();
        mem_buffer lbuf(size, UCS_MEMORY_TYPE_HOST);
        mapped_buffer rbuf(size, receiver());
        ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());

        ucp_ep_h ucp_ep_for_injection = get_ucp_ep_for_err_injection(failure_side);
        for (size_t lane_idx = 0; lane_idx < lanes.size() - 1; ++lane_idx) {
            std::vector<ucs_status_ptr_t> status_ptrs;
            ucp_lane_index_t lane = lanes[lane_idx];
            uct_ep_h uct_ep_for_injection = ucp_ep_get_lane(ucp_ep_for_injection, lane);
            ucs_status_t status = uct_ep_invalidate(uct_ep_for_injection, 0);
            if (status == UCS_ERR_UNSUPPORTED) {
                UCS_TEST_SKIP_R("uct_ep_invalidate is not supported");
            }

            EXPECT_EQ(UCS_OK, status) << "uct_ep_invalidate returned status: "
                                      << ucs_status_string(status);

            UCS_TEST_MESSAGE << "Attempting " << op_str
                             << " operation after failure injection on lane "
                             << size_t(lane) << '/' << lanes.size() << "...";

            status_ptrs.push_back(
                    ucp_put_nbx(sender().ep(0, INJECTED_EP_INDEX), lbuf.ptr(), size,
                    uintptr_t(rbuf.ptr()), rkey, &m_req_empty_param));
            status_ptrs.push_back(
                    ucp_am_send_nbx(sender().ep(0, INJECTED_EP_INDEX), AM_ID, NULL, 0,
                                       lbuf.ptr(), am_msg_size(), &m_req_empty_param));
            status_ptrs.push_back(
                    ucp_ep_flush_nbx(sender().ep(0, INJECTED_EP_INDEX), &m_req_empty_param));

            status = requests_wait(status_ptrs);
            EXPECT_EQ(UCS_OK, status) << "PUT, AM and FLUSH operations completed with status: "
                                      << ucs_status_string(status);

            // Check that no other lanes have been affected
            for (ucp_lane_index_t valid_lane = lane_idx + 1; valid_lane < lanes.size();
                 ++valid_lane) {
                const ucp_ep_config_t *ep_config = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX));
                ASSERT_FALSE(UCS_BIT(UCP_LANE_TYPE_FAILED) &
                             ep_config->key.lanes[lanes[valid_lane]].lane_types)
                    << "Lane " << size_t(valid_lane) << " has being marked as failed after "
                    << "failure injection on lane " << size_t(lane);
            }
        }

        short_progress_loop();
        ASSERT_EQ(0, m_err_count) << "Error callback invoked " << m_err_count << " times";
        UCS_TEST_MESSAGE << "Success";
    }

    /**
     * Common helper function to test AM send with injected failure
     */
    void test_am_with_injected_failure(failure_side_t failure_side, unsigned op_mask) {
        const std::string op_str = op_name(op_mask);

        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();

        std::vector<ucp_lane_index_t> am_bw_lanes = get_lanes(op_mask);

        UCS_TEST_MESSAGE << "Attempting " << op_str << " operation before failure injection...";
        ucs_status_t status = do_am_send_and_wait(sender().ep(0, INJECTED_EP_INDEX), am_msg_size(),
                                                  op_mask & TEST_OP_FLUSH);
        EXPECT_EQ(UCS_OK, status) << op_str << " operation returned status: "
                                  << ucs_status_string(status);

        ucp_ep_h ucp_ep_for_injection = get_ucp_ep_for_err_injection(failure_side);
        for (size_t lane_idx = 0; lane_idx < am_bw_lanes.size(); ++lane_idx) {
            ucp_lane_index_t lane = am_bw_lanes[lane_idx];
            uct_ep_h uct_ep_for_injection = ucp_ep_get_lane(ucp_ep_for_injection, lane);
            status = uct_ep_invalidate(uct_ep_for_injection, 0);
            if (status == UCS_ERR_UNSUPPORTED) {
                UCS_TEST_SKIP_R("uct_ep_invalidate is not supported");
            }

            EXPECT_EQ(UCS_OK, status) << "uct_ep_invalidate returned status: "
                                      << ucs_status_string(status);

            UCS_TEST_MESSAGE << "Attempting " << op_str
                             << " operation after failure injection on lane "
                             << size_t(lane) << '/' << am_bw_lanes.size() << "...";

            std::unique_ptr<scoped_log_handler> slh;
            if (lane_idx == (am_bw_lanes.size() - 1)) {
                slh.reset(new scoped_log_handler(hide_errors_logger));
            }

            status = do_am_send_and_wait(sender().ep(0, INJECTED_EP_INDEX), am_msg_size(),
                                         op_mask & TEST_OP_FLUSH);
            if (lane_idx < (am_bw_lanes.size() - 1)) {
                EXPECT_EQ(UCS_OK, status) << op_str << " operation returned status: "
                                        << ucs_status_string(status);
                ASSERT_EQ(0, m_err_count) << "Error callback invoked " << m_err_count << " times";
            } else {
                // The last lane is expected to fail
                short_progress_loop();
                if ((failure_side == FAILURE_SIDE_TARGET) &&
                    has_transport("dc_x")) {
                    // DC transport is not able to detect failure of remote DCI since DC is a connect2iface transport.
                    // This is a test limitation.
                } else {
                    ASSERT_EQ(1, m_err_count) << "Error callback invoked " << m_err_count << " times";
                }
            }
        }

        UCS_TEST_MESSAGE << "Success";
    }

    /**
     * Common helper function to test RMA operation with injected failure
     */
    void test_rma_with_injected_failure(failure_side_t failure_side, unsigned op_mask) {
        const size_t size        = rma_msg_size();
        const std::string op_str = op_name(op_mask);

        /* TODO: cover case when wireup is in progress, flush here is to complete wireup */
        flush_workers();

        std::vector<ucp_lane_index_t> rma_bw_lanes = get_lanes(op_mask);

        mem_buffer lbuf(size, UCS_MEMORY_TYPE_HOST);
        mapped_buffer rbuf(size, receiver());
        ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());

        if (op_mask & TEST_OP_PUT) {
            lbuf.pattern_fill(m_seed);
        } else {
            ASSERT_TRUE(op_mask & TEST_OP_GET);
            rbuf.pattern_fill(m_seed);
        }

        UCS_TEST_MESSAGE << "Attempting " << op_str << " operation before failure injection...";
        ucs_status_t status = do_rma_and_wait(sender().ep(0, INJECTED_EP_INDEX), op_mask,
                                              lbuf, rbuf, rkey.get(), size);
        EXPECT_EQ(UCS_OK, status) << op_str << " operation returned status: "
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

            UCS_TEST_MESSAGE << "Attempting " << op_str
                             << " operation after failure injection on lane "
                             << size_t(lane) << '/' << rma_bw_lanes.size() << "...";
            status = do_rma_and_wait(sender().ep(0, INJECTED_EP_INDEX), op_mask, lbuf, rbuf,
                                     rkey.get(), size);
            EXPECT_EQ(UCS_OK, status) << op_str << " operation returned status: "
                                    << ucs_status_string(status);

            for (ucp_lane_index_t valid_lane = lane_idx + 1; valid_lane < rma_bw_lanes.size();
                 ++valid_lane) {
                const ucp_ep_config_t *ep_config = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX));
                ASSERT_FALSE(UCS_BIT(UCP_LANE_TYPE_FAILED) &
                             ep_config->key.lanes[valid_lane].lane_types)
                    << "Lane " << size_t(valid_lane) << " has being marked as failed after "
                    << "failure injection on lane " << size_t(lane);
            }
        }

        short_progress_loop();
        ASSERT_EQ(0, m_err_count) << "Error callback invoked " << m_err_count << " times";
        UCS_TEST_MESSAGE << "Success";
    }

    void do_test(failure_side_t failure_side) {
        const unsigned op_mask = get_variant_value();

        if (ucs_test_all_flags(op_mask, TEST_OP_PUT | TEST_OP_AM | TEST_OP_FLUSH)) {
            test_put_am_flush_with_injected_failure(failure_side, op_mask);
        } else if (op_mask & TEST_OP_AM) {
            ASSERT_FALSE(op_mask & (TEST_OP_PUT|TEST_OP_GET));
            test_am_with_injected_failure(failure_side, op_mask);
        } else {
            ASSERT_TRUE(op_mask & (TEST_OP_PUT|TEST_OP_GET));
            test_rma_with_injected_failure(failure_side, op_mask);
        }
    }
private:
    static size_t rma_msg_size() {
        return ucs::limit_buffer_size((1000 * UCS_MBYTE) / ucs::test_time_multiplier());
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

        if (op_mask & TEST_OP_AM) {
            name += "AM|";
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
        if (flush_after) {
            ucs_status_t status = request_wait(ucp_ep_flush_nbx(ep, &param));
            if (status != UCS_OK) {
                request_wait(sptr);
                return status;
            }
        }

        ucs_status_t status = request_wait(sptr);
        if (status != UCS_OK) {
            return status;
        }

        wait_for_value(&m_am_received, true);
        mem_buffer::pattern_check(m_am_rbuf.data(), size, m_seed);
        return UCS_OK;
    }

    ucs_status_t do_put_and_wait(ucp_ep_h ep, mem_buffer &lbuf, mapped_buffer &rbuf,
                                 ucp_rkey_h rkey, size_t size, bool flush) {
        rbuf.memset(0);
        ucs_status_ptr_t put_status_ptr   = ucp_put_nbx(ep, lbuf.ptr(), size, uintptr_t(rbuf.ptr()),
                                                        rkey, &m_req_empty_param);
        ucs_status_ptr_t flush_status_ptr = flush ? ucp_ep_flush_nbx(ep, &m_req_empty_param) : NULL;
        ucs_status_t status               = request_wait(put_status_ptr);
        if (status == UCS_OK) {
            rbuf.pattern_check(m_seed, size);
        }

        EXPECT_EQ(UCS_OK, status) << "put operation returned status: " << ucs_status_string(status);
        if (flush) {
            status = request_wait(flush_status_ptr);
            EXPECT_EQ(UCS_OK, status) << "flush operation returned status: " << ucs_status_string(status);
        }

        return status;
    }

    ucs_status_t do_get_and_wait(ucp_ep_h ep, mem_buffer &lbuf, mapped_buffer &rbuf,
                                 ucp_rkey_h rkey, size_t size, bool flush) {
        ucp_request_param_t param;
        param.op_attr_mask = 0;

        lbuf.memset(0);
        ucs_status_ptr_t status_ptr       = ucp_get_nbx(ep, lbuf.ptr(), size, uintptr_t(rbuf.ptr()), rkey, &param);
        ucs_status_ptr_t flush_status_ptr = flush ? ucp_ep_flush_nbx(ep, &param) : NULL;
        ucs_status_t status               = request_wait(status_ptr);
        EXPECT_EQ(UCS_OK, status) << "get operation returned status: " << ucs_status_string(status);
        if (status == UCS_OK) {
            lbuf.pattern_check(m_seed, size);
        }

        if (flush) {
            status = request_wait(flush_status_ptr);
            EXPECT_EQ(UCS_OK, status) << "flush operation returned status: " << ucs_status_string(status);
        }

        return status;
    }

    ucs_status_t do_rma_and_wait(ucp_ep_h ep, unsigned op_mask, mem_buffer &lbuf, mapped_buffer &rbuf,
                                 ucp_rkey_h rkey, size_t size) {
        if (op_mask & TEST_OP_PUT) {
            return do_put_and_wait(ep, lbuf, rbuf, rkey, size, op_mask & TEST_OP_FLUSH);
        }

        if (op_mask & TEST_OP_GET) {
            return do_get_and_wait(ep, lbuf, rbuf, rkey, size, op_mask & TEST_OP_FLUSH);
        }

        return UCS_ERR_INVALID_PARAM;
    }

protected:
    static constexpr uint64_t m_seed = 0x12345678;

    const ucp_request_param_t m_req_empty_param = { 0 };
    std::vector<uint8_t> m_am_rbuf              = std::vector<uint8_t>(am_msg_size());
    volatile bool m_am_received                 = false;

private:
    size_t       m_err_count  = 0;
    ucs_status_t m_err_status = UCS_OK;
};

UCP_INSTANTIATE_TEST_CASE(test_ucp_fault_tolerance)

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure, "MAX_EAGER_LANES=8")
{
    do_test(FAILURE_SIDE_INITIATOR);
}

UCS_TEST_P(test_ucp_fault_tolerance, target_failure, "MAX_EAGER_LANES=8")
{
    do_test(FAILURE_SIDE_TARGET);
}
