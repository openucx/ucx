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
#include <ucp/proto/proto_multi.h>
#include <ucp/proto/proto_single.h>
#include <uct/api/v2/uct_v2.h>
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
        // reduce UD testing time 
        modify_config("KEEPALIVE_INTERVAL", "0.3s");
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

    enum failover_proto_t {
        TEST_FAILOVER_PROTO_AM_SHORT,
        TEST_FAILOVER_PROTO_AM_SHORT_REPLY,
        TEST_FAILOVER_PROTO_AM_SINGLE_BCOPY,
        TEST_FAILOVER_PROTO_AM_SINGLE_BCOPY_REPLY,
        TEST_FAILOVER_PROTO_AM_MULTI_BCOPY,
        TEST_FAILOVER_PROTO_PUT_SHORT,
        TEST_FAILOVER_PROTO_PUT_BCOPY,
        TEST_FAILOVER_PROTO_PUT_AM_BCOPY,
        TEST_FAILOVER_PROTO_LAST
    };

    enum {
        TEST_FAILOVER_PROTO_FLAG_AM     = UCS_BIT(0),
        TEST_FAILOVER_PROTO_FLAG_SINGLE = UCS_BIT(1),
        TEST_FAILOVER_PROTO_FLAG_REPLY  = UCS_BIT(2)
    };

    struct failover_proto_info_t {
        const char *proto_name;
        uint64_t tl_cap;
        size_t size;
        unsigned flags;
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

        EXPECT_EQ(0ul, header_length);
        EXPECT_FALSE(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV) <<
                "RNDV is not covered yet";
        EXPECT_TRUE(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA);
        EXPECT_EQ(self->m_am_expect_reply,
                  !!(param->recv_attr & UCP_AM_RECV_ATTR_FIELD_REPLY_EP));
        EXPECT_EQ(self->m_am_expect_reply, param->reply_ep != nullptr);
        if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
            const size_t msg_index = self->m_am_recv_count;
            if (self->m_am_expected_count > 0) {
                EXPECT_LT(msg_index, self->m_am_expected_count);
            }
            EXPECT_EQ(self->m_am_expected_size, length);
            mem_buffer::pattern_check(data, length,
                                      m_seed + ((self->m_am_expected_count > 1) ?
                                                msg_index : 0));
            self->m_am_rbuf.resize(length);
            memcpy(self->m_am_rbuf.data(), data, length);
            ++self->m_am_recv_count;
        }

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
        ucp_ep_h sender_ep = self->sender().ep(0, INJECTED_EP_INDEX);

        UCS_TEST_MESSAGE << "Error callback invoked: " << ucs_status_string(status);

        EXPECT_TRUE((UCS_ERR_CONNECTION_RESET == status) ||
                    (UCS_ERR_ENDPOINT_TIMEOUT == status) ||
                    (UCS_ERR_CANCELED == status));

        self->m_err_status = status;
        ++self->m_total_err_count;
        if (ep == sender_ep) {
            ++self->m_initiator_err_count;
        }
    }

    static void shuffle_lanes(std::vector<ucp_lane_index_t> &lanes, const std::string &lane_type) {
        if (lanes.size() < 2) {
            UCS_TEST_SKIP_R("At least 2 " + lane_type + "lanes are required, but only " + std::to_string(lanes.size()) +
                            " available");
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
        std::string lane_type_str;
        unsigned lane_types;
        const ucp_lane_index_t *lane_idx;
        const ucp_lane_index_t *lanes_key_p;

        unsigned lane_type_mask = 0;
        if (op_mask & (TEST_OP_PUT | TEST_OP_GET)) {
            lane_type_mask |= UCS_BIT(UCP_LANE_TYPE_RMA_BW);
        }

        if (op_mask & TEST_OP_AM) {
            lane_type_mask |= UCS_BIT(UCP_LANE_TYPE_AM_BW);
        }

        if (op_mask & (TEST_OP_PUT | TEST_OP_GET)) {
            lane_type_str  += "RMA BW ";
            lanes_key_p = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.rma_bw_lanes;
            ucs_carray_for_each(lane_idx, lanes_key_p, UCP_MAX_LANES) {
                if (*lane_idx == UCP_NULL_LANE) {
                    continue;
                }

                lane_types = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.lanes[*lane_idx].lane_types;
                if (ucs_test_all_flags(lane_types, lane_type_mask)) {
                    tmp_lanes.insert(*lane_idx);
                }
            }
        }

        if (op_mask & TEST_OP_AM) {
            lane_type_mask |= UCS_BIT(UCP_LANE_TYPE_AM_BW);
            lane_type_str  += "AM BW ";
            lanes_key_p = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.am_bw_lanes;
            ucs_carray_for_each(lane_idx, lanes_key_p, UCP_MAX_LANES) {
                if (*lane_idx == UCP_NULL_LANE) {
                    continue;
                }

                lane_types = ucp_ep_config(sender().ep(0, INJECTED_EP_INDEX))->key.lanes[*lane_idx].lane_types;
                if (ucs_test_all_flags(lane_types, lane_type_mask)) {
                    tmp_lanes.insert(*lane_idx);
                }
            }
        }

        std::vector<ucp_lane_index_t> lanes(tmp_lanes.begin(), tmp_lanes.end());
        shuffle_lanes(lanes, lane_type_str);
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

        lbuf.pattern_fill(m_seed);

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
        ASSERT_EQ(0, m_total_err_count) << "Error callback invoked " << m_total_err_count << " times";
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
            const bool last_lane = (lane_idx == (am_bw_lanes.size() - 1));
            if (last_lane && has_any_transport({"ud_v", "ud_x"}) &&
                (failure_side == FAILURE_SIDE_INITIATOR)) {
                /* TODO: remove this once UD ep purge assertions are fixed */
                UCS_TEST_MESSAGE << "Keep 1 live lane for UD transports since "
                                 << "local error injection on all lanes leads to "
                                 << "failed assertion in ud_ep_purge";
                break;
            }

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
            if (last_lane) {
                slh.reset(new scoped_log_handler(hide_errors_logger));
            }

            status = do_am_send_and_wait(sender().ep(0, INJECTED_EP_INDEX), am_msg_size(),
                                         op_mask & TEST_OP_FLUSH);
            if (!last_lane) {
                EXPECT_EQ(UCS_OK, status) << op_str << " operation returned status: "
                                          << ucs_status_string(status);
                ASSERT_EQ(0, m_total_err_count) << "Error callback invoked " << m_total_err_count << " times";
            } else {
                // The last lane is expected to fail
                short_progress_loop();
                if ((failure_side == FAILURE_SIDE_TARGET) &&
                    has_transport("dc_x")) {
                    // DC transport is not able to detect failure of remote DCI since DC is a connect2iface transport.
                    // This is a test limitation.
                } else {
                    ucs_time_t deadline = ucs::get_deadline();
                    while ((m_initiator_err_count == 0) && (ucs_get_time() < deadline)) {
                        short_progress_loop();
                    }

                    // Initiator EP should invoke error callback only once
                    ASSERT_EQ(1, m_initiator_err_count) << "Error callback invoked " << m_initiator_err_count << " times";
                    // Remote side may detect failure by keepalive or other control messages but not more than 1 time
                    ASSERT_LE(m_total_err_count - m_initiator_err_count, 1)
                            << "Error callback invoked " << m_total_err_count << " times";
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
                             ep_config->key.lanes[rma_bw_lanes[valid_lane]].lane_types)
                    << "Lane " << size_t(rma_bw_lanes[valid_lane]) << " has being marked as failed after "
                    << "failure injection on lane " << size_t(lane);
            }
        }

        short_progress_loop();
        ASSERT_EQ(0, m_total_err_count) << "Error callback invoked " << m_total_err_count << " times";
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
        return ucs::limit_buffer_size((100 * UCS_MBYTE) / ucs::test_time_multiplier());
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
        m_am_expected_size = size;
        m_am_expected_count = 1;
        m_am_recv_count    = 0;
        m_am_expect_reply  = false;

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

        wait_for_value(&m_am_recv_count, 1ul);
        EXPECT_EQ(1ul, m_am_recv_count);
        EXPECT_EQ(size, m_am_rbuf.size());
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
    static const failover_proto_info_t&
    get_failover_proto_info(failover_proto_t proto)
    {
        static const failover_proto_info_t proto_info[] = {
            {"am/egr/short", UCT_IFACE_FLAG_AM_SHORT, 8,
             TEST_FAILOVER_PROTO_FLAG_AM | TEST_FAILOVER_PROTO_FLAG_SINGLE},
            {"am/egr/short/reply", UCT_IFACE_FLAG_AM_SHORT, 8,
             TEST_FAILOVER_PROTO_FLAG_AM | TEST_FAILOVER_PROTO_FLAG_SINGLE |
             TEST_FAILOVER_PROTO_FLAG_REPLY},
            {"am/egr/single/bcopy", UCT_IFACE_FLAG_AM_BCOPY, UCS_KBYTE,
             TEST_FAILOVER_PROTO_FLAG_AM | TEST_FAILOVER_PROTO_FLAG_SINGLE},
            {"am/egr/single/bcopy/reply", UCT_IFACE_FLAG_AM_BCOPY, UCS_KBYTE,
             TEST_FAILOVER_PROTO_FLAG_AM | TEST_FAILOVER_PROTO_FLAG_SINGLE |
             TEST_FAILOVER_PROTO_FLAG_REPLY},
            {"am/egr/multi/bcopy", UCT_IFACE_FLAG_AM_BCOPY, 64 * UCS_KBYTE,
             TEST_FAILOVER_PROTO_FLAG_AM},
            {"put/offload/short", UCT_IFACE_FLAG_PUT_SHORT, 8,
             TEST_FAILOVER_PROTO_FLAG_SINGLE},
            {"put/offload/bcopy", UCT_IFACE_FLAG_PUT_BCOPY, 64 * UCS_KBYTE,
             0},
            {"put/am/bcopy", UCT_IFACE_FLAG_AM_BCOPY, 64 * UCS_KBYTE, 0}
        };

        UCS_STATIC_ASSERT((sizeof(proto_info) / sizeof(proto_info[0])) ==
                          TEST_FAILOVER_PROTO_LAST);
        ucs_assert(proto < TEST_FAILOVER_PROTO_LAST);
        return proto_info[proto];
    }

    failover_proto_t select_failover_proto(failover_proto_t am_proto,
                                           failover_proto_t put_proto) const
    {
        const unsigned op_mask = get_variant_value();

        if ((op_mask & TEST_OP_AM) && !(op_mask & TEST_OP_PUT) &&
            (am_proto != TEST_FAILOVER_PROTO_LAST)) {
            return am_proto;
        }

        if ((op_mask & TEST_OP_PUT) && (put_proto != TEST_FAILOVER_PROTO_LAST)) {
            return put_proto;
        }

        UCS_TEST_SKIP_R("operation variant has no matching failover protocol");
    }

    bool is_failover_proto_supported(ucp_ep_h ep, failover_proto_t proto) const
    {
        const failover_proto_info_t& info = get_failover_proto_info(proto);
        unsigned failover_lane_count = 0;
        unsigned native_put_count = 0;

        for (ucp_lane_index_t lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
            uct_ep_h uct_ep = ucp_ep_get_lane(ep, lane);
            if (uct_ep == nullptr) {
                continue;
            }

            const uct_iface_attr_t *attr = ucp_ep_get_iface_attr(ep, lane);
            native_put_count += !!ucs_test_flags(attr->cap.flags,
                                                 UCT_IFACE_FLAG_PUT_SHORT,
                                                 UCT_IFACE_FLAG_PUT_BCOPY,
                                                 UCT_IFACE_FLAG_PUT_ZCOPY);
            if (!ucs_test_all_flags(attr->cap.flags, info.tl_cap)) {
                continue;
            }

            uct_iface_attr_v2_t attr_v2;
            attr_v2.field_mask = UCT_IFACE_ATTR_FIELD_CAP_FLAGS;
            if ((uct_iface_query_v2(uct_ep->iface, &attr_v2) == UCS_OK) &&
                ucs_test_all_flags(attr_v2.cap.flags,
                                   UCT_IFACE_FLAG_V2_QUERY_TOKEN)) {
                ++failover_lane_count;
            }
        }

        if ((failover_lane_count < 2) ||
            ((proto == TEST_FAILOVER_PROTO_PUT_AM_BCOPY) &&
             (native_put_count > 0))) {
            return false;
        }

        return true;
    }

    static ucp_request_t *get_proto_request(void *status_ptr)
    {
        return UCS_PTR_IS_PTR(status_ptr) ?
               reinterpret_cast<ucp_request_t*>(status_ptr) - 1 : nullptr;
    }

    static ucp_lane_index_t get_request_lane_single(const ucp_request_t *req)
    {
        const ucp_proto_single_priv_t *spriv =
                static_cast<const ucp_proto_single_priv_t*>(
                        req->send.proto_config->priv);
        return spriv->super.lane;
    }

    static ucp_lane_index_t get_request_lane_multi(const ucp_request_t *req)
    {
        const ucp_proto_multi_priv_t *mpriv =
                static_cast<const ucp_proto_multi_priv_t*>(
                        req->send.proto_config->priv);
        return mpriv->lanes[0].super.lane;
    }

    ucp_request_t *check_failover_request(
            void *status_ptr, const failover_proto_info_t& info)
    {
        ucp_request_t *req = get_proto_request(status_ptr);
        EXPECT_NE(nullptr, req) <<
                "failover operation did not return a request";
        if (req == nullptr) {
            return nullptr;
        }

        EXPECT_TRUE(req->flags & UCP_REQUEST_FLAG_PROTO_SEND);
        if (!(req->flags & UCP_REQUEST_FLAG_PROTO_SEND)) {
            request_wait(status_ptr);
            return nullptr;
        }

        EXPECT_STREQ(info.proto_name, req->send.proto_config->proto->name);
        if (strcmp(info.proto_name, req->send.proto_config->proto->name)) {
            request_wait(status_ptr);
            return nullptr;
        }

        return req;
    }

    void test_outstanding_am(failover_proto_t proto, ucp_ep_h ep)
    {
        const failover_proto_info_t& info = get_failover_proto_info(proto);
        mem_buffer sbuf(info.size, UCS_MEMORY_TYPE_HOST);
        ucp_request_param_t param;

        sbuf.pattern_fill(m_seed, info.size);
        m_am_expected_size = info.size;
        m_am_expected_count = 1;
        m_am_recv_count    = 0;
        m_am_expect_reply  = ucs_test_all_flags(
                info.flags, TEST_FAILOVER_PROTO_FLAG_REPLY);
        param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
        if (m_am_expect_reply) {
            param.op_attr_mask |= UCP_OP_ATTR_FIELD_FLAGS;
            param.flags         = UCP_AM_SEND_FLAG_REPLY;
        }

        void *request = ucp_am_send_nbx(ep, AM_ID, nullptr, 0, sbuf.ptr(),
                                        info.size, &param);
        ucp_request_t *req = check_failover_request(request, info);
        ASSERT_NE(nullptr, req);

        ucp_lane_index_t lane =
                ucs_test_all_flags(info.flags, TEST_FAILOVER_PROTO_FLAG_SINGLE) ?
                get_request_lane_single(req) : get_request_lane_multi(req);
        ucs_status_t status = uct_ep_invalidate(ucp_ep_get_lane(ep, lane), 0);
        if (status == UCS_ERR_UNSUPPORTED) {
            request_wait(request);
            UCS_TEST_SKIP_R("uct_ep_invalidate is not supported");
        }
        ASSERT_UCS_OK(status);
        ASSERT_UCS_OK(request_wait(request));
        ASSERT_UCS_OK(request_wait(ucp_ep_flush_nbx(ep, &m_req_empty_param)));
        wait_for_value(&m_am_recv_count, 1ul);
        ASSERT_EQ(1ul, m_am_recv_count);
        ASSERT_EQ(info.size, m_am_rbuf.size());
        mem_buffer::pattern_check(m_am_rbuf.data(), info.size, m_seed);
    }

    void test_outstanding_queue(failover_proto_t proto)
    {
        static const size_t max_msg_count = 64;
        const failover_proto_info_t& info = get_failover_proto_info(proto);
        std::vector<std::unique_ptr<mem_buffer>> sbufs;
        std::vector<ucs_status_ptr_t> requests;
        ucp_lane_index_t request_lane = UCP_NULL_LANE;
        size_t completed_count        = 0;
        size_t pending_count          = 0;
        ucp_request_param_t param;
        ucp_ep_h ep;

        if (!is_proto_enabled()) {
            UCS_TEST_SKIP_R("proto v1");
        }

        flush_workers();
        ep = sender().ep(0, INJECTED_EP_INDEX);
        if (!is_failover_proto_supported(ep, proto)) {
            UCS_TEST_SKIP_R(
                    "failover protocol is not supported by the endpoint lanes");
        }

        ucs_assert(ucs_test_all_flags(info.flags,
                                     TEST_FAILOVER_PROTO_FLAG_AM |
                                     TEST_FAILOVER_PROTO_FLAG_SINGLE));
        m_am_expected_size  = info.size;
        m_am_expected_count = max_msg_count;
        m_am_recv_count     = 0;
        m_am_expect_reply   = false;
        param.op_attr_mask  = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

        while ((pending_count == 0) && (requests.size() < max_msg_count)) {
            size_t msg_index = requests.size();
            sbufs.emplace_back(new mem_buffer(info.size,
                                              UCS_MEMORY_TYPE_HOST));
            sbufs.back()->pattern_fill(m_seed + msg_index, info.size);

            ucs_status_ptr_t request = ucp_am_send_nbx(
                    ep, AM_ID, nullptr, 0, sbufs.back()->ptr(), info.size,
                    &param);
            ucp_request_t *req = check_failover_request(request, info);
            ASSERT_NE(nullptr, req);

            ucp_lane_index_t lane = get_request_lane_single(req);
            if (request_lane == UCP_NULL_LANE) {
                request_lane = lane;
            } else {
                ASSERT_EQ(request_lane, lane);
            }

            if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
                ++completed_count;
            } else {
                ++pending_count;
            }
            requests.push_back(request);
        }

        ASSERT_GT(completed_count, 0ul);
        ASSERT_GT(pending_count, 0ul);
        m_am_expected_count = requests.size();

        ucs_status_t status =
                uct_ep_invalidate(ucp_ep_get_lane(ep, request_lane), 0);
        ASSERT_UCS_OK(status);
        ASSERT_UCS_OK(requests_wait(requests));
        ASSERT_UCS_OK(request_wait(ucp_ep_flush_nbx(ep, &m_req_empty_param)));
        wait_for_value(&m_am_recv_count, m_am_expected_count);
        short_progress_loop();
        EXPECT_EQ(m_am_expected_count, m_am_recv_count);
        ASSERT_EQ(info.size, m_am_rbuf.size());
        mem_buffer::pattern_check(m_am_rbuf.data(), info.size,
                                  m_seed + m_am_expected_count - 1);
    }

    void test_outstanding_put(failover_proto_t proto, ucp_ep_h ep)
    {
        const failover_proto_info_t& info = get_failover_proto_info(proto);
        mem_buffer lbuf(info.size, UCS_MEMORY_TYPE_HOST);
        mapped_buffer rbuf(info.size, receiver());
        ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());
        ucp_request_param_t param;

        lbuf.pattern_fill(m_seed, info.size);
        rbuf.memset(0);
        param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

        void *request = ucp_put_nbx(ep, lbuf.ptr(), info.size,
                                    uintptr_t(rbuf.ptr()), rkey.get(), &param);
        ucp_request_t *req = check_failover_request(request, info);
        ASSERT_NE(nullptr, req);

        ucp_lane_index_t lane =
                ucs_test_all_flags(info.flags, TEST_FAILOVER_PROTO_FLAG_SINGLE) ?
                get_request_lane_single(req) : get_request_lane_multi(req);
        ucs_status_t status = uct_ep_invalidate(ucp_ep_get_lane(ep, lane), 0);
        if (status == UCS_ERR_UNSUPPORTED) {
            request_wait(request);
            UCS_TEST_SKIP_R("uct_ep_invalidate is not supported");
        }
        ASSERT_UCS_OK(status);
        ASSERT_UCS_OK(request_wait(request));
        ASSERT_UCS_OK(request_wait(ucp_ep_flush_nbx(ep, &m_req_empty_param)));
        rbuf.pattern_check(m_seed, info.size);
    }

    void test_outstanding(failover_proto_t am_proto,
                          failover_proto_t put_proto)
    {
        if (!is_proto_enabled()) {
            UCS_TEST_SKIP_R("proto v1");
        }

        flush_workers();
        ucp_ep_h ep = sender().ep(0, INJECTED_EP_INDEX);
        const failover_proto_t proto =
                select_failover_proto(am_proto, put_proto);
        const failover_proto_info_t& info = get_failover_proto_info(proto);

        if (!is_failover_proto_supported(ep, proto)) {
            UCS_TEST_SKIP_R(
                    "failover protocol is not supported by the endpoint lanes");
        }

        if (ucs_test_all_flags(info.flags, TEST_FAILOVER_PROTO_FLAG_AM)) {
            test_outstanding_am(proto, ep);
        } else {
            test_outstanding_put(proto, ep);
        }

        short_progress_loop();
        if (ucs_test_all_flags(info.flags, TEST_FAILOVER_PROTO_FLAG_AM)) {
            EXPECT_EQ(1ul, m_am_recv_count);
        }
    }

    static constexpr uint64_t m_seed = 0x12345678;

    const ucp_request_param_t m_req_empty_param = { 0 };
    std::vector<uint8_t> m_am_rbuf;
    size_t m_am_expected_size                   = am_msg_size();
    size_t m_am_expected_count                  = 0;
    volatile size_t m_am_recv_count             = 0;
    bool m_am_expect_reply                      = false;

private:
    size_t m_initiator_err_count = 0;
    size_t m_total_err_count     = 0;
    ucs_status_t m_err_status    = UCS_OK;
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

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_short_outstanding,
           "MAX_EAGER_LANES=8", "MAX_RMA_LANES=8", "MAX_RMA_RAILS=8",
           "BCOPY_THRESH=inf", "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_outstanding(TEST_FAILOVER_PROTO_AM_SHORT,
                     TEST_FAILOVER_PROTO_PUT_SHORT);
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_short_reply_outstanding,
           "MAX_EAGER_LANES=8", "BCOPY_THRESH=inf", "ZCOPY_THRESH=inf",
           "RNDV_THRESH=inf")
{
    test_outstanding(TEST_FAILOVER_PROTO_AM_SHORT_REPLY,
                     TEST_FAILOVER_PROTO_LAST);
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_bcopy_outstanding,
           "MAX_EAGER_LANES=8", "MAX_RMA_LANES=8", "MAX_RMA_RAILS=8",
           "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_outstanding(TEST_FAILOVER_PROTO_AM_SINGLE_BCOPY,
                     TEST_FAILOVER_PROTO_PUT_BCOPY);
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_bcopy_reply_outstanding,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_outstanding(TEST_FAILOVER_PROTO_AM_SINGLE_BCOPY_REPLY,
                     TEST_FAILOVER_PROTO_LAST);
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_bcopy_queue_outstanding,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=inf", "RNDV_THRESH=inf",
           "RC_TX_QUEUE_LEN?=8", "RC_TX_MAX_BB?=4")
{
    if ((get_variant_value() & TEST_OP_AM) &&
        !(get_variant_value() & TEST_OP_PUT)) {
        test_outstanding_queue(TEST_FAILOVER_PROTO_AM_SINGLE_BCOPY);
    } else {
        UCS_TEST_SKIP_R("AM operation variant is required");
    }
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_multi_bcopy_outstanding,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_outstanding(TEST_FAILOVER_PROTO_AM_MULTI_BCOPY,
                     TEST_FAILOVER_PROTO_LAST);
}

UCS_TEST_P(test_ucp_fault_tolerance,
           initiator_failure_put_am_bcopy_outstanding,
           "MAX_EAGER_LANES=8", "MAX_RMA_LANES=8", "MAX_RMA_RAILS=8",
           "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    test_outstanding(TEST_FAILOVER_PROTO_LAST,
                     TEST_FAILOVER_PROTO_PUT_AM_BCOPY);
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_bcopy,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    do_test(FAILURE_SIDE_INITIATOR);
}

UCS_TEST_P(test_ucp_fault_tolerance, target_failure_bcopy,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=inf", "RNDV_THRESH=inf")
{
    do_test(FAILURE_SIDE_TARGET);
}

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure_zcopy,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=0", "RNDV_THRESH=inf")
{
    do_test(FAILURE_SIDE_INITIATOR);
}

UCS_TEST_P(test_ucp_fault_tolerance, target_failure_zcopy,
           "MAX_EAGER_LANES=8", "ZCOPY_THRESH=0", "RNDV_THRESH=inf")
{
    do_test(FAILURE_SIDE_TARGET);
}
