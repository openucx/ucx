/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "test_ucp_memheap.h"
#include <common/test_helpers.h>
#include <algorithm>
#include <string>

extern "C" {
#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_rkey.h>
#include <ucp/core/ucp_worker.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_select.inl>
#include <ucp/wireup/wireup_ep.h>
#include <uct/base/uct_iface.h>
}

/**
 * Test class for fault tolerance with injected failures
 */
class test_ucp_fault_tolerance : public test_ucp_memheap {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants) {
        static constexpr unsigned msg_size_variants[] = {
            TEST_MSG_SIZE_SMALL, TEST_MSG_SIZE_MEDIUM, TEST_MSG_SIZE_LARGE
        };

        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_PUT,
                               op_name(TEST_OP_PUT));
        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_PUT | TEST_OP_FLUSH,
                               op_name(TEST_OP_PUT | TEST_OP_FLUSH));

        for (unsigned msg_size_variant : msg_size_variants) {
            add_variant_with_value(variants, UCP_FEATURE_RMA,
                                   TEST_OP_GET | msg_size_variant,
                                   op_name(TEST_OP_GET | msg_size_variant));
        }

        add_variant_with_value(variants, UCP_FEATURE_RMA, TEST_OP_GET | TEST_OP_FLUSH,
                               op_name(TEST_OP_GET | TEST_OP_FLUSH));
        add_variant_with_value(variants, UCP_FEATURE_AM,  TEST_OP_AM,
                               op_name(TEST_OP_AM));
        add_variant_with_value(variants, UCP_FEATURE_AM,
                               TEST_OP_AM | TEST_OP_ALL_LANES_FAILED,
                               op_name(TEST_OP_AM | TEST_OP_ALL_LANES_FAILED));
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
        TEST_OP_PUT              = UCS_BIT(0),
        TEST_OP_GET              = UCS_BIT(1),
        TEST_OP_AM               = UCS_BIT(2),
        TEST_OP_FLUSH            = UCS_BIT(3),
        TEST_OP_ALL_LANES_FAILED = UCS_BIT(4),
        TEST_MSG_SIZE_SMALL      = UCS_BIT(5),
        TEST_MSG_SIZE_MEDIUM     = UCS_BIT(6),
        TEST_MSG_SIZE_LARGE      = UCS_BIT(7)
    };

    /* Must stay below cap.get.min_zcopy, so GET cannot use zcopy */
    static constexpr size_t SMALL_MSG_SIZE  = 1;
    static constexpr size_t MEDIUM_MSG_SIZE = UCS_KBYTE;
    static constexpr size_t LARGE_MSG_SIZE  = 100 * UCS_MBYTE;

    void init() override {
        if (get_variant_value() & TEST_OP_ALL_LANES_FAILED) {
            modify_config("RECOVERY_RETRIES", "1");
            modify_config("KEEPALIVE_INTERVAL", std::to_string(3) + "s");
        }

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

        std::random_shuffle(lanes.begin(), lanes.end(), ucs::rand_range);
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
        check_proto_coverage(op_mask);
        return lanes;
    }

    void dump_proto_select_elem(ucp_worker_cfg_index_t rkey_cfg_index,
                                const ucp_proto_select_param_t *select_param,
                                const ucp_proto_select_elem_t *select_elem) {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
        char *line;

        ucp_proto_select_elem_info(sender().worker(),
                                   sender().ep(0, INJECTED_EP_INDEX)->cfg_index,
                                   rkey_cfg_index, select_param, select_elem, 1,
                                   0, &strb);
        ucs_string_buffer_for_each_token(line, &strb, "\n") {
            UCS_TEST_MESSAGE << line;
        }
        ucs_string_buffer_cleanup(&strb);
    }

    /**
     * Walk the selected message size ranges of 'op_id' and verify that every
     * range is served by a real protocol. A range left to the 'reconfig' stub
     * protocol can never be completed on this endpoint configuration.
     */
    void check_op_proto_coverage(ucp_operation_id_t op_id,
                                 ucp_proto_select_t *proto_select,
                                 ucp_worker_cfg_index_t rkey_cfg_index) {
        ucp_worker_h worker = sender().worker();
        ucp_proto_select_param_t select_param;
        ucp_proto_select_elem_t *select_elem;
        ucp_proto_query_attr_t query_attr;
        ucp_memory_info_t mem_info;
        size_t range_start;
        bool gap_found;

        ucp_memory_info_set_host(&mem_info);
        ucp_proto_select_param_init(&select_param, op_id, 0, 0,
                                    UCP_DATATYPE_CONTIG, &mem_info, 1);

        select_elem = ucp_proto_select_lookup_slow(
                worker, proto_select, 0,
                sender().ep(0, INJECTED_EP_INDEX)->cfg_index, rkey_cfg_index,
                &select_param);
        if (select_elem == nullptr) {
            ADD_FAILURE() << ucp_operation_names[op_id]
                          << ": protocol selection is not initialized";
            return;
        }

        gap_found   = false;
        range_start = 0;
        do {
            if (!ucp_proto_select_elem_query(worker, select_elem, range_start,
                                             &query_attr)) {
                ADD_FAILURE() << ucp_operation_names[op_id]
                              << ": no protocol for message sizes "
                              << range_start << ".."
                              << query_attr.max_msg_length;
                gap_found = true;
            }

            range_start = query_attr.max_msg_length + 1;
        } while (query_attr.max_msg_length != SIZE_MAX);

        if (gap_found) {
            dump_proto_select_elem(rkey_cfg_index, &select_param, select_elem);
        }
    }

    /**
     * Verify that the operations exercised by the current variant have no
     * message size range without a protocol.
     */
    void check_proto_coverage(unsigned op_mask) {
        ucp_ep_h ep = sender().ep(0, INJECTED_EP_INDEX);
        ucp_worker_cfg_index_t rkey_cfg_index;
        ucp_proto_select_t *proto_select;

        if (op_mask & TEST_OP_AM) {
            check_op_proto_coverage(UCP_OP_ID_AM_SEND,
                                    &ucp_ep_config(ep)->proto_select,
                                    UCP_WORKER_CFG_INDEX_NULL);
        }

        if (!(op_mask & (TEST_OP_PUT | TEST_OP_GET))) {
            return;
        }

        /* RMA protocols are selected per remote key configuration */
        mapped_buffer rbuf(1, receiver());
        ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());

        proto_select = ucp_proto_select_get(sender().worker(), ep->cfg_index,
                                            rkey->cfg_index, &rkey_cfg_index);
        ASSERT_NE(nullptr, proto_select) << "no rkey protocol selection";

        if (op_mask & TEST_OP_PUT) {
            check_op_proto_coverage(UCP_OP_ID_PUT, proto_select,
                                    rkey_cfg_index);
        }

        if (op_mask & TEST_OP_GET) {
            check_op_proto_coverage(UCP_OP_ID_GET, proto_select,
                                    rkey_cfg_index);
        }
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
        for (size_t num_lanes_to_fail = (op_mask & TEST_OP_ALL_LANES_FAILED) ? am_bw_lanes.size() :
                                        (am_bw_lanes.size() - 1),
             lane_idx = 0; lane_idx < num_lanes_to_fail; ++lane_idx) {
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
            } else if ((failure_side == FAILURE_SIDE_TARGET) &&
                       has_transport("dc_x")) {
                /* DC cannot detect remote DCI failure (connect2iface); test limitation. */
            } else if (status == UCS_OK) {
                /* Some lanes recovered; EP still operable, no error callback required. */
            } else {
                /* Operation failed => EP must fail with exactly one initiator err CB. */
                ucs_time_t deadline = ucs::get_deadline();
                while ((m_initiator_err_count == 0) &&
                       (ucs_get_time() < deadline)) {
                    short_progress_loop();
                }

                ASSERT_EQ(1, m_initiator_err_count)
                        << "Error callback invoked " << m_initiator_err_count
                        << " times";
                /* Remote may detect failure via KA/control msgs, at most once. */
                ASSERT_LE(m_total_err_count - m_initiator_err_count, 1)
                        << "Error callback invoked " << m_total_err_count
                        << " times";
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

    void test_recovery(unsigned op_mask) {
        if (op_mask & TEST_OP_ALL_LANES_FAILED) {
            // Recovery is not expected, it depends on timings
            return;
        }

        UCS_TEST_MESSAGE << "Checking recovery status...";

        wait_for_cond([this]() {
            return ucp_ep_get_failed_lanes(sender().ep(0, INJECTED_EP_INDEX)) == 0;
        }, [this]() {
            short_progress_loop();
        });

        const ucp_lane_map_t failed_lanes =
                ucp_ep_get_failed_lanes(sender().ep(0, INJECTED_EP_INDEX));
        ASSERT_EQ(0, failed_lanes)
            << "Failed lanes are not recovered" << std::hex << failed_lanes;
        for (ucp_lane_index_t lane_idx = 0;
             lane_idx < ucp_ep_num_lanes(sender().ep(0, INJECTED_EP_INDEX));) {
            if (ucp_wireup_ep_test(ucp_ep_get_lane(sender().ep(0, INJECTED_EP_INDEX), lane_idx))) {
                short_progress_loop();
                continue;
            }

            ++lane_idx;
        }

        if (op_mask & TEST_OP_AM) {
            ucs_status_t status = do_am_send_and_wait(sender().ep(0, INJECTED_EP_INDEX),
                                                      am_msg_size(), true);
            EXPECT_EQ(UCS_OK, status) << "AM operation returned status: "
                                      << ucs_status_string(status);
        }

        if (op_mask & TEST_OP_PUT) {
            mem_buffer lbuf(rma_msg_size(), UCS_MEMORY_TYPE_HOST);
            mapped_buffer rbuf(rma_msg_size(), receiver());
            ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());
            lbuf.pattern_fill(m_seed);
            ucs_status_t status = do_put_and_wait(sender().ep(0, INJECTED_EP_INDEX), lbuf, rbuf,
                                                  rkey, rma_msg_size(), true);
            EXPECT_EQ(UCS_OK, status) << "PUT operation returned status: "
                                      << ucs_status_string(status);
        }

        if (op_mask & TEST_OP_GET) {
            mem_buffer lbuf(rma_msg_size(), UCS_MEMORY_TYPE_HOST);
            mapped_buffer rbuf(rma_msg_size(), receiver());
            ucs::handle<ucp_rkey_h> rkey = rbuf.rkey(sender());
            rbuf.pattern_fill(m_seed);
            ucs_status_t status = do_get_and_wait(sender().ep(0, INJECTED_EP_INDEX), lbuf, rbuf,
                                                  rkey, rma_msg_size(), true);
            EXPECT_EQ(UCS_OK, status) << "GET operation returned status: "
                                      << ucs_status_string(status);
        }

        ASSERT_EQ(0, m_total_err_count) << "Error callback invoked " << m_total_err_count
                                        << " times";
        UCS_TEST_MESSAGE << "All lanes are operational";
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

        test_recovery(op_mask);
    }
protected:
    size_t rma_msg_size() const {
        const unsigned op_mask = get_variant_value();

        if (op_mask & TEST_MSG_SIZE_SMALL) {
            return SMALL_MSG_SIZE;
        }

        if (op_mask & TEST_MSG_SIZE_MEDIUM) {
            return ucs::limit_buffer_size(MEDIUM_MSG_SIZE);
        }

        return ucs::limit_buffer_size(LARGE_MSG_SIZE /
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

        if (op_mask & TEST_OP_AM) {
            name += "AM|";
        }

        if (op_mask & TEST_OP_FLUSH) {
            name += "FLUSH|";
        }

        if (op_mask & TEST_OP_ALL_LANES_FAILED) {
            name += "ALL_LANES_FAILED|";
        }

        if (op_mask & TEST_MSG_SIZE_SMALL) {
            name += "MSG_SMALL|";
        }

        if (op_mask & TEST_MSG_SIZE_MEDIUM) {
            name += "MSG_MEDIUM|";
        }

        if (op_mask & TEST_MSG_SIZE_LARGE) {
            name += "MSG_LARGE|";
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

    void skip_unless_rc_probe_gate() {
        if (get_variant_value() != TEST_OP_AM) {
            UCS_TEST_SKIP_R("pure AM variant only");
        }
        if (!has_any_transport({"rc_x", "rc_v", "rc_mlx5", "rc_verbs", "ib"})) {
            UCS_TEST_SKIP_R("probe gate applies to RC p2p lanes only");
        }
    }

    static ucs_status_t recovery_probe_fail(uct_ep_h ep, unsigned flags,
                                            uct_completion_t *comp)
    {
        return UCS_ERR_ENDPOINT_TIMEOUT;
    }

    static ucs_status_t recovery_probe_hold(uct_ep_h ep, unsigned flags,
                                            uct_completion_t *comp)
    {
        if ((m_held_probe_comp == NULL) && (comp != NULL)) {
            m_held_probe_comp = comp;
            return UCS_INPROGRESS;
        }

        return UCS_ERR_ENDPOINT_TIMEOUT;
    }

    static void mock_recovery_probe(ucp_worker_h worker, ucs::mock &mock,
                                    uct_ep_check_func_t ep_check)
    {
        ucp_context_h context = worker->context;
        ucp_rsc_index_t rsc_index;

        for (rsc_index = 0; rsc_index < context->num_tls; ++rsc_index) {
            if (!UCS_STATIC_BITMAP_GET(context->tl_bitmap, rsc_index)) {
                continue;
            }

            ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);
            if ((wiface == NULL) || (wiface->iface == NULL)) {
                continue;
            }

            uint64_t flags = wiface->attr.cap.flags;
            if ((flags & UCT_IFACE_FLAG_EP_CHECK) &&
                (flags & UCT_IFACE_FLAG_CONNECT_TO_IFACE)) {
                mock.setup(&wiface->iface->ops.ep_check, ep_check);
            }
        }
    }

    static void mock_invoke_completion(ucs_status_t status)
    {
        uct_completion_t *comp = m_held_probe_comp;
        m_held_probe_comp      = NULL;
        if (comp == NULL) {
            UCS_TEST_ABORT("hold mock did not capture the in-flight probe completion");
        }

        uct_invoke_completion(comp, status);
    }

    bool wait_for_recovery_probe_in_flight(ucp_ep_h ep, ucs_time_t deadline)
    {
        ucp_ep_recovery_arg_t *arg;
        ucp_lane_index_t lane;

        while (ucs_get_time() < deadline) {
            short_progress_loop();

            arg = ep->ext->recovery_arg;
            if (arg == NULL) {
                continue;
            }

            for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
                if (arg->probe[lane].comp.count != 0) {
                    return true;
                }
            }
        }

        return false;
    }

    const ucp_request_param_t m_req_empty_param = { 0 };
    std::vector<uint8_t> m_am_rbuf              = std::vector<uint8_t>(am_msg_size());
    volatile bool m_am_received                 = false;

    size_t total_err_count() const {
        return m_total_err_count;
    }

    static uct_completion_t *m_held_probe_comp;

private:
    size_t m_initiator_err_count = 0;
    size_t m_total_err_count     = 0;
    ucs_status_t m_err_status    = UCS_OK;
};

uct_completion_t *test_ucp_fault_tolerance::m_held_probe_comp = NULL;

UCP_INSTANTIATE_TEST_CASE(test_ucp_fault_tolerance)

UCS_TEST_P(test_ucp_fault_tolerance, initiator_failure, "MAX_EAGER_LANES=8",
           "RECOVERY_RETRIES=100")
{
    if ((get_variant_value() & TEST_OP_ALL_LANES_FAILED) && has_any_transport({"ud_v", "ud_x"})) {
        UCS_TEST_SKIP_R("UD transport BUG: local error injection on all lanes leads to "
                        "assertion failure in ud_ep_purge");
    }

    do_test(FAILURE_SIDE_INITIATOR);
}

UCS_TEST_P(test_ucp_fault_tolerance, target_failure, "MAX_EAGER_LANES=8",
           "RECOVERY_RETRIES=100")
{
    do_test(FAILURE_SIDE_TARGET);
}

UCS_TEST_P(test_ucp_fault_tolerance, probe_gated_recovery, "MAX_EAGER_LANES=8",
           "RECOVERY_RETRIES=100")
{
    skip_unless_rc_probe_gate();

    bool probe_armed = false;

    test_am_with_injected_failure(FAILURE_SIDE_TARGET, TEST_OP_AM);

    wait_for_cond([this, &probe_armed]() {
        ucp_ep_h ep = sender().ep(0, INJECTED_EP_INDEX);
        ucp_ep_recovery_arg_t *arg = ep->ext->recovery_arg;
        ucp_lane_index_t lane;

        if (arg != NULL) {
            for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
                if (arg->probe[lane].comp.func != NULL) {
                    probe_armed = true;
                    break;
                }
            }
        }

        return ucp_ep_get_failed_lanes(ep) == 0;
    }, [this]() {
        short_progress_loop();
    });

    EXPECT_TRUE(probe_armed)
            << "RC p2p lane recovery completed without arming an aux probe";
}

UCS_TEST_P(test_ucp_fault_tolerance, teardown_with_outstanding_probe,
           "MAX_EAGER_LANES=8", "RECOVERY_RETRIES=1000")
{
    skip_unless_rc_probe_gate();

    m_held_probe_comp = NULL;

    ucs::mock mock;
    mock_recovery_probe(sender().worker(), mock, recovery_probe_hold);

    test_am_with_injected_failure(FAILURE_SIDE_INITIATOR, TEST_OP_AM);

    ucp_ep_h ep = sender().ep(0, INJECTED_EP_INDEX);
    if (ucp_ep_get_failed_lanes(ep) == 0) {
        UCS_TEST_SKIP_R("no RC p2p lane was marked failed");
    }

    ASSERT_TRUE(wait_for_recovery_probe_in_flight(
            ep, ucs_get_time() + ucs_time_from_sec(5.0)))
            << "could not catch an aux probe in flight to exercise teardown";

    void *creq = sender().disconnect_nb(0, INJECTED_EP_INDEX,
                                        UCP_EP_CLOSE_FLAG_FORCE);
    mock_invoke_completion(UCS_ERR_CANCELED);
    ASSERT_FALSE(UCS_PTR_IS_ERR(creq))
            << "disconnect failed: "
            << ucs_status_string(UCS_PTR_STATUS(creq));
    if (UCS_PTR_IS_PTR(creq)) {
        EXPECT_EQ(UCS_OK, request_wait(creq));
    }
}

UCS_TEST_P(test_ucp_fault_tolerance, recovery_retries_exhausted_live_lanes,
           "MAX_EAGER_LANES=8", "RECOVERY_RETRIES=2", "KEEPALIVE_INTERVAL=0.1s")
{
    skip_unless_rc_probe_gate();

    ucs::mock mock;
    mock_recovery_probe(sender().worker(), mock, recovery_probe_fail);
    test_am_with_injected_failure(FAILURE_SIDE_INITIATOR, TEST_OP_AM);

    ucp_ep_h ep = sender().ep(0, INJECTED_EP_INDEX);
    if (ucp_ep_get_failed_lanes(ep) == 0) {
        UCS_TEST_SKIP_R("no RC p2p lane was marked failed");
    }

    ASSERT_NE(nullptr, ep->ext->recovery_arg);
    wait_for_cond([ep]() {
        return ep->ext->recovery_arg == NULL;
    }, [this]() {
        short_progress_loop();
    });
    ASSERT_EQ(nullptr, ep->ext->recovery_arg)
            << "recovery retries were not exhausted";

    EXPECT_EQ(0, total_err_count())
            << "EP was failed even though live lanes remained";
    EXPECT_NE(0, ucp_ep_get_failed_lanes(ep))
            << "failed lanes were cleared without a successful probe";
    EXPECT_EQ(UCS_OK, do_am_send_and_wait(ep, am_msg_size(), true))
            << "data did not flow on live lanes after recovery give-up";
}
