/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <string.h>
#include "ucp_test.h"
#include <common/mem_buffer.h>
extern "C" {
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_ep.inl>
#include <ucp/proto/proto_common.h>
}


class test_ucp_request : public ucp_test {
public:
    virtual void init() override
    {
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        int mem_type_pair_index = get_variant_value() %
                                  mem_buffer::supported_mem_types().size();
        m_mem_type              =
                mem_buffer::supported_mem_types()[mem_type_pair_index];
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        int count = 0;
        add_variant(variants, UCP_FEATURE_TAG);

        for (auto mem_type : mem_buffer::supported_mem_types()) {
            std::string name = ucs_memory_type_names[mem_type];
            add_variant_with_value(variants, UCP_FEATURE_TAG, count, name);
            ++count;
        }
    }

    static const size_t msg_size = 4;

protected:
    ucs_memory_type_t m_mem_type;
};


UCS_TEST_P(test_ucp_request, test_request_query)
{
    ucp_request_param_t param;
    ucp_request_attr_t attr;
    ucp_worker_attr_t worker_attr;
    void *reqs[2];

    mem_buffer m_recv_mem_buf(msg_size, m_mem_type);
    mem_buffer m_send_mem_buf(msg_size, m_mem_type);

    param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;

    void *sreq = ucp_tag_send_nbx(sender().ep(), m_send_mem_buf.ptr(), msg_size,
                                  0, &param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(sreq));
    reqs[0] = sreq;

    void *rreq = ucp_tag_recv_nbx(receiver().worker(), m_recv_mem_buf.ptr(),
                                  msg_size, 0, 0, &param);
    ASSERT_TRUE(UCS_PTR_IS_PTR(rreq));
    reqs[1] = rreq;

    while ((ucp_request_check_status(sreq) == UCS_INPROGRESS) ||
           (ucp_request_check_status(rreq) == UCS_INPROGRESS)) {
        progress();
    }

    worker_attr.field_mask = UCP_WORKER_ATTR_FIELD_MAX_INFO_STRING;
    ucp_worker_query(receiver().worker(), &worker_attr);

    char debug_string[worker_attr.max_debug_string];
    memset(&debug_string, 0, worker_attr.max_debug_string);

    attr.field_mask        = UCP_REQUEST_ATTR_FIELD_INFO_STRING      |
                             UCP_REQUEST_ATTR_FIELD_INFO_STRING_SIZE |
                             UCP_REQUEST_ATTR_FIELD_MEM_TYPE         |
                             UCP_REQUEST_ATTR_FIELD_STATUS;
    attr.debug_string      = debug_string;
    attr.debug_string_size = worker_attr.max_debug_string;

    for (int i = 0; i < 2; i++) {
        const char *req_type = (i == 0) ? "send" : "recv";
        ucp_request_query(UCS_STATUS_PTR(reqs[i]), &attr);
        UCS_TEST_MESSAGE << req_type << " req: " << attr.debug_string;
        std::string str(attr.debug_string);
        EXPECT_GT(str.size(), 0);
        EXPECT_NE(str.find(req_type), std::string::npos);
        EXPECT_NE(str.find(ucs_memory_type_names[m_mem_type]),
                  std::string::npos);
        ASSERT_EQ(attr.status, UCS_OK);
        ASSERT_EQ(attr.mem_type, m_mem_type);

        ucp_request_free(reqs[i]);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_request, all, "all")

class test_proto_reset : public ucp_test {
public:
    enum operation_e {
        TAG,
        RMA_GET,
        RMA_PUT,
        STREAM,
        AM
    };

    test_proto_reset() :
        m_msg_size(UCS_MBYTE * 10),
        m_sbuf(m_msg_size),
        m_rbuf(m_msg_size),
        m_completed(0),
        m_memh(NULL),
        m_rkey(NULL)
    {
    }

    void init() override
    {
        if (!m_ucp_config->ctx.proto_enable) {
            UCS_TEST_SKIP_R("reset is not supported for proto v1");
        }

        ucp_test::init();
        connect();
        init_rkey();
    }

    void cleanup() override
    {
        ucp_rkey_destroy(m_rkey);
        ASSERT_UCS_OK(ucp_mem_unmap(receiver().ucph(), m_memh));
        ucp_test::cleanup();
    }

    void connect()
    {
        ucp_request_param_t param = {0};
        param.op_attr_mask        = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
        static const size_t size  = 1024;
        std::vector<char> sbuf(size);
        std::vector<char> rbuf(size);

        sender().connect(&receiver(), get_ep_params());
        receiver().connect(&sender(), get_ep_params());

        void *pre_msg = ucp_tag_send_nbx(sender().ep(), sbuf.data(), size, 0,
                                         &param);
        void *pre_rcv = ucp_tag_recv_nbx(receiver().worker(), rbuf.data(), size,
                                         0, 0, &param);
        request_wait(pre_msg);
        request_wait(pre_rcv);
    }

    void get_stream_data()
    {
        size_t roffset = 0;
        ucs_status_ptr_t rdata;
        size_t length;
        do {
            progress();
            rdata = ucp_stream_recv_data_nb(receiver().ep(), &length);
            if (rdata == NULL) {
                continue;
            }

            memcpy(&m_rbuf[roffset], rdata, length);
            roffset += length;
            ucp_stream_data_release(receiver().ep(), rdata);
        } while (roffset < m_rbuf.size());
    }

    static ucs_status_t
    am_data_cb(void *arg, const void *header, size_t header_length, void *data,
               size_t length, const ucp_am_recv_param_t *param)
    {
        std::vector<char> *rbuf = (std::vector<char>*)arg;

        memcpy((*rbuf).data(), data, length);
        return UCS_OK;
    }

    void *send_am()
    {
        ucp_request_param_t req_param = {0};
        ucp_am_handler_param_t param;
        void *sreq;

        param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                           UCP_AM_HANDLER_PARAM_FIELD_CB |
                           UCP_AM_HANDLER_PARAM_FIELD_ARG;
        param.id         = 0;
        param.cb         = am_data_cb;
        param.arg        = &m_rbuf;

        ucs_status_t status;
        status = ucp_worker_set_am_recv_handler(receiver().worker(), &param);
        ASSERT_UCS_OK(status);
        sreq = ucp_am_send_nbx(sender().ep(), 0, NULL, 0, m_sbuf.data(),
                               m_msg_size, &req_param);
        return sreq;
    }

    void wait_receive(operation_e op)
    {
        const double timeout      = 10;
        const ucs_time_t deadline = ucs::get_deadline(timeout);

        if (op == STREAM) {
            get_stream_data();
        } else if (op == AM) {
            while ((ucs_get_time() < deadline) && (m_rbuf != m_sbuf)) {
                progress();
            }
            EXPECT_EQ(m_rbuf, m_sbuf);
        }
    }

    static void flushed_cb(ucp_request_t *request)
    {
        test_proto_reset *self = (test_proto_reset*)request->user_data;
        uct_pending_req_t *uct_req;
        ucp_request_t *ucp_req;

        ucs_queue_for_each_extract(uct_req, &self->m_pending, priv, 1) {
            ucp_req = ucs_container_of(uct_req, ucp_request_t, send.uct);
            ucp_proto_request_restart(ucp_req);
        }

        self->m_completed = 1;
        ucp_request_complete_send(request, request->status);
    }

    void restart(ucp_ep_h ep)
    {
        ucp_request_param_t param;
        param.op_attr_mask = UCP_OP_ATTR_FIELD_USER_DATA |
                             UCP_OP_ATTR_FIELD_CALLBACK,
        param.user_data    = this;
        param.cb.send      = (ucp_send_nbx_callback_t)ucs_empty_function;

        ucs_queue_head_init(&m_pending);
        ucp_ep_purge_lanes(ep, ucp_request_purge_enqueue_cb, &m_pending);

        void *request = ucp_ep_flush_internal(ep, 0, &param, NULL, flushed_cb,
                                              "ep_restart");

        ASSERT_FALSE(UCS_PTR_IS_ERR(request));
        wait_for_value(&m_completed, (unsigned)1);
        ucp_request_release(request);
    }

    void reset_protocol(operation_e op, const std::string &output_proto,
                        bool sync = false)
    {
        ucp_request_param_t param = {0};
        void *rreq                = NULL;
        void *sreq                = NULL;

        ucs::fill_random(m_sbuf);
        ucs::fill_random(m_rbuf);

        switch (op) {
        case TAG:
            param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
            if (sync) {
                sreq = ucp_tag_send_sync_nbx(sender().ep(), m_sbuf.data(),
                                             m_msg_size, 0, &param);
            } else {
                sreq = ucp_tag_send_nbx(sender().ep(), m_sbuf.data(),
                                        m_msg_size, 0, &param);
            }

            ASSERT_TRUE(UCS_PTR_IS_PTR(sreq));
            rreq = ucp_tag_recv_nbx(receiver().worker(), m_rbuf.data(),
                                    m_msg_size, 0, 0, &param);
            ASSERT_TRUE(UCS_PTR_IS_PTR(rreq));
            break;

        case RMA_GET:
            param.op_attr_mask = UCP_OP_ATTR_FLAG_NO_IMM_CMPL;
            sreq = ucp_get_nbx(sender().ep(), m_rbuf.data(), m_msg_size,
                               (uint64_t)m_sbuf.data(), m_rkey, &param);
            break;

        case RMA_PUT:
            sreq = ucp_put_nbx(sender().ep(), m_rbuf.data(), m_msg_size,
                               (uint64_t)m_sbuf.data(), m_rkey, &param);
            break;

        case STREAM:
            param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
            param.datatype     = ucp_dt_make_contig(1);
            sreq = ucp_stream_send_nbx(sender().ep(), m_sbuf.data(), m_msg_size,
                                       &param);
            break;

        case AM:
            sreq = send_am();
            break;
        }

        ASSERT_TRUE(UCS_PTR_IS_PTR(sreq));
        wait_and_restart(sreq, rreq, output_proto);
        wait_receive(op);

        request_wait(sreq);
        request_wait(rreq);
        flush_ep(sender());

        EXPECT_EQ(m_sbuf, m_rbuf);
    }

    virtual void
    wait_and_restart(void *sreq, void *rreq, const std::string &expected_proto)
    {
        ucp_request_t *req                 = (ucp_request_t*)sreq - 1;
        const ucp_datatype_iter_t *dt_iter = &req->send.state.dt_iter;

        while (dt_iter->offset == 0) {
            progress();
        }

        EXPECT_LT(dt_iter->offset, dt_iter->length);
        restart(sender().ep());

        EXPECT_STREQ(expected_proto.c_str(),
                     req->send.proto_config->proto->name);
    }

    void init_rkey()
    {
        const ucp_mem_map_params_t params = {
            .field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                          UCP_MEM_MAP_PARAM_FIELD_LENGTH,
            .address    = m_sbuf.data(),
            .length     = m_sbuf.size()
        };

        ASSERT_UCS_OK(ucp_mem_map(receiver().ucph(), &params, &m_memh));

        void *rkey_buffer;
        size_t rkey_buffer_size;
        ASSERT_UCS_OK(ucp_rkey_pack(receiver().ucph(), m_memh, &rkey_buffer,
                                    &rkey_buffer_size));

        ASSERT_UCS_OK(ucp_ep_rkey_unpack(sender().ep(), rkey_buffer, &m_rkey));
        ucp_rkey_buffer_release(rkey_buffer);
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant_with_value(variants,
                               UCP_FEATURE_TAG | UCP_FEATURE_RMA |
                               UCP_FEATURE_STREAM | UCP_FEATURE_AM,
                               0, "");
    }

    size_t m_msg_size;
    std::vector<char> m_sbuf;
    std::vector<char> m_rbuf;
    unsigned m_completed;
    ucp_mem_h m_memh;
    ucp_rkey_h m_rkey;
    ucs_queue_head_t m_pending;
};

UCS_TEST_P(test_proto_reset, tag_eager_multi_bcopy, "ZCOPY_THRESH=inf",
           "RNDV_THRESH=inf")
{
    reset_protocol(TAG, "egr/multi/bcopy");
}

UCS_TEST_SKIP_COND_P(test_proto_reset, get_offload_bcopy_to_get_am_bcopy,
                     !has_transport("ib"), "ZCOPY_THRESH=inf",
                     "RNDV_THRESH=inf")
{
    reset_protocol(RMA_GET, "get/am/bcopy");
}

UCS_TEST_SKIP_COND_P(test_proto_reset, put_offload_bcopy_to_put_am_bcopy,
                     !has_transport("ib"), "ZCOPY_THRESH=inf",
                     "RNDV_THRESH=inf")
{
    reset_protocol(RMA_PUT, "put/am/bcopy");
}

UCS_TEST_P(test_proto_reset, stream_multi_bcopy, "ZCOPY_THRESH=inf",
           "RNDV_THRESH=inf")
{
    reset_protocol(STREAM, "stream/multi/bcopy");
}

UCS_TEST_P(test_proto_reset, rndv_am_bcopy, "ZCOPY_THRESH=inf", "RNDV_THRESH=0",
           "RNDV_SCHEME=am")
{
    reset_protocol(TAG, "rndv/am/bcopy");
}

UCS_TEST_P(test_proto_reset, eager_sync_multi_bcopy, "ZCOPY_THRESH=inf",
           "RNDV_THRESH=inf")
{
    reset_protocol(TAG, "egrsnc/multi/bcopy", true);
}

UCS_TEST_P(test_proto_reset, am_eager_multi_bcopy, "ZCOPY_THRESH=inf",
           "RNDV_THRESH=inf")
{
    reset_protocol(AM, "am/egr/multi/bcopy");
}

UCS_TEST_P(test_proto_reset, tag_eager_multi_zcopy_to_bcopy, "ZCOPY_THRESH=0",
           "RNDV_THRESH=inf")
{
    reset_protocol(TAG, "egr/multi/bcopy");
}

UCS_TEST_SKIP_COND_P(test_proto_reset, get_offload_zcopy_to_get_am_bcopy,
                     !has_transport("ib"), "ZCOPY_THRESH=0", "RNDV_THRESH=inf",
                     "RMA_ZCOPY_SEG_SIZE=1024")
{
    reset_protocol(RMA_GET, "get/am/bcopy");
}

UCS_TEST_P(test_proto_reset, put_offload_zcopy_to_put_am_bcopy,
           "ZCOPY_THRESH=0", "RNDV_THRESH=inf", "RMA_ZCOPY_SEG_SIZE=1024")
{
    reset_protocol(RMA_PUT, "put/am/bcopy");
}

UCS_TEST_P(test_proto_reset, stream_multi_zcopy_to_bcopy, "ZCOPY_THRESH=0",
           "RNDV_THRESH=inf")
{
    reset_protocol(STREAM, "stream/multi/bcopy");
}

UCS_TEST_P(test_proto_reset, rndv_am_zcopy_to_bcopy, "ZCOPY_THRESH=0",
           "RNDV_THRESH=0", "RNDV_SCHEME=am")
{
    reset_protocol(TAG, "rndv/am/bcopy");
}

UCS_TEST_P(test_proto_reset, am_eager_multi_zcopy_to_bcopy, "ZCOPY_THRESH=0",
           "RNDV_THRESH=inf")
{
    reset_protocol(AM, "am/egr/multi/bcopy");
}

UCS_TEST_P(test_proto_reset, rndv_put, "RNDV_THRESH=0", "RNDV_SCHEME=put_zcopy",
           "RMA_ZCOPY_SEG_SIZE=1024")
{
    reset_protocol(TAG, "rndv/put/zcopy");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_proto_reset, ib, "ib")
UCP_INSTANTIATE_TEST_CASE_TLS(test_proto_reset, tcp, "tcp")

/* The following tests require ENABLE_DEBUG_DATA flag in order to access
 * req->recv.proto_rndv_request, which is only present with this flag. */
#if ENABLE_DEBUG_DATA
class test_proto_reset_rndv_get : public test_proto_reset {
protected:
    void wait_and_restart(void *sreq, void *rreq,
                          const std::string &expected_proto) override
    {
        ucp_request_t *req = (ucp_request_t*)rreq - 1;

        while (req->recv.proto_rndv_request == NULL ||
               req->recv.proto_rndv_request->send.state.dt_iter.offset == 0) {
            progress();
        }

        const ucp_request_t *rndv_req = req->recv.proto_rndv_request;
        ASSERT_LT(rndv_req->send.state.dt_iter.offset,
                  rndv_req->send.state.dt_iter.length);

        restart(receiver().ep());

        EXPECT_STREQ(expected_proto.c_str(),
                     rndv_req->send.proto_config->proto->name);
    }
};

UCS_TEST_P(test_proto_reset_rndv_get, rndv_get_to_rtr, "RNDV_THRESH=0",
           "RNDV_SCHEME=get_zcopy", "RMA_ZCOPY_SEG_SIZE=1024")
{
    reset_protocol(TAG, "rndv/rtr");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_proto_reset_rndv_get, ib, "ib")

class test_proto_reset_atp : public test_proto_reset {
public:
    void init() override
    {
        test_proto_reset::init();
        m_msg_size = 50 * UCS_KBYTE;
    }

    void cleanup() override
    {
        test_proto_reset::cleanup();
    }

private:
    void hook_uct_cbs()
    {
        ucp_ep_h ep                = sender().ep();
        ucp_lane_index_t num_lanes = ucp_ep_config(ep)->key.num_lanes;
        uct_ep_h uct_ep;
        uct_iface_ops_t *ops;

        for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
            uct_ep                = ucp_ep_get_lane(ep, lane);
            ops                   = &uct_ep->iface->ops;
            ops->ep_put_zcopy     = (uct_ep_put_zcopy_func_t)
                                    ucs_empty_function_return_no_resource;
            ops->ep_pending_add   = add_pending;
            ops->ep_pending_purge = purge_pending;
        }
    }

    void restore_uct_cbs()
    {
        ucp_ep_h ep                = sender().ep();
        ucp_lane_index_t num_lanes = ucp_ep_config(ep)->key.num_lanes;

        for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
            ucp_ep_get_lane(ep, lane)->iface->ops = m_ops[lane];
        }
    }

    static ucs_status_t
    add_pending(uct_ep_h tl_ep, uct_pending_req_t *n, unsigned flag)
    {
        return UCS_OK;
    }

    static void
    purge_pending(uct_ep_h ep, uct_pending_purge_callback_t cb, void *arg)
    {
        cb(&m_req->send.uct, arg);
    }

protected:
    void wait_and_restart(void *sreq, void *rreq,
                          const std::string &expected_proto) override
    {
        ucp_request_t *req         = (ucp_request_t*)sreq - 1;
        ucp_ep_h ep                = sender().ep();
        ucp_lane_index_t num_lanes = ucp_ep_config(ep)->key.num_lanes;

        /* Backup uct ops for all lanes */
        for (ucp_lane_index_t lane = 0; lane < num_lanes; ++lane) {
            m_ops.push_back(ucp_ep_get_lane(ep, lane)->iface->ops);
        }

        hook_uct_cbs();

        /* Wait for rndv_put initialization */
        std::string rndv_put_zcopy_name("rndv/put/zcopy");
        while (rndv_put_zcopy_name != req->send.proto_config->proto->name) {
            progress();
        }

        restore_uct_cbs();

        /* Wait until ATP stage starts */
        static const unsigned send_stage = 0;
        while (req->send.proto_stage == send_stage) {
            req->send.uct.func(&req->send.uct);
        }

        /* One more progress to send the first ATP */
        req->send.uct.func(&req->send.uct);

        ucp_request_t *ucp_rreq       = (ucp_request_t*)rreq - 1;
        const ucp_request_t *rndv_req = ucp_rreq->recv.proto_rndv_request;

        /* Wait until receiver gets the ATP message */
        while (rndv_req->send.state.completed_size == 0) {
            receiver().progress();
        }

        m_req = req;
        hook_uct_cbs();
        restart(ep);

        EXPECT_STREQ(expected_proto.c_str(),
                     req->send.proto_config->proto->name);

        restore_uct_cbs();
    }

    static ucp_request_t *m_req;
    std::vector<uct_iface_ops> m_ops;
};

ucp_request_t *test_proto_reset_atp::m_req;

UCS_TEST_P(test_proto_reset_atp, rndv_put_to_rndv_am, "RNDV_THRESH=0",
           "RNDV_SCHEME=put_zcopy", "RMA_ZCOPY_SEG_SIZE=1024")
{
    if (count_resources(sender(), "rc_mlx5") <= 1) {
        UCS_TEST_SKIP_R("Less than 2 RC resources are found");
    }

    reset_protocol(TAG, "rndv/am/bcopy");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_proto_reset_atp, ib, "ib")

#endif
