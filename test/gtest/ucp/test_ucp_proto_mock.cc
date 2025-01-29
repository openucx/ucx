/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <uct/base/uct_iface.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_select.inl>
}

class mock_iface {
public:
    /* Can't use std::function due to coverity errors */
    using iface_attr_func_t = void (*)(uct_iface_attr&);

    mock_iface() : m_tl(nullptr)
    {
        ucs_assert(m_self == nullptr);
        m_self = this;
    }

    ~mock_iface()
    {
        cleanup();
        m_self = nullptr;
    }

    void cleanup()
    {
        m_mock.cleanup();
    }

    void add_mock_iface(
            const std::string &dev_name = "mock",
            iface_attr_func_t cb = [](uct_iface_attr_t &iface_attr) {})
    {
        m_iface_attrs_funcs[dev_name] = cb;
    }

    void mock_transport(const std::string &tl_name)
    {
        uct_component_h component;

        /* Currently only one TL can be mocked */
        ucs_assert(nullptr == m_tl);

        ucs_list_for_each(component, &uct_components_list, list) {
            uct_tl_t *tl;
            ucs_list_for_each(tl, &component->tl_list, list) {
                if (tl_name == tl->name) {
                    m_mock.setup(&tl->query_devices, query_devices_mock);
                    m_mock.setup(&tl->iface_open, iface_open_mock);
                    m_tl = tl;
                    return;
                }
            }
        }

        FAIL() << "Transport " << tl_name << " not found";
    }

private:
    static ucs_status_t
    query_devices_mock(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                       unsigned *num_tl_devices_p)
    {
        UCS_MOCK_ORIG_FUNC(m_self->m_mock, &m_self->m_tl->query_devices, md,
                           tl_devices_p, num_tl_devices_p);
        if (*num_tl_devices_p == 0) {
            return UCS_OK;
        }

        /* Instantiate mock devices only for the first available device */
        const char *first_dev_name = (*tl_devices_p)[0].name;
        if (m_self->m_real_dev_name.empty()) {
            m_self->m_real_dev_name = first_dev_name;
        } else if (m_self->m_real_dev_name != first_dev_name) {
            *num_tl_devices_p = 0;
            return UCS_OK;
        }

        /*
         * The number of real devices (and their names) do not match the mocked
         * ones. In order to pretend that all the mocked devices are supported,
         * we remember the first real device name, and then substitute the
         * response with the mocked devices names. For each mocked device the
         * individual sys_dev must be set, so that they are treated as different
         * devices. Later on the iface_open_mock will use the real device name
         * (same for all mocks) to create the mocked iface.
         */
        auto mock_devices  = (uct_tl_device_resource_t*)ucs_calloc(
                                    m_self->m_iface_attrs_funcs.size(),
                                    sizeof(uct_tl_device_resource_t),
                                    "mock_tl_devices");
        unsigned dev_count = 0;
        for (const auto &it : m_self->m_iface_attrs_funcs) {
            ucs_strncpy_safe(mock_devices[dev_count].name, it.first.c_str(),
                             UCT_DEVICE_NAME_MAX);
            mock_devices[dev_count].type       = (*tl_devices_p)[0].type;
            mock_devices[dev_count].sys_device = dev_count + 1;
            ++dev_count;
        }

        ucs_free(*tl_devices_p);
        *tl_devices_p     = mock_devices;
        *num_tl_devices_p = dev_count;
        return UCS_OK;
    }

    static ucs_status_t
    iface_open_mock(uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *config, uct_iface_h *iface_p)
    {
        ucs_assert(params->field_mask & UCT_IFACE_PARAM_FIELD_DEVICE);
        uct_iface_params_t iface_open_params   = *params;
        iface_open_params.mode.device.dev_name = m_self->m_real_dev_name.c_str();

        UCS_MOCK_ORIG_FUNC(m_self->m_mock, &m_self->m_tl->iface_open, md,
                           worker, &iface_open_params, config, iface_p);

        uct_base_iface_t *base      = ucs_derived_of(*iface_p, uct_base_iface_t);
        m_self->m_iface_names[base] = params->mode.device.dev_name;
        m_self->m_mock.setup(&(*iface_p)->ops.iface_query, iface_query_mock);
        return UCS_OK;
    }

    static ucs_status_t
    iface_query_mock(uct_iface_h iface, uct_iface_attr_t *iface_attr)
    {
        UCS_MOCK_ORIG_FUNC(m_self->m_mock, &iface->ops.iface_query, iface,
                           iface_attr);

        uct_base_iface_t *base  = ucs_derived_of(iface, uct_base_iface_t);
        std::string &iface_name = m_self->m_iface_names[base];
        auto it                 = m_self->m_iface_attrs_funcs.find(iface_name);
        (it->second)(*iface_attr);
        return UCS_OK;
    }

    /* We have to use singleton to mock C functions */
    static mock_iface *m_self;

    ucs::mock                                           m_mock;
    uct_tl_t                                           *m_tl;
    std::unordered_map<uct_base_iface_t *, std::string> m_iface_names;
    std::map<std::string, iface_attr_func_t>            m_iface_attrs_funcs;
    std::string                                         m_real_dev_name;
};

mock_iface *mock_iface::m_self = nullptr;

class test_ucp_proto_mock : public ucp_test, public mock_iface {
public:
    const static size_t INF                               = UCS_MEMUNITS_INF;
    const static uint16_t AM_ID                           = 123;
    const static ucp_worker_cfg_index_t EP_CONFIG_INDEX   = 0;
    const static ucp_worker_cfg_index_t RKEY_CONFIG_INDEX = 0;

    struct proto_select_data {
        size_t      range_start{0};
        size_t      range_end{0};
        std::string desc{"unknown"};
        std::string config;

        proto_select_data() = default;

        proto_select_data(size_t range_start,
                          const ucp_proto_query_attr_t &attr) :
            proto_select_data(range_start, attr.max_msg_length, attr.desc,
                              attr.config)
        {
        }

        proto_select_data(size_t range_start, size_t range_end,
                          std::string desc, std::string config) :
            range_start(range_start),
            range_end(range_end),
            desc(desc),
            config(config){};

        bool operator==(const proto_select_data &other) const
        {
            return (range_start == other.range_start) &&
                   (range_end == other.range_end) && (desc == other.desc) &&
                   (config == other.config);
        }

        bool operator!=(const proto_select_data &other) const
        {
            return !(*this == other);
        }

        friend std::ostream &
        operator<<(std::ostream &os, const proto_select_data &data)
        {
            os << data.range_start << "..";
            if (data.range_end == SIZE_MAX) {
                os << "inf";
            } else {
                os << data.range_end;
            }
            return os << " desc: " << data.desc << ", config: " << data.config;
        }
    };

    struct worker_config {
        ucp_worker_h           worker;
        ucp_worker_cfg_index_t ep_cfg_index;
        ucp_worker_cfg_index_t rkey_cfg_index;
    };

    using proto_select_data_vec_t = std::vector<proto_select_data>;

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_AM);
    }

    virtual void init() override
    {
        /* This test is for dynamic protocol selection available only in v2 */
        if (!is_proto_enabled()) {
            UCS_TEST_SKIP_R("Proto v2 is disabled");
        }

        /* Reset topo provider to force reload from config */
        ucs_sys_topo_reset_provider();

        /*
         * By default TOPO_PRIO="sysfs,default"
         * We keep only default config to always have the same topo distances
         * when test is being executed on different machines.
         */
        modify_config("TOPO_PRIO", "default");

        ucp_test::init();
        connect();
    }

    virtual void cleanup() override
    {
        mock_iface::cleanup();
        ucp_test::cleanup();
        /* Reset topo provider to not affect subsequent tests */
        ucs_sys_topo_reset_provider();
    }

    static void check_ep_config(entity &e,
                                const proto_select_data_vec_t &data_vec,
                                ucp_proto_select_key_t key)
    {
        worker_config worker_cfg = {e.worker(), EP_CONFIG_INDEX,
                                    UCP_WORKER_CFG_INDEX_NULL};
        ucp_ep_config_t *config  = ucp_worker_ep_config(e.worker(),
                                                        worker_cfg.ep_cfg_index);
        check_proto_select(worker_cfg, config->proto_select, data_vec, key);
    }

    static void check_rkey_config(entity &e,
                                  const proto_select_data_vec_t &data_vec,
                                  ucp_proto_select_key_t key)
    {
        worker_config worker_cfg = {e.worker(), EP_CONFIG_INDEX,
                                    RKEY_CONFIG_INDEX};
        ucp_rkey_config_t *config =
                &e.worker()->rkey_config[worker_cfg.rkey_cfg_index];
        check_proto_select(worker_cfg, config->proto_select, data_vec, key);
    }

    /*
     * Helper function that returns a key that matches any protocol selection.
     * It is used to first create the default key matching any protocol, and
     * then the fine-grained setting can be applied to match specific use case.
     */
    static ucp_proto_select_key_t any_key()
    {
        ucp_proto_select_key_t key;
        /* We set all key params to UINT8_MAX, meaning default value, was not
         * explicitly configured with a fine-grained setting */
        key.u64 = UINT64_MAX;
        return key;
    }

    void connect()
    {
        sender().connect(&receiver(), get_ep_params());
        wait_for_cond(
                [this] {
                    return (sender().worker()->ep_config.length >
                            EP_CONFIG_INDEX);
                },
                [this] { progress(); });

        EXPECT_EQ(1, sender().worker()->ep_config.length);
        EXPECT_EQ(1, sender().worker()->rkey_config_count);
    }

protected:
    static bool
    select_elem_match(ucp_worker_h worker, const ucp_proto_select_elem_t &elem,
                      const proto_select_data &data, size_t range_start)
    {
        ucp_proto_query_attr_t attr;
        if (!ucp_proto_select_elem_query(worker, &elem, range_start, &attr)) {
            return false;
        }

        return (data.range_end == attr.max_msg_length) &&
               (data.desc == attr.desc) && (data.config == attr.config);
    }

    static proto_select_data
    select_data_for_range(ucp_worker_h worker,
                          const ucp_proto_select_elem_t &elem,
                          size_t range_start)
    {
        ucp_proto_query_attr_t attr;

        if (ucp_proto_select_elem_query(worker, &elem, range_start, &attr)) {
            return proto_select_data(range_start, attr);
        }

        return proto_select_data();
    }

    static void dump_select_info(const worker_config &worker_cfg,
                                 const ucp_proto_select_param_t &select_param,
                                 const ucp_proto_select_elem_t &select_elem)
    {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
        ucp_proto_select_elem_info(worker_cfg.worker, worker_cfg.ep_cfg_index,
                                   worker_cfg.rkey_cfg_index, &select_param,
                                   &select_elem, 1, &strb);

        char *line;
        ucs_string_buffer_for_each_token(line, &strb, "\n") {
            UCS_TEST_MESSAGE << line;
        }
        ucs_string_buffer_cleanup(&strb);
    }

    static void
    check_proto_select_elem(const worker_config &worker_cfg,
                            const ucp_proto_select_param_t &select_param,
                            const ucp_proto_select_elem_t &select_elem,
                            const proto_select_data_vec_t &data_vec)
    {
        bool failed        = false;
        unsigned range_idx = 0;
        for (auto &expected_data : data_vec) {
            size_t range_start = expected_data.range_start;
            auto actual_data   = select_data_for_range(worker_cfg.worker,
                                                       select_elem, range_start);
            EXPECT_EQ(expected_data, actual_data)
                    << "unexpected difference at range["
                    << (failed = true, range_idx) << "]";

            /* As we cannot get range_start directly, we assert that protocol
             * is different at that range */
            if (range_start > 0) {
                auto prev_actual_data = select_data_for_range(worker_cfg.worker,
                                                              select_elem,
                                                              range_start - 1);
                EXPECT_NE(expected_data, prev_actual_data)
                        << "unexpected equality at range["
                        << (failed = true, range_idx) << "]";
            }
            ++range_idx;
        }
        if (failed) {
            dump_select_info(worker_cfg, select_param, select_elem);
        }
    }

    static void check_proto_select(const worker_config &worker_cfg,
                                   const ucp_proto_select_t &proto_select,
                                   const proto_select_data_vec_t &data_vec,
                                   const ucp_proto_select_key_t &key)
    {
        ucp_proto_select_elem_t select_elem;
        ucp_proto_select_key_t select_key;

        kh_foreach(proto_select.hash, select_key.u64, select_elem, {
            if (key_match(key, select_key)) {
                check_proto_select_elem(worker_cfg, select_key.param,
                                        select_elem, data_vec);
            }
        })
    }

    static bool
    key_match(ucp_proto_select_key_t req, ucp_proto_select_key_t actual)
    {
        /*
         * For each field of the protocol selection key check whether it has a
         * default value (UINT8_MAX - meaning a wildcard), or the fine-grained
         * setting matches was provided and matches the actual value.
         */
#define CMP_FIELD(FIELD) \
        if ((UINT8_MAX != req.param.FIELD) && \
            (req.param.FIELD != actual.param.FIELD)) { \
            return false; \
        }

        CMP_FIELD(op_id_flags);
        CMP_FIELD(op_attr);
        CMP_FIELD(dt_class);
        CMP_FIELD(mem_type);
        return true;
    }

    void send_recv_am(size_t size)
    {
        /* Prepare receiver data handler */
        mem_buffer recv_buf(size, UCS_MEMORY_TYPE_HOST);
        struct ctx_t {
            mem_buffer                     *buf;
            bool                            received;
            ucp_worker_h                    worker;
            ucp_am_recv_data_nbx_callback_t cmpl;
        } ctx = {&recv_buf, false, receiver().worker()};

        ctx.cmpl = [](void *req, ucs_status_t status, size_t len, void *arg) {
            ((ctx_t *)arg)->received = true;
            ucp_request_free(req);
        };

        auto cb = [](void *arg, const void *header, size_t header_length,
                     void *data, size_t len, const ucp_am_recv_param_t *param) {
            ctx_t *ctx = (ctx_t *)arg;

            if (param->recv_attr & UCP_AM_RECV_ATTR_FLAG_DATA) {
                memcpy(ctx->buf->ptr(), data, len);
                ctx->received = true;
                return UCS_OK;
            }

            ucs_assert(param->recv_attr & UCP_AM_RECV_ATTR_FLAG_RNDV);
            ucp_request_param_t params;
            params.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_USER_DATA;
            params.user_data    = arg;
            params.cb.recv_am   = ctx->cmpl;

            auto sptr = ucp_am_recv_data_nbx(ctx->worker, data, ctx->buf->ptr(),
                                             len, &params);
            EXPECT_FALSE(UCS_PTR_IS_ERR(sptr));
            return UCS_INPROGRESS;
        };

        /* Set receiver callback */
        ucp_am_handler_param_t am_param;
        am_param.field_mask = UCP_AM_HANDLER_PARAM_FIELD_ID |
                              UCP_AM_HANDLER_PARAM_FIELD_CB |
                              UCP_AM_HANDLER_PARAM_FIELD_ARG |
                              UCP_AM_HANDLER_PARAM_FIELD_FLAGS;
        am_param.id         = AM_ID;
        am_param.cb         = cb;
        am_param.arg        = &ctx;
        am_param.flags      = UCP_AM_FLAG_PERSISTENT_DATA;
        ASSERT_UCS_OK(ucp_worker_set_am_recv_handler(ctx.worker, &am_param));

        /* Send data */
        mem_buffer buf(size, UCS_MEMORY_TYPE_HOST);
        ucp_request_param_t param = {};
        auto sptr = ucp_am_send_nbx(sender().ep(), AM_ID, NULL, 0ul, buf.ptr(),
                                    buf.size(), &param);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sptr));

        /* Wait for completion */
        EXPECT_EQ(UCS_OK, request_wait(sptr));
        wait_for_flag(&ctx.received);
        EXPECT_TRUE(ctx.received);
    }
};

class test_ucp_proto_mock_rcx : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        /* Device with higher BW and latency */
        add_mock_iface("mock_0:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 2000;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 600e-9;
            iface_attr.latency.m         = 1e-9;
            iface_attr.cap.get.max_zcopy = 16384;
        });
        /* Device with smaller BW but lower latency */
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 208;
            iface_attr.bandwidth.shared = 24e9;
            iface_attr.latency.c        = 500e-9;
            iface_attr.latency.m        = 1e-9;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_1_lane, "IB_NUM_PATHS?=1",
           "MAX_RNDV_LANES=1")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* Prefer mock_0:1 iface for RNDV because it has larger BW */
    check_ep_config(sender(), {
        {0,     200,   "short",                "rc_mlx5/mock_1:1"},
        {201,   6650,  "copy-in",              "rc_mlx5/mock_1:1"},
        {6651,  8246,  "zero-copy",            "rc_mlx5/mock_1:1"},
        {8247,  21991, "multi-frag zero-copy", "rc_mlx5/mock_1:1"},
        {21992, INF,   "rendezvous zero-copy read from remote",
                       "rc_mlx5/mock_0:1"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, zero_rndv_perf_diff, "IB_NUM_PATHS?=1",
           "MAX_RNDV_LANES=1", "RNDV_PERF_DIFF=0")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {0,     200,   "short",                "rc_mlx5/mock_1:1"},
        {201,   6650,  "copy-in",              "rc_mlx5/mock_1:1"},
        {6651,  8246,  "zero-copy",            "rc_mlx5/mock_1:1"},
        {8247,  22502, "multi-frag zero-copy", "rc_mlx5/mock_1:1"},
        {22503, INF,   "rendezvous zero-copy read from remote",
                       "rc_mlx5/mock_0:1"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_2_lanes, "IB_NUM_PATHS?=2",
           "MAX_RNDV_LANES=2")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* The optimal RNDV config must use mock_0:1 and mock_1:1 proportionally. */
    check_ep_config(sender(), {
        {0,     200,   "short",                "rc_mlx5/mock_1:1/path0"},
        {201,   6650,  "copy-in",              "rc_mlx5/mock_1:1/path0"},
        {6651,  8246,  "zero-copy",            "rc_mlx5/mock_1:1/path0"},
        {8247,  19883, "multi-frag zero-copy", "rc_mlx5/mock_1:1/path0"},
        {19884, INF,   "rendezvous zero-copy read from remote",
         "47% on rc_mlx5/mock_1:1/path0 and 53% on rc_mlx5/mock_0:1/path0"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_send_recv_small_frag,
           "IB_NUM_PATHS?=2", "MAX_RNDV_LANES=2", "RNDV_THRESH=0")
{
    for (size_t i = 1024; i <= 65536; i += 1024) {
        send_recv_am(i);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx, rcx, "rc_x")

class test_ucp_proto_mock_cma : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_cma()
    {
        mock_transport("cma");
    }

    virtual void init() override
    {
        add_mock_iface();
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_cma, am_send_1_lane)
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {0,      92, "short",                                 "posix/memory"},
        {93,   5345, "copy-in",                               "posix/memory"},
        {5346, INF,  "rendezvous zero-copy read from remote", "cma/mock"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_cma, mm_cma, "posix,cma")

class test_ucp_proto_mock_tcp : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_tcp()
    {
        mock_transport("tcp");
    }

    virtual void init() override
    {
        add_mock_iface("mock", [](uct_iface_attr_t &iface_attr) {
            iface_attr.bandwidth.dedicated = 0;
            iface_attr.bandwidth.shared    = 100e9 / 8; /* 100Gb/s */
            iface_attr.latency.c           = 20e-6;
            iface_attr.latency.m           = 0;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_tcp, am_send_1_lane)
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {0,      8184,   "short",                                       "tcp/mock"},
        {8185,   65528,  "zero-copy",                                   "tcp/mock"},
        {65529,  366985, "multi-frag zero-copy",                        "tcp/mock"},
        {366986, INF,    "rendezvous zero-copy fenced write to remote", "tcp/mock"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_tcp, tcp, "tcp")

class test_ucp_proto_mock_self : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_self()
    {
        mock_transport("self");
    }

    virtual void init() override
    {
        add_mock_iface();
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_self, rkey_ptr)
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {0,    2284, "short",                                     "self/mock"},
        {2285, INF,  "rendezvous copy from mapped remote memory", "self/mock"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_self, self, "self")
