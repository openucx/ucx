/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <uct/base/uct_iface.h>
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

    void add_mock_iface(const std::string &dev_name, iface_attr_func_t cb)
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
                    m_mock.setup(&component->query_md_resources, query_md_mock);
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
    static ucs_status_t query_md_mock(uct_component_t *component,
                                      uct_md_resource_desc_t **resources_p,
                                      unsigned *num_resources_p)
    {
        UCS_MOCK_ORIG_FUNC(m_self->m_mock, &component->query_md_resources,
                           component, resources_p, num_resources_p);
        /* Keep only the first available MD */
        *num_resources_p = ucs_min(*num_resources_p, 1);
        return UCS_OK;
    }

    static ucs_status_t
    query_devices_mock(uct_md_h md, uct_tl_device_resource_t **tl_devices_p,
                       unsigned *num_tl_devices_p)
    {
        UCS_MOCK_ORIG_FUNC(m_self->m_mock, &m_self->m_tl->query_devices, md,
                           tl_devices_p, num_tl_devices_p);
        if (*num_tl_devices_p == 0) {
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
        m_self->m_real_dev_name  = (*tl_devices_p)[0].name;
        unsigned size            = m_self->m_iface_attrs_funcs.size();
        auto mock_devices        = (uct_tl_device_resource_t *)ucs_malloc(
                                        sizeof(uct_tl_device_resource_t) * size,
                                        "mock_tl_devices");
        ucs_sys_device_t sys_dev = 1;
        unsigned i               = 0;
        for (const auto &it : m_self->m_iface_attrs_funcs) {
            ucs_strncpy_safe(mock_devices[i].name, it.first.c_str(),
                             UCT_DEVICE_NAME_MAX);
            mock_devices[i].type         = (*tl_devices_p)[0].type;
            mock_devices[i++].sys_device = sys_dev++;
        }

        ucs_free(*tl_devices_p);
        *tl_devices_p     = mock_devices;
        *num_tl_devices_p = size;
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

struct proto_select_data {
    size_t      range_start;
    size_t      range_end;
    std::string desc;
    std::string config;
};

using proto_select_data_vec_t = std::vector<proto_select_data>;

class test_ucp_proto_mock : public ucp_test, public mock_iface {
public:
    const static size_t INF = UCS_MEMUNITS_INF;

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

    static void check_ep_config(entity &e, const proto_select_data_vec_t &data,
                                ucp_proto_select_key_t key)
    {
        ucp_ep_config_t *config = ucp_worker_ep_config(e.worker(), 0);
        check_proto_select(e.worker(), config->proto_select, data, key);
    }

    static void check_rkey_config(entity &e,
                                  const proto_select_data_vec_t &data,
                                  ucp_proto_select_key_t key)
    {
        ucp_rkey_config_t *config = &e.worker()->rkey_config[0];
        check_proto_select(e.worker(), config->proto_select, data, key);
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
        wait_for_cond([this] {
            return (receiver().worker()->ep_config.length > 0);
        }, [this] { progress(); });

        EXPECT_EQ(1, sender().worker()->ep_config.length);
        EXPECT_EQ(1, sender().worker()->rkey_config_count);
        EXPECT_EQ(1, receiver().worker()->ep_config.length);
        EXPECT_EQ(1, receiver().worker()->rkey_config_count);
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

    static void dump_select_elem(ucp_worker_h worker,
                                 const ucp_proto_select_elem_t &elem)
    {
        size_t range_end = -1;
        size_t range_start;
        do {
            range_start = range_end + 1;
            ucp_proto_query_attr_t attr;
            if (ucp_proto_select_elem_query(worker, &elem, range_start, &attr)) {
                UCS_TEST_MESSAGE << range_start <<  "-" << attr.max_msg_length
                                 << " desc: " << attr.desc << ", config: "
                                 << attr.config;
            }

            range_end = attr.max_msg_length;
        } while (range_end != SIZE_MAX);
    }

    static void check_proto_select_elem(ucp_worker_h worker,
                                        const ucp_proto_select_elem_t &elem,
                                        const proto_select_data_vec_t &data)
    {
        for (auto &it : data) {
            EXPECT_TRUE(select_elem_match(worker, elem, it, it.range_start));
            /* As we cannot get range_start directly, we assert that protocol
             * is different at that range */
            if (it.range_start > 0) {
                EXPECT_FALSE(select_elem_match(worker, elem, it,
                                               it.range_start - 1));
            }
        }
    }

    static void check_proto_select(ucp_worker_h worker,
                                   const ucp_proto_select_t &proto_select,
                                   const proto_select_data_vec_t &data,
                                   ucp_proto_select_key_t key)
    {
        ucp_proto_select_elem_t select_elem;
        ucp_proto_select_key_t select_key;

        kh_foreach(proto_select.hash, select_key.u64, select_elem,
            if (key_match(key, select_key)) {
                check_proto_select_elem(worker, select_elem, data);
            });

        if (testing::Test::HasFailure()) {
            kh_foreach(proto_select.hash, select_key.u64, select_elem,
                if (key_match(key, select_key)) {
                    UCS_TEST_MESSAGE << "Key op flags: "
                                     << (int)select_key.param.op_id_flags
                                     << ", attr: "
                                     << (int)select_key.param.op_attr;
                    dump_select_elem(worker, select_elem);
                });
        }
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
            iface_attr.cap.am.max_short = 2000;
            iface_attr.bandwidth.shared = 28000000000;
            iface_attr.latency.c        = 0.0000006;
            iface_attr.latency.m        = 0.000000001;
        });
        /* Device with smaller BW but lower latency */
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 208;
            iface_attr.bandwidth.shared = 24000000000;
            iface_attr.latency.c        = 0.0000005;
            iface_attr.latency.m        = 0.000000001;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_1_lane,
           "IB_NUM_PATHS?=1", "MAX_RNDV_LANES=1")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* Prefer mock_0:1 iface for RNDV because it has larger BW */
    check_ep_config(sender(), {
        {0,     200,   "short",                "rc_mlx5/mock_1:1"},
        {201,   6650,  "copy-in",              "rc_mlx5/mock_1:1"},
        {6651,  8246,  "zero-copy",            "rc_mlx5/mock_1:1"},
        {8247,  22502, "multi-frag zero-copy", "rc_mlx5/mock_1:1"},
        {22503, INF,   "rendezvous zero-copy read from remote",
                       "rc_mlx5/mock_0:1"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_2_lanes,
           "IB_NUM_PATHS?=2", "MAX_RNDV_LANES=2")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* The optimal RNDV config must use mock_0:1 and mock_1:1 proportionally. */
    check_ep_config(sender(), {
        {0,     200,   "short",                "rc_mlx5/mock_1:1/path0"},
        {201,   6650,  "copy-in",              "rc_mlx5/mock_1:1/path0"},
        {6651,  8246,  "zero-copy",            "rc_mlx5/mock_1:1/path0"},
        {8247,  20300, "multi-frag zero-copy", "rc_mlx5/mock_1:1/path0"},
        {20301, INF,   "rendezvous zero-copy read from remote",
         "47% on rc_mlx5/mock_1:1/path0 and 53% on rc_mlx5/mock_0:1/path0"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx, rcx, "rc_x")
