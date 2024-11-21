/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <uct/base/uct_component.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucp/proto/proto_select.inl>
}

class mock_iface {
public:
    /* Can't use std::function due to coverity errors */
    using perf_attr_func_t  = void (*)(uct_perf_attr_t&);
    using iface_attr_func_t = void (*)(uct_iface_attr&);

    mock_iface()
    {
        ucs_assert(singleton == nullptr);
        singleton = this;
    }

    ~mock_iface()
    {
        cleanup();
        singleton = nullptr;
    }

    void cleanup()
    {
        m_mock.cleanup();
    }

    void
    set_mock_perf_attr(const std::string &tl_name, perf_attr_func_t cb)
    {
        mock_tl(tl_name);
        m_perf_attrs_funcs[tl_name] = cb;
    }

    void
    set_mock_iface_attr(const std::string &tl_name, iface_attr_func_t cb)
    {
        mock_tl(tl_name);
        m_iface_attrs_funcs[tl_name] = cb;
    }

private:
    void mock_tl(const std::string &tl_name)
    {
        uct_component_h component;
        ucs_list_for_each(component, &uct_components_list, list) {
            uct_tl_t *tl;
            ucs_list_for_each(tl, &component->tl_list, list) {
                if (tl_name == tl->name) {
                    EXPECT_EQ(0, m_tls.count(tl_name));
                    m_tls[tl_name] = tl;
                    m_mock.setup(&tl->iface_open, iface_open_mock);
                }
            }
        }
    }

    static ucs_status_t
    iface_open_mock(uct_md_h md, uct_worker_h worker,
                    const uct_iface_params_t *params,
                    const uct_iface_config_t *config, uct_iface_h *iface_p)
    {
        mock_iface *self = singleton;
        uct_tl_t *tl     = self->m_tls[params->mode.device.tl_name];
        ucs_status_t status;

        status = self->m_mock.orig_func(&tl->iface_open, md, worker, params,
                                        config, iface_p);
        if (status != UCS_OK) {
            return status;
        }

        uct_base_iface_t *base    = ucs_derived_of(*iface_p, uct_base_iface_t);
        self->m_iface_names[base] = params->mode.device.tl_name;

        self->m_mock.setup(&base->internal_ops->iface_estimate_perf,
                           iface_estimate_perf_mock);
        self->m_mock.setup(&(*iface_p)->ops.iface_query, iface_query_mock);
        return status;
    }

    static ucs_status_t
    iface_estimate_perf_mock(uct_iface_h iface, uct_perf_attr_t *perf_attr)
    {
        mock_iface *self       = singleton;
        uct_base_iface_t *base = ucs_derived_of(iface, uct_base_iface_t);
        ucs_status_t status    = self->m_mock.orig_func(
                                    &base->internal_ops->iface_estimate_perf,
                                    iface, perf_attr);
        if (status != UCS_OK) {
            return status;
        }

        auto it = self->m_perf_attrs_funcs.find(self->m_iface_names[base]);
        if (it != self->m_perf_attrs_funcs.end()) {
            (it->second)(*perf_attr);
        }

        return status;
    }

    static ucs_status_t
    iface_query_mock(uct_iface_h iface, uct_iface_attr_t *iface_attr)
    {
        mock_iface *self    = singleton;
        ucs_status_t status = self->m_mock.orig_func(
                                    &iface->ops.iface_query, iface, iface_attr);
        if (status != UCS_OK) {
            return status;
        }

        uct_base_iface_t *base = ucs_derived_of(iface, uct_base_iface_t);
        auto it = self->m_iface_attrs_funcs.find(self->m_iface_names[base]);
        if (it != self->m_iface_attrs_funcs.end()) {
            (it->second)(*iface_attr);
        }

        return status;
    }

    /* We have to use singleton to mock C functions */
    static mock_iface *singleton;

    ucs::mock                                           m_mock;
    std::unordered_map<std::string, uct_tl_t *>         m_tls;
    std::unordered_map<uct_base_iface_t *, std::string> m_iface_names;
    std::unordered_map<std::string, iface_attr_func_t>  m_iface_attrs_funcs;
    std::unordered_map<std::string, perf_attr_func_t>   m_perf_attrs_funcs;
};

mock_iface *mock_iface::singleton = nullptr;

struct proto_select_data {
    std::string range_start;
    std::string range_end;
    std::string desc;
};

using proto_select_data_vec_t = std::vector<proto_select_data>;

class test_ucp_proto_mock : public ucp_test, public mock_iface {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_AM);
    }

    virtual void init() override
    {
        modify_config("PROTO_ENABLE", "y");
        modify_config("MAX_RNDV_LANES", "1");

        /* Reset topo provider to force reload from config */
        ucs_sys_topo_reset_provider();
        modify_config("TOPO_PRIO", "default");
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

    static ucp_proto_select_key_t any_key()
    {
        ucp_proto_select_key_t key;
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
    static size_t string_to_memunits(const std::string &str)
    {
        size_t range;
        EXPECT_UCS_OK(ucs_str_to_memunits(str.c_str(), &range));
        return range;
    }

    static bool select_elem_match(ucp_worker_h worker,
                                  const ucp_proto_select_elem_t &select_elem,
                                  const proto_select_data &data,
                                  size_t range_start)
    {
        ucp_proto_query_attr_t proto_attr;
        if (!ucp_proto_select_elem_query(worker, &select_elem, range_start,
                                         &proto_attr)) {
            return false;
        }

        size_t range_end = string_to_memunits(data.range_end);
        return (range_end == proto_attr.max_msg_length) &&
               (data.desc == proto_attr.desc);
    }

    static void dump_select_elem(ucp_worker_h worker,
                                 const ucp_proto_select_elem_t &select_elem)
    {
        size_t range_end = -1;
        size_t range_start;
        do {
            range_start = range_end + 1;
            ucp_proto_query_attr_t proto_attr;
            if (ucp_proto_select_elem_query(worker, &select_elem, range_start,
                                            &proto_attr)) {
                UCS_TEST_MESSAGE << range_start <<  "-"
                                 << proto_attr.max_msg_length
                                 << " desc: " << proto_attr.desc << ", config: "
                                 << proto_attr.config;
            }

            range_end = proto_attr.max_msg_length;
        } while (range_end != SIZE_MAX);
    }

    static void
    check_proto_select_elem(ucp_worker_h worker,
                            const ucp_proto_select_elem_t &select_elem,
                            const proto_select_data_vec_t &data)
    {
        for (auto &it : data) {
            size_t start = string_to_memunits(it.range_start);
            EXPECT_TRUE(select_elem_match(worker, select_elem, it, start));
            /* As we cannot get range_start directly, we assert that protocol
             * is different at that range */
            if (start > 0) {
                EXPECT_FALSE(
                        select_elem_match(worker, select_elem, it, start - 1));
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

UCS_TEST_P(test_ucp_proto_mock, mock_iface_attr)
{
    set_mock_iface_attr("rc_mlx5",
        [](uct_iface_attr_t &iface_attr) {
            iface_attr.dev_num_paths    = 1;
            iface_attr.cap.am.max_short = 208;
            iface_attr.bandwidth.shared = 10000000000;
            iface_attr.latency.c        = 0.000006;
            iface_attr.latency.m        = 0.000000001;
        });

    ucp_test::init();
    connect();

    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {"0",      "200",    "short"},
        {"201",    "8246",   "copy-in"},
        {"8247",   "377094", "multi-frag copy-in"},
        {"377095", "inf",    "rendezvous zero-copy read from remote"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock, rcx, "rc_x")
