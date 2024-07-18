/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"
#include <unordered_map>

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <uct/base/uct_component.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucp/proto/proto_select.inl>
}

struct mock_t {
public:
    using func_t     = void (*)();
    using mock_map_t = std::unordered_map<func_t *, std::unique_ptr<mock_t>>;

    static void cleanup()
    {
        g_mock_map.clear();
    }

    template <typename Fn>
    static mock_t *setup(Fn *orig_ptr, Fn mock)
    {
        func_t *ptr    = reinterpret_cast<func_t *>(orig_ptr);
        func_t mock_fn = reinterpret_cast<func_t>(mock);
        auto it        = g_mock_map.find(ptr);
        if (it == g_mock_map.end()) {
            mock_t * new_mock = new mock_t(ptr, mock_fn);
            g_mock_map[ptr] = std::unique_ptr<mock_t>(new_mock);
            return new_mock;
        }

        it->second->m_mock = mock_fn;
        return it->second.get();
    }

    template <typename Fn, typename... Args>
    static ucs_status_t actual_call(Fn *orig_ptr, Args&&... args)
    {
        auto it = g_mock_map.find(reinterpret_cast<func_t *>(orig_ptr));
        Fn orig_func = (it == g_mock_map.end()) ?
                            *orig_ptr :
                            reinterpret_cast<Fn>(it->second->m_orig);
        return orig_func(std::forward<Args>(args)...);
    }

private:
    mock_t(func_t *orig_ptr, func_t mock) :
        m_orig_ptr(orig_ptr), m_orig(*orig_ptr), m_mock(mock)
    {
        *m_orig_ptr = m_mock;
    }

    ~mock_t()
    {
        *m_orig_ptr = m_orig;
    }

    friend std::unique_ptr<mock_t>::deleter_type;

    func_t *m_orig_ptr;
    func_t  m_orig;
    func_t  m_mock;

    static mock_map_t g_mock_map;
};

mock_t::mock_map_t mock_t::g_mock_map;


class mock_component {
public:
    using perf_attr_func_t  = std::function<void(uct_perf_attr_t &)>;
    using iface_attr_func_t = std::function<void(uct_iface_attr &)>;

    static mock_component &setup(const std::string &name)
    {
        uct_component_h component = find_uct_component(name);
        if (nullptr == component) {
            UCS_TEST_SKIP_R("Component " + name + " not found");
        }

        mock_component *mock_comp = find(component);
        if (nullptr == mock_comp) {
            mock_comp = new mock_component(component);
            g_component_map.emplace(component,
                                    std::unique_ptr<mock_component>(mock_comp));
        }

        return *mock_comp;
    }

    static void cleanup()
    {
        g_component_map.clear();
        mock_t::cleanup();
    }

    uct_component_h operator->() const
    {
        return m_component;
    }

    void set_mock_perf_attr(const std::string &iface_name, perf_attr_func_t cb)
    {
        m_perf_attrs_funcs[iface_name] = cb;
    }

    void set_mock_iface_attr(const std::string &iface_name,
                             iface_attr_func_t cb)
    {
        m_iface_attrs_funcs[iface_name] = cb;
    }

private:
    mock_component(uct_component_h component) :
        m_component(component),
        m_flags_orig(component->flags)
    {
        uct_tl_t *tl;
        ucs_list_for_each(tl, &component->tl_list, list) {
            mock_t::setup(&tl->iface_open, iface_open_mock);
        }
    }

    ~mock_component()
    {
        m_component->flags = m_flags_orig;
    }

    static uct_component_h find_uct_component(const std::string &name)
    {
        uct_component_h component;
        ucs_list_for_each(component, &uct_components_list, list) {
            if (name == component->name) {
                return component;
            }
        }

        return nullptr;
    }

    static mock_component * find(uct_component_h component)
    {
        auto it = g_component_map.find(component);
        return (it == g_component_map.end()) ? nullptr : it->second.get();
    }

    uct_tl_t * find_tl(const uct_iface_params_t *params) const
    {
        uct_tl_t *tl;
        ucs_list_for_each(tl, &m_component->tl_list, list) {
            if (0 == strcmp(tl->name, params->mode.device.tl_name)) {
                return tl;
            }
        }

        return nullptr;
    }

    static ucs_status_t iface_open_mock(uct_md_h md, uct_worker_h worker,
                                        const uct_iface_params_t *params,
                                        const uct_iface_config_t *config,
                                        uct_iface_h *iface_p)
    {
        mock_component *self = find(md->component);
        uct_tl_t *tl         = self->find_tl(params);
        ucs_status_t status  = mock_t::actual_call(&tl->iface_open, md, worker,
                                                   params, config, iface_p);
        if (status != UCS_OK) {
            return status;
        }

        uct_base_iface_t *base    = ucs_derived_of(*iface_p, uct_base_iface_t);
        self->m_iface_names[base] = std::string(params->mode.device.tl_name) +
                                    "/" + params->mode.device.dev_name;

        mock_t::setup(&base->internal_ops->iface_estimate_perf,
                      iface_estimate_perf_mock);
        mock_t::setup(&(*iface_p)->ops.iface_query, iface_query_mock);
        return status;
    }

    static ucs_status_t
    iface_estimate_perf_mock(uct_iface_h iface, uct_perf_attr_t *perf_attr)
    {
        uct_base_iface_t *base = ucs_derived_of(iface, uct_base_iface_t);
        ucs_status_t status    =
            mock_t::actual_call(&base->internal_ops->iface_estimate_perf, iface,
                                perf_attr);
        if (status != UCS_OK) {
            return status;
        }

        mock_component *self = find(base->md->component);
        auto it = self->m_perf_attrs_funcs.find(self->m_iface_names[base]);
        if (it != self->m_perf_attrs_funcs.end()) {
            (it->second)(*perf_attr);
        }

        return status;
    }

    static ucs_status_t
    iface_query_mock(uct_iface_h iface, uct_iface_attr_t *iface_attr)
    {
        ucs_status_t status =
            mock_t::actual_call(&iface->ops.iface_query, iface, iface_attr);
        if (status != UCS_OK) {
            return status;
        }

        uct_base_iface_t *base = ucs_derived_of(iface, uct_base_iface_t);
        mock_component *self   = find(base->md->component);
        auto it = self->m_iface_attrs_funcs.find(self->m_iface_names[base]);
        if (it != self->m_iface_attrs_funcs.end()) {
            (it->second)(*iface_attr);
        }

        return status;
    }

    friend std::unique_ptr<mock_component>::deleter_type;

    uct_component_h                           m_component;
    uint64_t                                  m_flags_orig;
    std::unordered_map<uct_base_iface_t *, std::string> m_iface_names;
    std::unordered_map<std::string, iface_attr_func_t>  m_iface_attrs_funcs;
    std::unordered_map<std::string, perf_attr_func_t>   m_perf_attrs_funcs;

    using component_map_t = std::unordered_map<uct_component_h,
                                               std::unique_ptr<mock_component>>;
    static component_map_t g_component_map;
};

mock_component::component_map_t mock_component::g_component_map;


struct proto_select_data {
    std::string range_start;
    std::string range_end;
    std::string desc;
    std::string config;
};

using proto_select_data_vec_t = std::vector<proto_select_data>;

class test_ucp_proto_select : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant>& variants)
    {
        add_variant(variants, UCP_FEATURE_TAG | UCP_FEATURE_AM);
    }

    virtual void init() override
    {
        modify_config("PROTO_ENABLE", "y");
        modify_config("MAX_RNDV_LANES", "1");
        modify_config("TOPO_PRIO", "default");
    }

    virtual void cleanup() override
    {
        ucp_test::cleanup();
        mock_component::cleanup();
    }

    static void
    check_ep_config(entity &e, const proto_select_data_vec_t &data,
                    ucp_proto_select_key_t key)
    {
        ucp_ep_config_t *config = ucp_worker_ep_config(e.worker(), 0);
        check_proto_select(e.worker(), config->proto_select, data, key);
    }

    static void
    check_rkey_config(entity &e, const proto_select_data_vec_t &data,
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
        ucp_test::init();
        sender().connect(&receiver(), get_ep_params());
        for (int i = 0; i < 10; ++i) {
            progress();
        }

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
        int valid = ucp_proto_select_elem_query(worker, &select_elem,
                                                range_start, &proto_attr);
        if (!valid) {
            return false;
        }

        size_t range_end = string_to_memunits(data.range_end);
        return (range_end == proto_attr.max_msg_length) &&
               (data.desc == proto_attr.desc) &&
               (data.config == proto_attr.config);
    }

    static void dump_select_elem(ucp_worker_h worker,
                                 const ucp_proto_select_elem_t &select_elem)
    {
        size_t range_end = -1;
        size_t range_start;
        do {
            range_start = range_end + 1;

            ucp_proto_query_attr_t proto_attr;
            int valid = ucp_proto_select_elem_query(worker, &select_elem,
                                                    range_start, &proto_attr);
            range_end = proto_attr.max_msg_length;
            if (valid) {
            char buf[1024];
                sprintf(buf, "  %zu-%zu desc: %s, config: %s", range_start,
                range_end, proto_attr.desc, proto_attr.config);
                UCS_TEST_MESSAGE << buf;
            }
        } while (range_end != SIZE_MAX);
    }

    static void
    check_proto_select_elem(ucp_worker_h worker,
                            const ucp_proto_select_elem_t &select_elem,
                            const proto_select_data_vec_t &data)
    {
        dump_select_elem(worker, select_elem);
        for (auto &it : data) {
            size_t start = string_to_memunits(it.range_start);
            EXPECT_TRUE(select_elem_match(worker, select_elem, it, start));
            /* As we cannot get range_start directly, we assert that protocol
             * is different at that range */
            if (start > 0) {
                EXPECT_FALSE(select_elem_match(worker, select_elem, it,
                             start - 1));
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
                UCS_TEST_MESSAGE << "Key op flags: "
                                 << (int)select_key.param.op_id_flags
                                 << ", attr: " << (int)select_key.param.op_attr;
                check_proto_select_elem(worker, select_elem, data);
            });
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

UCS_TEST_P(test_ucp_proto_select, mock_perf_attr, "NET_DEVICES=mlx5_0:1")
{
    mock_component::setup("ib").set_mock_iface_attr("rc_mlx5/mlx5_0:1",
        [](uct_iface_attr_t &iface_attr) {
            iface_attr.dev_num_paths    = 1;
            iface_attr.bandwidth.shared = 10000000000;
            iface_attr.latency.c        = 0.000006;
            iface_attr.latency.m        = 0.000000001;
        });

    connect();

    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {"0",          "2038",   "short",                "rc_mlx5/mlx5_0:1"},
        {"2039",       "8246",   "zero-copy",            "rc_mlx5/mlx5_0:1"},
        {"8247",       "288939", "multi-frag zero-copy", "rc_mlx5/mlx5_0:1"},
        {"288940",     "1G",     "rendezvous zero-copy read from remote",
                                                         "rc_mlx5/mlx5_0:1"},
        {"1073741825", "inf",    "rendezvous zero-copy fenced write to remote",
                                                         "rc_mlx5/mlx5_0:1"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_select, rcx, "rc_x")
