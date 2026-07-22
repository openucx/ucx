/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucp_test.h"

extern "C" {
#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_types.h>
#include <uct/base/uct_iface.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_select.inl>
#include <ucs/memory/numa.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/topo/base/topo.h>
#include <ucs/type/cpu_set.h>

#if HAVE_IB
#include <uct/ib/base/ib_md.h>
#endif
}

class mock_iface {
public:
    using iface_attr_func_t = std::function<void(uct_iface_attr&)>;
    using perf_attr_func_t  = std::function<void(uct_perf_attr_t&)>;

    mock_iface() : m_tl(nullptr), m_real_md(nullptr)
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

    void add_mock_iface(const std::string &dev_name = "mock",
                        iface_attr_func_t cb =
                                [](uct_iface_attr_t &iface_attr) {},
                        perf_attr_func_t perf_cb = default_perf_mock,
                        ucs_sys_device_t sys_device = UCS_SYS_DEVICE_ID_UNKNOWN)
    {
        m_iface_attrs_funcs[dev_name] = std::move(cb);
        m_perf_attrs_funcs[dev_name]  = std::move(perf_cb);
        m_sys_devices[dev_name]       = sys_device;
    }

    void add_mock_iface_on_sys_device(
            const std::string &dev_name, ucs_sys_device_t sys_device,
            iface_attr_func_t cb = [](uct_iface_attr_t &iface_attr) {},
            perf_attr_func_t perf_cb = default_perf_mock)
    {
        add_mock_iface(dev_name, std::move(cb), std::move(perf_cb),
                       sys_device);
    }

    /* Return the sys_dev assigned to a mock device during topology
     * registration in query_devices_mock(). */
    ucs_sys_device_t get_mock_sys_dev_by_name(const std::string &dev_name) const
    {
        return m_sys_devs_by_name.at(dev_name);
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

    void mock_cuda_ipc_remote_pid(ucp_worker_h worker)
    {
        ucp_context_h context = worker->context;

        for (ucp_rsc_index_t rsc_index = 0; rsc_index < context->num_tls;
             ++rsc_index) {
            const uct_tl_resource_desc_t *tl_rsc =
                    &context->tl_rscs[rsc_index].tl_rsc;

            if (std::string(tl_rsc->tl_name) != "cuda_ipc") {
                continue;
            }

            if (!UCS_STATIC_BITMAP_GET(context->tl_bitmap, rsc_index)) {
                continue;
            }

            ucp_worker_iface_t *wiface = ucp_worker_iface(worker, rsc_index);
            m_mock.setup(&wiface->iface->ops.iface_get_address,
                         cuda_ipc_get_address_mock);
        }
    }

#if HAVE_IB
    void ib_event(enum ibv_event_type event_type, uint8_t port_num)
    {
        uct_ib_async_event_t event = {};
        event.event_type = event_type;
        event.port_num   = port_num;
        uct_ib_md_t *md  = reinterpret_cast<uct_ib_md_t *>(m_self->m_real_md);
        uct_ib_handle_async_event(&md->dev, &event);
    }
#endif

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
            m_self->m_real_md       = md;
        } else if (m_self->m_real_dev_name != first_dev_name) {
            *num_tl_devices_p = 0;
            return UCS_OK;
        }

        /*
         * The number of real devices (and their names) do not match the mocked
         * ones. In order to pretend that all the mocked devices are supported,
         * we remember the first real device name, and then substitute the
         * response with the mocked devices names. Each mocked device is
         * assigned a distinct sys_device: either the one explicitly requested
         * via add_mock_iface(), or a freshly registered synthetic topology
         * device (so that mocked distances can be set) when none was requested.
         * The resulting sys_device is recorded per name for later lookup. Later
         * on the iface_open_mock will use the real device name (same for all
         * mocks) to create the mocked iface.
         */
        auto mock_devices  = (uct_tl_device_resource_t*)ucs_calloc(
                                    m_self->m_iface_attrs_funcs.size(),
                                    sizeof(uct_tl_device_resource_t),
                                    "mock_tl_devices");
        unsigned dev_count = 0;
        for (const auto &it : m_self->m_iface_attrs_funcs) {
            ucs_strncpy_safe(mock_devices[dev_count].name, it.first.c_str(),
                             UCT_DEVICE_NAME_MAX);
            mock_devices[dev_count].type = (*tl_devices_p)[0].type;

            ucs_sys_device_t sys_dev = m_self->m_sys_devices[it.first];
            if (sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN) {
                ucs_sys_bus_id_t bus_id = {
                    .domain   = 0xffff,
                    .bus      = 0xff,
                    .slot     = 0xff,
                    .function = static_cast<uint8_t>(dev_count),
                };

                auto status = ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev);
                ucs_assert_always(status == UCS_OK);
                ucs_assert_always(sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN);
            }

            mock_devices[dev_count].sys_device   = sys_dev;
            m_self->m_sys_devs_by_name[it.first] = sys_dev;
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
        m_self->m_mock.setup(&base->internal_ops->iface_estimate_perf, perf_mock);
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

    static ucs_status_t perf_mock(uct_iface_h iface, uct_perf_attr_t *perf_attr)
    {
        uct_base_iface_t *base = ucs_derived_of(iface, uct_base_iface_t);
        uct_iface_attr_t iface_attr;
        ucs_status_t status;

        UCS_MOCK_ORIG_FUNC(m_self->m_mock,
                           &base->internal_ops->iface_estimate_perf, iface,
                           perf_attr);

        if (perf_attr->field_mask & (UCT_PERF_ATTR_FIELD_BANDWIDTH |
                                     UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH)) {
            status = iface_query_mock(iface, &iface_attr);
            if (status != UCS_OK) {
                return status;
            }

            if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_BANDWIDTH) {
                perf_attr->bandwidth = iface_attr.bandwidth;
            }

            if (perf_attr->field_mask & UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH) {
                perf_attr->path_bandwidth = iface_attr.bandwidth;
            }
        }

        std::string &iface_name = m_self->m_iface_names[base];
        auto it                 = m_self->m_perf_attrs_funcs.find(iface_name);
        (it->second)(*perf_attr);
        return UCS_OK;
    }

    static ucs_status_t
    cuda_ipc_get_address_mock(uct_iface_h iface, uct_iface_addr_t *iface_addr)
    {
        UCS_MOCK_ORIG_FUNC(m_self->m_mock, &iface->ops.iface_get_address, iface,
                           iface_addr);

        *(pid_t*)iface_addr = getpid() + 1;
        return UCS_OK;
    }

    static void default_perf_mock(uct_perf_attr_t& perf_attr)
    {
        if (ucs_test_all_flags(perf_attr.field_mask,
                               UCT_PERF_ATTR_FIELD_BANDWIDTH |
                               UCT_PERF_ATTR_FIELD_PATH_BANDWIDTH)) {
            perf_attr.path_bandwidth = perf_attr.bandwidth;
        }
    }

    /* We have to use singleton to mock C functions */
    static mock_iface *m_self;

    ucs::mock                                           m_mock;
    uct_tl_t                                           *m_tl;
    std::unordered_map<uct_base_iface_t *, std::string> m_iface_names;
    std::map<std::string, iface_attr_func_t>            m_iface_attrs_funcs;
    std::map<std::string, perf_attr_func_t>             m_perf_attrs_funcs;
    std::map<std::string, ucs_sys_device_t>             m_sys_devices;
    std::map<std::string, ucs_sys_device_t>             m_sys_devs_by_name;
    std::string                                         m_real_dev_name;
    uct_md_h                                            m_real_md;
};

mock_iface *mock_iface::m_self = nullptr;

class test_ucp_proto_mock : public ucp_test, public mock_iface {
public:
    const static size_t INF                               = UCS_MEMUNITS_INF;
    const static uint16_t AM_ID                           = 123;

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

    using proto_select_data_vec_t = std::vector<proto_select_data>;

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants,
                    UCP_FEATURE_TAG | UCP_FEATURE_AM | UCP_FEATURE_RMA);
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
        modify_config("TOPO_PRIO", topo_prio());

        ucp_test::init();
        post_ucp_init();
        connect();
    }

    virtual void cleanup() override
    {
        mock_iface::cleanup();
        ucp_test::cleanup();
        /* Reset topo provider to not affect subsequent tests */
        ucs_sys_topo_reset_provider();
    }

    static void check_ep_config(const entity &e,
                                const proto_select_data_vec_t &data_vec,
                                ucp_proto_select_key_t key)
    {
        ucp_ep_config_t *config = ucp_worker_ep_config(e.worker(),
                                                       ep_config_index(e));
        check_proto_select(e, config->proto_select, data_vec, key);
    }

    static void check_rkey_config(const entity &e,
                                  const proto_select_data_vec_t &data_vec,
                                  ucp_proto_select_key_t key,
                                  ucp_worker_cfg_index_t rkey_cfg_index)
    {
        ucp_rkey_config_t *config = &ucs_array_elem(&e.worker()->rkey_config,
                                                    rkey_cfg_index);
        check_proto_select(e, config->proto_select, data_vec, key,
                           rkey_cfg_index);
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
        send_recv_am(1); /* Wait for connection establishment */
    }

protected:
    virtual const char *topo_prio() const
    {
        return "default";
    }

    virtual void post_ucp_init()
    {
    }

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

    static void dump_select_info(const entity &e,
                                 const ucp_proto_select_param_t &select_param,
                                 const ucp_proto_select_elem_t &select_elem,
                                 ucp_worker_cfg_index_t rkey_cfg_index)
    {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
        ucp_proto_select_elem_info(e.worker(), ep_config_index(e),
                                   rkey_cfg_index, &select_param,
                                   &select_elem, 1, 0, &strb);

        char *line;
        ucs_string_buffer_for_each_token(line, &strb, "\n") {
            UCS_TEST_MESSAGE << line;
        }
        ucs_string_buffer_cleanup(&strb);
    }

    static void
    check_proto_select_elem(const entity &e,
                            const ucp_proto_select_param_t &select_param,
                            const ucp_proto_select_elem_t &select_elem,
                            const proto_select_data_vec_t &data_vec,
                            ucp_worker_cfg_index_t rkey_cfg_index)
    {
        bool failed        = false;
        unsigned range_idx = 0;
        for (auto &expected_data : data_vec) {
            size_t range_start = expected_data.range_start;
            auto actual_data   = select_data_for_range(e.worker(), select_elem,
                                                       range_start);
            EXPECT_EQ(expected_data, actual_data)
                    << "unexpected difference at range["
                    << (failed = true, range_idx) << "]";

            /* As we cannot get range_start directly, we assert that protocol
             * is different at that range */
            if (range_start > 0) {
                auto prev_actual_data = select_data_for_range(e.worker(),
                                                              select_elem,
                                                              range_start - 1);
                EXPECT_NE(expected_data, prev_actual_data)
                        << "unexpected equality at range["
                        << (failed = true, range_idx) << "]";
            }
            ++range_idx;
        }
        if (failed) {
            dump_select_info(e, select_param, select_elem, rkey_cfg_index);
        }
    }

    static void check_proto_select(
            const entity &e, const ucp_proto_select_t &proto_select,
            const proto_select_data_vec_t &data_vec,
            const ucp_proto_select_key_t &key,
            ucp_worker_cfg_index_t rkey_cfg_index = UCP_WORKER_CFG_INDEX_NULL)
    {
        ucp_proto_select_elem_t select_elem;
        ucp_proto_select_key_t select_key;

        bool found = false;
        kh_foreach(proto_select.hash, select_key.u64, select_elem, {
            if (key_match(key, select_key)) {
                check_proto_select_elem(e, select_key.param, select_elem,
                                        data_vec, rkey_cfg_index);
                found = true;
            }
        })
        if (!found) {
            FAIL() << "Did not find matching protocol selection keys";
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

    void
    send_recv_am(size_t size, ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST)
    {
        /* Prepare receiver data handler */
        mem_buffer recv_buf(size, mem_type);
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
        mem_buffer buf(size, mem_type);
        ucp_request_param_t param = {};
        auto sptr = ucp_am_send_nbx(sender().ep(), AM_ID, NULL, 0ul, buf.ptr(),
                                    buf.size(), &param);
        EXPECT_FALSE(UCS_PTR_IS_ERR(sptr));

        /* Wait for completion */
        EXPECT_EQ(UCS_OK, request_wait(sptr));
        wait_for_flag(&ctx.received);
        EXPECT_TRUE(ctx.received);
    }

    void send_recv_am_range(size_t msg_start, size_t msg_end, size_t msg_step,
                            ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST)
    {
        for (size_t msg_size = msg_start; msg_size <= msg_end;
             msg_size += msg_step) {
            send_recv_am(msg_size, mem_type);
        }
    }

    static ucp_worker_cfg_index_t ep_config_index(const entity &e)
    {
        return e.ep()->cfg_index;
    }

    static ucs::handle<ucp_mem_h, ucp_context_h>
    mem_map(entity &e, void *address, size_t length,
            ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST)
    {
        ucp_mem_map_params_t mem_map_params;
        mem_map_params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                     UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                                     UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE;
        mem_map_params.address     = address;
        mem_map_params.length      = length;
        mem_map_params.memory_type = mem_type;
        ucp_mem_h mem;
        ASSERT_UCS_OK(ucp_mem_map(e.ucph(), &mem_map_params, &mem));
        return {
            mem,
            [](ucp_mem_h mem, ucp_context_h context) {
                 static_cast<void>(ucp_mem_unmap(context, mem));
            },
            e.ucph()
        };
    }

    static ucs::handle<ucp_mem_h, ucp_context_h>
    mem_map(entity &e, mem_buffer &buf)
    {
        return mem_map(e, buf.ptr(), buf.size(), buf.mem_type());
    }

    static ucs::handle<void*> rkey_pack(entity &e, ucp_mem_h memh)
    {
        void *rkey_buffer;
        size_t rkey_buffer_size;
        ASSERT_UCS_OK(ucp_rkey_pack(e.ucph(), memh, &rkey_buffer,
                                    &rkey_buffer_size));
        return {rkey_buffer, ucp_rkey_buffer_release};
    }

    static ucs::handle<ucp_rkey_h> rkey_unpack(ucp_ep_h ep, void *rkey_buffer)
    {
        ucp_rkey_h rkey;
        ASSERT_UCS_OK(ucp_ep_rkey_unpack(ep, rkey_buffer, &rkey));
        return {rkey, ucp_rkey_destroy};
    }

    ucp_worker_cfg_index_t
    send_recv_rma(size_t size, ucp_operation_id_t op_id,
                  ucs_memory_type_t mem_type = UCS_MEMORY_TYPE_HOST,
                  unsigned rkey_cfg_index = 1)
    {
        mem_buffer recv_buf(size, mem_type);
        recv_buf.pattern_fill(1);
        auto memh        = mem_map(receiver(), recv_buf);
        auto rkey_packed = rkey_pack(receiver(), memh);
        auto rkey        = rkey_unpack(sender().ep(), rkey_packed);

        mem_buffer send_buf(size, mem_type);
        send_buf.pattern_fill(2);

        ucp_request_param_t req_param;
        req_param.op_attr_mask = 0;
        ucs_status_ptr_t sptr;
        if (op_id == UCP_OP_ID_PUT) {
            sptr = ucp_put_nbx(sender().ep(), send_buf.ptr(), size,
                               (uint64_t)recv_buf.ptr(), rkey, &req_param);
        } else if (op_id == UCP_OP_ID_GET) {
            sptr = ucp_get_nbx(sender().ep(), send_buf.ptr(), size,
                               (uint64_t)recv_buf.ptr(), rkey, &req_param);
        } else {
            sptr = nullptr;
            ADD_FAILURE() << "Invalid operation ID: " << op_id;
            return UCP_WORKER_CFG_INDEX_NULL;
        }

        EXPECT_EQ(UCS_OK, request_wait(sptr));

        if (op_id == UCP_OP_ID_PUT) {
            recv_buf.pattern_check(2);
        } else if (op_id == UCP_OP_ID_GET) {
            send_buf.pattern_check(1);
        }

        auto actual_rkey_cfg_index = rkey->cfg_index;
        if (mem_type == UCS_MEMORY_TYPE_HOST) {
            EXPECT_EQ(actual_rkey_cfg_index, rkey_cfg_index);
        }

        return actual_rkey_cfg_index;
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
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 600e-9;
            iface_attr.latency.m         = 1e-9;
            iface_attr.cap.get.max_zcopy = 16384;
        });
        /* Device with smaller BW but lower latency */
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 24e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx, memtype_copy_enable,
           "IB_NUM_PATHS?=1", "MAX_RNDV_LANES=1",
           "MEMTYPE_COPY_ENABLE=n")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {0,       0, "rendezvous no data fetch", ""},
        {1,      64, "rendezvous zero-copy fenced write to remote",
                     "rc_mlx5/mock_0:1"},
        {21992, INF, "rendezvous zero-copy read from remote",
                     "rc_mlx5/mock_0:1"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_1_lane, "IB_NUM_PATHS?=1",
           "MAX_RNDV_LANES=1")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* Prefer mock_0:1 iface for RNDV because it has larger BW */
    check_ep_config(sender(), {
        {0,     200,   "short",                "rc_mlx5/mock_1:1"},
        {201,   404,   "copy-in",              "rc_mlx5/mock_1:1"},
        {405,   8246,  "zero-copy",            "rc_mlx5/mock_1:1"},
        {8247,  21145, "multi-frag zero-copy", "rc_mlx5/mock_1:1"},
        {21146, INF,   "rendezvous zero-copy read from remote",
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
        {201,   404,   "copy-in",              "rc_mlx5/mock_1:1"},
        {405,   8246,  "zero-copy",            "rc_mlx5/mock_1:1"},
        {8247,  21563, "multi-frag zero-copy", "rc_mlx5/mock_1:1"},
        {21564, INF,   "rendezvous zero-copy read from remote",
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
        {201,   404,   "copy-in",              "rc_mlx5/mock_1:1/path0"},
        {405,   8246,  "zero-copy",            "rc_mlx5/mock_1:1/path0"},
        {8247,  19149, "multi-frag zero-copy", "rc_mlx5/mock_1:1/path0"},
        {19150, INF,   "rendezvous zero-copy read from remote",
         "47% on rc_mlx5/mock_1:1/path0 and 53% on rc_mlx5/mock_0:1/path0"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_send_recv_small_frag,
           "IB_NUM_PATHS?=2", "MAX_RNDV_LANES=2", "RNDV_THRESH=0")
{
    send_recv_am_range(UCS_KBYTE, 64 * UCS_KBYTE, UCS_KBYTE);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rndv_4_paths,
           "IB_NUM_PATHS?=4", "MAX_RNDV_LANES=8", "RNDV_THRESH=0")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* All existing IB paths should be selected. */
    check_ep_config(sender(), {
        {1,       477,    "rendezvous fragmented copy-in copy-out",
         "rc_mlx5/mock_1:1/path0"},
        {478,     3813,   "rendezvous zero-copy", "rc_mlx5/mock_1:1/path0"},
        {3814,    283699, "rendezvous zero-copy read from remote",
         "12% on rc_mlx5/mock_1:1/path0, 14% on rc_mlx5/mock_0:1/path0, "
         "14% on rc_mlx5/mock_0:1/path1, 12% on rc_mlx5/mock_1:1/path1, 14%"},
        {283700,  INF,    "rendezvous zero-copy fenced write to remote",
         "12% on rc_mlx5/mock_1:1/path0, 14% on rc_mlx5/mock_0:1/path0, "
         "14% on rc_mlx5/mock_0:1/path1, 12% on rc_mlx5/mock_1:1/path1, 14%"},
    }, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx, rma_put_2_lanes,
           "IB_NUM_PATHS?=1", "MAX_RMA_RAILS=2")
{
    send_recv_rma(64 * UCS_KBYTE, UCP_OP_ID_PUT);

    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_PUT;
    key.param.op_attr          = 0;

    check_rkey_config(sender(), {
        {0,    2048, "short",     "rc_mlx5/mock_1:1"},
        {2049, INF,  "zero-copy", "47% on rc_mlx5/mock_1:1 and 53% on rc_mlx5/mock_0:1"},
    }, key, 0);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx, rcx, "rc_x")

class test_ucp_proto_mock_rcx2 : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx2()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        /* Device with high BW and lower latency */
        add_mock_iface("mock_0:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
            iface_attr.cap.get.max_zcopy = 16384;
        });
        /* Device with lower BW and higher latency */
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 2000;
            iface_attr.bandwidth.shared = 24e9;
            iface_attr.latency.c        = 600e-9;
            iface_attr.latency.m        = 1e-9;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx2, rndv_send_recv_small_frag,
           "IB_NUM_PATHS?=2", "MAX_RNDV_LANES=2", "RNDV_THRESH=0")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), {
        {1,   433, "rendezvous fragmented copy-in copy-out",
         "rc_mlx5/mock_0:1/path0"},
        {434, INF, "rendezvous zero-copy read from remote",
         "54% on rc_mlx5/mock_0:1/path0 and 46% on rc_mlx5/mock_1:1/path0"},
    }, key);

    send_recv_am_range(UCS_KBYTE, 64 * UCS_KBYTE, UCS_KBYTE);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx2, rcx, "rc_x")

class test_ucp_proto_mock_rcx3 : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx3()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        /* Device with high BW and lower latency, but 0 get_zcopy.
         * This use case is similar to cuda_ipc when NVLink is not available. */
        add_mock_iface("mock_0:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
            iface_attr.cap.get.max_zcopy = 0;
        });
        /* Device with lower BW and higher latency */
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 2000;
            iface_attr.bandwidth.shared = 24e9;
            iface_attr.latency.c        = 600e-9;
            iface_attr.latency.m        = 1e-9;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx3, single_lane_no_zcopy,
           "IB_NUM_PATHS?=1", "MAX_RNDV_LANES=2", "RNDV_THRESH=0")
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;

    /* Check that get_zcopy is selected on slower device */
    check_ep_config(sender(), {
        {1,    94,    "rendezvous fragmented copy-in copy-out", "rc_mlx5/mock_0:1"},
        {95,   53753, "rendezvous zero-copy read from remote",  "rc_mlx5/mock_1:1"},
        {53754, INF,  "rendezvous zero-copy fenced write to remote",
         "54% on rc_mlx5/mock_0:1 and 46% on rc_mlx5/mock_1:1"},
    }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx3, rcx, "rc_x")

class test_ucp_proto_mock_rcx_numa : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx_numa() :
        m_topo_state(nullptr), m_local_sys_dev(UCS_SYS_DEVICE_ID_UNKNOWN),
        m_remote_sys_dev(UCS_SYS_DEVICE_ID_UNKNOWN), m_affinity_set(false)
    {
        UCS_CPU_ZERO(&m_worker_cpu_mask);
        mock_transport("rc_mlx5");
    }

    virtual ucp_worker_params_t get_worker_params() override
    {
        ucp_worker_params_t params = ucp_test::get_worker_params();

        params.field_mask |= UCP_WORKER_PARAM_FIELD_CPU_MASK;
        params.cpu_mask    = m_worker_cpu_mask;
        return params;
    }

    virtual void init() override
    {
        auto iface_attr_func = [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 2000;
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
            iface_attr.cap.get.max_zcopy = 16384;
        };

        setup_numa_topology();
        add_mock_iface_on_sys_device("mock_0:1", m_remote_sys_dev,
                                     iface_attr_func);
        add_mock_iface_on_sys_device("mock_1:1", m_local_sys_dev,
                                     iface_attr_func);
        test_ucp_proto_mock::init();
    }

    virtual void cleanup() override
    {
        test_ucp_proto_mock::cleanup();
        cleanup_numa_topology();
    }

protected:
    virtual const char *topo_prio() const override
    {
        return "sysfs";
    }

private:
    void setup_numa_topology()
    {
        ucs_sys_cpuset_t test_affinity;
        ucs_numa_node_t local_node, remote_node;
        int local_cpu, remote_cpu;

        if (ucs_sys_getaffinity(&m_orig_affinity) != 0) {
            UCS_TEST_SKIP_R("failed to get process affinity");
        }

        find_numa_cpus(local_cpu, remote_cpu, local_node, remote_node);

        CPU_ZERO(&test_affinity);
        CPU_SET(local_cpu, &test_affinity);
        CPU_SET(remote_cpu, &test_affinity);
        if (ucs_sys_setaffinity(&test_affinity) != 0) {
            UCS_TEST_SKIP_R("failed to set process affinity");
        }

        m_affinity_set = true;
        UCS_CPU_ZERO(&m_worker_cpu_mask);
        UCS_CPU_SET(local_cpu, &m_worker_cpu_mask);

        m_topo_state = ucs_topo_extract_state();
        ASSERT_TRUE(m_topo_state != nullptr);

        add_fake_sys_device("mock_remote", 0, remote_node, &m_remote_sys_dev);
        add_fake_sys_device("mock_local", 1, local_node, &m_local_sys_dev);
    }

    void cleanup_numa_topology()
    {
        if (m_topo_state != nullptr) {
            ucs_topo_restore_state(m_topo_state);
            m_topo_state = nullptr;
            ucs_sys_topo_reset_provider();
        }

        if (m_affinity_set) {
            ucs_sys_setaffinity(&m_orig_affinity);
            m_affinity_set = false;
        }
    }

    void find_numa_cpus(int &local_cpu, int &remote_cpu,
                        ucs_numa_node_t &local_node,
                        ucs_numa_node_t &remote_node) const
    {
        unsigned cpu, num_cpus;

        local_cpu   = -1;
        remote_cpu  = -1;
        local_node  = UCS_NUMA_NODE_UNDEFINED;
        remote_node = UCS_NUMA_NODE_UNDEFINED;

        num_cpus = ucs_min(ucs_numa_num_configured_cpus(), UCS_CPU_SETSIZE);
        for (cpu = 0; cpu < num_cpus; ++cpu) {
            ucs_numa_node_t cpu_node;

            if (!CPU_ISSET(cpu, &m_orig_affinity)) {
                continue;
            }

            cpu_node = ucs_numa_node_of_cpu(cpu);
            if (cpu_node == UCS_NUMA_NODE_UNDEFINED) {
                continue;
            }

            if (local_cpu < 0) {
                local_cpu  = cpu;
                local_node = cpu_node;
            } else if (cpu_node != local_node) {
                remote_cpu  = cpu;
                remote_node = cpu_node;
                break;
            }
        }

        if (remote_cpu < 0) {
            UCS_TEST_SKIP_R("need CPUs from at least two NUMA nodes");
        }
    }

    static void
    add_fake_sys_device(const char *name, uint8_t function,
                        ucs_numa_node_t numa_node,
                        ucs_sys_device_t *sys_dev_p)
    {
        ucs_sys_bus_id_t bus_id;

        bus_id.domain   = 0xfffe;
        bus_id.bus      = 0xfe;
        bus_id.slot     = 0x1f;
        bus_id.function = function;

        ASSERT_UCS_OK(ucs_topo_find_device_by_bus_id(&bus_id, sys_dev_p));
        ASSERT_UCS_OK(ucs_topo_sys_device_set_name(*sys_dev_p, name, 10));
        ASSERT_UCS_OK(ucs_topo_sys_device_set_numa_node(*sys_dev_p,
                                                        numa_node));
    }

protected:
    ucs_global_state_t *m_topo_state;
    ucs_sys_cpuset_t    m_orig_affinity;
    ucs_cpu_set_t       m_worker_cpu_mask;
    ucs_sys_device_t    m_local_sys_dev;
    ucs_sys_device_t    m_remote_sys_dev;
    bool                m_affinity_set;
};

UCS_TEST_P(test_ucp_proto_mock_rcx_numa, worker_cpu_mask_affects_score,
           "IB_NUM_PATHS?=1", "MAX_RNDV_LANES=1", "MAX_EAGER_LANES=1")
{
    ucp_ep_config_t *config = ucp_worker_ep_config(sender().worker(),
                                                   ep_config_index(sender()));
    ucp_lane_index_t lane   = config->key.am_lane;
    ucp_rsc_index_t rsc_index;
    ucs_sys_device_t sys_dev;

    ASSERT_NE(UCP_NULL_LANE, lane);

    rsc_index = config->key.lanes[lane].rsc_index;
    sys_dev   = sender().worker()->context->tl_rscs[rsc_index].
                tl_rsc.sys_device;
    EXPECT_EQ(m_local_sys_dev, sys_dev);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_numa, rcx, "rc_x")

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
        {93,   5028, "copy-in",                               "posix/memory"},
        {5029, INF,  "rendezvous zero-copy read from remote", "cma/mock"},
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
            iface_attr.cap.am.max_zcopy    = 64 * UCS_KBYTE;
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
        {0,      0,      "short",                                       "tcp/mock"},
        {1,      65528,  "zero-copy",                                   "tcp/mock"},
        {65529,  367108, "multi-frag zero-copy",                        "tcp/mock"},
        {367109, INF,    "rendezvous zero-copy fenced write to remote", "tcp/mock"},
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

class test_ucp_proto_mock_gpu : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_gpu()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        add_mock_iface("mock", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 2000;
            iface_attr.bandwidth.shared = 28e9;
            iface_attr.latency.c        = 600e-9;
            iface_attr.latency.m        = 1e-9;
        });
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_gpu, cuda_managed_ppln_host_frag,
           "RNDV_FRAG_MEM_TYPES=host", "IB_NUM_PATHS?=1", "MAX_RNDV_LANES=1")
{
    send_recv_am(1024, UCS_MEMORY_TYPE_CUDA_MANAGED);

    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
    key.param.op_attr          = 0;
    key.param.mem_type         = UCS_MEMORY_TYPE_CUDA_MANAGED;

    check_ep_config(sender(), {
        {0, 0,    "short",   "rc_mlx5/mock"},
        {1, 8246, "copy-in", "rc_mlx5/mock"},
        {8247, 512 * UCS_KBYTE,
         "rendezvous cuda_copy, fenced write to remote, frag host, cuda_copy, frag host",
         "rc_mlx5/mock"},
        {(512 * UCS_KBYTE) + 1, INF,
         "rendezvous pipeline cuda_copy, fenced write to remote, frag host, cuda_copy, frag host",
         "rc_mlx5/mock"},
        }, key);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_gpu, rcx_gpu,
                              "rc_x,cuda,rocm")

class test_ucp_proto_mock_cuda_ipc : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_cuda_ipc()
    {
        if (has_transport("rc_x")) {
            mock_transport("rc_mlx5");
        }
    }

    virtual void init() override
    {
        if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA)) {
            UCS_TEST_SKIP_R("CUDA memory is not supported");
        }

        if (!has_transport("rc_x")) {
            UCS_TEST_SKIP_R("rc_mlx5 transport is not supported");
        }

        /* Keep protocol selection independent of NVLink probing. */
        modify_config("CUDA_IPC_ENABLE_GET_ZCOPY", "on", SETENV_IF_NOT_EXIST);

        add_mock_iface("mock", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
        });
        test_ucp_proto_mock::init();
    }

    virtual void post_ucp_init() override
    {
        mock_cuda_ipc_remote_pid(sender().worker());
        mock_cuda_ipc_remote_pid(receiver().worker());
    }

    void test_cuda_rma(ucp_operation_id_t op_id,
                       const proto_select_data_vec_t &data_vec)
    {
        auto rkey_cfg_index = send_recv_rma(UCS_MBYTE, op_id,
                                            UCS_MEMORY_TYPE_CUDA);
        ASSERT_NE(rkey_cfg_index, UCP_WORKER_CFG_INDEX_NULL);

        ucp_proto_select_key_t key = any_key();
        key.param.op_id_flags      = op_id;
        key.param.op_attr          = 0;
        key.param.mem_type         = UCS_MEMORY_TYPE_CUDA;

        check_rkey_config(sender(), data_vec, key, rkey_cfg_index);
    }
};

UCS_TEST_P(test_ucp_proto_mock_cuda_ipc, put, "IB_NUM_PATHS?=1")
{
    test_cuda_rma(UCP_OP_ID_PUT, {
        {0, 0,   "short",     "rc_mlx5/mock"},
        {1, INF, "zero-copy", "cuda_ipc/cuda"},
    });
}

UCS_TEST_P(test_ucp_proto_mock_cuda_ipc, get, "IB_NUM_PATHS?=1")
{
    test_cuda_rma(UCP_OP_ID_GET, {
        {0, 0,   "copy-out",  "rc_mlx5/mock"},
        {1, INF, "zero-copy", "cuda_ipc/cuda"},
    });
}

UCP_INSTANTIATE_TEST_CASE_TLS_GPU_AWARE(test_ucp_proto_mock_cuda_ipc,
                                        shm_rc_ipc, "rc_x,cuda_ipc,rocm_ipc")

class test_ucp_proto_mock_rcx_twins : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx_twins()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        auto iface_attr_func = [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
        };

        add_mock_iface("mock_0:1", iface_attr_func);
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 2000;
            iface_attr.bandwidth.shared = 24e9;
            iface_attr.latency.c        = 600e-9;
            iface_attr.latency.m        = 1e-9;
        });
        add_mock_iface("mock_2:1", iface_attr_func);
        test_ucp_proto_mock::init();
    }
};

class test_ucp_proto_mock_rcx_twins_tag : public test_ucp_proto_mock_rcx_twins {
protected:
    void check_config(const proto_select_data_vec_t &data_vec);
};

void test_ucp_proto_mock_rcx_twins_tag::check_config(
        const proto_select_data_vec_t &data_vec)
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_TAG_SEND;
    key.param.op_attr          = 0;

    check_ep_config(sender(), data_vec, key);
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_tag, use_all_net_devices,
           "IB_NUM_PATHS?=2")
{
    check_config(
            {{0, 200, "eager short", "rc_mlx5/mock_0:1/path0"},
             {201, 404, "eager copy-in copy-out", "rc_mlx5/mock_0:1/path0"},
             {405, 8246, "eager zero-copy copy-out", "rc_mlx5/mock_0:1/path0"},
             {8247, 18542, "multi-frag eager zero-copy copy-out",
              "rc_mlx5/mock_0:1/path0"},
             // SINGLE_NET_DEVICE=n [default]
             // Use two network devices with the same bandwidth in 50/50 ratio
             {18543, INF, "rendezvous zero-copy read from remote",
              "50% on rc_mlx5/mock_0:1/path0 and 50% on "
              "rc_mlx5/mock_2:1/path0"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_tag, use_single_net_device_0,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=0")
{
    check_config(
            {{0, 200, "eager short", "rc_mlx5/mock_0:1/path0"},
             {201, 404, "eager copy-in copy-out", "rc_mlx5/mock_0:1/path0"},
             {405, 8246, "eager zero-copy copy-out", "rc_mlx5/mock_0:1/path0"},
             {8247, 18542, "multi-frag eager zero-copy copy-out",
              "rc_mlx5/mock_0:1/path0"},
             // SINGLE_NET_DEVICE=y, NODE_LOCAL_ID=0
             // Use two paths of the zero network device in 50/50 ratio
             {18543, INF, "rendezvous zero-copy read from remote",
              "rc_mlx5/mock_0:1 50% on path0 and 50% on path1"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_tag, use_single_net_device_1,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=1")
{
    check_config(
            {{0, 200, "eager short", "rc_mlx5/mock_0:1/path0"},
             {201, 404, "eager copy-in copy-out", "rc_mlx5/mock_0:1/path0"},
             {405, 8246, "eager zero-copy copy-out", "rc_mlx5/mock_0:1/path0"},
             {8247, 18542, "multi-frag eager zero-copy copy-out",
              "rc_mlx5/mock_0:1/path0"},
             // SINGLE_NET_DEVICE=y, NODE_LOCAL_ID=1
             // Use two paths of the second network device in 50/50
             {18543, INF, "rendezvous zero-copy read from remote",
              "rc_mlx5/mock_2:1 50% on path0 and 50% on path1"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_tag, use_single_net_device_2,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=2")
{
    check_config(
            {{0, 200, "eager short", "rc_mlx5/mock_0:1/path0"},
             {201, 404, "eager copy-in copy-out", "rc_mlx5/mock_0:1/path0"},
             {405, 8246, "eager zero-copy copy-out", "rc_mlx5/mock_0:1/path0"},
             {8247, 18542, "multi-frag eager zero-copy copy-out",
              "rc_mlx5/mock_0:1/path0"},
             // SINGLE_NET_DEVICE=y, NODE_LOCAL_ID=2
             // Use two paths of the zero network device in 50/50 ratio
             {18543, INF, "rendezvous zero-copy read from remote",
              "rc_mlx5/mock_0:1 50% on path0 and 50% on path1"}});
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_twins_tag, rcx, "rc_x")

class test_ucp_proto_mock_rcx_twins_rndv_min :
    public test_ucp_proto_mock_rcx_twins_tag {
public:
    virtual void init() override
    {
        auto iface_attr_func = [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.cap.get.min_zcopy = 65;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
        };

        add_mock_iface("mock_0:1", iface_attr_func);
        add_mock_iface("mock_1:1", iface_attr_func);
        add_mock_iface("mock_2:1", iface_attr_func);
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_rndv_min,
           rndv_respects_lane_min_length,
           "IB_NUM_PATHS?=1", "RNDV_THRESH=1", "RNDV_SCHEME=get_zcopy",
           "MAX_RNDV_LANES=2")
{
    check_config(
            {{0, 129, "eager short", "rc_mlx5/mock_0:1"},
             {130, INF, "rendezvous zero-copy read from remote",
              "50% on rc_mlx5/mock_0:1 and 50% on rc_mlx5/mock_1:1"}});
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_twins_rndv_min, rcx,
                              "rc_x")

class test_ucp_proto_mock_rcx_twins_put : public test_ucp_proto_mock_rcx_twins {
protected:
    void check_config(const proto_select_data_vec_t &data_vec);
};

void test_ucp_proto_mock_rcx_twins_put::check_config(
        const proto_select_data_vec_t &data_vec)
{
    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_PUT;
    key.param.op_attr          = 0;

    check_rkey_config(sender(), data_vec, key, 0);
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_put, use_single_net_device_rank_0,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=0")
{
    check_config({{0, 2048, "short", "rc_mlx5/mock_0:1/path0"},
                  {2049, INF, "zero-copy", "rc_mlx5/mock_0:1/path0"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_put, use_single_net_device_rank_1,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=1")
{
    check_config({{0, 2048, "short", "rc_mlx5/mock_0:1/path0"},
                  {2049, INF, "zero-copy", "rc_mlx5/mock_2:1/path0"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_put, use_single_net_device_rank_2,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=2")
{
    check_config({{0, 2048, "short", "rc_mlx5/mock_0:1/path0"},
                  {2049, INF, "zero-copy", "rc_mlx5/mock_0:1/path0"}});
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_twins_put, rcx, "rc_x")

class test_ucp_proto_mock_rcx_same_bdf_put : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx_same_bdf_put() :
        m_low_user_value_sys_dev(UCS_SYS_DEVICE_ID_UNKNOWN),
        m_high_user_value_sys_dev(UCS_SYS_DEVICE_ID_UNKNOWN)
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        auto iface_attr_func = [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
        };

        setup_same_bdf_topology();
        add_mock_iface_on_sys_device("mock_0:1", m_high_user_value_sys_dev,
                                     iface_attr_func);
        add_mock_iface_on_sys_device("mock_1:1", m_low_user_value_sys_dev,
                                     iface_attr_func);
        test_ucp_proto_mock::init();
    }

protected:
    void check_config(const proto_select_data_vec_t &data_vec)
    {
        ucp_proto_select_key_t key = any_key();
        key.param.op_id_flags      = UCP_OP_ID_PUT;
        key.param.op_attr          = 0;

        check_rkey_config(sender(), data_vec, key, 0);
    }

private:
    void setup_same_bdf_topology()
    {
        ucs_sys_bus_id_t bus_id;

        bus_id.domain   = 0xfffd;
        bus_id.bus      = 0xfd;
        bus_id.slot     = 0x1f;
        bus_id.function = 0;

        /* Do not extract/restore topology here: UCP init still probes real
         * transports, and clearing global topology can break hardware caches. */
        ASSERT_UCS_OK(ucs_topo_find_device_by_bus_id_and_user_value(
                &bus_id, 2, &m_high_user_value_sys_dev));
        ASSERT_UCS_OK(ucs_topo_sys_device_set_name(
                m_high_user_value_sys_dev, "mock_high_user_value", 10));

        ASSERT_UCS_OK(ucs_topo_find_device_by_bus_id_and_user_value(
                &bus_id, 1, &m_low_user_value_sys_dev));
        ASSERT_UCS_OK(ucs_topo_sys_device_set_name(
                m_low_user_value_sys_dev, "mock_low_user_value", 10));
    }

    ucs_sys_device_t m_low_user_value_sys_dev;
    ucs_sys_device_t m_high_user_value_sys_dev;
};

UCS_TEST_P(test_ucp_proto_mock_rcx_same_bdf_put,
           single_net_device_orders_same_bdf_by_user_value_0,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=0")
{
    check_config({{0, 2048, "short", "rc_mlx5/mock_0:1/path0"},
                  {2049, INF, "zero-copy", "rc_mlx5/mock_1:1/path0"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_same_bdf_put,
           single_net_device_orders_same_bdf_by_user_value_1,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=1")
{
    check_config({{0, 2048, "short", "rc_mlx5/mock_0:1/path0"},
                  {2049, INF, "zero-copy", "rc_mlx5/mock_0:1/path0"}});
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_same_bdf_put, rcx,
                              "rc_x")

class test_ucp_proto_mock_rcx_twins_get : public test_ucp_proto_mock_rcx_twins {
protected:
    void check_config(const proto_select_data_vec_t &data_vec);
};

void test_ucp_proto_mock_rcx_twins_get::check_config(
        const proto_select_data_vec_t &data_vec)
{
    uint8_t remote   = 42;
    auto memh        = mem_map(receiver(), &remote, sizeof(remote));
    auto rkey_packed = rkey_pack(receiver(), memh);
    auto rkey        = rkey_unpack(sender().ep(), rkey_packed);

    uint8_t local = 0;
    ucp_request_param_t req_param;
    req_param.op_attr_mask = 0;
    auto status            = ucp_get_nbx(sender().ep(), &local, sizeof(local),
                                         (uint64_t)&remote, rkey, &req_param);
    request_wait(status);
    ASSERT_EQ(local, 42);

    ucp_proto_select_key_t key = any_key();
    key.param.op_id_flags      = UCP_OP_ID_GET;
    key.param.op_attr          = 0;

    check_rkey_config(sender(), data_vec, key, rkey->cfg_index);
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_get, use_single_net_device_rank_0,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=0")
{
    check_config({{0, 624, "copy-out", "rc_mlx5/mock_0:1/path0"},
                  {625, INF, "zero-copy", "rc_mlx5/mock_0:1/path0"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_get, use_single_net_device_rank_1,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=1")
{
    check_config({{0, 624, "copy-out", "rc_mlx5/mock_2:1/path0"},
                  {625, INF, "zero-copy", "rc_mlx5/mock_2:1/path0"}});
}

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_get, use_single_net_device_rank_2,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=2")
{
    check_config({{0, 624, "copy-out", "rc_mlx5/mock_0:1/path0"},
                  {625, INF, "zero-copy", "rc_mlx5/mock_0:1/path0"}});
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_twins_get, rcx, "rc_x")

class test_ucp_proto_mock_rcx_twins_get_inline_0 :
    public test_ucp_proto_mock_rcx_twins_get {
protected:
    test_ucp_proto_mock_rcx_twins_get_inline_0()
    {
        modify_config("IB_NUM_PATHS", "1", SETENV_IF_NOT_EXIST);
        modify_config("IB_TX_INLINE_RESP", "0", SETENV_IF_NOT_EXIST);
    }
};

UCS_TEST_P(test_ucp_proto_mock_rcx_twins_get_inline_0,
           multi_rail_max_min_size_one, "MAX_RMA_RAILS=2", "ZCOPY_THRESH=0")
{
    check_config({{1, INF, "zero-copy",
                   "50% on rc_mlx5/mock_0:1 and 50% on rc_mlx5/mock_2:1"}});
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_twins_get_inline_0, rcx,
                              "rc_x")

/*
 * Tests the min-distance filter and deterministic sort in
 * ucp_proto_multi_filter_single_net_device(). Sets up three mock NICs
 * with identical bandwidth (so all survive the filter's bandwidth
 * tiebreaker) and gives mock_1:1 a lower iface latency so it wins the
 * wireup AM/RMA-lane race and lands at lane[0]. This makes the
 * lane-index insertion order (mock_1:1 first) differ from the
 * deterministic bus-id sorted order [mock_0:1, mock_1:1, mock_2:1].
 *
 * Installs a "proto_mock" topology provider that gives any known sys_dev
 * pair a 100ns latency penalty unless it is one of the explicit adjacent
 * pairs with mock_1:1. This makes mock_0:1 and mock_2:1 adjacent to the
 * buffer while mock_1:1 (the buffer's own device) is "far".
 *
 * Each test asserts two ranges in the proto cache:
 *   - 0..64 (get/bcopy "copy-out"): reg_mem_info unknown, so all NICs
 *     have sys_latency=0 and survive the filter; the seed runs over
 *     the sorted set [mock_0:1, mock_1:1, mock_2:1].
 *   - 65..INF (get/zcopy "zero-copy"): reg_mem_info is filled from
 *     select_param; the filter drops mock_1:1 and the seed runs over
 *     [mock_0:1, mock_2:1].
 */
class test_ucp_proto_mock_rcx_trio_local_distance_get :
    public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx_trio_local_distance_get()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        auto iface_attr_slow = [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 600e-9;
            iface_attr.latency.m         = 1e-9;
        };

        auto iface_attr_fast = [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short  = 208;
            iface_attr.cap.put.max_short = 2048;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 300e-9;
            iface_attr.latency.m         = 1e-9;
        };

        add_mock_iface("mock_0:1", iface_attr_slow); /* bus 0 */
        add_mock_iface("mock_1:1", iface_attr_fast); /* bus 1, lane[0] */
        add_mock_iface("mock_2:1", iface_attr_slow); /* bus 2 */
        test_ucp_proto_mock::init();

        s_low_adjacent_sys_dev  = get_mock_sys_dev_by_name("mock_0:1");
        s_local_sys_dev         = get_mock_sys_dev_by_name("mock_1:1");
        s_high_adjacent_sys_dev = get_mock_sys_dev_by_name("mock_2:1");

        const ucs_sys_topo_ops_t topo_ops = {
            .get_distance                   = get_distance,
            .get_memory_distance            = get_memory_distance,
            .get_memory_distance_for_cpuset = get_memory_distance_for_cpuset
        };
        ASSERT_UCS_OK(ucs_sys_topo_provider_push(&topo_ops));
    }

    virtual void cleanup() override
    {
        ucs_sys_topo_provider_pop();
        s_low_adjacent_sys_dev  = UCS_SYS_DEVICE_ID_UNKNOWN;
        s_local_sys_dev         = UCS_SYS_DEVICE_ID_UNKNOWN;
        s_high_adjacent_sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        test_ucp_proto_mock::cleanup();
    }

protected:
    void check_get_picks(const std::string &zcopy_mock,
                         const std::string &bcopy_mock)
    {
        uint8_t remote   = 42;
        auto memh        = mem_map(receiver(), &remote, sizeof(remote));
        auto rkey_packed = rkey_pack(receiver(), memh);
        auto rkey        = rkey_unpack(sender().ep(), rkey_packed);

        uint8_t local       = 0;
        auto local_memh     = mem_map(sender(), &local, sizeof(local));
        local_memh->sys_dev = get_mock_sys_dev_by_name("mock_1:1");

        ucp_request_param_t req_param;
        req_param.op_attr_mask = UCP_OP_ATTR_FIELD_MEMH;
        req_param.memh         = local_memh;

        auto *status = ucp_get_nbx(sender().ep(), &local, sizeof(local),
                                   (uint64_t)&remote, rkey, &req_param);
        request_wait(status);
        ASSERT_EQ(local, 42);

        ucp_proto_select_key_t key = any_key();
        key.param.op_id_flags      = UCP_OP_ID_GET;
        key.param.op_attr          = 0;

        const std::string bcopy_config = "rc_mlx5/" + bcopy_mock + "/path0";
        const std::string zcopy_config = "rc_mlx5/" + zcopy_mock + "/path0";
        check_rkey_config(sender(),
                          {{0, 64, "copy-out", bcopy_config},
                           {65, INF, "zero-copy", zcopy_config}},
                          key, rkey->cfg_index);
    }

    static int is_adjacent_to_local(ucs_sys_device_t device1,
                                    ucs_sys_device_t device2)
    {
        if (device1 == s_local_sys_dev) {
            return (device2 == s_low_adjacent_sys_dev) ||
                   (device2 == s_high_adjacent_sys_dev);
        }

        if (device2 == s_local_sys_dev) {
            return (device1 == s_low_adjacent_sys_dev) ||
                   (device1 == s_high_adjacent_sys_dev);
        }

        return 0;
    }

    static ucs_status_t get_distance(ucs_sys_device_t device1,
                                     ucs_sys_device_t device2,
                                     ucs_sys_dev_distance_t *distance)
    {
        *distance = ucs_topo_default_distance;
        if ((device1 != UCS_SYS_DEVICE_ID_UNKNOWN) &&
            (device2 != UCS_SYS_DEVICE_ID_UNKNOWN) &&
            !is_adjacent_to_local(device1, device2)) {
            distance->latency = 100e-9;
        }
        return UCS_OK;
    }

    static void
    get_memory_distance(ucs_sys_device_t, ucs_sys_dev_distance_t *distance)
    {
        *distance = ucs_topo_default_distance;
    }

    static void
    get_memory_distance_for_cpuset(ucs_sys_device_t, const ucs_cpu_set_t *,
                                   ucs_sys_dev_distance_t *distance)
    {
        *distance = ucs_topo_default_distance;
    }

private:
    static ucs_sys_device_t s_low_adjacent_sys_dev;
    static ucs_sys_device_t s_local_sys_dev;
    static ucs_sys_device_t s_high_adjacent_sys_dev;
};

ucs_sys_device_t
test_ucp_proto_mock_rcx_trio_local_distance_get::s_low_adjacent_sys_dev =
        UCS_SYS_DEVICE_ID_UNKNOWN;
ucs_sys_device_t
test_ucp_proto_mock_rcx_trio_local_distance_get::s_local_sys_dev =
        UCS_SYS_DEVICE_ID_UNKNOWN;
ucs_sys_device_t
test_ucp_proto_mock_rcx_trio_local_distance_get::s_high_adjacent_sys_dev =
        UCS_SYS_DEVICE_ID_UNKNOWN;

UCS_TEST_P(test_ucp_proto_mock_rcx_trio_local_distance_get,
           single_net_dev_local_id_0_picks_lowest_adjacent_sys_dev,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=0",
           "ZCOPY_THRESH=0")
{
    /* zcopy: 0 % 2 = 0 -> mock_0:1.   bcopy: 0 % 3 = 0 -> mock_0:1. */
    check_get_picks("mock_0:1", "mock_0:1");
}

UCS_TEST_P(test_ucp_proto_mock_rcx_trio_local_distance_get,
           single_net_dev_local_id_1_picks_highest_adjacent_sys_dev,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=1",
           "ZCOPY_THRESH=0")
{
    /* zcopy: 1 % 2 = 1 -> mock_2:1.   bcopy: 1 % 3 = 1 -> mock_1:1. */
    check_get_picks("mock_2:1", "mock_1:1");
}

UCS_TEST_P(test_ucp_proto_mock_rcx_trio_local_distance_get,
           single_net_dev_local_id_2_wraps_to_lowest_adjacent_sys_dev,
           "IB_NUM_PATHS?=2", "SINGLE_NET_DEVICE=y", "NODE_LOCAL_ID=2",
           "ZCOPY_THRESH=0")
{
    /* zcopy: 2 % 2 = 0 -> mock_0:1.   bcopy: 2 % 3 = 2 -> mock_2:1. */
    check_get_picks("mock_0:1", "mock_2:1");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_trio_local_distance_get,
                              rcx, "rc_x")

class test_ucp_proto_mock_am_tiebreak : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_am_tiebreak()
    {
        mock_transport("rc_mlx5");
    }

protected:
    void add_mock_device(const std::string &dev_name, double bandwidth,
                         double latency)
    {
        add_mock_iface(dev_name,
                       [bandwidth, latency](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.am.max_short = 208;
            iface_attr.bandwidth.shared = bandwidth;
            iface_attr.latency.c        = latency;
            iface_attr.latency.m        = 1e-9;
        });
    }

    void check_config(const std::string &config)
    {
        ucp_proto_select_key_t key = any_key();
        key.param.op_id_flags      = UCP_OP_ID_AM_SEND;
        key.param.op_attr          = 0;

        check_ep_config(sender(), {{0, 200, "short", config}}, key);
    }
};

class test_ucp_proto_mock_am_tiebreak_equal_score :
    public test_ucp_proto_mock_am_tiebreak {
public:
    virtual void init() override
    {
        add_mock_device("mock_0:1", 10e9, 500e-9);
        add_mock_device("mock_1:1", 28e9, 500e-9);
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_am_tiebreak_equal_score, higher_bandwidth_wins,
           "IB_NUM_PATHS?=1")
{
    check_config("rc_mlx5/mock_1:1");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_am_tiebreak_equal_score, rcx,
                              "rc_x")

class test_ucp_proto_mock_am_tiebreak_score_dominates :
    public test_ucp_proto_mock_am_tiebreak {
public:
    virtual void init() override
    {
        add_mock_device("mock_0:1", 10e9, 500e-9);
        add_mock_device("mock_1:1", 28e9, 2000e-9);
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_am_tiebreak_score_dominates,
           score_beats_bandwidth, "IB_NUM_PATHS?=1")
{
    check_config("rc_mlx5/mock_0:1");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_am_tiebreak_score_dominates,
                              rcx, "rc_x")

class test_ucp_proto_mock_am_tiebreak_within_window :
    public test_ucp_proto_mock_am_tiebreak {
public:
    virtual void init() override
    {
        add_mock_device("mock_0:1", 10e9, 500e-9);
        add_mock_device("mock_1:1", 28e9, 508e-9);
        test_ucp_proto_mock::init();
    }
};

UCS_TEST_P(test_ucp_proto_mock_am_tiebreak_within_window,
           latency_beats_bandwidth, "IB_NUM_PATHS?=1")
{
    check_config("rc_mlx5/mock_0:1");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_am_tiebreak_within_window,
                              rcx, "rc_x")

class test_ucp_proto_mock_keepalive_tiebreak :
    public test_ucp_proto_mock_am_tiebreak {
public:
    test_ucp_proto_mock_keepalive_tiebreak()
    {
        modify_config("KEEPALIVE_INTERVAL", "1s");
    }

    virtual ucp_ep_params_t get_ep_params() override
    {
        ucp_ep_params_t params = test_ucp_proto_mock::get_ep_params();

        params.field_mask     |= UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                                 UCP_EP_PARAM_FIELD_ERR_HANDLER;
        params.err_mode        = UCP_ERR_HANDLING_MODE_PEER;
        params.err_handler.cb  = reinterpret_cast<ucp_err_handler_cb_t>(
                                         ucs_empty_function);
        params.err_handler.arg = reinterpret_cast<void*>(this);

        return params;
    }

    virtual void init() override
    {
        const size_t lower_max_inflight_eps = SIZE_MAX - (SIZE_MAX / 100);

        add_keepalive_mock_device("mock_0:1", 10e9, 500e-9, SIZE_MAX);
        add_keepalive_mock_device("mock_1:1", 28e9, 500e-9,
                                  lower_max_inflight_eps);
        test_ucp_proto_mock::init();
    }

protected:
    void add_keepalive_mock_device(const std::string &dev_name,
                                   double bandwidth, double latency,
                                   size_t max_inflight_eps)
    {
        add_mock_iface(dev_name,
                       [bandwidth, latency](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.flags &= ~UCT_IFACE_FLAG_EP_KEEPALIVE;
            iface_attr.cap.flags |= UCT_IFACE_FLAG_EP_CHECK |
                                    UCT_IFACE_FLAG_CONNECT_TO_EP |
                                    UCT_IFACE_FLAG_ERRHANDLE_PEER_FAILURE;
            iface_attr.cap.am.max_short = 208;
            iface_attr.bandwidth.shared = bandwidth;
            iface_attr.latency.c        = latency;
            iface_attr.latency.m        = 1e-9;
        }, [max_inflight_eps](uct_perf_attr_t &perf_attr) {
            if (perf_attr.field_mask &
                UCT_PERF_ATTR_FIELD_MAX_INFLIGHT_EPS) {
                perf_attr.max_inflight_eps = max_inflight_eps;
            }
        });
    }

    void check_keepalive_lane(const std::string &dev_name)
    {
        const ucp_ep_config_t *config = ucp_worker_ep_config(sender().worker(),
                                                             ep_config_index(
                                                                     sender()));
        const ucp_lane_index_t lane   = config->key.keepalive_lane;

        ASSERT_NE(UCP_NULL_LANE, lane);
        EXPECT_STREQ(dev_name.c_str(),
                     ucp_ep_get_tl_rsc(sender().ep(), lane)->dev_name);
    }
};

UCS_TEST_P(test_ucp_proto_mock_keepalive_tiebreak,
           higher_bandwidth_within_score_window, "IB_NUM_PATHS?=1")
{
    check_keepalive_lane("mock_1:1");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_keepalive_tiebreak, rcx,
                              "rc_x")


#if HAVE_DECL_IBV_EVENT_PORT_SPEED_CHANGE

class test_ucp_proto_mock_rcx_speed_change : public test_ucp_proto_mock {
public:
    test_ucp_proto_mock_rcx_speed_change()
    {
        mock_transport("rc_mlx5");
    }

    virtual void init() override
    {
        m_port_speed["mock_0:1"] = 28e9;
        add_mock_iface("mock_0:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.get.min_zcopy = 0;
            iface_attr.bandwidth.shared  = 28e9;
            iface_attr.latency.c         = 600e-9;
            iface_attr.latency.m         = 1e-9;
        }, [this](uct_perf_attr_t &perf_attr) {
            perf_attr.bandwidth.shared = this->m_port_speed["mock_0:1"];
            perf_attr.path_bandwidth   = perf_attr.bandwidth;
        });

        m_port_speed["mock_1:1"] = 24e9;
        add_mock_iface("mock_1:1", [](uct_iface_attr_t &iface_attr) {
            iface_attr.cap.get.min_zcopy = 0;
            iface_attr.bandwidth.shared  = 24e9;
            iface_attr.latency.c         = 500e-9;
            iface_attr.latency.m         = 1e-9;
        }, [this](uct_perf_attr_t &perf_attr) {
            perf_attr.bandwidth.shared = this->m_port_speed["mock_1:1"];
            perf_attr.path_bandwidth   = perf_attr.bandwidth;
        });
        test_ucp_proto_mock::init();
    }

    void set_port_speed(const std::string &iface_name, double port_speed)
    {
        m_port_speed[iface_name] = port_speed;

        ib_event(IBV_EVENT_PORT_SPEED_CHANGE, 1);
        while (progress());
    }

    void test_port_speed(std::function<void(unsigned)> send,
                         ucp_operation_id_t op_id)
    {
        // One EP & rkey config created during connection establishment
        ucp_worker_h worker = sender().worker();
        EXPECT_EQ(worker->rkey_config_count, 1);
        EXPECT_EQ(worker->ep_config.length, 1);

        // New rkey config created during first operation
        send(1);
        EXPECT_EQ(worker->rkey_config_count, 2);
        EXPECT_EQ(worker->ep_config.length, 1);

        // Existing rkey config is used during second operation
        send(1);
        EXPECT_EQ(worker->rkey_config_count, 2);
        EXPECT_EQ(worker->ep_config.length, 1);

        ucp_proto_select_key_t key = any_key();
        key.param.op_id_flags      = op_id;
        key.param.op_attr          = 0;

        check_rkey_config(sender(), {
            {0, INF,  "zero-copy", "47% on rc_mlx5/mock_1:1 and 53% on rc_mlx5/mock_0:1"},
        }, key, 1);

        // Reduce port_speed of mock_0:1 by 50%, new EP & rkey configs are created
        set_port_speed("mock_0:1", 14e9);
        send(2);
        EXPECT_EQ(worker->rkey_config_count, 3);
        EXPECT_EQ(worker->ep_config.length, 2);

        // Slightly change port_speed, so that quantized value remains the same
        // This shouldn't affect EP or rkey config
        set_port_speed("mock_0:1", 14.5e9);
        send(2);
        EXPECT_EQ(worker->rkey_config_count, 3);
        EXPECT_EQ(worker->ep_config.length, 2);

        check_rkey_config(sender(), {
            {0, INF,  "zero-copy", "64% on rc_mlx5/mock_1:1 and 36% on rc_mlx5/mock_0:1"},
        }, key, 2);

        // Reduce port_speed of mock_1:1 to be equal with mock_0:1,
        // new EP & rkey configs are created
        set_port_speed("mock_1:1", 14e9);
        send(3);
        EXPECT_EQ(worker->rkey_config_count, 4);
        EXPECT_EQ(worker->ep_config.length, 3);

        check_rkey_config(sender(), {
            {0, INF,  "zero-copy", "50% on rc_mlx5/mock_1:1 and 50% on rc_mlx5/mock_0:1"},
        }, key, 3);

        // Reset port_speeds to initial values, should switch to initial configs
        set_port_speed("mock_0:1", 28e9);
        set_port_speed("mock_1:1", 24e9);
        send(1);
        EXPECT_EQ(worker->rkey_config_count, 4);
        EXPECT_EQ(worker->ep_config.length, 3);
    }

private:
    std::map<std::string, double> m_port_speed;
};

UCS_TEST_P(test_ucp_proto_mock_rcx_speed_change, rma_put,
           "IB_NUM_PATHS?=1", "MAX_RMA_RAILS=2", "ZCOPY_THRESH=0")
{
    test_port_speed([this](unsigned rkey_cfg_index) {
        send_recv_rma(64 * UCS_KBYTE, UCP_OP_ID_PUT, UCS_MEMORY_TYPE_HOST,
                      rkey_cfg_index);
    }, UCP_OP_ID_PUT);
}

UCS_TEST_P(test_ucp_proto_mock_rcx_speed_change, rma_get,
           "IB_NUM_PATHS?=1", "MAX_RMA_RAILS=2", "ZCOPY_THRESH=0")
{
    test_port_speed([this](unsigned rkey_cfg_index) {
        send_recv_rma(64 * UCS_KBYTE, UCP_OP_ID_GET, UCS_MEMORY_TYPE_HOST,
                      rkey_cfg_index);
    }, UCP_OP_ID_GET);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_proto_mock_rcx_speed_change, rcx, "rc_x")

#endif // HAVE_DECL_IBV_EVENT_PORT_SPEED_CHANGE
