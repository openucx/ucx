/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>

#include <string>
#include <vector>

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_tl_info.h>
#include <ucs/datastruct/string_buffer.h>
#include <ucs/sys/string.h>
#include <ucs/sys/topo/base/topo.h>
}


class test_ucp_tl_info : public ucs::test {
public:
    /* Isolate the global topology subsystem so that system devices registered
     * by a test are deterministic (and removed afterwards). */
    virtual void init()
    {
        ucs::test::init();
        m_topo_state = ucs_topo_extract_state();
    }

    virtual void cleanup()
    {
        ucs_topo_restore_state(m_topo_state);
        ucs::test::cleanup();
    }

protected:
    ucs_global_state_t *m_topo_state;

    /* Owns a ucp_context_t plus the backing component array built from a list
     * of component names, keeping tl_cmpts valid for the object's lifetime. */
    class dummy_context {
    public:
        dummy_context(const std::vector<std::string> &cmpt_names,
                      const std::string &name = "")
        {
            auto num_cmpts = cmpt_names.size() + 1;
            m_cmpts        = std::vector<ucp_tl_cmpt_t>(num_cmpts);

            for (size_t i = 0; i < cmpt_names.size(); ++i) {
                EXPECT_NE(std::string("rdmacm"), cmpt_names[i]);

                ucs_strncpy_safe(m_cmpts[i].attr.name, cmpt_names[i].c_str(),
                                 sizeof(m_cmpts[i].attr.name));

                /* Set the MD resource count to 1 for all components. */
                m_cmpts[i].attr.md_resource_count = 1;
            }

            /* Add rdmacm with no MD resources. */
            ucs_strncpy_safe(m_cmpts[cmpt_names.size()].attr.name, "rdmacm",
                             sizeof("rdmacm"));
            m_cmpts[cmpt_names.size()].attr.md_resource_count = 0;

            memset(&m_ctx, 0, sizeof(m_ctx));
            m_ctx.tl_cmpts  = m_cmpts.data();
            m_ctx.num_cmpts = num_cmpts;
            ucs_strncpy_safe(m_ctx.name, name.c_str(), sizeof(m_ctx.name));
        }

        ucp_context_h get()
        {
            return &m_ctx;
        }

    private:
        ucp_context_t m_ctx;
        std::vector<ucp_tl_cmpt_t> m_cmpts;
    };

    /* Search the context's components for the given name and return its index. */
    ucp_rsc_index_t find_cmpt_index(ucp_context_h context, const char *name)
    {
        for (ucp_rsc_index_t i = 0; i < context->num_cmpts; ++i) {
            if (strcmp(context->tl_cmpts[i].attr.name, name) == 0) {
                return i;
            }
        }

        ADD_FAILURE() << "unknown component '" << name << "'";
        return 0;
    }

    void add_rsc(dummy_context &context, ucp_tl_info_array_t *rscs,
                 uct_device_type_t dev_type, const char *cmpt_name,
                 const char *tl_name, const char *dev_name, bool enabled,
                 ucs_sys_device_t sys_device = UCS_SYS_DEVICE_ID_UNKNOWN)
    {
        ucp_tl_info_entry_t *entry = ucs_array_append(rscs, FAIL());

        memset(entry, 0, sizeof(*entry));
        ucs_strncpy_safe(entry->rsc.tl_name, tl_name,
                         sizeof(entry->rsc.tl_name));
        ucs_strncpy_safe(entry->rsc.dev_name, dev_name,
                         sizeof(entry->rsc.dev_name));
        entry->rsc.dev_type   = dev_type;
        entry->rsc.sys_device = sys_device;
        entry->cmpt_index     = find_cmpt_index(context.get(), cmpt_name);
        entry->enabled        = enabled;
    }

    /* Register a system device with a known name and return its id. */
    ucs_sys_device_t add_sys_device(uint8_t bus, const char *name)
    {
        ucs_sys_bus_id_t bus_id = {};
        ucs_sys_device_t sys_dev;

        bus_id.bus = bus;
        EXPECT_EQ(UCS_OK, ucs_topo_find_device_by_bus_id(&bus_id, &sys_dev));
        EXPECT_EQ(UCS_OK, ucs_topo_sys_device_set_name(sys_dev, name, 1));
        return sys_dev;
    }

    std::string render(dummy_context &context, ucp_tl_info_array_t *rscs)
    {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
        ucs_status_t status;

        status = ucp_context_render_tl_info(context.get(), rscs, &strb);
        EXPECT_EQ(UCS_OK, status);

        std::string out(ucs_string_buffer_cstr(&strb));
        ucs_string_buffer_cleanup(&strb);
        return out;
    }
};


/* Basic table: title, header, a single enabled transport with one device, and
 * the trailing legend. */
UCS_TEST_F(test_ucp_tl_info, single_transport) {
    dummy_context context({"posix"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_SHM, "posix", "posix", "memory",
            true);

    /* clang-format off */
    EXPECT_EQ(
        "+------------+-----------+-----------+------------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+------------+-----------+-----------+------------------------------------------------------------+\n"
        "| Type       | Component | Transport | Device (System device)                                     |\n"
        "+------------+-----------+-----------+------------------------------------------------------------+\n"
        "| intra-node | posix     | + posix   | + memory                                                   |\n"
        "+------------+-----------+-----------+------------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+------------+-----------+-----------+------------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* More devices than fit on one line wrap to a continuation row whose leading
 * cells are blank, and a disabled device is marked with '-'. */
UCS_TEST_F(test_ucp_tl_info, device_line_wrap) {
    dummy_context context({"ib"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_0:1",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_1:1",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_2:1",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_3:1",
            false);

    /* clang-format off */
    EXPECT_EQ(
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Type    | Component | Transport  | Device (System device)                                       |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| network | ib        | + rc_verbs | + mlx5_0:1  + mlx5_1:1  + mlx5_2:1                           |\n"
        "|         |           |            | - mlx5_3:1                                                   |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* Components that contributed no resources are listed in an "<unavailable>"
 * block, with a merged separator that carries over the blank "Type" column. */
UCS_TEST_F(test_ucp_tl_info, unavailable_components) {
    dummy_context context({"tcp", "gdr_copy", "cuda_ipc"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "tcp", "tcp", "eth0", true);

    /* clang-format off */
    EXPECT_EQ(
        "+---------------+-----------+-----------+---------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+---------------+-----------+-----------+---------------------------------------------------------+\n"
        "| Type          | Component | Transport | Device (System device)                                  |\n"
        "+---------------+-----------+-----------+---------------------------------------------------------+\n"
        "| network       | tcp       | + tcp     | + eth0                                                  |\n"
        "+---------------+-----------+-----------+---------------------------------------------------------+\n"
        "| <unavailable> | gdr_copy  |           |                                                         |\n"
        "|               +-----------+-----------+---------------------------------------------------------+\n"
        "|               | cuda_ipc  |           |                                                         |\n"
        "+---------------+-----------+-----------+---------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+---------------+-----------+-----------+---------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* Adjacent groups are separated by merged separators whose blank carry-over
 * width depends on whether the device type, component, or transport changed:
 * 2 cols for a new transport in the same component, 1 col for a new component
 * in the same device type, and 0 cols (full separator) for a new device type. */
UCS_TEST_F(test_ucp_tl_info, multiple_groups) {
    dummy_context context({"ib", "tcp", "posix"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_0:1",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "ud_verbs", "mlx5_0:1",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "tcp", "tcp", "eth0", true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_SHM, "posix", "posix", "memory",
            true);

    /* clang-format off */
    EXPECT_EQ(
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| Type       | Component | Transport  | Device (System device)                                    |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| network    | ib        | + rc_verbs | + mlx5_0:1                                                |\n"
        "|            |           +------------+-----------------------------------------------------------+\n"
        "|            |           | + ud_verbs | + mlx5_0:1                                                |\n"
        "|            +-----------+------------+-----------------------------------------------------------+\n"
        "|            | tcp       | + tcp      | + eth0                                                    |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| intra-node | posix     | + posix    | + memory                                                  |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* The context name, when set, is appended to the table title. */
UCS_TEST_F(test_ucp_tl_info, title_includes_context_name) {
    dummy_context context({"posix"}, "myctx");
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_SHM, "posix", "posix", "memory",
            true);

    std::string out = render(context, &rscs);
    EXPECT_NE(std::string::npos,
              out.find("Available Transports and Devices (ctx: myctx)"));

    ucs_array_cleanup_dynamic(&rscs);
}

/* A device with a known system device prints "dev_name (system_device_name)",
 * while a device without one (UNKNOWN) prints just its name. */
UCS_TEST_F(test_ucp_tl_info, device_with_system_device) {
    dummy_context context({"ib"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_sys_device_t sys_dev = add_sys_device(0x03, "mlx5_0");

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "ib0", true,
            sys_dev);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "ib1", true);

    /* clang-format off */
    EXPECT_EQ(
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Type    | Component | Transport  | Device (System device)                                       |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| network | ib        | + rc_verbs | + ib0 (mlx5_0)  + ib1                                        |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* The redundant system-device suffix is omitted when the device name already
 * starts with the system-device name, and kept otherwise. */
UCS_TEST_F(test_ucp_tl_info, device_name_matches_system_device) {
    dummy_context context({"ib"});
    ucp_tl_info_array_t rscs    = UCS_ARRAY_DYNAMIC_INITIALIZER;
    ucs_sys_device_t sd_mlx5_2  = add_sys_device(0x01, "mlx5_2");
    ucs_sys_device_t sd_ens10f0 = add_sys_device(0x02, "ens10f0");
    ucs_sys_device_t sd_mlx5_0  = add_sys_device(0x03, "mlx5_0");

    /* dev_name starts with the system-device name -> suffix dropped. */
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_2:1",
            true, sd_mlx5_2);
    /* dev_name equals the system-device name -> suffix dropped. */
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "ens10f0",
            true, sd_ens10f0);
    /* dev_name does not start with the system-device name -> suffix kept. */
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "ibs2", true,
            sd_mlx5_0);

    /* clang-format off */
    EXPECT_EQ(
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Type    | Component | Transport  | Device (System device)                                       |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| network | ib        | + rc_verbs | + mlx5_2:1  + ens10f0  + ibs2 (mlx5_0)                       |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* A transport whose devices are all disabled is itself marked disabled ('-'). */
UCS_TEST_F(test_ucp_tl_info, transport_transport_disabled) {
    dummy_context context({"ib"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_0:1",
            false);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_1:1",
            false);

    /* clang-format off */
    EXPECT_EQ(
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Type    | Component | Transport  | Device (System device)                                       |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| network | ib        | - rc_verbs | - mlx5_0:1  - mlx5_1:1                                       |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+---------+-----------+------------+--------------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}

/* Resources added out of order are sorted by (device type, component,
 * transport) before rendering. */
UCS_TEST_F(test_ucp_tl_info, sorted_output) {
    dummy_context context({"ib", "self"});
    ucp_tl_info_array_t rscs = UCS_ARRAY_DYNAMIC_INITIALIZER;

    /* Intentionally added out of sorted order: SHM before NET, and within the
     * "ib" component, "ud_verbs" before "rc_verbs". */
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_SHM, "self", "self", "memory",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "ud_verbs", "mlx5_0:1",
            true);
    add_rsc(context, &rscs, UCT_DEVICE_TYPE_NET, "ib", "rc_verbs", "mlx5_0:1",
            true);

    /* clang-format off */
    EXPECT_EQ(
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| Available Transports and Devices                                                                |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| Type       | Component | Transport  | Device (System device)                                    |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| network    | ib        | + rc_verbs | + mlx5_0:1                                                |\n"
        "|            |           +------------+-----------------------------------------------------------+\n"
        "|            |           | + ud_verbs | + mlx5_0:1                                                |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| intra-node | self      | + self     | + memory                                                  |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n"
        "| Legend: + = enabled, - = disabled                                                               |\n"
        "| All of the available transports are listed, some may be disabled or unsupported on your system. |\n"
        "| All of the visible devices are listed per transport, some may be disabled.                      |\n"
        "+------------+-----------+------------+-----------------------------------------------------------+\n",
        render(context, &rscs));
    /* clang-format on */

    ucs_array_cleanup_dynamic(&rscs);
}
