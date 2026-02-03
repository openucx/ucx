/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <set>

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucs/sys/sys.h>
}

class test_ucp_lib_query : public ucs::test {
};

UCS_TEST_F(test_ucp_lib_query, test_max_thread_support) {
    ucs_status_t status;
    ucp_lib_attr_t params;
    memset(&params, 0, sizeof(ucp_lib_attr_t));
    params.field_mask = UCP_LIB_ATTR_FIELD_MAX_THREAD_LEVEL;
    status            = ucp_lib_query(&params);
    ASSERT_EQ(UCS_OK, status);
#if ENABLE_MT
    EXPECT_EQ(UCS_THREAD_MODE_MULTI, params.max_thread_level);
#else
    EXPECT_EQ(UCS_THREAD_MODE_SERIALIZED, params.max_thread_level);
#endif
}

UCS_TEST_P(test_ucp_context, minimal_field_mask) {
    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    ucs::handle<ucp_context_h> ucph;
    ucs::handle<ucp_worker_h> worker;

    {
        /* Features ONLY */
        ucp_params_t params;
        VALGRIND_MAKE_MEM_UNDEFINED(&params, sizeof(params));
        params.field_mask = UCP_PARAM_FIELD_FEATURES;
        params.features   = get_variant_ctx_params().features;

        UCS_TEST_CREATE_HANDLE(ucp_context_h, ucph, ucp_cleanup,
                               ucp_init, &params, config.get());
    }

    {
        /* Empty set */
        ucp_worker_params_t params;
        VALGRIND_MAKE_MEM_UNDEFINED(&params, sizeof(params));
        params.field_mask = 0;

        UCS_TEST_CREATE_HANDLE(ucp_worker_h, worker, ucp_worker_destroy,
                               ucp_worker_create, ucph.get(), &params);
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_context, all, "all")

class test_ucp_aliases : public test_ucp_context {
};

UCS_TEST_P(test_ucp_aliases, aliases) {
    create_entity();
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, rcv, "rc_v")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, rcx, "rc_x")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, ud, "ud")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, srd, "srd")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, ud_mlx5, "ud_mlx5")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, ugni, "ugni")
UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_aliases, shm, "shm")


class test_ucp_version : public test_ucp_context {
};

UCS_TEST_P(test_ucp_version, wrong_api_version) {

    ucs::handle<ucp_config_t*> config;
    UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                           ucp_config_read, NULL, NULL);

    ucp_context_h ucph;
    ucs_status_t status;
    size_t warn_count;
    {
        scoped_log_handler slh(hide_warns_logger);
        warn_count = m_warnings.size();
        status = ucp_init_version(99, 99, &get_variant_ctx_params(),
                                  config.get(), &ucph);
    }
    if (status != UCS_OK) {
        ADD_FAILURE() << "Failed to create UCP with wrong version";
    } else {
        if (m_warnings.size() == warn_count) {
            ADD_FAILURE() << "Missing wrong version warning";
        }
        ucp_cleanup(ucph);
    }
}

UCS_TEST_P(test_ucp_version, version_string) {

    unsigned major_version, minor_version, release_number;

    ucp_get_version(&major_version, &minor_version, &release_number);

    std::string string_version     = std::to_string(major_version) + '.' +
                                     std::to_string(minor_version) + '.' +
                                     std::to_string(release_number);
    std::string ucp_string_version = ucp_get_version_string();

    EXPECT_EQ(string_version,
              ucp_string_version.substr(0, string_version.length()));
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_version, all, "all")

class test_ucp_net_devices_config : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants) {
        add_variant(variants, UCP_FEATURE_TAG);
    }

protected:
    static const char DELIMITER = ':';

    /* Iterate over all network devices and apply action to each */
    template<typename Action>
    static void for_each_net_device(const entity &e, Action action) {
        ucp_context_h ctx = e.ucph();
        for (ucp_rsc_index_t i = 0; i < ctx->num_tls; ++i) {
            const uct_tl_resource_desc_t *rsc = &ctx->tl_rscs[i].tl_rsc;
            if (rsc->dev_type == UCT_DEVICE_TYPE_NET) {
                action(rsc);
            }
        }
    }

    /* Check if a specific device name exists in the set */
    static bool has_device(const std::set<std::string> &devices,
                           const std::string &dev_name)
    {
        return devices.find(dev_name) != devices.end();
    }


    /* Get all network device names from the context */
    static std::set<std::string> get_net_device_names(const entity &e)
    {
        std::set<std::string> device_names;
        for_each_net_device(e, [&](const uct_tl_resource_desc_t *rsc) {
            device_names.insert(rsc->dev_name);
        });
        return device_names;
    }

    /* Get all network device names from the context with delimiter */
    static std::set<std::string>
    get_net_device_names_with_delimiter(const entity &e)
    {
        std::set<std::string> device_names;
        for_each_net_device(e, [&](const uct_tl_resource_desc_t *rsc) {
            std::string dev_name(rsc->dev_name);
            size_t delimiter_pos = dev_name.find(DELIMITER);
            if (delimiter_pos != std::string::npos) {
                device_names.insert(dev_name);
            }
        });
        return device_names;
    }

    static std::set<std::string>
    get_device_base_names(const std::set<std::string> &dev_names)
    {
        std::set<std::string> base_names;

        for (const std::string &dev_name : dev_names) {
            size_t delimiter_pos = dev_name.find(DELIMITER);
            if (delimiter_pos != std::string::npos) {
                base_names.insert(dev_name.substr(0, delimiter_pos));
            } else {
                base_names.insert(dev_name);
            }
        }

        return base_names;
    }

    /* Join strings with a delimiter */
    static std::string
    join(const std::set<std::string> &strings, const std::string &delimiter)
    {
        std::string result;
        for (auto it = strings.begin(); it != strings.end(); ++it) {
            if (it != strings.begin()) {
                result += delimiter;
            }
            result += *it;
        }
        return result;
    }

    /* Test that net device selection works correctly */
    void
    test_net_device_selection(const std::set<std::string> &test_net_devices,
                              const std::set<std::string> &expected_net_devices)
    {
        std::string net_devices_config = join(test_net_devices, ",");
        modify_config("NET_DEVICES", net_devices_config.c_str());
        entity *e = create_entity();

        std::set<std::string> selected_devices = get_net_device_names(*e);
        EXPECT_EQ(selected_devices.size(), expected_net_devices.size());

        for (const std::string &net_device : expected_net_devices) {
            EXPECT_TRUE(has_device(selected_devices, net_device))
                    << "Device '" << net_device << "' should be selected when "
                    << "UCX_NET_DEVICES=" << net_devices_config;
        }
    }

    /* Test that a device config triggers duplicate device warning */
    void test_duplicate_device_warning(const std::string &required_dev_name,
                                       const std::string &devices_config,
                                       const std::string &duplicate_dev_name)
    {
        entity *e = create_entity();

        std::set<std::string> net_devices = get_net_device_names(*e);
        ASSERT_FALSE(net_devices.empty());

        if (!has_device(net_devices, required_dev_name)) {
            UCS_TEST_SKIP_R(required_dev_name + " device not available");
        }

        m_entities.clear();

        modify_config("NET_DEVICES", devices_config.c_str());

        size_t warn_count;
        {
            scoped_log_handler slh(hide_warns_logger);
            warn_count = m_warnings.size();
            create_entity();
        }

        EXPECT_EQ(m_warnings.size() - warn_count, 1)
                << "Expected exactly one warning";

        /* Check that the warning about duplicate device was printed */
        std::string expected_warn = "device '" + duplicate_dev_name +
                                    "' is specified multiple times";
        EXPECT_NE(m_warnings[warn_count].find(expected_warn), std::string::npos)
                << "Expected warning about duplicate device '"
                << duplicate_dev_name << "' with config '" << devices_config
                << "'";
    }
};

/*
 * Test that when UCX_NET_DEVICES is set to a base name (e.g., "mlx5_0"),
 * devices with the same base name are selected (e.g., "mlx5_0:1").
 */
UCS_TEST_P(test_ucp_net_devices_config, base_name_selects_device)
{
    entity *e = create_entity();

    std::set<std::string> net_devices = get_net_device_names_with_delimiter(*e);
    if (net_devices.empty()) {
        UCS_TEST_SKIP_R("No network devices available with delimiter");
    }

    m_entities.clear();

    std::set<std::string> base_names = get_device_base_names(net_devices);
    test_net_device_selection(base_names, net_devices);
}

/*
 * Test that explicit suffix specification works correctly.
 */
UCS_TEST_P(test_ucp_net_devices_config, explicit_suffix)
{
    entity *e = create_entity();

    std::set<std::string> net_devices = get_net_device_names_with_delimiter(*e);
    if (net_devices.empty()) {
        UCS_TEST_SKIP_R("No network devices available with delimiter");
    }

    m_entities.clear();

    test_net_device_selection(net_devices, net_devices);
}

/*
 * Test that specifying a device multiple times produces a warning
 */
UCS_TEST_P(test_ucp_net_devices_config, duplicate_device_warning_simple)
{
    test_duplicate_device_warning("mlx5_0:1", "mlx5_0:1,mlx5_0:1", "mlx5_0:1");
}

UCS_TEST_P(test_ucp_net_devices_config, duplicate_device_warning_base_name)
{
    test_duplicate_device_warning("mlx5_0:1", "mlx5_0:1,mlx5_0", "mlx5_0");
}

UCS_TEST_P(test_ucp_net_devices_config, duplicate_device_warning_two_base_name)
{
    test_duplicate_device_warning("mlx5_0:1", "mlx5_0,mlx5_0", "mlx5_0");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_net_devices_config, all, "all")
