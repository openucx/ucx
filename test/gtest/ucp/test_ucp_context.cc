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

    /* Get all mlx5 network device names from the context */
    static std::set<std::string> get_mlx5_device_names(const entity &e) {
        std::set<std::string> device_names;
        for_each_net_device(e, [&](const uct_tl_resource_desc_t *rsc) {
            std::string dev_name(rsc->dev_name);
            if (dev_name.compare(0, 5, "mlx5_") == 0) {
                device_names.insert(rsc->dev_name);
            }
        });
        return device_names;
    }

    /* Get a list of all mlx5 device base names (without port suffix) */
    static std::set<std::string>
    get_mlx5_base_names(const std::set<std::string> &mlx5_devices) {
        std::set<std::string> base_names;

        for (const std::string &dev_name : mlx5_devices) {
            size_t colon_pos = dev_name.find(':');
            if (colon_pos != std::string::npos) {
                base_names.insert(dev_name.substr(0, colon_pos));
            } else {
                base_names.insert(dev_name);
            }
        }

        return base_names;
    }

    /* Count mlx5 resources matching a device name prefix */
    static size_t
    count_mlx5_resources_with_prefix(const std::set<std::string> &mlx5_devices,
                                     const std::string &prefix) {
        size_t count = 0;

        for (const std::string &dev_name : mlx5_devices) {
            if (dev_name.compare(0, prefix.length(), prefix) == 0) {
                ++count;
            }
        }

        return count;
    }

    /* Check if a specific device name exists in the set */
    static bool has_device(const std::set<std::string> &devices,
                           const std::string &dev_name) {
        return devices.find(dev_name) != devices.end();
    }
};

/*
 * Test that when UCX_NET_DEVICES is set to a base name (e.g., "mlx5_0"),
 * devices with the default port suffix ":1" are selected.
 */
UCS_TEST_P(test_ucp_net_devices_config, base_name_selects_default_port)
{
    entity *e = create_entity();

    std::set<std::string> mlx5_devices = get_mlx5_device_names(*e);
    if (mlx5_devices.empty()) {
        UCS_TEST_SKIP_R("No mlx5 network device available");
    }

    std::set<std::string> base_names = get_mlx5_base_names(mlx5_devices);
    ASSERT_FALSE(base_names.empty());

    /* Pick the first base name for testing */
    std::string test_base_name = *base_names.begin();

    m_entities.clear();

    /* Now create a new context with NET_DEVICES set to the base name */
    modify_config("NET_DEVICES", test_base_name.c_str());
    e = create_entity();

    /* Verify that devices matching the base name were selected */
    std::set<std::string> selected_devices = get_mlx5_device_names(*e);
    size_t count = count_mlx5_resources_with_prefix(selected_devices, test_base_name);
    EXPECT_EQ(count, 1) << "Expected exactly one device with base name '"
                        << test_base_name << "' to be selected, found: "
                        << testing::PrintToString(selected_devices);

    std::string expected_dev = test_base_name + ":1";
    EXPECT_TRUE(has_device(selected_devices, expected_dev))
            << "Device '" << expected_dev << "' should be selected when "
            << "UCX_NET_DEVICES=" << test_base_name;
}

/*
 * Test that explicit port suffix specification works correctly.
 */
UCS_TEST_P(test_ucp_net_devices_config, explicit_port_suffix)
{
    entity *e = create_entity();

    std::set<std::string> mlx5_devices = get_mlx5_device_names(*e);
    if (mlx5_devices.empty()) {
        UCS_TEST_SKIP_R("No mlx5 network device available");
    }

    /* Find a device with port suffix (contains ':') */
    std::string test_dev_name = *mlx5_devices.begin();
    ASSERT_NE(test_dev_name.find(':'), std::string::npos)
            << "No port suffix found in device name";

    m_entities.clear();

    /* Create context with explicit device:port specification */
    modify_config("NET_DEVICES", test_dev_name.c_str());
    e = create_entity();

    /* Verify the specific device was selected */
    std::set<std::string> selected_devices = get_mlx5_device_names(*e);
    EXPECT_TRUE(has_device(selected_devices, test_dev_name))
        << "Device '" << test_dev_name << "' should be selected";
}

/*
 * Test that device name range specification works with base names.
 * E.g., "mlx5_[0-1]" should match mlx5_0:1 and mlx5_1:1
 */
UCS_TEST_P(test_ucp_net_devices_config, range_with_base_names)
{
    entity *e = create_entity();

    std::set<std::string> mlx5_devices = get_mlx5_device_names(*e);
    if (mlx5_devices.empty()) {
        UCS_TEST_SKIP_R("No mlx5 network device available");
    }

    size_t num_mlx5_devices = mlx5_devices.size();
    if (num_mlx5_devices < 2) {
        UCS_TEST_SKIP_R("Need at least 2 mlx5 devices for range test");
    }

    m_entities.clear();

    /* Use a range that should match all mlx devices */
    modify_config("NET_DEVICES", "mlx5_[0-99]");
    e = create_entity();

    /* Verify that mlx5 devices were selected */
    std::set<std::string> selected_devices = get_mlx5_device_names(*e);
    EXPECT_EQ(selected_devices.size(), num_mlx5_devices)
            << "Expected " << num_mlx5_devices
            << " mlx5 devices to be selected with range";
}

/*
 * Test that specifying a device multiple times (e.g., via range and explicit)
 * produces a warning about duplicate device specification.
 */
UCS_TEST_P(test_ucp_net_devices_config, duplicate_device_warning)
{
    entity *e = create_entity();

    std::set<std::string> mlx5_devices = get_mlx5_device_names(*e);
    if (mlx5_devices.empty()) {
        UCS_TEST_SKIP_R("No mlx5 network device available");
    }

    std::set<std::string> base_names = get_mlx5_base_names(mlx5_devices);
    if (base_names.empty()) {
        UCS_TEST_SKIP_R("No mlx5 devices with port suffix found");
    }

    /* Pick the first base name for testing */
    std::string test_base_name = *base_names.begin();

    m_entities.clear();

    /* Set NET_DEVICES to include both a range and an explicit device that
     * overlaps with the range, e.g., "mlx5_[0-99],mlx5_0" */
    std::string devices_config = "mlx5_[0-99]," + test_base_name;
    modify_config("NET_DEVICES", devices_config.c_str());

    size_t warn_count;
    {
        scoped_log_handler slh(hide_warns_logger);
        warn_count = m_warnings.size();
        create_entity();
    }

    /* Check that a warning about duplicate device was printed */
    std::string expected_warn = "device '" + test_base_name +
                                "' is specified multiple times";
    bool found_warning        = false;
    for (size_t i = warn_count; i < m_warnings.size(); ++i) {
        if (m_warnings[i].find(expected_warn) != std::string::npos) {
            found_warning = true;
            break;
        }
    }

    EXPECT_TRUE(found_warning) << "Expected warning about duplicate device '"
                               << test_base_name << "'";
}

/*
 * Test that a range not covering all devices only selects matching devices.
 * E.g., "mlx5_[1-2]" should match mlx5_1 and mlx5_2, but not mlx5_0.
 */
UCS_TEST_P(test_ucp_net_devices_config, partial_range_selection)
{
    entity *e = create_entity();

    std::set<std::string> mlx5_devices = get_mlx5_device_names(*e);
    if (mlx5_devices.empty()) {
        UCS_TEST_SKIP_R("No mlx5 network device available");
    }

    std::set<std::string> base_names = get_mlx5_base_names(mlx5_devices);

    if (!has_device(base_names, "mlx5_0") ||
        !has_device(base_names, "mlx5_1") ||
        !has_device(base_names, "mlx5_2")) {
        UCS_TEST_SKIP_R(
                "Need mlx5_0, mlx5_1, and mlx5_2 devices for this test");
    }

    m_entities.clear();

    modify_config("NET_DEVICES", "mlx5_[1-2]");
    e = create_entity();

    std::set<std::string> selected_devices = get_mlx5_device_names(*e);

    std::set<std::string> selected_base_names = get_mlx5_base_names(
            selected_devices);

    std::set<std::string> expected_base_names = {"mlx5_1", "mlx5_2"};
    EXPECT_EQ(selected_base_names, expected_base_names);
}

/*
 * Test that non-mlx devices are not affected by the mlx default port logic.
 */
UCS_TEST_P(test_ucp_net_devices_config, non_mlx_device_unaffected)
{
    std::string devices_list;
    std::set<std::string> non_mlx_devices;

    entity *e = create_entity();

    std::set<std::string> mlx5_devices = get_mlx5_device_names(*e);

    /* Find all non-mlx network devices */
    for_each_net_device(*e, [&](const uct_tl_resource_desc_t *rsc) {
        if (!has_device(mlx5_devices, rsc->dev_name)) {
            non_mlx_devices.insert(rsc->dev_name);
        }
    });

    if (non_mlx_devices.empty()) {
        GTEST_SKIP() << "No non-mlx network devices available";
    }

    /* Build comma-separated list of all non-mlx devices */
    for (const std::string &dev : non_mlx_devices) {
        if (!devices_list.empty()) {
            devices_list += ",";
        }
        devices_list += dev;
    }

    m_entities.clear();

    /* Create context with all non-mlx devices */
    modify_config("NET_DEVICES", devices_list.c_str());
    e = create_entity();

    /* Verify all devices were selected */
    std::set<std::string> selected_devices;
    for_each_net_device(*e, [&](const uct_tl_resource_desc_t *rsc) {
        selected_devices.insert(rsc->dev_name);
    });

    for (const std::string &dev : non_mlx_devices) {
        EXPECT_TRUE(has_device(selected_devices, dev))
                << "Non-mlx device '" << dev << "' should be selected";
    }
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_net_devices_config, all, "all")
