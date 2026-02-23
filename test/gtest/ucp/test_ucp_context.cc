/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <set>
#include "absl/strings/str_join.h"

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
        const scoped_log_handler slh(hide_warns_logger);
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
    const std::string ucp_string_version = ucp_get_version_string();

    EXPECT_EQ(string_version,
              ucp_string_version.substr(0, string_version.length()));
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_version, all, "all")

namespace {

/* Iterate over all devices of a given type and apply action to each */
template<typename Action>
static void for_each_device(const ucp_test_base::entity &e,
                            uct_device_type_t dev_type, Action action)
{
    ucp_context_h ctx = e.ucph();
    for (ucp_rsc_index_t i = 0; i < ctx->num_tls; ++i) {
        const uct_tl_resource_desc_t *rsc = &ctx->tl_rscs[i].tl_rsc;
        if (rsc->dev_type == dev_type) {
            action(rsc);
        }
    }
}

/* Check if a specific device name exists in the set */
static bool
has_device(const std::set<std::string> &devices, const std::string &dev_name)
{
    return devices.find(dev_name) != devices.end();
}

/* Get all device names of a given type from the context */
static std::set<std::string>
get_device_names(const ucp_test_base::entity &e, uct_device_type_t dev_type)
{
    std::set<std::string> device_names;
    for_each_device(e, dev_type, [&](const uct_tl_resource_desc_t *rsc) {
        device_names.insert(rsc->dev_name);
    });
    return device_names;
}

} // namespace

class test_ucp_devices_config : public ucp_test {
public:
    static void get_test_variants(std::vector<ucp_test_variant> &variants) {
        for (int dev_type = 0; dev_type < UCT_DEVICE_TYPE_LAST; ++dev_type) {
            add_variant_with_value(
                    variants, UCP_FEATURE_TAG, dev_type,
                    device_type_name(static_cast<uct_device_type_t>(dev_type)));
        }
    }

protected:
    static const char DELIMITER = ':';

    /* Get the device type for the current test variant */
    uct_device_type_t device_type() const
    {
        return static_cast<uct_device_type_t>(get_variant_value());
    }

    /* Get the config variable name for a device type */
    static std::string device_type_config_name(uct_device_type_t dev_type)
    {
        switch (dev_type) {
        case UCT_DEVICE_TYPE_NET:
            return "NET_DEVICES";
        case UCT_DEVICE_TYPE_SHM:
            return "SHM_DEVICES";
        case UCT_DEVICE_TYPE_ACC:
            return "ACC_DEVICES";
        case UCT_DEVICE_TYPE_SELF:
            return "SELF_DEVICES";
        case UCT_DEVICE_TYPE_LAST:
            break;
        }

        UCS_TEST_ABORT("Invalid device type: " << dev_type);
    }

    /* Get device type name for messages */
    static std::string device_type_name(uct_device_type_t dev_type)
    {
        if (dev_type >= UCT_DEVICE_TYPE_LAST) {
            UCS_TEST_ABORT("Invalid device type: " << dev_type);
        }
        return uct_device_type_names[dev_type];
    }

    /* Get all device names of a given type from the context with delimiter */
    static std::set<std::string>
    get_device_names_with_delimiter(const entity &e, uct_device_type_t dev_type)
    {
        std::set<std::string> device_names;
        for_each_device(e, dev_type, [&](const uct_tl_resource_desc_t *rsc) {
            const std::string dev_name(rsc->dev_name);
            const size_t delimiter_pos = dev_name.find(DELIMITER);
            if (delimiter_pos != std::string::npos) {
                device_names.insert(dev_name);
            }
        });
        return device_names;
    }

    static bool
    get_device_base_name(const std::string &dev_name, std::string &base_name)
    {
        const size_t delimiter_pos = dev_name.find(DELIMITER);
        if (delimiter_pos != std::string::npos) {
            base_name = dev_name.substr(0, delimiter_pos);
            return true;
        }
        return false;
    }

    static std::set<std::string>
    get_device_base_names(const std::set<std::string> &dev_names)
    {
        std::set<std::string> base_names;
        for (const std::string &dev_name : dev_names) {
            std::string base_name;
            if (get_device_base_name(dev_name, base_name)) {
                base_names.insert(base_name);
            }
        }
        return base_names;
    }

    /* Test that device selection works correctly */
    void test_device_selection(const std::set<std::string> &test_devices,
                               const std::set<std::string> &expected_devices)
    {
        const uct_device_type_t dev_type = device_type();
        const std::string config_name    = device_type_config_name(dev_type);
        const std::string devices_config = absl::StrJoin(test_devices, ",");

        modify_config(config_name.c_str(), devices_config.c_str());
        entity *e = create_entity();

        const std::set<std::string> selected_devices =
                get_device_names(*e, dev_type);
        EXPECT_EQ(selected_devices.size(), expected_devices.size());

        for (const std::string &device : expected_devices) {
            EXPECT_TRUE(has_device(selected_devices, device))
                    << "Device '" << device << "' should be selected when "
                    << "UCX_" << config_name << "=" << devices_config;
        }
    }

    /* Test that a device config triggers duplicate device warning */
    void
    test_duplicate_device_warning(const std::set<std::string> &required_devices,
                                  const std::string &devices_config,
                                  const std::string &duplicate_dev_name)
    {
        const uct_device_type_t dev_type = device_type();
        const std::string config_name    = device_type_config_name(dev_type);

        entity *e = create_entity();

        const std::set<std::string> devices = get_device_names(*e, dev_type);
        if (devices.empty()) {
            UCS_TEST_SKIP_R("No " + device_type_name(dev_type) +
                            " devices available");
        }

        for (const std::string &required_device : required_devices) {
            if (!has_device(devices, required_device)) {
                UCS_TEST_SKIP_R(required_device + " device not available");
            }
        }

        m_entities.clear();

        modify_config(config_name.c_str(), devices_config.c_str());

        size_t warn_count;
        {
            const scoped_log_handler slh(hide_warns_logger);
            warn_count = m_warnings.size();
            create_entity();
        }

        ASSERT_EQ(m_warnings.size() - warn_count, 1)
                << "Expected exactly one warning";

        /* Check that the warning about duplicate device was printed */
        const std::string expected_warn = "device '" + duplicate_dev_name +
                                    "' is specified multiple times";
        EXPECT_NE(m_warnings[warn_count].find(expected_warn), std::string::npos)
                << "Expected warning about duplicate device '"
                << duplicate_dev_name << "' with config '" << devices_config
                << "'";
    }
};

/*
 * Test that when UCX_*_DEVICES is set to a base name (e.g., "mlx5_0"),
 * devices with the same base name are selected (e.g., "mlx5_0:1").
 */
UCS_TEST_P(test_ucp_devices_config, base_name_selects_device)
{
    const uct_device_type_t dev_type = device_type();
    entity *e                        = create_entity();

    const std::set<std::string> devices = get_device_names_with_delimiter(*e,
                                                                          dev_type);
    if (devices.empty()) {
        UCS_TEST_SKIP_R("No " + device_type_name(dev_type) +
                        " devices available with delimiter");
    }

    m_entities.clear();

    const std::set<std::string> base_names = get_device_base_names(devices);

    test_device_selection(base_names, devices);
}

/*
 * Test that explicit suffix specification works correctly.
 */
UCS_TEST_P(test_ucp_devices_config, explicit_suffix)
{
    const uct_device_type_t dev_type = device_type();
    entity *e                        = create_entity();

    const std::set<std::string> devices = get_device_names_with_delimiter(*e,
                                                                          dev_type);
    if (devices.empty()) {
        UCS_TEST_SKIP_R("No " + device_type_name(dev_type) +
                        " devices available with delimiter");
    }

    m_entities.clear();

    test_device_selection(devices, devices);
}

/*
 * Test that specifying a device multiple times produces a warning
 */
UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_simple)
{
    test_duplicate_device_warning({"mlx5_0:1"}, "mlx5_0:1,mlx5_0:1",
                                  "mlx5_0:1");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_base_name)
{
    test_duplicate_device_warning({"mlx5_0:1"}, "mlx5_0:1,mlx5_0", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_two_base_name)
{
    test_duplicate_device_warning({"mlx5_0:1"}, "mlx5_0,mlx5_0", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_range_full_name_1)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"},
                                  "mlx5_[0-1],mlx5_0:1", "mlx5_0:1");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_range_full_name_2)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"},
                                  "mlx5_0:1,mlx5_[0-1]", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_range_base_name_1)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"}, "mlx5_[0-1],mlx5_0",
                                  "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_range_base_name_2)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"}, "mlx5_0,mlx5_[0-1]",
                                  "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_two_ranges)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1", "mlx5_2:1"},
                                  "mlx5_[0-1],mlx5_[1-2]", "mlx5_1");
}

/*
 * Test that negate mode excludes a single device
 */
UCS_TEST_P(test_ucp_devices_config, negate_single_device)
{
    const uct_device_type_t dev_type = device_type();
    const std::string config_name    = device_type_config_name(dev_type);
    entity *e                        = create_entity();

    const std::set<std::string> devices = get_device_names(*e, dev_type);
    if (devices.size() < 2) {
        UCS_TEST_SKIP_R(std::string("Need at least 2 ") +
                        device_type_name(dev_type) +
                        " devices to test negate mode");
    }

    const std::string excluded_device = *devices.begin();
    m_entities.clear();

    /* Set negate mode - exclude one device */
    modify_config(config_name.c_str(), ("^" + excluded_device).c_str());
    e = create_entity();

    const std::set<std::string> selected_devices = get_device_names(*e,
                                                                    dev_type);

    /* Verify excluded device is not selected */
    EXPECT_FALSE(has_device(selected_devices, excluded_device))
            << "Device '" << excluded_device << "' should be excluded";

    /* Verify that other devices were selected */
    EXPECT_EQ(selected_devices.size(), devices.size() - 1)
            << "Expected 1 device to be excluded";
}

/*
 * Test that negate mode excludes multiple devices
 */
UCS_TEST_P(test_ucp_devices_config, negate_multiple_devices)
{
    const uct_device_type_t dev_type = device_type();
    const std::string config_name    = device_type_config_name(dev_type);
    entity *e                        = create_entity();

    const std::set<std::string> devices = get_device_names(*e, dev_type);
    if (devices.size() < 3) {
        UCS_TEST_SKIP_R(std::string("Need at least 3 ") +
                        device_type_name(dev_type) +
                        " devices to test negate multiple devices");
    }

    /* Get first two devices to exclude */
    auto it                           = devices.begin();
    const std::string excluded_dev1   = *it++;
    const std::string excluded_dev2   = *it++;
    const std::string excluded_config = excluded_dev1 + "," + excluded_dev2;

    m_entities.clear();

    /* Set negate mode - exclude two devices */
    modify_config(config_name.c_str(), ("^" + excluded_config).c_str());
    e = create_entity();

    const std::set<std::string> selected_devices = get_device_names(*e,
                                                                    dev_type);

    /* Verify both excluded devices are not selected */
    EXPECT_FALSE(has_device(selected_devices, excluded_dev1))
            << "Device '" << excluded_dev1 << "' should be excluded";
    EXPECT_FALSE(has_device(selected_devices, excluded_dev2))
            << "Device '" << excluded_dev2 << "' should be excluded";

    /* Verify that other devices were selected */
    EXPECT_EQ(selected_devices.size(), devices.size() - 2)
            << "Expected 2 devices to be excluded";
}

/*
 * Test that negate mode with base name excludes the device
 */
UCS_TEST_P(test_ucp_devices_config, negate_base_name)
{
    const uct_device_type_t dev_type = device_type();
    const std::string config_name    = device_type_config_name(dev_type);
    entity *e                        = create_entity();

    const std::set<std::string> devices = get_device_names_with_delimiter(*e,
                                                                          dev_type);
    if (devices.empty()) {
        UCS_TEST_SKIP_R("No " + device_type_name(dev_type) +
                        " devices available with delimiter");
    }

    const std::set<std::string> base_names = get_device_base_names(devices);
    if (base_names.size() < 2) {
        UCS_TEST_SKIP_R(std::string("Need at least 2 base names to test negate mode"));
    }

    const std::string dev_name = *devices.begin();
    std::string dev_basename;
    ASSERT_TRUE(get_device_base_name(dev_name, dev_basename));

    m_entities.clear();

    /* Set negate mode - exclude by base name */
    modify_config(config_name.c_str(), ("^" + dev_basename).c_str());
    e = create_entity();

    const std::set<std::string> selected_devices = get_device_names(*e,
                                                                    dev_type);

    /* Verify device with that base name is excluded */
    EXPECT_FALSE(has_device(selected_devices, dev_name))
            << "Device '" << dev_name << "' should be excluded";

    /* Verify that other devices were selected */
    EXPECT_GE(selected_devices.size(), base_names.size() - 1)
            << "At least one device should be excluded";

    const std::set<std::string> selected_base_names = get_device_base_names(
            selected_devices);
    EXPECT_EQ(selected_base_names.size(), base_names.size() - 1)
            << "Expected 1 base name to be excluded";
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_devices_config, all, "all")

class test_ucp_devices_config_mlx5_range : public ucp_test {
public:
    test_ucp_devices_config_mlx5_range()
    {
        m_warn_invalid_config = true;
    }

    static void get_test_variants(std::vector<ucp_test_variant> &variants)
    {
        add_variant(variants, UCP_FEATURE_TAG);
    }

protected:
    /* Count how many mlx5 devices exist */
    static unsigned get_mlx5_device_count(const entity &e)
    {
        unsigned max_idx = 0;
        unsigned count   = 0;

        for (const std::string &dev_name :
             get_device_names(e, UCT_DEVICE_TYPE_NET)) {
            unsigned idx, port;
            if (sscanf(dev_name.c_str(), "mlx5_%u:%u", &idx, &port) == 2) {
                EXPECT_EQ(port, 1)
                        << "Expected port 1 for mlx5 device, found: " << port;
                max_idx = std::max(max_idx, idx);
                count++;
            }
        }

        EXPECT_EQ(max_idx + 1, count) << "Expected " << (max_idx + 1)
                                      << " mlx5 devices, found: " << count;

        return count;
    }

    static std::string build_mlx5_device_name(unsigned index)
    {
        return "mlx5_" + std::to_string(index) + ":1";
    }

    /* Build set of device names for mlx5_[start-end] */
    static std::set<std::string>
    build_mlx5_range_set(unsigned start, unsigned end)
    {
        std::set<std::string> result;
        for (unsigned i = start; i <= end; ++i) {
            result.insert(build_mlx5_device_name(i));
        }
        return result;
    }

    /* Build range config string like "mlx5_[0-2]" */
    static std::string build_mlx5_range_config(unsigned start, unsigned end)
    {
        return "mlx5_[" + std::to_string(start) + "-" + std::to_string(end) +
               "]";
    }

    /* Verify that only the expected device is selected */
    void
    verify_single_mlx5_device_selected(unsigned count,
                                       const std::set<std::string> &selected,
                                       unsigned expected_device_idx)
    {
        for (unsigned i = 0; i < count; ++i) {
            std::string const device_name = build_mlx5_device_name(i);
            bool const should_be_selected = (i == expected_device_idx);

            EXPECT_EQ(should_be_selected, has_device(selected, device_name))
                    << device_name << " should be"
                    << (should_be_selected ? "selected" : "excluded");
        }
    }

    /* Test that a range exceeding the device count produces a warning */
    void test_range_exceeds_device_count(bool negate)
    {
        entity *e = create_entity();

        unsigned const count = get_mlx5_device_count(*e);
        if (count < 1) {
            UCS_TEST_SKIP_R("Need at least 1 mlx5 device");
        }

        m_entities.clear();

        /* Range that extends one beyond the last device index:
         * e.g. mlx5_[0-N] when only mlx5_0 through mlx5_(N-1) exist */
        std::string config = ((negate) ? "^" : "") +
                             build_mlx5_range_config(0, count);
        modify_config("NET_DEVICES", config.c_str());

        size_t warn_count;
        {
            const scoped_log_handler slh(hide_warns_logger);
            warn_count = m_warnings.size();
            create_entity();
        }

        ASSERT_EQ(m_warnings.size() - warn_count, 1)
                << "Expected exactly one warning";

        /* Check that the warning about device not available was printed */
        const std::string expected_warn = "device 'mlx5_" +
                                          std::to_string(count) +
                                          "' is not available";
        EXPECT_NE(m_warnings[warn_count].find(expected_warn), std::string::npos)
                << "Expected warning about device not available: 'mlx5_"
                << std::to_string(count) << "'";
    }
};

/*
 * Test that range syntax selects the correct devices.
 */
UCS_TEST_P(test_ucp_devices_config_mlx5_range, range)
{
    entity *e = create_entity();

    unsigned const count = get_mlx5_device_count(*e);
    if (count < 2) {
        UCS_TEST_SKIP_R("Need at least 2 mlx5 devices");
    }

    m_entities.clear();

    /* Exclude all but last device using negated range [0-(count-2)] */
    const std::string config = build_mlx5_range_config(0, count - 2);
    modify_config("NET_DEVICES", config.c_str());
    e = create_entity();

    const std::set<std::string> selected =
            get_device_names(*e, UCT_DEVICE_TYPE_NET);
    const std::set<std::string> expected = build_mlx5_range_set(0, count - 2);

    EXPECT_EQ(expected, selected);
}

/*
 * Test that negated range excludes the correct devices.
 */
UCS_TEST_P(test_ucp_devices_config_mlx5_range, range_negate)
{
    entity *e = create_entity();

    unsigned const count = get_mlx5_device_count(*e);
    if (count < 2) {
        UCS_TEST_SKIP_R("Need at least 2 mlx5 devices");
    }

    m_entities.clear();

    /* Exclude all but last device using negated range [0-(count-2)] */
    const std::string config = "^" + build_mlx5_range_config(0, count - 2);
    modify_config("NET_DEVICES", config.c_str());
    e = create_entity();

    const std::set<std::string> selected =
            get_device_names(*e, UCT_DEVICE_TYPE_NET);

    verify_single_mlx5_device_selected(count, selected, count - 1);
}

/*
 * Test that negated range excludes the correct devices (multiple ranges)
 */
UCS_TEST_P(test_ucp_devices_config_mlx5_range, range_negate_multiple)
{
    entity *e = create_entity();

    unsigned const count = get_mlx5_device_count(*e);
    if (count < 3) {
        UCS_TEST_SKIP_R("Need at least 2 mlx5 devices");
    }

    m_entities.clear();

    /* Exclude all mlx devices but mlx5_1 */
    const std::string config = "^" + build_mlx5_range_config(0, 0) + "," +
                               build_mlx5_range_config(2, count - 1);
    modify_config("NET_DEVICES", config.c_str());
    e = create_entity();

    const std::set<std::string> selected =
            get_device_names(*e, UCT_DEVICE_TYPE_NET);

    verify_single_mlx5_device_selected(count, selected, 1);
}

/*
 * Test that an invalid range (end < start) produces a configuration error.
 */
UCS_TEST_P(test_ucp_devices_config_mlx5_range, range_invalid_produces_error)
{
    ucs_status_t status;
    {
        scoped_log_handler const slh(hide_errors_logger);
        status = ucp_config_modify_internal(m_ucp_config, "NET_DEVICES",
                                            "mlx5_[5-2]");
    }
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_P(test_ucp_devices_config_mlx5_range,
           range_exceeds_device_count_produces_error)
{
    test_range_exceeds_device_count(false);
}

UCS_TEST_P(test_ucp_devices_config_mlx5_range,
           range_negate_exceeds_device_count_produces_error)
{
    test_range_exceeds_device_count(true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_devices_config_mlx5_range, ib, "ib")
