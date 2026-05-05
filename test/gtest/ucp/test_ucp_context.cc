/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <cstddef>
#include <set>
#include <vector>

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
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

UCS_TEST_P(test_ucp_context, dmabuf_reg_devices_config)
{
    modify_config("DMABUF_REG_DEVICES", "2");

    entity *e = create_entity();
    EXPECT_EQ(UCP_DMABUF_REG_DEVICES_LIMIT,
              e->ucph()->config.ext.dmabuf_reg_devices_mode);
    EXPECT_EQ(2u, e->ucph()->config.ext.dmabuf_reg_devices_count);
}

UCS_TEST_P(test_ucp_context, dmabuf_reg_devices_limits_memh_md_map)
{
    if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("CUDA is not supported");
    }

    /* Create entity with dmabuf_reg_devices=all to discover how many dmabuf MDs
     * are available on this system */
    modify_config("DMABUF_REG_DEVICES", "all");
    entity *e_all         = create_entity();
    ucp_context_h ctx_all = e_all->ucph();

    if (ucs_popcount(ctx_all->dmabuf_reg_md_map) < 2) {
        UCS_TEST_SKIP_R("need at least 2 dmabuf-capable MDs");
    }

    const size_t size = 4096;
    void *ptr         = mem_buffer::allocate(size, UCS_MEMORY_TYPE_CUDA);

    ucp_mem_map_params_t params = {};
    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                  UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address              = ptr;
    params.length               = size;

    ucp_mem_h memh_all;
    ASSERT_UCS_OK(ucp_mem_map(ctx_all, &params, &memh_all));
    ucp_md_map_t dmabuf_all = memh_all->md_map & ctx_all->dmabuf_reg_md_map;
    ucp_mem_unmap(ctx_all, memh_all);

    /* Create a second entity with dmabuf_reg_devices=1 and verify that fewer
     * dmabuf MDs end up on the memh */
    modify_config("DMABUF_REG_DEVICES", "1");
    entity *e_lim         = create_entity();
    ucp_context_h ctx_lim = e_lim->ucph();

    ucp_mem_h memh_lim;
    ASSERT_UCS_OK(ucp_mem_map(ctx_lim, &params, &memh_lim));
    ucp_md_map_t dmabuf_lim = memh_lim->md_map & ctx_lim->dmabuf_reg_md_map;
    ucp_mem_unmap(ctx_lim, memh_lim);

    EXPECT_GT(ucs_popcount(dmabuf_all), 0);
    EXPECT_LE(ucs_popcount(dmabuf_lim), 1);
    EXPECT_LE(ucs_popcount(dmabuf_lim), ucs_popcount(dmabuf_all));

    mem_buffer::release(ptr, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_ucp_context, dmabuf_reg_devices_closest_is_subset)
{
    if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("CUDA is not supported");
    }

    modify_config("DMABUF_REG_DEVICES", "all");
    entity *e_all         = create_entity();
    ucp_context_h ctx_all = e_all->ucph();

    if (ucs_popcount(ctx_all->dmabuf_reg_md_map) < 2) {
        UCS_TEST_SKIP_R("need at least 2 dmabuf-capable MDs");
    }

    const size_t size = 4096;
    void *ptr         = mem_buffer::allocate(size, UCS_MEMORY_TYPE_CUDA);

    ucp_mem_map_params_t params = {};
    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                  UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address              = ptr;
    params.length               = size;

    ucp_mem_h memh_all;
    ASSERT_UCS_OK(ucp_mem_map(ctx_all, &params, &memh_all));
    ucp_md_map_t dmabuf_all = memh_all->md_map & ctx_all->dmabuf_reg_md_map;
    ucp_mem_unmap(ctx_all, memh_all);

    modify_config("DMABUF_REG_DEVICES", "closest");
    entity *e_close         = create_entity();
    ucp_context_h ctx_close = e_close->ucph();

    ucp_mem_h memh_close;
    ASSERT_UCS_OK(ucp_mem_map(ctx_close, &params, &memh_close));
    ucp_md_map_t dmabuf_close = memh_close->md_map &
                                ctx_close->dmabuf_reg_md_map;
    ucp_mem_unmap(ctx_close, memh_close);

    EXPECT_GT(ucs_popcount(dmabuf_close), 0);
    EXPECT_LE(ucs_popcount(dmabuf_close), ucs_popcount(dmabuf_all));

    mem_buffer::release(ptr, UCS_MEMORY_TYPE_CUDA);
}

UCS_TEST_P(test_ucp_context, dmabuf_reg_devices_all_preserves_non_dmabuf)
{
    if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("CUDA is not supported");
    }

    modify_config("DMABUF_REG_DEVICES", "all");
    entity *e_all         = create_entity();
    ucp_context_h ctx_all = e_all->ucph();

    modify_config("DMABUF_REG_DEVICES", "1");
    entity *e_lim         = create_entity();
    ucp_context_h ctx_lim = e_lim->ucph();

    if (ctx_all->dmabuf_reg_md_map == 0) {
        UCS_TEST_SKIP_R("no dmabuf-capable MDs");
    }

    const size_t size = 4096;
    void *ptr         = mem_buffer::allocate(size, UCS_MEMORY_TYPE_CUDA);

    ucp_mem_map_params_t params = {};
    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                  UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address              = ptr;
    params.length               = size;

    ucp_mem_h memh_all;
    ASSERT_UCS_OK(ucp_mem_map(ctx_all, &params, &memh_all));

    ucp_mem_h memh_lim;
    ASSERT_UCS_OK(ucp_mem_map(ctx_lim, &params, &memh_lim));

    /* Non-dmabuf MDs (e.g. cuda_copy, self) should be registered identically
     * regardless of the dmabuf device policy */
    ucp_md_map_t non_dmabuf_all = memh_all->md_map &
                                  ~ctx_all->dmabuf_reg_md_map;
    ucp_md_map_t non_dmabuf_lim = memh_lim->md_map &
                                  ~ctx_lim->dmabuf_reg_md_map;
    EXPECT_EQ(non_dmabuf_all, non_dmabuf_lim);

    ucp_mem_unmap(ctx_all, memh_all);
    ucp_mem_unmap(ctx_lim, memh_lim);
    mem_buffer::release(ptr, UCS_MEMORY_TYPE_CUDA);
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

class test_ucp_dmabuf_reg_devices : public ucs::test {
protected:
    static ucp_context_config_t
    config(ucp_dmabuf_reg_devices_mode_t mode, unsigned count = UCP_MAX_MDS)
    {
        ucp_context_config_t config = {};

        config.dmabuf_reg_devices_mode  = mode;
        config.dmabuf_reg_devices_count = count;
        return config;
    }

    static ucp_dmabuf_reg_select_md_t md(ucp_md_index_t md_index,
                                         const char *name, double latency,
                                         uint32_t last_used = 0)
    {
        ucp_dmabuf_reg_select_md_t md = {};

        md.md_index  = md_index;
        md.name      = name;
        md.latency   = latency;
        md.last_used = last_used;
        return md;
    }

    static void set_context_md(ucp_tl_md_t *tl_mds, ucp_md_index_t md_index,
                               const char *name)
    {
        ucs_strncpy_safe(tl_mds[md_index].rsc.md_name, name,
                         sizeof(tl_mds[md_index].rsc.md_name));
    }

    static void init_context(ucp_context_t &context, ucp_tl_md_t *tl_mds,
                             ucp_dmabuf_reg_devices_mode_t mode, unsigned count)
    {
        context.tl_mds                              = tl_mds;
        context.num_mds                             = 4;
        context.dmabuf_reg_md_map                   = UCS_MASK(4);
        context.dmabuf_reg_timestamp                = 0;
        context.config.ext.dmabuf_reg_devices_mode  = mode;
        context.config.ext.dmabuf_reg_devices_count = count;

        set_context_md(tl_mds, 0, "mlx5_0");
        set_context_md(tl_mds, 1, "mlx5_1");
        set_context_md(tl_mds, 2, "mlx5_2");
        set_context_md(tl_mds, 3, "mlx5_3");
    }
};

UCS_TEST_F(test_ucp_dmabuf_reg_devices, closest_by_latency) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_CLOSEST);
    std::vector<ucp_dmabuf_reg_select_md_t> mds = {md(0, "mlx5_2", 5e-7),
                                                   md(1, "mlx5_0", 3e-7),
                                                   md(2, "mlx5_1", 3e-7),
                                                   md(3, "mlx5_3", 1e-6)};

    EXPECT_EQ(UCS_BIT(1) | UCS_BIT(2),
              ucp_dmabuf_reg_select(&cfg, mds.data(), mds.size()));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices, closest_selects_all_equal_latency) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_CLOSEST);
    std::vector<ucp_dmabuf_reg_select_md_t> mds = {md(0, "mlx5_0", 3e-7),
                                                   md(1, "mlx5_1", 3e-7)};

    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(1),
              ucp_dmabuf_reg_select(&cfg, mds.data(), mds.size()));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices, limit_uses_latency_before_last_used) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_LIMIT, 2);
    std::vector<ucp_dmabuf_reg_select_md_t> mds = {md(0, "mlx5_0", 3e-7, 9),
                                                   md(1, "mlx5_1", 5e-7, 0),
                                                   md(2, "mlx5_2", 3e-7, 8),
                                                   md(3, "mlx5_3", 5e-7, 0)};

    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(2),
              ucp_dmabuf_reg_select(&cfg, mds.data(), mds.size()));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices,
           limit_larger_than_available_selects_all) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_LIMIT, 8);
    std::vector<ucp_dmabuf_reg_select_md_t> mds = {md(0, "mlx5_0", 3e-7),
                                                   md(1, "mlx5_1", 5e-7)};

    EXPECT_EQ(UCS_BIT(0) | UCS_BIT(1),
              ucp_dmabuf_reg_select(&cfg, mds.data(), mds.size()));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices, empty_selects_none) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_LIMIT, 1);

    EXPECT_EQ((ucp_md_map_t)0, ucp_dmabuf_reg_select(&cfg, NULL, 0));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices, limit_prefers_least_recently_used) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_LIMIT, 1);
    std::vector<ucp_dmabuf_reg_select_md_t> mds = {md(0, "mlx5_0", 3e-7, 9),
                                                   md(1, "mlx5_1", 3e-7, 0)};

    EXPECT_EQ(UCS_BIT(1), ucp_dmabuf_reg_select(&cfg, mds.data(), mds.size()));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices, limit_uses_name_after_last_used) {
    ucp_context_config_t cfg = config(UCP_DMABUF_REG_DEVICES_LIMIT, 1);
    std::vector<ucp_dmabuf_reg_select_md_t> mds = {md(0, "mlx5_1", 3e-7, 0),
                                                   md(1, "mlx5_0", 3e-7, 0)};

    EXPECT_EQ(UCS_BIT(1), ucp_dmabuf_reg_select(&cfg, mds.data(), mds.size()));
}

UCS_TEST_F(test_ucp_dmabuf_reg_devices, context_selection_rotates_mds) {
    ucp_context_t context = {};
    ucp_tl_md_t tl_mds[4] = {};
    ucp_md_map_t md_map;

    init_context(context, tl_mds, UCP_DMABUF_REG_DEVICES_LIMIT, 1);

    md_map = ucp_context_select_dmabuf_reg_md_map(&context,
                                                  UCS_BIT(1) | UCS_BIT(3),
                                                  UCS_SYS_DEVICE_ID_UNKNOWN);
    EXPECT_EQ(UCS_BIT(1), md_map);

    ucp_context_dmabuf_reg_mark_used(&context, UCS_BIT(1));
    md_map = ucp_context_select_dmabuf_reg_md_map(&context,
                                                  UCS_BIT(1) | UCS_BIT(3),
                                                  UCS_SYS_DEVICE_ID_UNKNOWN);
    EXPECT_EQ(UCS_BIT(3), md_map);
}

namespace {

const std::string NONEXISTENT_DEV1 = "nonexistent_dev1";
const std::string NONEXISTENT_DEV2 = "nonexistent_dev2";

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

    /* Iterate over all devices of a given type and apply action to each */
    template<typename Action>
    static void
    for_each_device(const entity &e, uct_device_type_t dev_type, Action action)
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
    static bool has_device(const std::set<std::string> &devices,
                           const std::string &dev_name)
    {
        return devices.find(dev_name) != devices.end();
    }


    /* Get all device names of a given type from the context */
    static std::set<std::string>
    get_device_names(const entity &e, uct_device_type_t dev_type)
    {
        std::set<std::string> device_names;
        for_each_device(e, dev_type, [&](const uct_tl_resource_desc_t *rsc) {
            device_names.insert(rsc->dev_name);
        });
        return device_names;
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
        const std::string devices_config = ucs::join(test_devices, ",");

        modify_config(config_name.c_str(), devices_config.c_str());
        entity *e = create_entity();

        const std::set<std::string> selected_devices = get_device_names(*e, dev_type);
        EXPECT_EQ(selected_devices.size(), expected_devices.size());

        for (const std::string &device : expected_devices) {
            EXPECT_TRUE(has_device(selected_devices, device))
                    << "Device '" << device << "' should be selected when "
                    << "UCX_" << config_name << "=" << devices_config;
        }
    }

    /* Test that a device config triggers duplicate device warning */
    void test_duplicate_device_warning(const std::string &required_dev_name,
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

        if (!has_device(devices, required_dev_name)) {
            UCS_TEST_SKIP_R(required_dev_name + " device not available");
        }

        m_entities.clear();

        modify_config(config_name.c_str(), devices_config.c_str());

        size_t warn_count;
        {
            const scoped_log_handler slh(hide_warns_logger);
            warn_count = m_warnings.size();
            create_entity();
        }

        const size_t num_warnings = m_warnings.size() - warn_count;
        if (num_warnings != 1) {
            ADD_FAILURE() << "Expected exactly one warning, got "
                          << num_warnings << " warnings";
            return;
        }

        /* Check that the warning about duplicate device was printed */
        const std::string expected_warn = "device '" + duplicate_dev_name +
                                    "' is specified multiple times";
        EXPECT_NE(m_warnings[warn_count].find(expected_warn), std::string::npos)
                << "Expected warning about duplicate device '"
                << duplicate_dev_name << "' with config '" << devices_config
                << "'";
    }

    /* Test that a device config triggers nonexistent device warning */
    void test_nonexistent_device_warning(bool negate)
    {
        const uct_device_type_t dev_type = device_type();
        entity *e                        = create_entity();

        const std::set<std::string> devices = get_device_names(*e, dev_type);
        if (devices.empty()) {
            UCS_TEST_SKIP_R("No " + device_type_name(dev_type) +
                            " devices available");
        }

        const std::string existing_device = *devices.begin();
        m_entities.clear();

        std::string devices_config;
        if (negate) {
            devices_config = "^";
        }
        devices_config += ucs::join({NONEXISTENT_DEV1, existing_device,
                                     NONEXISTENT_DEV2},
                                    ",");

        const std::string config_name = device_type_config_name(dev_type);
        modify_config(config_name.c_str(), devices_config.c_str());
        modify_config("WARN_INVALID_CONFIG", "y");

        size_t warn_count;
        {
            const scoped_log_handler slh(hide_warns_logger);
            warn_count = m_warnings.size();
            create_entity();
        }

        const size_t num_warnings = m_warnings.size() - warn_count;
        if (num_warnings != 1) {
            ADD_FAILURE() << "Expected exactly one warning, got "
                          << num_warnings << " warnings";
            return;
        }

        const std::string expected_warn =
                device_type_name(dev_type) + " devices '" + NONEXISTENT_DEV1 +
                "','" + NONEXISTENT_DEV2 +
                "' are not available, please use one or more of:";

        EXPECT_NE(m_warnings[warn_count].find(expected_warn), std::string::npos)
                << "Expected warning about nonexistent devices '"
                << NONEXISTENT_DEV1 << "','" << NONEXISTENT_DEV2 << "'";
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
    test_duplicate_device_warning("mlx5_0:1", "mlx5_0:1,mlx5_0:1", "mlx5_0:1");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_base_name)
{
    test_duplicate_device_warning("mlx5_0:1", "mlx5_0:1,mlx5_0", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config, duplicate_device_warning_two_base_name)
{
    test_duplicate_device_warning("mlx5_0:1", "mlx5_0,mlx5_0", "mlx5_0");
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

    const std::set<std::string> selected_devices = get_device_names(*e, dev_type);

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

    const std::set<std::string> selected_devices = get_device_names(*e, dev_type);

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

    const std::set<std::string> selected_devices = get_device_names(*e, dev_type);

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

UCS_TEST_P(test_ucp_devices_config, nonexistent_device_warning)
{
    test_nonexistent_device_warning(false);
}

UCS_TEST_P(test_ucp_devices_config, negate_nonexistent_device_warning)
{
    test_nonexistent_device_warning(true);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_devices_config, all, "all")
