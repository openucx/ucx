/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_test.h"

#include <cstddef>
#include <set>

extern "C" {
#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_mm.h>
#include <uct/base/uct_iface.h>
#include <uct/base/uct_md.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/topo/base/topo.h>
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

UCS_TEST_P(test_ucp_context, max_hca_per_gpu_config)
{
    modify_config("MAX_HCA_PER_GPU", "2");

    entity *e = create_entity();
    EXPECT_EQ(UCP_REG_DEVICES_LIMIT,
              ucp_reg_devices_mode(e->ucph()->config.ext.max_hca_per_gpu));
    EXPECT_EQ(2u, ucp_reg_devices_count(e->ucph()->config.ext.max_hca_per_gpu));
}

UCS_TEST_P(test_ucp_context, controller_device_metadata)
{
    const uint64_t resource = UCT_TL_RESOURCE_DESC_FIELD_RESOURCE;
    const uint64_t control  = UCT_TL_RESOURCE_DESC_FIELD_CTL_DEVICE;
    ucp_test_param param    = GetParam();
    uct_tl_t *tl            = nullptr;
    uct_tl_t *candidate;
    uct_md_h md;
    ucp_context_h context;
    uct_tl_resource_desc_t *legacy;
    uct_tl_resource_desc_v2_t *resources;
    unsigned num_resources;

    param.transports = {"self"};
    context          = create_entity(false, param)->ucph();
    ASSERT_EQ(1, context->num_tls);
    EXPECT_EQ(UCS_SYS_DEVICE_ID_UNKNOWN, context->tl_rscs[0].ctl_device);
    md = context->tl_mds[context->tl_rscs[0].md_index].md;
    ASSERT_UCS_OK(uct_md_query_tl_resources_v2(
            md, control, sizeof(*resources),
            &resources, &num_resources));
    ASSERT_EQ(1u, num_resources);
    EXPECT_EQ(UCS_SYS_DEVICE_ID_UNKNOWN, resources[0].ctl_device);
    uct_release_tl_resource_list_v2(resources);

    ucs_list_for_each(candidate, &md->component->tl_list, list) {
        if (!strcmp(candidate->name, "self")) {
            tl = candidate;
            break;
        }
    }
    ASSERT_NE(nullptr, tl);
    ucs::mock mock;
    auto query = +[](uct_md_h, uct_tl_device_resource_t **devices,
                     unsigned *count) -> ucs_status_t {
        *devices = static_cast<uct_tl_device_resource_t*>(
                ucs_calloc(2, sizeof(**devices), "test_devices"));
        if (*devices == nullptr) {
            return UCS_ERR_NO_MEMORY;
        }

        for (unsigned i = 0; i < 2; ++i) {
            ucs_snprintf_safe((*devices)[i].name, sizeof((*devices)[i].name),
                              "memory%u", i);
            (*devices)[i].type       = UCT_DEVICE_TYPE_SELF;
            (*devices)[i].sys_device = UCS_SYS_DEVICE_ID_UNKNOWN;
            (*devices)[i].ctl_device = 42 + i;
        }
        *count = 2;
        return UCS_OK;
    };
    mock.setup(&tl->query_devices, query);
    for (uint64_t mask : {resource | UCS_BIT(63), control, uint64_t(0)}) {
        const size_t size = (mask & resource) ?
                offsetof(uct_tl_resource_desc_v2_t, ctl_device) :
                sizeof(*resources);
        ASSERT_UCS_OK(uct_md_query_tl_resources_v2(md, mask, size, &resources,
                                                  &num_resources));
        ASSERT_EQ(2u, num_resources);
        EXPECT_EQ(mask & ~UCS_BIT(63), resources[0].field_mask);
        if (mask & resource) {
            EXPECT_STREQ("self", resources[0].resource.tl_name);
            EXPECT_STREQ("memory0", resources[0].resource.dev_name);
        }
        if (mask & control) {
            EXPECT_EQ(42, resources[0].ctl_device);
            EXPECT_EQ(43, resources[1].ctl_device);
        }
        uct_release_tl_resource_list_v2(resources);
    }
    ASSERT_UCS_OK(uct_md_query_tl_resources(md, &legacy, &num_resources));
    EXPECT_EQ(2u, num_resources);
    EXPECT_STREQ("memory0", legacy[0].dev_name);
    EXPECT_STREQ("memory1", legacy[1].dev_name);
    uct_release_tl_resource_list(legacy);
    {
        scoped_log_handler slh(hide_errors_logger);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, uct_md_query_tl_resources_v2(
                md, resource, sizeof(uint64_t), &resources, &num_resources));
    }
    context = create_entity(false, param)->ucph();
    ASSERT_EQ(2, context->num_tls);
    EXPECT_EQ(42, context->tl_rscs[0].ctl_device);
    EXPECT_EQ(43, context->tl_rscs[1].ctl_device);
}

UCS_TEST_P(test_ucp_context, gda_registration_assignment)
{
    ucp_test_param param = GetParam();
    bool has_gda        = false;
    unsigned checked    = 0;
    entity *e;
    ucp_context_h context;

    if (!mem_buffer::is_mem_type_supported(UCS_MEMORY_TYPE_CUDA)) {
        UCS_TEST_SKIP_R("CUDA is not supported");
    }

    modify_config("MAX_HCA_PER_GPU", "1");
    modify_config("IB_GDA_MAX_SYS_LATENCY", "1us");
    param.variant.ctx_params.features |= UCP_FEATURE_DEVICE;
    param.transports                   = {"rc_mlx5", "rc_gda", "cuda_copy"};
    e                                  = create_entity(false, param);
    context                            = e->ucph();
    for (ucp_rsc_index_t i = 0; i < context->num_tls; ++i) {
        has_gda |= !strcmp(context->tl_rscs[i].tl_rsc.tl_name, "rc_gda");
    }

    if (!has_gda) {
        UCS_TEST_SKIP_R("no GDA resources available");
    }

    for (int gpu = 0; gpu < mem_buffer::get_device_count(); ++gpu) {
        scoped_cuda_device guard(gpu);
        mapped_buffer buffer(4096, *e, 0, UCS_MEMORY_TYPE_CUDA);
        ucp_mem_h memh           = buffer.memh();
        ucp_md_map_t assignment  = 0;
        const std::string prefix = "cuda" + std::to_string(gpu) + "-";
        ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, memh->sys_dev);
        for (ucp_rsc_index_t i = 0; i < context->num_tls; ++i) {
            const auto &rsc = context->tl_rscs[i];
            if (!strcmp(rsc.tl_rsc.tl_name, "rc_gda") &&
                !strncmp(rsc.tl_rsc.dev_name, prefix.c_str(), prefix.size())) {
                EXPECT_EQ(memh->sys_dev, rsc.ctl_device);
                assignment |= UCS_BIT(rsc.md_index);
            }
        }

        if (assignment != 0) {
            EXPECT_EQ(1, ucs_popcount(memh->md_map & assignment));
            EXPECT_EQ(1, ucs_popcount(memh->md_map &
                                     ucp_context_get_net_md_map(context)));
            ++checked;
        }
    }

    EXPECT_GT(checked, 0u);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_context, all, "all")

class test_ucp_context_reg_device : public ucs::test {
public:
    virtual void init() override
    {
        ucs::test::init();
        m_topo_state = ucs_topo_extract_state();
        ASSERT_NE(nullptr, m_topo_state);
        s_self = this;

        for (unsigned i = 0; i < ucs_static_array_size(m_devices); ++i) {
            const ucs_sys_bus_id_t bus_id = {0xffff, 0, uint8_t(i), 0};
            ASSERT_UCS_OK(ucs_topo_find_device_by_bus_id(&bus_id,
                                                       &m_devices[i]));
        }

        m_context.config.features = UCP_FEATURE_DEVICE;
        m_context.tl_mds          = m_mds;
        m_context.num_mds         = ucs_static_array_size(m_mds);
        m_context.tl_rscs         = m_resources;
        m_context.num_tls         = ucs_static_array_size(m_resources);
        for (unsigned i = 0; i < ucs_static_array_size(m_mds); ++i) {
            ucs_snprintf_safe(m_mds[i].rsc.md_name,
                              sizeof(m_mds[i].rsc.md_name), "md%u", i);
            m_distances[i] = {10e-9, 20e9};
            for (unsigned j = 0; j < 2; ++j) {
                auto &resource            = m_resources[2 * i + j];
                resource.md_index         = i;
                resource.ctl_device       = (j == 0) ?
                        UCS_SYS_DEVICE_ID_UNKNOWN : m_devices[i];
                resource.tl_rsc.sys_device = m_devices[2 + i];
                resource.tl_rsc.dev_type   = UCT_DEVICE_TYPE_NET;
            }
        }

        const ucs_sys_topo_ops_t ops = {get_distance, nullptr, nullptr};
        ASSERT_UCS_OK(ucs_sys_topo_provider_push(&ops));
        m_provider_pushed = true;
    }

    virtual void cleanup() override
    {
        if (m_provider_pushed) {
            ucs_sys_topo_provider_pop();
        }
        if (m_topo_state != nullptr) {
            ucs_topo_restore_state(m_topo_state);
        }
        s_self = nullptr;
        ucs::test::cleanup();
    }

protected:
    ucp_md_map_t select(unsigned gpu, unsigned long limit,
                        ucp_md_map_t md_map = UCS_BIT(0) | UCS_BIT(1))
    {
        m_context.config.ext.max_hca_per_gpu = limit;
        return ucp_context_select_reg_mds(&m_context, md_map, m_devices[gpu]);
    }

    static ucs_status_t get_distance(ucs_sys_device_t, ucs_sys_device_t device,
                                    ucs_sys_dev_distance_t *distance)
    {
        *distance = s_self->m_distances[device == s_self->m_devices[3]];
        return UCS_OK;
    }

    ucp_context_t m_context               = {};
    ucp_tl_md_t m_mds[2]                  = {};
    ucp_tl_resource_desc_t m_resources[4] = {};
    ucs_sys_device_t m_devices[4]         = {};
    ucs_global_state_t *m_topo_state      = nullptr;
    bool m_provider_pushed                = false;
    ucs_sys_dev_distance_t m_distances[2];
    static test_ucp_context_reg_device *s_self;
};

test_ucp_context_reg_device *test_ucp_context_reg_device::s_self = nullptr;

UCS_TEST_F(test_ucp_context_reg_device, symmetric_assignment_stays_stable)
{
    for (unsigned gpu : {1u, 0u, 0u, 1u, 1u, 0u}) {
        EXPECT_EQ(UCS_BIT(gpu), select(gpu, 1));
    }
}

UCS_TEST_F(test_ucp_context_reg_device, assignment_precedes_distance)
{
    m_distances[0].bandwidth = 40e9;
    EXPECT_EQ(UCS_BIT(1), select(1, 1));
    m_distances[1].latency = 20e-9;
    EXPECT_EQ(UCS_BIT(1), select(1, 1));
    EXPECT_EQ(UCS_BIT(1), select(1, UCS_ULUNITS_AUTO));
}

UCS_TEST_F(test_ucp_context_reg_device, registration_limits)
{
    const ucp_md_map_t both = UCS_BIT(0) | UCS_BIT(1);

    EXPECT_EQ(0, select(1, 0));
    EXPECT_EQ(UCS_BIT(1), select(1, 1));
    EXPECT_EQ(both, select(1, 2));
    EXPECT_EQ(UCS_BIT(1), select(1, UCS_ULUNITS_AUTO));
    EXPECT_EQ(both, select(1, UCS_ULUNITS_INF));

    /* GDA may assign more NICs than the registration limit. */
    m_resources[1].ctl_device = m_devices[1];
    EXPECT_EQ(UCS_BIT(0), select(1, 1));
    EXPECT_EQ(both, select(1, UCS_ULUNITS_AUTO));
    m_distances[1].bandwidth = 40e9;
    EXPECT_EQ(UCS_BIT(1), select(1, 1));
    EXPECT_EQ(UCS_BIT(1), select(1, UCS_ULUNITS_AUTO));
    EXPECT_EQ(both, select(1, 2));
}

UCS_TEST_F(test_ucp_context_reg_device, unassigned_candidates_use_distance)
{
    EXPECT_EQ(UCS_BIT(0), select(1, 1, UCS_BIT(0)));
    m_resources[3].flags = UCP_TL_RSC_FLAG_AUX;
    EXPECT_EQ(UCS_BIT(0), select(1, 1));
    m_resources[3].flags      = 0;
    m_resources[3].ctl_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    EXPECT_EQ(UCS_BIT(0), select(1, 1));
    m_distances[1].bandwidth = 40e9;
    EXPECT_EQ(UCS_BIT(1), select(1, 1));
}

UCS_TEST_F(test_ucp_context_reg_device, no_device_feature_uses_distance)
{
    m_context.config.features = UCP_FEATURE_TAG;
    EXPECT_EQ(UCS_BIT(0), select(1, 1));
    m_distances[1].bandwidth = 40e9;
    EXPECT_EQ(UCS_BIT(1), select(0, 1));
}

UCS_TEST_F(test_ucp_context_reg_device, unknown_device_uses_distance)
{
    m_resources[3].ctl_device = UCS_SYS_DEVICE_ID_UNKNOWN;
    m_distances[0].bandwidth  = 40e9;
    m_devices[1]             = UCS_SYS_DEVICE_ID_UNKNOWN;
    EXPECT_EQ(UCS_BIT(0), select(1, 1));
}

class test_ucp_context_gpu_nic_assignment : public test_ucp_context {
protected:
    /* Run ucp_init() with the given assignment mode, and optionally one more
     * configuration option */
    void init_context(const char *mode, const char *name, const char *value,
                      ucs_status_t *status_p, ucp_context_h *ucph_p)
    {
        ucs::handle<ucp_config_t*> config;

        /* A failed assertion returns before ucp_init() */
        *status_p = UCS_ERR_NO_PROGRESS;

        UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                               ucp_config_read, NULL, NULL);
        ASSERT_UCS_OK(ucp_config_modify(config.get(), "GPU_NIC_ASSIGNMENT_MODE",
                                        mode));
        if (name != NULL) {
            ASSERT_UCS_OK(ucp_config_modify(config.get(), name, value));
        }

        const scoped_log_handler slh(hide_errors_logger);
        *status_p = ucp_init(&get_variant_ctx_params(), config.get(), ucph_p);
    }

    void check_no_assignable_nic(const char *mode, const char *tls,
                                 ucs_status_t exp_status)
    {
        ucp_context_h ucph;
        ucs_status_t status;

        init_context(mode, "TLS", tls, &status, &ucph);

        EXPECT_EQ(exp_status, status);
        if (status == UCS_OK) {
            EXPECT_EQ(nullptr, ucph->gpu_nic_assignment);
            ucp_cleanup(ucph);
        }
    }

    /* Expect ucp_init() to fail when the assignment is combined with a
     * conflicting configuration */
    void expect_init_rejected(const char *name, const char *value)
    {
        ucp_context_h ucph;
        ucs_status_t status;
        bool has_assignment;

        /* Without an assignment, ucp_init() fails or succeeds regardless of
         * the conflicting configuration */
        init_context("flip", NULL, NULL, &status, &ucph);
        if (status == UCS_ERR_INVALID_PARAM) {
            UCS_TEST_SKIP_R("no assignable network devices on this host");
        }

        ASSERT_UCS_OK(status);
        has_assignment = (ucph->gpu_nic_assignment != nullptr);
        ucp_cleanup(ucph);
        if (!has_assignment) {
            UCS_TEST_SKIP_R("no gpu-nic assignment on this host");
        }

        init_context("flip", name, value, &status, &ucph);
        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
        if (status == UCS_OK) {
            ucp_cleanup(ucph);
        }
    }
};

/* shm has no network devices, and the tcp md has no UCT_MD_FLAG_REG, so
 * neither provides a assignable NICs */

UCS_TEST_P(test_ucp_context_gpu_nic_assignment, auto_mode_shm)
{
    check_no_assignable_nic("auto", "shm", UCS_OK);
}

UCS_TEST_P(test_ucp_context_gpu_nic_assignment, auto_mode_tcp)
{
    check_no_assignable_nic("auto", "tcp", UCS_OK);
}

UCS_TEST_P(test_ucp_context_gpu_nic_assignment, explicit_mode_shm)
{
    check_no_assignable_nic("flip", "shm", UCS_ERR_INVALID_PARAM);
}

UCS_TEST_P(test_ucp_context_gpu_nic_assignment, explicit_mode_tcp)
{
    check_no_assignable_nic("flip", "tcp", UCS_ERR_INVALID_PARAM);
}

UCS_TEST_P(test_ucp_context_gpu_nic_assignment, rejects_single_net_device)
{
    expect_init_rejected("SINGLE_NET_DEVICE", "y");
}

UCS_TEST_P(test_ucp_context_gpu_nic_assignment, rejects_old_protocols)
{
    expect_init_rejected("PROTO_ENABLE", "n");
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_context_gpu_nic_assignment, all, "all")

class test_ucp_context_max_rails : public test_ucp_context {
protected:
    void expect_init_rejected(const char *name)
    {
        const std::string value = std::to_string(UCP_MAX_LANES + 1);
        ucs::handle<ucp_config_t*> config;
        ucp_context_h ucph;
        ucs_status_t status;
        size_t num_errors;

        UCS_TEST_CREATE_HANDLE(ucp_config_t*, config, ucp_config_release,
                               ucp_config_read, NULL, NULL);
        ASSERT_UCS_OK(ucp_config_modify(config.get(), name, value.c_str()));

        num_errors = m_errors.size();
        {
            const scoped_log_handler slh(hide_errors_logger);
            status = ucp_init(&get_variant_ctx_params(), config.get(), &ucph);
        }

        EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
        if (status == UCS_OK) {
            ucp_cleanup(ucph);
        }

        ASSERT_GT(m_errors.size(), num_errors);
        EXPECT_NE(std::string::npos,
                  m_errors.back().find(std::string("UCX_") + name + " " +
                                       value + " exceeds"))
                << m_errors.back();
    }
};

UCS_TEST_P(test_ucp_context_max_rails, max_eager_rails_above_limit)
{
    expect_init_rejected("MAX_EAGER_RAILS");
}

UCS_TEST_P(test_ucp_context_max_rails, max_rndv_rails_above_limit)
{
    expect_init_rejected("MAX_RNDV_RAILS");
}

UCS_TEST_P(test_ucp_context_max_rails, max_rma_rails_above_limit)
{
    expect_init_rejected("MAX_RMA_RAILS");
}

UCS_TEST_P(test_ucp_context_max_rails, max_rails_at_limit)
{
    const unsigned max_lanes = UCP_MAX_LANES;
    const std::string value  = std::to_string(max_lanes);

    modify_config("MAX_EAGER_RAILS", value);
    modify_config("MAX_RNDV_RAILS", value);
    modify_config("MAX_RMA_RAILS", value);

    entity *e = create_entity();
    EXPECT_EQ(max_lanes, e->ucph()->config.ext.max_eager_lanes);
    EXPECT_EQ(max_lanes, e->ucph()->config.ext.max_rndv_lanes);
    EXPECT_EQ(max_lanes, e->ucph()->config.ext.max_rma_lanes);
}

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_context_max_rails, all, "all")

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

const std::string NONEXISTENT_DEV = "nonexistent_dev";

/* Iterate over all devices of a given type and apply action to each */
template<typename Action>
void for_each_device(const ucp_test_base::entity &e, uct_device_type_t dev_type,
                     Action action)
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
bool has_device(const std::set<std::string> &devices,
                const std::string &dev_name)
{
    return devices.find(dev_name) != devices.end();
}

/* Get all device names of a given type from the context */
std::set<std::string>
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
        devices_config += ucs::join({NONEXISTENT_DEV + "1", existing_device,
                                     NONEXISTENT_DEV + "[2-3]"},
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
                device_type_name(dev_type) + " devices '" + NONEXISTENT_DEV +
                "1" + "','" + NONEXISTENT_DEV + "2" + "','" + NONEXISTENT_DEV +
                "3" + "' are not available, please use one or more of:";

        EXPECT_NE(m_warnings[warn_count].find(expected_warn), std::string::npos)
                << "Expected warning about 3 nonexistent devices '"
                << NONEXISTENT_DEV + "[1-3]'";
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

class test_ucp_devices_config_mlx5 : public ucp_test {
public:
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

         /* Range tests assume devices mlx5_0 .. mlx5_<max_idx>. This assumption
         * can break in virtualized environments with devices that may use any
         * idx. */
        if ((count > 0) && ((max_idx + 1) != count)) {
            UCS_TEST_SKIP_R("mlx5 devices are not numbered from 0 to " +
                            std::to_string(max_idx));
        }

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
            const std::string device_name = build_mlx5_device_name(i);
            const bool should_be_selected = (i == expected_device_idx);

            EXPECT_EQ(should_be_selected, has_device(selected, device_name))
                    << device_name << " should be "
                    << (should_be_selected ? "selected" : "excluded");
        }
    }

    /* Test that a device config triggers duplicate device warning */
    void
    test_duplicate_device_warning(const std::set<std::string> &required_devices,
                                  const std::string &devices_config,
                                  const std::string &duplicate_dev_name)
    {
        entity *e = create_entity();

        const std::set<std::string> devices =
                get_device_names(*e, UCT_DEVICE_TYPE_NET);

        for (const std::string &required_device : required_devices) {
            if (!has_device(devices, required_device)) {
                UCS_TEST_SKIP_R(required_device + " device not available");
            }
        }

        m_entities.clear();

        modify_config("NET_DEVICES", devices_config.c_str());

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
 * Test that specifying a device multiple times produces a warning
 */
UCS_TEST_P(test_ucp_devices_config_mlx5, duplicate_device_warning_simple)
{
    test_duplicate_device_warning({"mlx5_0:1"}, "mlx5_0:1,mlx5_0:1",
                                  "mlx5_0:1");
}

UCS_TEST_P(test_ucp_devices_config_mlx5, duplicate_device_warning_base_name)
{
    test_duplicate_device_warning({"mlx5_0:1"}, "mlx5_0:1,mlx5_0", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config_mlx5, duplicate_device_warning_two_base_name)
{
    test_duplicate_device_warning({"mlx5_0:1"}, "mlx5_0,mlx5_0", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config_mlx5,
           duplicate_device_warning_range_full_name_1)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"},
                                  "mlx5_[0-1],mlx5_0:1", "mlx5_0:1");
}

UCS_TEST_P(test_ucp_devices_config_mlx5,
           duplicate_device_warning_range_full_name_2)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"},
                                  "mlx5_0:1,mlx5_[0-1]", "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config_mlx5,
           duplicate_device_warning_range_base_name_1)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"}, "mlx5_[0-1],mlx5_0",
                                  "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config_mlx5,
           duplicate_device_warning_range_base_name_2)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1"}, "mlx5_0,mlx5_[0-1]",
                                  "mlx5_0");
}

UCS_TEST_P(test_ucp_devices_config_mlx5, duplicate_device_warning_two_ranges)
{
    test_duplicate_device_warning({"mlx5_0:1", "mlx5_1:1", "mlx5_2:1"},
                                  "mlx5_[0-1],mlx5_[1-2]", "mlx5_1");
}

/*
 * Test that range syntax selects the correct devices.
 */
UCS_TEST_P(test_ucp_devices_config_mlx5, range)
{
    entity *e = create_entity();

    unsigned const count = get_mlx5_device_count(*e);
    if (count < 2) {
        UCS_TEST_SKIP_R("Need at least 2 mlx5 devices");
    }

    m_entities.clear();

    /* Select all but last device using range [0-(count-2)] */
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
UCS_TEST_P(test_ucp_devices_config_mlx5, range_negate)
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
UCS_TEST_P(test_ucp_devices_config_mlx5, range_negate_multiple)
{
    entity *e = create_entity();

    unsigned const count = get_mlx5_device_count(*e);
    if (count < 3) {
        UCS_TEST_SKIP_R("Need at least 3 mlx5 devices");
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

UCP_INSTANTIATE_TEST_CASE_TLS(test_ucp_devices_config_mlx5, ib, "ib")
