/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
#include <ucp/core/ucp_gpu_nic_assignment.h>

#include <cstring>
#include <vector>

class test_ucp_gpu_nic : public ucs::test {
public:
    test_ucp_gpu_nic() :
        m_groups_initialized(false), m_assignment_initialized(false)
    {
    }

    virtual void cleanup()
    {
        if (m_assignment_initialized) {
            ucp_gpu_nic_assignment_release(&m_assignment);
            m_assignment_initialized = false;
        }

        if (m_groups_initialized) {
            ucs_topo_release_groups(&m_groups);
            m_groups_initialized = false;
        }

        ucs::test::cleanup();
    }

protected:
    struct topology_shape_t {
        ucs_topo_groups_type_t type;
        size_t                 num_groups;
        size_t                 num_gpus_per_group;
        size_t                 num_nics_per_group;
        size_t                 num_gpu_devices;
        size_t                 num_nic_ports;

        size_t                 num_gpus() const noexcept
        {
            return num_groups * num_gpus_per_group;
        }

        size_t num_nics() const noexcept
        {
            return num_groups * num_nics_per_group;
        }

        size_t first_nic_port_sys_dev() const noexcept
        {
            return num_gpus() * num_gpu_devices;
        }
    };

    static ucs_sys_device_t gpu_sys_dev(const topology_shape_t &config,
                                        size_t gpu_idx,
                                        size_t gpu_device_idx) noexcept
    {
        return static_cast<ucs_sys_device_t>(
                (gpu_idx * config.num_gpu_devices) + gpu_device_idx);
    }

    static ucs_sys_device_t nic_port_sys_dev(const topology_shape_t &config,
                                             size_t nic_idx,
                                             size_t port_idx) noexcept
    {
        return static_cast<ucs_sys_device_t>(config.first_nic_port_sys_dev() +
                                             (nic_idx * config.num_nic_ports) +
                                             port_idx);
    }

    void build_groups(const topology_shape_t &config)
    {
        ucs_topo_group_t *group;
        ucs_topo_gpu_t *gpu;
        ucs_topo_nic_t *nic;

        if (m_groups_initialized) {
            ucs_topo_release_groups(&m_groups);
            m_groups_initialized = false;
        }

        ucs_array_init_dynamic(&m_groups.groups);
        m_groups.type        = config.type;
        m_groups_initialized = true;

        for (size_t group_idx = 0; group_idx < config.num_groups; ++group_idx) {
            group = ucs_array_append(&m_groups.groups,
                                     FAIL() << "Failed to append group");
            ucs_topo_init_group(group);

            for (size_t local_idx = 0; local_idx < config.num_gpus_per_group;
                 ++local_idx) {
                gpu = ucs_array_append(&group->gpus,
                                       FAIL() << "Failed to append GPU");
                std::memset(gpu, 0, sizeof(*gpu));
                const size_t gpu_idx = (group_idx * config.num_gpus_per_group) +
                                       local_idx;
                gpu->num_devices     = config.num_gpu_devices;

                for (size_t gpu_device_idx = 0;
                     gpu_device_idx < config.num_gpu_devices;
                     ++gpu_device_idx) {
                    gpu->devices[gpu_device_idx] = gpu_sys_dev(config, gpu_idx,
                                                               gpu_device_idx);
                }
            }

            for (size_t local_idx = 0; local_idx < config.num_nics_per_group;
                 ++local_idx) {
                nic = ucs_array_append(&group->nics,
                                       FAIL() << "Failed to append NIC");
                std::memset(nic, 0, sizeof(*nic));
                const size_t nic_idx = (group_idx * config.num_nics_per_group) +
                                       local_idx;
                nic->num_ports       = config.num_nic_ports;

                for (size_t port_idx = 0; port_idx < config.num_nic_ports;
                     ++port_idx) {
                    nic->ports[port_idx] = nic_port_sys_dev(config, nic_idx,
                                                            port_idx);
                }
            }
        }
    }

    void build_assignment(ucp_gpu_nic_policy_t policy)
    {
        ucs_status_t status;

        if (m_assignment_initialized) {
            ucp_gpu_nic_assignment_release(&m_assignment);
            m_assignment_initialized = false;
        }

        status = ucp_gpu_nic_assignment_build(&m_groups, policy, &m_assignment);
        ASSERT_UCS_OK(status);
        m_assignment_initialized = true;
    }

    void check_gpu_device_aliases(const topology_shape_t &config)
    {
        for (size_t gpu_idx = 0; gpu_idx < config.num_gpus(); ++gpu_idx) {
            size_t gpu_device_idx = 0;
            const ucp_gpu_nic_sys_dev_bitmap_t *expected_bitmap =
                    ucp_gpu_nic_assignment_lookup(&m_assignment,
                                                  gpu_sys_dev(config, gpu_idx,
                                                              gpu_device_idx));
            ASSERT_NE(nullptr, expected_bitmap);

            for (gpu_device_idx = 1; gpu_device_idx < config.num_gpu_devices;
                 ++gpu_device_idx) {
                const ucp_gpu_nic_sys_dev_bitmap_t *actual_bitmap =
                        ucp_gpu_nic_assignment_lookup(
                                &m_assignment,
                                gpu_sys_dev(config, gpu_idx, gpu_device_idx));
                ASSERT_NE(nullptr, actual_bitmap);
                EXPECT_EQ(expected_bitmap, actual_bitmap);
            }
        }
    }

    void check_nic_owners(const topology_shape_t &config,
                          const std::vector<size_t> &expected_owners)
    {
        /* Expected number of assigned NIC ports per physical GPU. */
        std::vector<size_t> expected_port_counts(config.num_gpus(), 0);

        /* Verify that each NIC has exactly one expected physical GPU owner. */
        for (size_t nic_idx = 0; nic_idx < config.num_nics(); ++nic_idx) {
            const size_t expected_owner = expected_owners[nic_idx];
            size_t actual_owner         = config.num_gpus();

            expected_port_counts[expected_owner] += config.num_nic_ports;

            for (size_t gpu_idx = 0; gpu_idx < config.num_gpus(); ++gpu_idx) {
                const ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap =
                        ucp_gpu_nic_assignment_lookup(&m_assignment,
                                                      gpu_sys_dev(config,
                                                                  gpu_idx, 0));
                ASSERT_NE(nullptr, nic_sys_dev_bitmap);

                size_t port_idx         = 0;
                const bool gpu_owns_nic = ucp_gpu_nic_bitmap_test(
                        nic_sys_dev_bitmap,
                        nic_port_sys_dev(config, nic_idx, port_idx));

                /* All ports of a NIC must have the same owner. */
                for (port_idx = 1; port_idx < config.num_nic_ports;
                     ++port_idx) {
                    EXPECT_EQ(gpu_owns_nic,
                              ucp_gpu_nic_bitmap_test(
                                      nic_sys_dev_bitmap,
                                      nic_port_sys_dev(config, nic_idx,
                                                       port_idx)));
                }

                if (gpu_owns_nic) {
                    EXPECT_EQ(config.num_gpus(), actual_owner);
                    actual_owner = gpu_idx;
                }

                EXPECT_EQ(gpu_idx == expected_owner, gpu_owns_nic);
            }

            EXPECT_EQ(expected_owner, actual_owner);
        }

        /* Verify each GPU has the expected number of assigned ports. */
        for (size_t gpu_idx = 0; gpu_idx < config.num_gpus(); ++gpu_idx) {
            const ucp_gpu_nic_sys_dev_bitmap_t *nic_sys_dev_bitmap =
                    ucp_gpu_nic_assignment_lookup(&m_assignment,
                                                  gpu_sys_dev(config, gpu_idx,
                                                              0));
            ASSERT_NE(nullptr, nic_sys_dev_bitmap);
            EXPECT_EQ(expected_port_counts[gpu_idx],
                      static_cast<size_t>(
                              UCS_STATIC_BITMAP_POPCOUNT(*nic_sys_dev_bitmap)));
        }
    }

    void check_assignment(const topology_shape_t &config,
                          ucp_gpu_nic_policy_t policy,
                          const std::vector<size_t> &expected_owners)
    {
        ASSERT_NE(config.num_groups, 0);
        ASSERT_NE(config.num_gpus_per_group, 0);
        ASSERT_NE(config.num_nics_per_group, 0);
        ASSERT_NE(config.num_nic_ports, 0);
        ASSERT_NE(config.num_gpu_devices, 0);

        ASSERT_LE(config.num_gpu_devices, UCS_TOPO_MAX_DEVICES_PER_GPU);
        ASSERT_LE(config.num_nic_ports, UCS_TOPO_MAX_PORTS_PER_NIC);

        ASSERT_LE(config.first_nic_port_sys_dev() +
                          (config.num_nics() * config.num_nic_ports),
                  UCS_SYS_DEVICE_ID_COUNT);
        ASSERT_EQ(config.num_nics(), expected_owners.size());

        for (size_t owner_idx = 0; owner_idx < expected_owners.size();
             ++owner_idx) {
            ASSERT_LT(expected_owners[owner_idx], config.num_gpus());
        }

        build_groups(config);
        build_assignment(policy);
        check_gpu_device_aliases(config);
        check_nic_owners(config, expected_owners);
    }

    void check_vera_rubin_assignment(ucp_gpu_nic_policy_t policy,
                                     const std::vector<size_t> &expected_owners)
    {
        topology_shape_t config;

        /* Vera-Rubin topology is built of groups, each with 2 GPUs and 4 NICs 
         * that are equally distant from each other. */
        config.type               = UCS_TOPO_GROUPS_TYPE_VERA_RUBIN;
        config.num_groups         = 2;
        config.num_gpus_per_group = 2;
        config.num_nics_per_group = 4;

        /* Each GPU may appear as two devices if MPS MLOPart is enabled. */
        for (size_t num_gpu_devices = 1; num_gpu_devices <= 2;
             ++num_gpu_devices) {
            /* Each NIC may appear as two ports if dual-port mode is enabled. */
            for (size_t num_nic_ports = 1; num_nic_ports <= 2;
                 ++num_nic_ports) {
                config.num_gpu_devices = num_gpu_devices;
                config.num_nic_ports   = num_nic_ports;
                check_assignment(config, policy, expected_owners);
            }
        }
    }

private:
    ucs_topo_groups_t m_groups;
    ucp_gpu_nic_assignment_t m_assignment;
    bool m_groups_initialized;
    bool m_assignment_initialized;
};

UCS_TEST_F(test_ucp_gpu_nic, vera_rubin_flip) {
    check_vera_rubin_assignment(UCP_GPU_NIC_POLICY_FLIP,
                                {0, 1, 1, 0, 2, 3, 3, 2});
}

UCS_TEST_F(test_ucp_gpu_nic, vera_rubin_alt) {
    check_vera_rubin_assignment(UCP_GPU_NIC_POLICY_ALT,
                                {0, 1, 0, 1, 2, 3, 2, 3});
}
