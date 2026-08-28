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
    test_ucp_gpu_nic() : m_assignment_initialized(false)
    {
    }

    virtual void init()
    {
        ucs::test::init();

        m_groups.type = UCS_TOPO_GROUPS_TYPE_VERA_RUBIN;
        ucs_array_init_dynamic(&m_groups.groups);
    }

    virtual void cleanup()
    {
        if (m_assignment_initialized) {
            ucp_gpu_nic_assignment_release(&m_assignment);
        }

        ucs_topo_release_groups(&m_groups);
        ucs::test::cleanup();
    }

protected:
    struct topology_shape_t {
        const size_t num_groups;
        const size_t num_gpus_per_group;
        const size_t num_nics_per_group;
        const size_t num_gpu_devices;
        const size_t num_nic_ports;

        size_t       num_gpus() const noexcept
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

    ucs_status_t build_groups(const topology_shape_t &config)
    {
        ucs_topo_group_t *group;
        ucs_topo_gpu_t *gpu;
        ucs_topo_nic_t *nic;

        for (size_t group_idx = 0; group_idx < config.num_groups; ++group_idx) {
            group = ucs_array_append(&m_groups.groups,
                                     return UCS_ERR_NO_MEMORY);
            ucs_topo_init_group(group);

            for (size_t local_idx = 0; local_idx < config.num_gpus_per_group;
                 ++local_idx) {
                gpu = ucs_array_append(&group->gpus, return UCS_ERR_NO_MEMORY);
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
                nic = ucs_array_append(&group->nics, return UCS_ERR_NO_MEMORY);
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

        return UCS_OK;
    }

    ucs_status_t build_assignment(ucp_gpu_nic_policy_t policy)
    {
        ucs_status_t status;

        status = ucp_gpu_nic_assignment_build(&m_groups, policy, &m_assignment);
        if (status == UCS_OK) {
            m_assignment_initialized = true;
        }

        return status;
    }

    void check_nic_owners(const topology_shape_t &config,
                          const std::vector<size_t> &expected_owners)
    {
        std::vector<size_t> expected_port_counts(config.num_gpus(), 0);

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
                const bool owns_nic = UCS_STATIC_BITMAP_GET(
                        *nic_sys_dev_bitmap,
                        nic_port_sys_dev(config, nic_idx, 0));

                if (owns_nic) {
                    EXPECT_EQ(config.num_gpus(), actual_owner);
                    actual_owner = gpu_idx;
                }

                EXPECT_EQ(gpu_idx == expected_owner, owns_nic);
                for (size_t port_idx = 1; port_idx < config.num_nic_ports;
                     ++port_idx) {
                    EXPECT_EQ(owns_nic,
                              UCS_STATIC_BITMAP_GET(
                                      *nic_sys_dev_bitmap,
                                      nic_port_sys_dev(config, nic_idx,
                                                       port_idx)));
                }
            }

            EXPECT_EQ(expected_owner, actual_owner);
        }

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

        ASSERT_UCS_OK(build_groups(config));
        ASSERT_UCS_OK(build_assignment(policy));
        check_nic_owners(config, expected_owners);
    }

    void check_vera_assignment(ucp_gpu_nic_policy_t policy,
                               const std::vector<size_t> &expected_owners)
    {
        const topology_shape_t config = {
            2, // num_groups
            2, // num_gpus_per_group
            4, // num_nics_per_group
            2, // num_gpu_devices
            2, // num_nic_ports
        };

        check_assignment(config, policy, expected_owners);
    }

private:
    ucs_topo_groups_t m_groups;
    ucp_gpu_nic_assignment_t m_assignment;
    bool m_assignment_initialized;
};

UCS_TEST_F(test_ucp_gpu_nic, vera_flip) {
    check_vera_assignment(UCP_GPU_NIC_POLICY_FLIP, {0, 1, 1, 0, 2, 3, 3, 2});
}

UCS_TEST_F(test_ucp_gpu_nic, vera_alt) {
    check_vera_assignment(UCP_GPU_NIC_POLICY_ALT, {0, 1, 0, 1, 2, 3, 2, 3});
}
