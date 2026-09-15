/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>

extern "C" {
#include <ucs/sys/topo/base/topo_groups.h>
}

#include <algorithm>
#include <deque>
#include <random>


static constexpr ucs_sys_pci_id_t pci_id_cx9 = {
    UCS_TOPO_GROUPS_MELLANOX_VENDOR_ID,
    UCS_TOPO_GROUPS_CX9_DEVICE_ID,
};


class test_topo_groups : public ucs::test {
protected:
    using physical_device = std::vector<ucs_sys_device_t>;

    static constexpr int num_devices_per_gpu = 2;
    static constexpr int num_ports_per_nic   = 2;

    struct numa_devices {
        ucs_numa_node_t              numa_node;
        std::vector<physical_device> gpus;
        std::vector<physical_device> nics;
    };

    virtual void cleanup()
    {
        if (m_groups_initialized) {
            ucs_topo_release_groups(&m_groups);
        }

        ucs::test::cleanup();
    }

    ucs_sys_bus_id_t generate_bus_id()
    {
        ucs_sys_bus_id_t bus_id = {};
        ucs_bus_id_bit_rep_t bus_id_bits;

        do {
            bus_id.domain = static_cast<uint16_t>(m_domain_distribution(m_rng));
            bus_id.bus    = static_cast<uint8_t>(m_bus_distribution(m_rng));
            bus_id.slot   = static_cast<uint8_t>(m_slot_distribution(m_rng));
            bus_id.function = 0;
            bus_id_bits     = ucs_topo_get_bus_id_bit_repr(&bus_id);
        } while (!m_bus_id_bits.insert(bus_id_bits).second);

        return bus_id;
    }

    ucs_sys_device_t
    add_device(const std::string &name, const ucs_sys_bus_id_t &bus_id,
               ucs_topo_device_class_t device_class, ucs_numa_node_t numa_node,
               const ucs_sys_pci_id_t *pci_id, uintptr_t user_value)
    {
        ucs_topo_sys_device_info_t device = {};
        ucs_sys_device_t sys_dev;

        m_device_names.push_back(name);
        char *name_cstr = const_cast<char*>(m_device_names.back().c_str());

        device.bus_id          = bus_id;
        device.name            = name_cstr;
        device.numa_node       = numa_node;
        device.user_value      = user_value;
        device.device_class    = device_class;
        device.class_ordinal   = UCS_SYS_DEVICE_ORDINAL_INVALID;
        device.sys_dev_aux     = UCS_SYS_DEVICE_ID_UNKNOWN;
        device.sibling_sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;

        if (pci_id != nullptr) {
            device.pci_id = *pci_id;
        }

        sys_dev = static_cast<ucs_sys_device_t>(m_devices.size());
        m_devices.push_back(device);
        return sys_dev;
    }

    void sort_expected_result(std::vector<numa_devices> &expected_result) const
    {
        const auto sys_dev_less = [this](ucs_sys_device_t lhs,
                                         ucs_sys_device_t rhs) {
            const ucs_bus_id_bit_rep_t lhs_bus = ucs_topo_get_bus_id_bit_repr(
                    &m_devices[lhs].bus_id);
            const ucs_bus_id_bit_rep_t rhs_bus = ucs_topo_get_bus_id_bit_repr(
                    &m_devices[rhs].bus_id);

            return (lhs_bus != rhs_bus) ? (lhs_bus < rhs_bus) :
                                          (m_devices[lhs].user_value <
                                           m_devices[rhs].user_value);
        };

        for (auto &numa : expected_result) {
            for (auto &gpu : numa.gpus) {
                std::sort(gpu.begin(), gpu.end(), sys_dev_less);
            }

            for (auto &nic : numa.nics) {
                std::sort(nic.begin(), nic.end(), sys_dev_less);
            }

            std::sort(numa.gpus.begin(), numa.gpus.end(),
                      [&sys_dev_less](const physical_device &lhs,
                                      const physical_device &rhs) {
                          return sys_dev_less(lhs.front(), rhs.front());
                      });
            std::sort(numa.nics.begin(), numa.nics.end(),
                      [&sys_dev_less](const physical_device &lhs,
                                      const physical_device &rhs) {
                          return sys_dev_less(lhs.front(), rhs.front());
                      });
        }

        std::sort(expected_result.begin(), expected_result.end(),
                  [&sys_dev_less](const numa_devices &lhs,
                                  const numa_devices &rhs) {
                      return sys_dev_less(lhs.gpus.front().front(),
                                          rhs.gpus.front().front());
                  });
    }

    void add_vera_rubin_gpus(unsigned numa_idx, numa_devices &numa)
    {
        for (unsigned gpu_idx = 0; gpu_idx < 2; ++gpu_idx) {
            const ucs_sys_bus_id_t bus_id = generate_bus_id();
            physical_device gpu_devices;

            /* Add devices in reverse order to test that the sorting is correct */
            for (int device_idx = num_devices_per_gpu - 1; device_idx >= 0;
                 --device_idx) {
                const std::string &name = "numa" + std::to_string(numa_idx) +
                                          "_gpu" + std::to_string(gpu_idx) +
                                          "." + std::to_string(device_idx);
                uintptr_t const user_value = static_cast<uintptr_t>(device_idx);

                gpu_devices.push_back(
                        add_device(name, bus_id, UCS_TOPO_DEVICE_CLASS_ACC,
                                   numa.numa_node, nullptr, user_value));
            }

            /* Add an alias with no user value, it should be filtered out */
            add_device("numa" + std::to_string(numa_idx) + "_gpu" +
                               std::to_string(gpu_idx) + ".alias",
                       bus_id, UCS_TOPO_DEVICE_CLASS_ACC, numa.numa_node,
                       nullptr, UCS_SYS_DEVICE_USER_VALUE_EMPTY);

            numa.gpus.push_back(gpu_devices);
        }
    }

    void add_vera_rubin_nics(unsigned numa_idx, numa_devices &numa)
    {
        for (unsigned nic_idx = 0; nic_idx < 5; ++nic_idx) {
            const bool is_cx9                  = nic_idx < 4;
            const ucs_sys_bus_id_t slot_bus_id = generate_bus_id();
            physical_device nic_ports;

            /* Add ports in reverse order to test that the sorting is correct */
            for (int port_idx = num_ports_per_nic - 1; port_idx >= 0;
                 --port_idx) {
                const std::string &name = "numa" + std::to_string(numa_idx) +
                                          "_nic" + std::to_string(nic_idx) +
                                          "." + std::to_string(port_idx);

                /* NICs ports bus ID differs only by the function */
                ucs_sys_bus_id_t port_bus_id = slot_bus_id;
                port_bus_id.function         = static_cast<uint8_t>(port_idx);

                const ucs_sys_device_t sys_dev = add_device(
                        name, port_bus_id, UCS_TOPO_DEVICE_CLASS_NET,
                        numa.numa_node, is_cx9 ? &pci_id_cx9 : nullptr,
                        UCS_SYS_DEVICE_USER_VALUE_EMPTY);

                if (is_cx9) {
                    nic_ports.push_back(sys_dev);
                }
            }

            if (is_cx9) {
                numa.nics.push_back(nic_ports);
            }
        }
    }

    std::vector<numa_devices> add_vera_rubin_devices()
    {
        std::vector<numa_devices> expected_result(2);

        for (unsigned numa_idx = 0; numa_idx < expected_result.size();
             ++numa_idx) {
            numa_devices &numa = expected_result[numa_idx];

            numa.numa_node = static_cast<ucs_numa_node_t>(numa_idx);
            add_vera_rubin_gpus(numa_idx, numa);
            add_vera_rubin_nics(numa_idx, numa);
        }

        sort_expected_result(expected_result);
        return expected_result;
    }

    ucs_status_t build()
    {
        ucs_status_t const status = ucs_topo_build_groups_inner(
                m_devices.data(), static_cast<unsigned>(m_devices.size()),
                &m_groups);

        if (status == UCS_OK) {
            m_groups_initialized = true;
        }

        return status;
    }

    std::string render()
    {
        ucs_string_buffer_t strb = UCS_STRING_BUFFER_INITIALIZER;
        ucs_status_t status;

        status = ucs_topo_groups_render(m_devices.data(), &m_groups, &strb);
        EXPECT_EQ(UCS_OK, status);

        std::string out(ucs_string_buffer_cstr(&strb));
        ucs_string_buffer_cleanup(&strb);
        return out;
    }

    void expect_group(const ucs_topo_group_t &group,
                      const std::vector<physical_device> &expected_gpus,
                      const std::vector<physical_device> &expected_nics)
    {
        ASSERT_EQ(expected_gpus.size(), ucs_array_length(&group.gpus));
        ASSERT_EQ(expected_nics.size(), ucs_array_length(&group.nics));

        for (size_t i = 0; i < expected_gpus.size(); ++i) {
            const ucs_topo_group_element_t &gpu =
                    ucs_array_elem(&group.gpus, i);

            ASSERT_EQ(expected_gpus[i].size(), gpu.num_sys_devs);
            for (size_t j = 0; j < expected_gpus[i].size(); ++j) {
                EXPECT_EQ(expected_gpus[i][j], gpu.sys_devs[j]);
            }
        }

        for (size_t i = 0; i < expected_nics.size(); ++i) {
            const ucs_topo_group_element_t &nic =
                    ucs_array_elem(&group.nics, i);

            ASSERT_EQ(expected_nics[i].size(), nic.num_sys_devs);
            for (size_t j = 0; j < expected_nics[i].size(); ++j) {
                EXPECT_EQ(expected_nics[i][j], nic.sys_devs[j]);
            }
        }
    }

    std::mt19937 m_rng{42};
    std::uniform_int_distribution<unsigned> m_domain_distribution{0,
                                                                  UINT16_MAX};
    std::uniform_int_distribution<unsigned> m_bus_distribution{0, UINT8_MAX};
    std::uniform_int_distribution<unsigned> m_slot_distribution{0, 0x1f};
    std::set<ucs_bus_id_bit_rep_t> m_bus_id_bits;
    std::deque<std::string> m_device_names;
    std::vector<ucs_topo_sys_device_info_t> m_devices;
    ucs_topo_groups_t m_groups;
    bool m_groups_initialized = false;
};


UCS_TEST_F(test_topo_groups, vera_groups_by_numa) {
    const std::vector<numa_devices> expected_groups = add_vera_rubin_devices();

    ASSERT_UCS_OK(build());

    ASSERT_EQ(expected_groups.size(), ucs_array_length(&m_groups));

    for (size_t i = 0; i < expected_groups.size(); ++i) {
        expect_group(ucs_array_elem(&m_groups, i),
                     expected_groups[i].gpus, expected_groups[i].nics);
    }
}

UCS_TEST_F(test_topo_groups, undefined_numa_elements_are_skipped) {
    /* Add devices with valid numa node */
    ucs_numa_node_t numa_node    = 0;
    ucs_sys_device_t gpu_sys_dev = add_device("gpu", generate_bus_id(),
                                              UCS_TOPO_DEVICE_CLASS_ACC,
                                              numa_node, nullptr, 0);
    ucs_sys_device_t nic_sys_dev = add_device("nic", generate_bus_id(),
                                              UCS_TOPO_DEVICE_CLASS_NET,
                                              numa_node, &pci_id_cx9,
                                              UCS_SYS_DEVICE_USER_VALUE_EMPTY);

    /* Add devices with undefined numa node */
    add_device("undefined_gpu", generate_bus_id(), UCS_TOPO_DEVICE_CLASS_ACC,
               UCS_NUMA_NODE_UNDEFINED, nullptr, 0);
    add_device("undefined_nic", generate_bus_id(), UCS_TOPO_DEVICE_CLASS_NET,
               UCS_NUMA_NODE_UNDEFINED, &pci_id_cx9,
               UCS_SYS_DEVICE_USER_VALUE_EMPTY);

    ASSERT_UCS_OK(build());
    ASSERT_EQ(1, ucs_array_length(&m_groups));
    expect_group(ucs_array_elem(&m_groups, 0), {{gpu_sys_dev}},
                 {{nic_sys_dev}});
}

UCS_TEST_F(test_topo_groups, nic_only_group) {
    ucs_sys_bus_id_t port0_bus_id = generate_bus_id();
    ucs_sys_bus_id_t port1_bus_id = port0_bus_id;

    port1_bus_id.function = 1;

    const ucs_sys_device_t port1 = add_device("nic.1", port1_bus_id,
                                              UCS_TOPO_DEVICE_CLASS_NET, 0,
                                              &pci_id_cx9,
                                              UCS_SYS_DEVICE_USER_VALUE_EMPTY);
    const ucs_sys_device_t port0 = add_device("nic.0", port0_bus_id,
                                              UCS_TOPO_DEVICE_CLASS_NET, 0,
                                              &pci_id_cx9,
                                              UCS_SYS_DEVICE_USER_VALUE_EMPTY);


    ASSERT_UCS_OK(build());
    ASSERT_EQ(1, ucs_array_length(&m_groups));
    expect_group(ucs_array_elem(&m_groups, 0), {}, {{port0, port1}});
}

UCS_TEST_F(test_topo_groups, table) {
    const ucs_sys_bus_id_t gpu0_bus_id    = {0, 1, 0, 0};
    const ucs_sys_bus_id_t gpu1_bus_id    = {0, 2, 0, 0};
    const ucs_sys_bus_id_t nic0_port0_bdf = {0, 3, 0, 0};
    const ucs_sys_bus_id_t nic0_port1_bdf = {0, 3, 0, 1};
    const ucs_sys_bus_id_t nic1_bus_id    = {0, 4, 0, 0};
    const ucs_sys_bus_id_t gpu2_bus_id    = {0, 5, 0, 0};
    const ucs_sys_bus_id_t nic2_bus_id    = {0, 6, 0, 0};

    add_device("gpu0.1", gpu0_bus_id, UCS_TOPO_DEVICE_CLASS_ACC, 0, nullptr, 1);
    add_device("gpu0.0", gpu0_bus_id, UCS_TOPO_DEVICE_CLASS_ACC, 0, nullptr, 0);
    add_device("gpu1", gpu1_bus_id, UCS_TOPO_DEVICE_CLASS_ACC, 0, nullptr, 0);
    add_device("gpu2", gpu2_bus_id, UCS_TOPO_DEVICE_CLASS_ACC, 1, nullptr, 0);
    add_device("nic0.1", nic0_port1_bdf, UCS_TOPO_DEVICE_CLASS_NET, 0,
               &pci_id_cx9, UCS_SYS_DEVICE_USER_VALUE_EMPTY);
    add_device("nic0.0", nic0_port0_bdf, UCS_TOPO_DEVICE_CLASS_NET, 0,
               &pci_id_cx9, UCS_SYS_DEVICE_USER_VALUE_EMPTY);
    add_device("nic1", nic1_bus_id, UCS_TOPO_DEVICE_CLASS_NET, 0, &pci_id_cx9,
               UCS_SYS_DEVICE_USER_VALUE_EMPTY);
    add_device("nic2", nic2_bus_id, UCS_TOPO_DEVICE_CLASS_NET, 1, &pci_id_cx9,
               UCS_SYS_DEVICE_USER_VALUE_EMPTY);

    {
        /* Exercise the debug-log integration in addition to the renderer. */
        ucs::scoped_log_level log_level(UCS_LOG_LEVEL_DEBUG);
        ASSERT_UCS_OK(build());
    }

    /* clang-format off */
    EXPECT_EQ("+---------+----------------------+----------------------+\n"
              "| Group # | GPUs                 | NICs                 |\n"
              "+---------+----------------------+----------------------+\n"
              "|       0 | [gpu0.0;gpu0.1] gpu1 | [nic0.0;nic0.1] nic1 |\n"
              "|       1 | gpu2                 | nic2                 |\n"
              "+---------+----------------------+----------------------+",
              render());
    /* clang-format on */
}

UCS_TEST_F(test_topo_groups, table_empty_columns) {
    const ucs_sys_bus_id_t gpu_bus_id = {0, 1, 0, 0};
    const ucs_sys_bus_id_t nic_bus_id = {0, 2, 0, 0};
    ucs_topo_group_t *empty_group;

    add_device("gpu", gpu_bus_id, UCS_TOPO_DEVICE_CLASS_ACC, 0, nullptr, 0);
    add_device("nic", nic_bus_id, UCS_TOPO_DEVICE_CLASS_NET, 1, &pci_id_cx9,
               UCS_SYS_DEVICE_USER_VALUE_EMPTY);

    ASSERT_UCS_OK(build());

    empty_group = ucs_array_append(&m_groups, FAIL());
    ucs_topo_init_group(empty_group);

    EXPECT_EQ("+---------+------+------+\n"
              "| Group # | GPUs | NICs |\n"
              "+---------+------+------+\n"
              "|       0 | gpu  |      |\n"
              "|       1 |      | nic  |\n"
              "|       2 |      |      |\n"
              "+---------+------+------+",
              render());
}

UCS_TEST_F(test_topo_groups, table_empty) {
    ASSERT_UCS_OK(build());

    EXPECT_EQ("+---------+------+------+\n"
              "| Group # | GPUs | NICs |\n"
              "+---------+------+------+\n"
              "| <empty>               |\n"
              "+---------+------+------+",
              render());
}
