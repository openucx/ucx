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
#include <map>
#include <random>


class test_topo_groups : public ucs::test {
protected:
    using physical_device = std::vector<ucs_sys_device_t>;

    static constexpr int num_devices_per_gpu = 2;
    static constexpr int num_gpus_per_numa   = 2;
    static constexpr int num_ports_per_nic   = 2;
    static constexpr int num_nics_per_numa   = 4;

    struct numa_devices {
        ucs_numa_node_t              numa_node;
        std::vector<physical_device> gpus;
        std::vector<physical_device> nics;
    };

    virtual void cleanup()
    {
        s_pci_path.clear();
        ucs_topo_release_groups(&m_groups);
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

    ucs_sys_device_t add_device(const std::string &name,
                                const ucs_sys_bus_id_t &bus_id,
                                ucs_topo_device_class_t device_class,
                                ucs_numa_node_t numa_node, uintptr_t user_value)
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
        for (unsigned gpu_idx = 0; gpu_idx < num_gpus_per_numa; ++gpu_idx) {
            const ucs_sys_bus_id_t bus_id = generate_bus_id();
            physical_device gpu_devices;

            /* Add devices in reverse order to test that the sorting is correct */
            for (int device_idx = num_devices_per_gpu - 1; device_idx >= 0;
                 --device_idx) {
                const std::string &name = "numa" + std::to_string(numa_idx) +
                                          "_gpu" + std::to_string(gpu_idx) +
                                          "." + std::to_string(device_idx);
                uintptr_t const user_value = static_cast<uintptr_t>(device_idx);

                gpu_devices.push_back(add_device(name, bus_id,
                                                 UCS_TOPO_DEVICE_CLASS_ACC,
                                                 numa.numa_node, user_value));
            }

            numa.gpus.push_back(gpu_devices);
        }
    }

    void add_vera_rubin_nics(unsigned numa_idx, numa_devices &numa)
    {
        for (unsigned nic_idx = 0; nic_idx < num_nics_per_numa; ++nic_idx) {
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

                nic_ports.push_back(
                        add_device(name, port_bus_id, UCS_TOPO_DEVICE_CLASS_NET,
                                   numa.numa_node,
                                   UCS_SYS_DEVICE_USER_VALUE_EMPTY));
            }

            numa.nics.push_back(nic_ports);
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

    ucs_status_t
    build(const ucs_sys_device_bitmap_t *net_device_filter = nullptr,
          ucs_topo_group_by_t group_by = UCS_TOPO_GROUP_BY_NUMA)
    {
        return ucs_topo_build_groups_inner(m_devices.data(),
                                           static_cast<unsigned>(
                                                   m_devices.size()),
                                           net_device_filter, group_by,
                                           pci_switch_depth, &m_groups);
    }

    ucs_status_t build_by_pci_switch()
    {
        return build(nullptr, UCS_TOPO_GROUP_BY_PCI_SWITCH);
    }

    ucs_sys_device_t add_switch_device(const std::string &name,
                                       ucs_topo_device_class_t device_class,
                                       int pci_switch)
    {
        return add_switch_device(name, device_class,
                                 std::vector<int>{pci_switch});
    }

    /* pci_path lists the PCIe switches above the device, from the PCI root */
    ucs_sys_device_t add_switch_device(const std::string &name,
                                       ucs_topo_device_class_t device_class,
                                       const std::vector<int> &pci_path)
    {
        const uintptr_t user_value = (device_class ==
                                      UCS_TOPO_DEVICE_CLASS_ACC) ?
                                             0 :
                                             UCS_SYS_DEVICE_USER_VALUE_EMPTY;
        const ucs_sys_device_t sys_dev = add_device(name, generate_bus_id(),
                                                    device_class, 0,
                                                    user_value);

        s_pci_path[sys_dev] = pci_path;
        return sys_dev;
    }

    /* Devices without a PCIe path do not share a switch with any device */
    static unsigned pci_switch_depth(ucs_sys_device_t sys_dev1,
                                     ucs_sys_device_t sys_dev2)
    {
        auto it1 = s_pci_path.find(sys_dev1);
        auto it2 = s_pci_path.find(sys_dev2);
        unsigned depth = 0;

        if ((it1 == s_pci_path.end()) || (it2 == s_pci_path.end())) {
            return 0;
        }

        while ((depth < it1->second.size()) && (depth < it2->second.size()) &&
               (it1->second[depth] == it2->second[depth])) {
            ++depth;
        }

        return depth;
    }

    /* Order devices by bus id, as the inventory does */
    void sort_devices(std::vector<physical_device> &devices) const
    {
        std::sort(devices.begin(), devices.end(),
                  [this](const physical_device &lhs,
                         const physical_device &rhs) {
                      return ucs_topo_get_bus_id_bit_repr(
                                     &m_devices[lhs.front()].bus_id) <
                             ucs_topo_get_bus_id_bit_repr(
                                     &m_devices[rhs.front()].bus_id);
                  });
    }

    static std::map<ucs_sys_device_t, std::vector<int>> s_pci_path;

    void expect_groups(const std::vector<numa_devices> &expected_groups)
    {
        ASSERT_EQ(expected_groups.size(), ucs_array_length(&m_groups));

        for (size_t i = 0; i < expected_groups.size(); ++i) {
            expect_group(ucs_array_elem(&m_groups, i), expected_groups[i].gpus,
                         expected_groups[i].nics);
        }
    }

    void expect_group(const ucs_topo_group_t &group,
                      const std::vector<physical_device> &expected_gpus,
                      const std::vector<physical_device> &expected_nics)
    {
        ASSERT_EQ(expected_gpus.size(), ucs_array_length(&group.gpus));
        ASSERT_EQ(expected_nics.size(), ucs_array_length(&group.nics));

        for (size_t i = 0; i < expected_gpus.size(); ++i) {
            const ucs_topo_group_element_t &gpu = ucs_array_elem(&group.gpus,
                                                                 i);

            ASSERT_EQ(expected_gpus[i].size(), gpu.num_sys_devs);
            for (size_t j = 0; j < expected_gpus[i].size(); ++j) {
                EXPECT_EQ(expected_gpus[i][j], gpu.sys_devs[j]);
            }
        }

        for (size_t i = 0; i < expected_nics.size(); ++i) {
            const ucs_topo_group_element_t &nic = ucs_array_elem(&group.nics,
                                                                 i);

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
    ucs_topo_groups_t m_groups = {};
};

std::map<ucs_sys_device_t, std::vector<int>> test_topo_groups::s_pci_path;


UCS_TEST_F(test_topo_groups, vera_groups_by_numa) {
    const std::vector<numa_devices> expected_groups = add_vera_rubin_devices();

    ASSERT_UCS_OK(build());
    expect_groups(expected_groups);
}

UCS_TEST_F(test_topo_groups, net_device_filter) {
    std::vector<numa_devices> expected_groups = add_vera_rubin_devices();
    ucs_sys_device_bitmap_t net_device_filter;

    /* Include all network devices in the filter */
    UCS_STATIC_BITMAP_RESET_ALL(&net_device_filter);
    for (const auto &group : expected_groups) {
        for (const physical_device &nic : group.nics) {
            for (const ucs_sys_device_t sys_dev : nic) {
                UCS_STATIC_BITMAP_SET(&net_device_filter, sys_dev);
            }
        }
    }

    /* Filter out all ports of a NIC */
    auto &group0_nics = expected_groups[0].nics;
    for (const ucs_sys_device_t sys_dev : group0_nics[1]) {
        UCS_STATIC_BITMAP_RESET(&net_device_filter, sys_dev);
    }
    group0_nics.erase(group0_nics.begin() + 1);

    /* Filter out one port of a NIC, the NIC keeps its other port */
    physical_device &partial_nic = expected_groups[1].nics[0];
    UCS_STATIC_BITMAP_RESET(&net_device_filter, partial_nic.back());
    partial_nic.pop_back();

    ASSERT_UCS_OK(build(&net_device_filter));
    expect_groups(expected_groups);
}

UCS_TEST_F(test_topo_groups, undefined_numa_elements_are_skipped) {
    /* Add devices with valid numa node */
    ucs_numa_node_t numa_node    = 0;
    ucs_sys_device_t gpu_sys_dev = add_device("gpu", generate_bus_id(),
                                              UCS_TOPO_DEVICE_CLASS_ACC,
                                              numa_node, 0);
    ucs_sys_device_t nic_sys_dev = add_device("nic", generate_bus_id(),
                                              UCS_TOPO_DEVICE_CLASS_NET,
                                              numa_node,
                                              UCS_SYS_DEVICE_USER_VALUE_EMPTY);

    /* Add devices with undefined numa node */
    add_device("undefined_gpu", generate_bus_id(), UCS_TOPO_DEVICE_CLASS_ACC,
               UCS_NUMA_NODE_UNDEFINED, 0);
    add_device("undefined_nic", generate_bus_id(), UCS_TOPO_DEVICE_CLASS_NET,
               UCS_NUMA_NODE_UNDEFINED, UCS_SYS_DEVICE_USER_VALUE_EMPTY);

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
                                              UCS_SYS_DEVICE_USER_VALUE_EMPTY);
    const ucs_sys_device_t port0 = add_device("nic.0", port0_bus_id,
                                              UCS_TOPO_DEVICE_CLASS_NET, 0,
                                              UCS_SYS_DEVICE_USER_VALUE_EMPTY);


    ASSERT_UCS_OK(build());
    ASSERT_EQ(1, ucs_array_length(&m_groups));
    expect_group(ucs_array_elem(&m_groups, 0), {}, {{port0, port1}});
}

UCS_TEST_F(test_topo_groups, pci_switch_groups) {
    std::vector<numa_devices> expected_groups(3);

    /* One GPU with two NICs, one GPU with one NIC, two GPUs with two NICs */
    const std::vector<std::pair<int, int>> switches = {{1, 2}, {1, 1}, {2, 2}};

    for (size_t sw = 0; sw < switches.size(); ++sw) {
        const std::string prefix = "switch" + std::to_string(sw);

        for (int i = 0; i < switches[sw].first; ++i) {
            expected_groups[sw].gpus.push_back({add_switch_device(
                    prefix + "_gpu" + std::to_string(i),
                    UCS_TOPO_DEVICE_CLASS_ACC, sw)});
        }

        for (int i = 0; i < switches[sw].second; ++i) {
            expected_groups[sw].nics.push_back({add_switch_device(
                    prefix + "_nic" + std::to_string(i),
                    UCS_TOPO_DEVICE_CLASS_NET, sw)});
        }
    }

    sort_expected_result(expected_groups);

    ASSERT_UCS_OK(build_by_pci_switch());
    expect_groups(expected_groups);
}

UCS_TEST_F(test_topo_groups, pci_switch_nested_switches) {
    std::vector<numa_devices> expected_groups(2);

    /* Both GPUs share upstream switch 0, every NIC must join the GPU below
     * its deepest switch only */
    expected_groups[0].gpus = {{add_switch_device(
            "gpu0", UCS_TOPO_DEVICE_CLASS_ACC, std::vector<int>{0, 1})}};
    expected_groups[0].nics = {
        {add_switch_device("nic0", UCS_TOPO_DEVICE_CLASS_NET,
                           std::vector<int>{0, 1})},
        {add_switch_device("nic1", UCS_TOPO_DEVICE_CLASS_NET,
                           std::vector<int>{0, 1, 3})}};
    expected_groups[1].gpus = {{add_switch_device(
            "gpu1", UCS_TOPO_DEVICE_CLASS_ACC, std::vector<int>{0, 2})}};
    expected_groups[1].nics = {
        {add_switch_device("nic2", UCS_TOPO_DEVICE_CLASS_NET,
                           std::vector<int>{0, 2})},
        {add_switch_device("nic3", UCS_TOPO_DEVICE_CLASS_NET,
                           std::vector<int>{0, 2, 4})}};

    sort_expected_result(expected_groups);

    ASSERT_UCS_OK(build_by_pci_switch());
    expect_groups(expected_groups);
}

UCS_TEST_F(test_topo_groups, pci_switch_upstream_nic_not_merged) {
    std::vector<numa_devices> expected_groups(2);

    /* The GPUs decided switches 1 and 2, nic1 is only below switch 0, so it is
     * not assigned and the GPUs are not merged */
    expected_groups[0].gpus = {{add_switch_device(
            "gpu0", UCS_TOPO_DEVICE_CLASS_ACC, std::vector<int>{0, 1})}};
    expected_groups[0].nics = {{add_switch_device(
            "nic0", UCS_TOPO_DEVICE_CLASS_NET, std::vector<int>{0, 1})}};
    expected_groups[1].gpus = {{add_switch_device(
            "gpu1", UCS_TOPO_DEVICE_CLASS_ACC, std::vector<int>{0, 2})}};
    expected_groups[1].nics = {{add_switch_device(
            "nic2", UCS_TOPO_DEVICE_CLASS_NET, std::vector<int>{0, 2})}};
    const ucs_sys_device_t nic1 = add_switch_device(
            "nic1", UCS_TOPO_DEVICE_CLASS_NET, std::vector<int>{0});

    sort_expected_result(expected_groups);

    ASSERT_UCS_OK(build_by_pci_switch());
    ASSERT_EQ(3, ucs_array_length(&m_groups));
    for (size_t i = 0; i < expected_groups.size(); ++i) {
        expect_group(ucs_array_elem(&m_groups, i), expected_groups[i].gpus,
                     expected_groups[i].nics);
    }

    expect_group(ucs_array_elem(&m_groups, 2), {}, {{nic1}});
}

UCS_TEST_F(test_topo_groups, pci_switch_deepest_switch_wins) {
    std::vector<numa_devices> expected_groups(2);

    /* gpu0 decided switch 0, which is also above nic2, but nic2 joins gpu1
     * which decided the deeper switch 2 */
    expected_groups[0].gpus = {{add_switch_device(
            "gpu0", UCS_TOPO_DEVICE_CLASS_ACC, std::vector<int>{0})}};
    expected_groups[0].nics = {{add_switch_device(
            "nic0", UCS_TOPO_DEVICE_CLASS_NET, std::vector<int>{0})}};
    expected_groups[1].gpus = {{add_switch_device(
            "gpu1", UCS_TOPO_DEVICE_CLASS_ACC, std::vector<int>{0, 2})}};
    expected_groups[1].nics = {{add_switch_device(
            "nic2", UCS_TOPO_DEVICE_CLASS_NET, std::vector<int>{0, 2})}};

    sort_expected_result(expected_groups);

    ASSERT_UCS_OK(build_by_pci_switch());
    expect_groups(expected_groups);
}

UCS_TEST_F(test_topo_groups, pci_switch_unmatched_devices) {
    const ucs_sys_device_t local_gpu = add_switch_device(
            "local_gpu", UCS_TOPO_DEVICE_CLASS_ACC, 0);
    const ucs_sys_device_t local_nic = add_switch_device(
            "local_nic", UCS_TOPO_DEVICE_CLASS_NET, 0);
    const ucs_sys_device_t lonely_gpu = add_switch_device(
            "lonely_gpu", UCS_TOPO_DEVICE_CLASS_ACC, 1);
    const ucs_sys_device_t lonely_nic = add_switch_device(
            "lonely_nic", UCS_TOPO_DEVICE_CLASS_NET, 2);
    const ucs_sys_device_t root_nic = add_device(
            "root_nic", generate_bus_id(), UCS_TOPO_DEVICE_CLASS_NET, 0,
            UCS_SYS_DEVICE_USER_VALUE_EMPTY);
    std::vector<physical_device> gpus           = {{local_gpu}, {lonely_gpu}};
    std::vector<physical_device> unmatched_nics = {{lonely_nic}, {root_nic}};

    sort_devices(gpus);
    sort_devices(unmatched_nics);

    ASSERT_UCS_OK(build_by_pci_switch());
    ASSERT_EQ(3, ucs_array_length(&m_groups));
    for (size_t i = 0; i < gpus.size(); ++i) {
        const std::vector<physical_device> nics =
                (gpus[i].front() == local_gpu) ?
                        std::vector<physical_device>{{local_nic}} :
                        std::vector<physical_device>{};

        expect_group(ucs_array_elem(&m_groups, i), {gpus[i]}, nics);
    }

    expect_group(ucs_array_elem(&m_groups, 2), {}, unmatched_nics);
}

UCS_TEST_F(test_topo_groups, pci_switch_no_shared_switch) {
    const std::vector<numa_devices> numa_groups = add_vera_rubin_devices();
    std::vector<physical_device> gpus, nics;

    for (const auto &numa : numa_groups) {
        gpus.insert(gpus.end(), numa.gpus.begin(), numa.gpus.end());
        nics.insert(nics.end(), numa.nics.begin(), numa.nics.end());
    }

    sort_devices(gpus);
    sort_devices(nics);

    /* Every GPU forms its own group, and all NICs are unmatched */
    ASSERT_UCS_OK(build_by_pci_switch());
    ASSERT_EQ(gpus.size() + 1, ucs_array_length(&m_groups));
    for (size_t i = 0; i < gpus.size(); ++i) {
        expect_group(ucs_array_elem(&m_groups, i), {gpus[i]}, {});
    }

    expect_group(ucs_array_elem(&m_groups, gpus.size()), {}, nics);
}
