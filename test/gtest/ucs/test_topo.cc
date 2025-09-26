/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <common/test.h>
extern "C" {
#include <ucs/memory/numa.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/topo/base/topo.h>
}

static std::string get_sysfs_device_path(const std::string &bdf)
{
    std::string symlink = "/sys/bus/pci/devices/" + bdf;
    char resolved[PATH_MAX];
    if (realpath(symlink.c_str(), resolved)) {
        return std::string(resolved);
    } else {
        return ""; // Not found or invalid BDF
    }
}

class test_topo : public ucs::test {
protected:
    std::vector<std::string> m_hcas, m_gpus, m_dmas;
    ucs_global_state_t *m_topo_state;

    ucs_sys_device_t
    register_device(const std::string &name, const std::string &bdf)
    {
        auto path = get_sysfs_device_path(bdf);
        return ucs_topo_get_sysfs_dev(name.c_str(), path.c_str(), 0);
    }

    void read_pcie_devices();

    // Find a sibling DMA engine for a GPU
    void get_siblings(const std::string &hca_bdf, std::string &gpu_bdf,
                      std::string &dma_bdf)
    {
        std::string hca_path = get_sysfs_device_path(hca_bdf);
        for (const auto &gpu : m_gpus) {
            std::string gpu_path = get_sysfs_device_path(gpu);

            for (const auto &dma : m_dmas) {
                auto gpu_dev = register_device("gpu0", gpu);
                ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, gpu_dev);

                ASSERT_UCS_OK(ucs_topo_sys_device_enable_aux_path(gpu_dev));

                auto hca_dev = register_device("hca0", hca_bdf);
                ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, hca_dev);

                auto dma_dev = register_device("dma", dma);
                ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, dma_dev);

                ASSERT_UCS_OK(
                        ucs_topo_sys_device_set_sys_dev_aux(hca_dev, dma_dev));
                bool is_sibling = ucs_topo_is_sibling(hca_dev, gpu_dev);

                ucs_topo_cleanup();
                ucs_topo_init();

                if (is_sibling) {
                    gpu_bdf = gpu;
                    dma_bdf = dma;
                    return;
                }
            }
        }
    }

public:
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
};

UCS_TEST_F(test_topo, find_device_by_bus_id) {
    ucs_status_t status;
    ucs_sys_device_t dev1;
    ucs_sys_device_t dev2;
    ucs_sys_bus_id_t dummy_bus_id;
    ucs_sys_bus_id_t bus_id1;
    ucs_sys_bus_id_t bus_id2;

    dummy_bus_id.domain   = 0xffff;
    dummy_bus_id.bus      = 0xff;
    dummy_bus_id.slot     = 0xff;
    dummy_bus_id.function = 1;

    status = ucs_topo_find_device_by_bus_id(&dummy_bus_id, &dev1);
    ASSERT_UCS_OK(status);
    EXPECT_LT(dev1, UCS_SYS_DEVICE_ID_MAX);
    status = ucs_topo_sys_device_set_name(dev1, "test_bus_id_1", 10);
    ASSERT_UCS_OK(status);

    status = ucs_topo_get_device_bus_id(dev1, &bus_id1);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(bus_id1.domain, dummy_bus_id.domain);
    EXPECT_EQ(bus_id1.bus, dummy_bus_id.bus);
    EXPECT_EQ(bus_id1.slot, dummy_bus_id.slot);
    EXPECT_EQ(bus_id1.function, dummy_bus_id.function);

    dummy_bus_id.function = 2;

    status = ucs_topo_find_device_by_bus_id(&dummy_bus_id, &dev2);
    ASSERT_UCS_OK(status);
    EXPECT_EQ((unsigned)dev1 + 1, dev2);
    EXPECT_LT(dev2, UCS_SYS_DEVICE_ID_MAX);
    status = ucs_topo_sys_device_set_name(dev2, "test_bus_id_2", 10);
    ASSERT_UCS_OK(status);

    status = ucs_topo_get_device_bus_id(dev2, &bus_id2);
    ASSERT_UCS_OK(status);

    EXPECT_EQ(bus_id2.domain, dummy_bus_id.domain);
    EXPECT_EQ(bus_id2.bus, dummy_bus_id.bus);
    EXPECT_EQ(bus_id2.slot, dummy_bus_id.slot);
    EXPECT_EQ(bus_id2.function, dummy_bus_id.function);

    EXPECT_GE(ucs_topo_num_devices(), 2);
}

UCS_TEST_F(test_topo, get_distance) {
    ucs_status_t status;
    ucs_sys_dev_distance_t distance;

    status = ucs_topo_get_distance(UCS_SYS_DEVICE_ID_UNKNOWN,
                                   UCS_SYS_DEVICE_ID_UNKNOWN, &distance);
    ASSERT_EQ(UCS_OK, status);
    EXPECT_NEAR(distance.latency, 0.0, 1e-9);

    char buf[128];
    UCS_TEST_MESSAGE << "distance: "
                     << ucs_topo_distance_str(&distance, buf, sizeof(buf));
}

UCS_TEST_F(test_topo, print_info) {
    // Restore the state to print the info
    ucs_topo_restore_state(m_topo_state);
    ucs_topo_print_info(stdout);
    // Extract the state again
    m_topo_state = ucs_topo_extract_state();
}

UCS_TEST_F(test_topo, bdf_name) {
    static const char *bdf_name = "0002:8f:5c.0";
    static const char *dev_name = "test_bdf_name";
    static const uintptr_t user_value = 1337;

    ucs_sys_device_t sys_dev    = UCS_SYS_DEVICE_ID_UNKNOWN;

    ucs_status_t status = ucs_topo_find_device_by_bdf_name(bdf_name, &sys_dev);
    ASSERT_UCS_OK(status);
    ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, sys_dev);

    status = ucs_topo_sys_device_set_name(sys_dev, dev_name, 10);
    ASSERT_UCS_OK(status);

    status = ucs_topo_sys_device_set_user_value(sys_dev, user_value);
    ASSERT_UCS_OK(status);

    const char *result_name = ucs_topo_sys_device_get_name(sys_dev);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(std::string(dev_name), std::string(result_name));
    UCS_TEST_MESSAGE << "name: " << result_name;

    uintptr_t result_user_value = ucs_topo_sys_device_get_user_value(sys_dev);
    ASSERT_UCS_OK(status);
    EXPECT_EQ(user_value, result_user_value);
    UCS_TEST_MESSAGE << "user value: " << result_user_value;

    char name_buffer[UCS_SYS_BDF_NAME_MAX];
    const char *found_name = ucs_topo_sys_device_bdf_name(sys_dev, name_buffer,
                                                          sizeof(name_buffer));
    ASSERT_UCS_OK(status);
    EXPECT_EQ(std::string(bdf_name), std::string(found_name));
}

UCS_TEST_F(test_topo, bdf_name_zero_domain) {
    static const char *bdf_name = "0000:8f:5c.0";
    ucs_sys_device_t sys_dev    = UCS_SYS_DEVICE_ID_UNKNOWN;

    const char *short_bdf = strchr(bdf_name, ':') + 1;
    ucs_status_t status = ucs_topo_find_device_by_bdf_name(short_bdf, &sys_dev);
    ASSERT_UCS_OK(status);
    ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, sys_dev);
    status = ucs_topo_sys_device_set_name(sys_dev, "test_bdf_name_zd", 10);
    ASSERT_UCS_OK(status);

    char name_buffer[UCS_SYS_BDF_NAME_MAX];
    const char *found_name = ucs_topo_sys_device_bdf_name(sys_dev, name_buffer,
                                                          sizeof(name_buffer));
    ASSERT_UCS_OK(status);
    EXPECT_EQ(std::string(bdf_name), std::string(found_name));
}

UCS_TEST_F(test_topo, bdf_name_invalid) {
    ucs_sys_device_t sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    ucs_status_t status;

    status = ucs_topo_find_device_by_bdf_name("0000:8f:5c!0", &sys_dev);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    status = ucs_topo_find_device_by_bdf_name("0000:8t:5c.0", &sys_dev);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    status = ucs_topo_find_device_by_bdf_name("5c.0", &sys_dev);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);

    status = ucs_topo_find_device_by_bdf_name("1:2:3", &sys_dev);
    EXPECT_EQ(UCS_ERR_INVALID_PARAM, status);
}

UCS_TEST_F(test_topo, numa_distance) {
    ucs_numa_node_t num_of_nodes;

    num_of_nodes = ucs_numa_num_configured_nodes();
    for (auto node1 = 0; node1 < num_of_nodes; ++node1) {
        for (auto node2 = 0; node2 < num_of_nodes; ++node2) {
            UCS_TEST_MESSAGE << "Test distance: node" << node1 << " to node"
                             << node2;
            if (node1 == node2) {
                EXPECT_EQ(ucs_numa_distance(node1, node2), 10);
            } else {
                EXPECT_EQ(ucs_numa_distance(node1, node2),
                          ucs_numa_distance(node2, node1));
            }
            
            EXPECT_LE(ucs_numa_distance(node1, node1),
                      ucs_numa_distance(node1, node2));
        }
    }
}

// Scan and classify PCI devices
void test_topo::read_pcie_devices()
{
    const char *path = "/sys/bus/pci/devices";

    DIR *dir = opendir(path);
    if (!dir) {
        perror("opendir failed");
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr) {
        if ((entry->d_type != DT_DIR) && (entry->d_type != DT_LNK)) {
            continue;
        }

        std::string bdf        = entry->d_name;
        std::string class_path = std::string(path) + "/" + bdf + "/class";

        std::ifstream class_file(class_path.c_str());
        if (!class_file.is_open()) {
            continue;
        }

        std::string class_code;
        class_file >> class_code;
        class_file.close();

        std::string gpu_class = "0x030200";
        std::string hca_class = "0x020700";
        std::string dma_class = "0x080100";

        // Only keep GPUs, HCAs and their DMA PF
        if ((class_code != hca_class) && (class_code != gpu_class) &&
            (class_code != dma_class)) {
            continue;
        }

        if (class_code == hca_class) {
            m_hcas.push_back(bdf);
        } else if (class_code == gpu_class) {
            m_gpus.push_back(bdf);
        } else {
            m_dmas.push_back(bdf);
        }

        UCS_TEST_MESSAGE << "bdf=" << bdf << " class=" << class_code;
    }

    closedir(dir);
}

UCS_TEST_F(test_topo, sibling_error) {
    scoped_log_handler slh(hide_errors_logger);
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, ucs_topo_sys_device_set_sys_dev_aux(1, 0));
    ASSERT_EQ(UCS_ERR_INVALID_PARAM, ucs_topo_sys_device_enable_aux_path(1));
}

UCS_TEST_F(test_topo, sibling) {
    constexpr int count = 3;

    read_pcie_devices();
    if ((m_hcas.size() < count) || (m_gpus.size() == 0) ||
        (m_dmas.size() == 0)) {
        UCS_TEST_SKIP_R("Not enough HCA, GPU and DMA PCIe device");
    }

    std::string sibling_gpu, sibling_dma;
    get_siblings(m_hcas[0], sibling_gpu, sibling_dma);

    std::vector<ucs_sys_device_t> hca_devs;
    for (int i = 0; i < count; ++i) {
        hca_devs.push_back(
                register_device("hca" + std::to_string(i), m_hcas[i]));
        ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, hca_devs.back());
    }

    auto dma = m_dmas[0];
    auto gpu = m_gpus[0];
    if (!sibling_dma.empty()) {
        dma = sibling_dma;
        gpu = sibling_gpu;
        UCS_TEST_MESSAGE << "Found sibling "
                         << "dma=" << dma << " gpu=" << gpu;
    }

    auto dma_dev = register_device("dma", dma);
    ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, dma_dev);
    auto gpu_dev = register_device("gpu0", gpu);
    ASSERT_NE(UCS_SYS_DEVICE_ID_UNKNOWN, gpu_dev);

    // Link DMA with its HCA
    ASSERT_UCS_OK(ucs_topo_sys_device_set_sys_dev_aux(hca_devs[0], dma_dev));
    // Link fake DMA with its HCA
    ASSERT_UCS_OK(
            ucs_topo_sys_device_set_sys_dev_aux(hca_devs[1], hca_devs[1]));

    // Associate GPU with HCA
    ASSERT_UCS_OK(ucs_topo_sys_device_enable_aux_path(gpu_dev));

    ASSERT_TRUE(ucs_topo_is_reachable(hca_devs[0], gpu_dev));
    // Reachable as there is no auxiliary capability (cuda_ipc)
    ASSERT_TRUE(ucs_topo_is_reachable(hca_devs[2], gpu_dev));
    ASSERT_FALSE(ucs_topo_is_sibling(hca_devs[1], gpu_dev));
    ASSERT_FALSE(ucs_topo_is_sibling(hca_devs[2], gpu_dev));

    ASSERT_TRUE(ucs_topo_is_reachable(hca_devs[1], gpu_dev));

    if (!sibling_dma.empty()) {
        ASSERT_FALSE(ucs_topo_is_reachable(hca_devs[1], gpu_dev));
        ASSERT_TRUE(ucs_topo_is_sibling(hca_devs[0], gpu_dev));
        ASSERT_TRUE(ucs_topo_is_sibling(gpu_dev, hca_devs[0]));
    } else {
        ASSERT_TRUE(ucs_topo_is_reachable(hca_devs[1], gpu_dev));
        ASSERT_FALSE(ucs_topo_is_sibling(hca_devs[0], gpu_dev));
        ASSERT_FALSE(ucs_topo_is_sibling(gpu_dev, hca_devs[0]));
    }
}
