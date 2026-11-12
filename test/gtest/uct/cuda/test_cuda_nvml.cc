/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025-2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <uct/cuda/base/cuda_nvml.h>
#include <uct/cuda/base/cuda_util.h>
#include <ucs/sys/topo/base/topo.h>
}

#include <map>

class test_cuda_nvml : public ucs::test {
};

UCS_TEST_F(test_cuda_nvml, device_get_field_values) {
    nvmlDevice_t device;
    auto status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, 0,
                                          &device);
    EXPECT_EQ(status, UCS_OK);

    nvmlFieldValue_t value;
    value.fieldId = NVML_FI_DEV_NVLINK_LINK_COUNT;
    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetFieldValues, device, 1,
                                     &value);
    EXPECT_EQ(status, UCS_OK);

    nvmlPciInfo_t pci;
    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetNvLinkRemotePciInfo, device,
                                     0, &pci);
    EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_IO_ERROR));
}

UCS_TEST_F(test_cuda_nvml, device_rubin_flag) {
    std::map<ucs_bus_id_bit_rep_t, nvmlDeviceArchitecture_t> nvml_dev_archs;
    unsigned num_nvml_devices;
    nvmlDeviceArchitecture_t arch;
    ucs_sys_bus_id_t bus_id;
    nvmlDevice_t device;
    nvmlPciInfo_t pci;

    uct_cuda_init_devices();

    if (ucs_topo_device_class_is_incomplete(UCS_TOPO_DEVICE_CLASS_ACC)) {
        UCS_TEST_SKIP_R("gpu discovery is incomplete");
    }

    /* Collect the NVML device architectures */
    ASSERT_UCS_OK(
            UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetCount_v2, &num_nvml_devices));
    for (unsigned i = 0; i < num_nvml_devices; ++i) {
        ASSERT_UCS_OK(UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, i,
                                              &device));
        ASSERT_UCS_OK(
                UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetPciInfo_v3, device, &pci));
        if (UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetArchitecture, device, &arch) !=
            UCS_OK) {
            arch = NVML_DEVICE_ARCH_UNKNOWN;
        }

        bus_id.domain   = pci.domain;
        bus_id.bus      = pci.bus;
        bus_id.slot     = pci.device;
        bus_id.function = 0;

        nvml_dev_archs[ucs_topo_get_bus_id_bit_repr(&bus_id)] = arch;
    }

    /* Check the flags of topo GPU devices */
    const unsigned num_devices = ucs_topo_num_devices();
    for (unsigned i = 0; i < num_devices; ++i) {
        auto sys_dev = static_cast<ucs_sys_device_t>(i);
        if (ucs_topo_sys_device_get_class(sys_dev) !=
            UCS_TOPO_DEVICE_CLASS_ACC) {
            continue;
        }

        ASSERT_UCS_OK(ucs_topo_get_device_bus_id(sys_dev, &bus_id));

        auto it = nvml_dev_archs.find(ucs_topo_get_bus_id_bit_repr(&bus_id));
        ASSERT_NE(nvml_dev_archs.end(), it)
                << "no nvml device for "
                << ucs_topo_sys_device_get_name(sys_dev);

        EXPECT_EQ(it->second == UCT_CUDA_NVML_DEVICE_ARCH_RUBIN,
                  (ucs_topo_sys_device_get_flags(sys_dev) &
                   UCS_TOPO_DEVICE_FLAG_RUBIN) != 0)
                << ucs_topo_sys_device_get_name(sys_dev);
    }
}

#if HAVE_NVML_FABRIC_INFO
UCS_TEST_F(test_cuda_nvml, device_get_fabric_info) {
    nvmlDevice_t device;
    auto status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetHandleByIndex, 0,
                                          &device);
    EXPECT_EQ(status, UCS_OK);

    nvmlGpuFabricInfoV_t fabric_info;
    fabric_info.version = nvmlGpuFabricInfo_v2;
    status = UCT_CUDA_NVML_WRAP_CALL(nvmlDeviceGetGpuFabricInfoV, device,
                                     &fabric_info);
    EXPECT_TRUE((status == UCS_OK) || (status == UCS_ERR_IO_ERROR));
}
#endif
