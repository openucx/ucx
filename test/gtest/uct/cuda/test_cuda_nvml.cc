/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <common/test.h>
extern "C" {
#include <uct/cuda/base/cuda_nvml.h>
}

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
    EXPECT_EQ(status, UCS_OK);
}
#endif
