/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 * Copyright (C) Intel Corporation, 2023. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "memory_type.h"

#include <stddef.h>


const char *ucs_memory_type_names[] = {
    [UCS_MEMORY_TYPE_HOST]         = "host",
    [UCS_MEMORY_TYPE_CUDA]         = "cuda",
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = "cuda-managed",
    [UCS_MEMORY_TYPE_ROCM]         = "rocm",
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = "rocm-managed",
    [UCS_MEMORY_TYPE_RDMA]         = "rdma",
    [UCS_MEMORY_TYPE_ZE_HOST]      = "ze-host",
    [UCS_MEMORY_TYPE_ZE_DEVICE]    = "ze-device",
    [UCS_MEMORY_TYPE_ZE_MANAGED]   = "ze-managed",
    [UCS_MEMORY_TYPE_LAST]         = "unknown",
    [UCS_MEMORY_TYPE_LAST + 1]     = NULL
};

const char *ucs_memory_type_descs[] = {
    [UCS_MEMORY_TYPE_HOST]         = "System memory",
    [UCS_MEMORY_TYPE_CUDA]         = "NVIDIA GPU memory",
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = "NVIDIA GPU managed/unified memory",
    [UCS_MEMORY_TYPE_ROCM]         = "AMD/ROCm GPU memory",
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = "AMD/ROCm GPU managed memory",
    [UCS_MEMORY_TYPE_RDMA]         = "RDMA device memory",
    [UCS_MEMORY_TYPE_ZE_HOST]      = "Intel/Ze USM host memory",
    [UCS_MEMORY_TYPE_ZE_DEVICE]    = "Intel/Ze GPU memory",
    [UCS_MEMORY_TYPE_ZE_MANAGED]   = "Intel/Ze GPU managed memory",
    [UCS_MEMORY_TYPE_LAST]         = "unknown"
};
