/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
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
    [UCS_MEMORY_TYPE_CUDA]         = "cuda" ,
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = "cuda-managed",
    [UCS_MEMORY_TYPE_ROCM]         = "rocm",
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = "rocm-managed",
    [UCS_MEMORY_TYPE_LAST]         = "unknown",
    [UCS_MEMORY_TYPE_LAST + 1]     = NULL
};

const char *ucs_memory_type_descs[] = {
    [UCS_MEMORY_TYPE_HOST]         = "System memory",
    [UCS_MEMORY_TYPE_CUDA]         = "NVIDIA GPU memory" ,
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = "NVIDIA GPU managed/unified memory",
    [UCS_MEMORY_TYPE_ROCM]         = "AMD/ROCm GPU memory",
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = "AMD/ROCm GPU managed memory",
    [UCS_MEMORY_TYPE_LAST]         = "unknown"
};

