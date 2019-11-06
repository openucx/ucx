/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "memory_type.h"

#include <stddef.h>


const char *ucs_memory_type_names[] = {
    [UCS_MEMORY_TYPE_HOST]         = "host",
    [UCS_MEMORY_TYPE_CUDA]         = "cuda" ,
    [UCS_MEMORY_TYPE_CUDA_MANAGED] = "cuda-managed",
    [UCS_MEMORY_TYPE_ROCM]         = "rocm",
    [UCS_MEMORY_TYPE_ROCM_MANAGED] = "rocm-managed",
    [UCS_MEMORY_TYPE_LAST]         = "unknown"
};

