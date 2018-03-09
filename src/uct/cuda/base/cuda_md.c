/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <uct/cuda/base/cuda_md.h>

#include <cuda_runtime.h>
#include <cuda.h>

int uct_cuda_is_mem_type_owned(uct_md_h md, void *addr, size_t length)
{
    int memory_type;
    CUresult cu_err;

    if (addr == NULL) {
        return 0;
    }

    cu_err = cuPointerGetAttribute(&memory_type,
                                   CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                   (CUdeviceptr)addr);
    return ((cu_err == CUDA_SUCCESS) && (memory_type == CU_MEMORYTYPE_DEVICE));
}
