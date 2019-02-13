/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cuda_md.h"

#include <ucs/sys/module.h>
#include <cuda_runtime.h>
#include <cuda.h>


int uct_cuda_is_mem_type_owned(uct_md_h md, void *addr, size_t length)
{
    CUmemorytype memType = 0;
    uint32_t isManaged = 0;
    void *attrdata[] = {(void *)&memType, (void *)&isManaged};
    CUpointer_attribute attributes[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         CU_POINTER_ATTRIBUTE_IS_MANAGED};
    CUresult cu_err;

    if (addr == NULL) {
        return 0;
    }

    cu_err = cuPointerGetAttributes(2, attributes, attrdata, (CUdeviceptr)addr);
    return ((cu_err == CUDA_SUCCESS) && (!isManaged && (memType == CU_MEMORYTYPE_DEVICE)));
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda);
    return UCS_OK;
}
