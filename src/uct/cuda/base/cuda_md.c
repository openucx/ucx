/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cuda_md.h"

#include <ucs/sys/module.h>
#include <cuda_runtime.h>
#include <cuda.h>


ucs_status_t uct_cuda_base_detect_memory_type(uct_md_h md, void *addr, size_t length,
                                              uct_memory_type_t *mem_type_p)
{
    CUmemorytype memType = 0;
    uint32_t isManaged = 0;
    void *attrdata[] = {(void *)&memType, (void *)&isManaged};
    CUpointer_attribute attributes[2] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                                         CU_POINTER_ATTRIBUTE_IS_MANAGED};
    CUresult cu_err;

    if (addr == NULL) {
        *mem_type_p = UCT_MD_MEM_TYPE_HOST;
        return UCS_OK;
    }

    cu_err = cuPointerGetAttributes(2, attributes, attrdata, (CUdeviceptr)addr);
    if ((cu_err == CUDA_SUCCESS) && (memType == CU_MEMORYTYPE_DEVICE)) {
        if (isManaged) {
            *mem_type_p = UCT_MD_MEM_TYPE_CUDA_MANAGED;
        } else {
            *mem_type_p = UCT_MD_MEM_TYPE_CUDA;
        }
        return UCS_OK;
    }

    return UCS_ERR_INVALID_ADDR;
}

UCS_MODULE_INIT() {
    /* TODO make gdrcopy independent of cuda */
    UCS_MODULE_FRAMEWORK_DECLARE(uct_cuda);
    UCS_MODULE_FRAMEWORK_LOAD(uct_cuda, 0);
    return UCS_OK;
}
