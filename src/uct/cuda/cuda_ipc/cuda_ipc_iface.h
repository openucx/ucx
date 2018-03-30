/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_H
#define UCT_CUDA_IPC_IFACE_H

#include <uct/base/uct_iface.h>
#include <ucs/arch/cpu.h>
#include <cuda_runtime.h>
#include <cuda.h>


#define UCT_CUDA_IPC_TL_NAME    "cuda_ipc"
#define UCT_CUDA_DEV_NAME   "cudaipc0"

#define CUDA_FUNC(func)  ({                             \
ucs_status_t _status = UCS_OK;                          \
do {                                                    \
    CUresult _result = (func);                          \
    if (CUDA_SUCCESS != _result) {                      \
        ucs_error("[%s:%d] cuda failed with %d \n",     \
                  __FILE__, __LINE__,_result);          \
        _status = UCS_ERR_IO_ERROR;                     \
    }                                                   \
} while (0);                                            \
_status;                                                \
})


typedef struct uct_cuda_ipc_iface {
    uct_base_iface_t           super;
} uct_cuda_ipc_iface_t;


typedef struct uct_cuda_ipc_iface_config {
    uct_iface_config_t      super;
} uct_cuda_ipc_iface_config_t;


#endif
