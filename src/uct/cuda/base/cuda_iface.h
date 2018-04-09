/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <ucs/sys/preprocessor.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define UCT_CUDA_FUNC(_func)  ({                        \
ucs_status_t _status = UCS_OK;                          \
do {                                                    \
    cudaError_t _result = (_func);                      \
    if (cudaSuccess != _result) {                       \
        ucs_error("%s failed with %d \n",               \
                  UCS_PP_MAKE_STRING(_func), _result);  \
        _status = UCS_ERR_IO_ERROR;                     \
    }                                                   \
} while (0);                                            \
_status;                                                \
})


#define UCT_CUDADRV_FUNC(_func)  ({                     \
ucs_status_t _status = UCS_OK;                          \
do {                                                    \
    CUresult _result = (_func);                         \
    if (CUDA_SUCCESS != _result) {                      \
        ucs_error("%s failed with %d \n",               \
                  UCS_PP_MAKE_STRING(_func), _result);  \
        _status = UCS_ERR_IO_ERROR;                     \
    }                                                   \
} while (0);                                            \
_status;                                                \
})

#endif
