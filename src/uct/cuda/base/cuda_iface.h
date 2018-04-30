/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IFACE_H
#define UCT_CUDA_IFACE_H

#include <ucs/sys/preprocessor.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define UCT_CUDA_FUNC(_func)                                    \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            cudaError_t _result = (_func);                      \
            if (cudaSuccess != _result) {                       \
                ucs_error("%s is failed. ret:%s",               \
                          UCS_PP_MAKE_STRING(_func),            \
                          cudaGetErrorString(_result));         \
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })


#define UCT_CUDADRV_FUNC(_func)                                 \
    ({                                                          \
        ucs_status_t _status = UCS_OK;                          \
        do {                                                    \
            CUresult _result = (_func);                         \
            const char *cu_err_str;                             \
            if (CUDA_ERROR_NOT_READY == _result) {              \
                _status = UCS_INPROGRESS;                       \
            }                                                   \
            else if (CUDA_SUCCESS != _result) {                 \
                cuGetErrorString(_result, &cu_err_str);         \
                ucs_error("%s is failed. ret:%s",               \
                          UCS_PP_MAKE_STRING(_func),cu_err_str);\
                _status = UCS_ERR_IO_ERROR;                     \
            }                                                   \
        } while (0);                                            \
        _status;                                                \
    })

#endif
