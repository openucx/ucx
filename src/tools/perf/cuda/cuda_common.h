/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_COMMON_H_
#define CUDA_COMMON_H_

#include <ucs/debug/log_def.h>

BEGIN_C_DECLS

/* TODO: move it to some common place */
#define CUDA_CALL_HANDLER(_handler, _ret, _func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            _handler("%s() failed: %d (%s)", UCS_PP_MAKE_STRING(_func), \
                     (int)_cerr, cudaGetErrorString(_cerr)); \
            return _ret; \
        } \
    } while (0)

#define CUDA_CALL(_ret, _func, ...) \
    CUDA_CALL_HANDLER(ucs_error, _ret, _func, __VA_ARGS__)

END_C_DECLS

#endif /* CUDA_COMMON_H_ */
