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
#define CUDA_CALL(_handler, _log_level, _func, ...) \
    do { \
        cudaError_t _cerr = _func(__VA_ARGS__); \
        if (_cerr != cudaSuccess) { \
            ucs_log(_log_level, "%s() failed: %d (%s)", UCS_PP_MAKE_STRING(_func), \
                    (int)_cerr, cudaGetErrorString(_cerr)); \
            _handler; \
        } \
    } while (0)

#define CUDA_CALL_RET(_ret, _func, ...) \
    CUDA_CALL(return _ret, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CUDA_CALL_ERR(_func, ...) \
    CUDA_CALL(, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CUDA_CALL_WARN(_func, ...) \
    CUDA_CALL(, UCS_LOG_LEVEL_WARN, _func, __VA_ARGS__)

END_C_DECLS

#endif /* CUDA_COMMON_H_ */
