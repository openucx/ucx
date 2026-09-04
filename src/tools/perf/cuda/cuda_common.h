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

/* Same as CUDA_CALL* above, but for CUresult-returning CUDA driver API calls
 * (e.g. cuMemCreate, cuMemMap), as opposed to cudaError_t-returning CUDA
 * runtime API calls. */
#define CUDA_DRV_CALL(_handler, _log_level, _func, ...) \
    do { \
        CUresult _cerr = _func(__VA_ARGS__); \
        if (_cerr != CUDA_SUCCESS) { \
            const char *_name = "unknown", *_desc = "no description"; \
            cuGetErrorName(_cerr, &_name); \
            cuGetErrorString(_cerr, &_desc); \
            ucs_log(_log_level, "%s() failed: %s (%s)", \
                    UCS_PP_MAKE_STRING(_func), _name, _desc); \
            _handler; \
        } \
    } while (0)

#define CUDA_DRV_CALL_RET(_ret, _func, ...) \
    CUDA_DRV_CALL(return _ret, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CUDA_DRV_CALL_ERR(_func, ...) \
    CUDA_DRV_CALL(, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CUDA_DRV_CALL_WARN(_func, ...) \
    CUDA_DRV_CALL(, UCS_LOG_LEVEL_WARN, _func, __VA_ARGS__)

END_C_DECLS

#endif /* CUDA_COMMON_H_ */
