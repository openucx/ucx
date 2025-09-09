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
#define _CALL(_handler, _log_level, _err_type, _ok, _err_func, _func, ...) \
    do { \
        _err_type _cerr = _func(__VA_ARGS__); \
        if (_cerr != _ok) { \
            ucs_log(_log_level, "%s() failed: %d (%s)", \
                    UCS_PP_MAKE_STRING(_func), (int)_cerr, _err_func(_cerr)); \
            _handler; \
        } \
    } while (0)

#define CUDA_CALL(_handler, _log_level, _func, ...) \
    _CALL(_handler, _log_level, cudaError_t, cudaSuccess, cudaGetErrorString, _func, __VA_ARGS__)

#define CUDA_CALL_RET(_ret, _func, ...) \
    CUDA_CALL(return _ret, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CUDA_CALL_ERR(_func, ...) \
    CUDA_CALL(, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CUDA_CALL_WARN(_func, ...) \
    CUDA_CALL(, UCS_LOG_LEVEL_WARN, _func, __VA_ARGS__)

#define CU_ERR_STR(res) \
    ({ \
        const char* _str = NULL; \
        cuGetErrorString((res), &_str); \
        _str ? _str : "(unknown)"; \
    })

#define CU_CALL(_handler, _log_level, _func, ...) \
    _CALL(_handler, _log_level, CUresult, CUDA_SUCCESS, CU_ERR_STR, _func, __VA_ARGS__)

#define CU_CALL_RET(_ret, _func, ...) \
    CU_CALL(return _ret, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CU_CALL_ERR(_func, ...) \
    CU_CALL(, UCS_LOG_LEVEL_ERROR, _func, __VA_ARGS__)

#define CU_CALL_WARN(_func, ...) \
    CU_CALL(, UCS_LOG_LEVEL_WARN, _func, __VA_ARGS__)

END_C_DECLS

#endif /* CUDA_COMMON_H_ */
