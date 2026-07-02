/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_CUDA_CHECK_DEF_H
#define TEST_CUDA_CHECK_DEF_H

#include "test_lib_check_def.h"

#include <cuda.h>

#define _CUDA_ERROR_STRING(_err) \
    ({ \
        const char *_err_str; \
        cuGetErrorString(_err, &_err_str); \
        _err_str; \
    })

#define CUDA_CHECK(_func) \
    LIB_CHECK(CUresult, _func, CUDA_SUCCESS, _CUDA_ERROR_STRING)

#endif
