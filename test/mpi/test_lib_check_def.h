/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_LIB_CHECK_DEF_H
#define TEST_LIB_CHECK_DEF_H

#include "ucs/sys/preprocessor.h"

#include <stdio.h>
#include <stdlib.h>

#define LIB_CHECK(_err_t, _func, _success_code, _lib_error_string) \
    do { \
        _err_t _err = (_func); \
        if (_err != _success_code) { \
            fprintf(stderr, "%s failed: %d (%s)\n", UCS_PP_MAKE_STRING(_func), \
                    _err, _lib_error_string(_err)); \
            exit(_err); \
        } \
    } while (0)

#endif
