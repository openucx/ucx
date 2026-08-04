/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_UCX_CHECK_DEF_H
#define TEST_UCX_CHECK_DEF_H

#include "test_lib_check_def.h"

#include "ucs/type/status.h"

#define UCX_CHECK(_func) \
    LIB_CHECK(ucs_status_t, _func, UCS_OK, ucs_status_string)

#endif
