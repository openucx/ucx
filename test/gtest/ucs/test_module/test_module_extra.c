/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/sys/compiler.h>

extern int ucs_test_module_extra_loaded;

UCS_STATIC_INIT {
    ++ucs_test_module_extra_loaded;
}
