/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucs/sys/compiler.h>

extern int ucs_test_module_base_loaded;

int ucs_test_module_base_value(void);

int ucs_test_module_base_value(void)
{
    return 42;
}

UCS_STATIC_INIT {
    ++ucs_test_module_base_loaded;
}
