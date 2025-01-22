/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucx_info.h"

#include <ucs/sys/preprocessor.h>


void print_build_config()
{
    typedef struct {
        const char *name;
        const char *value;
    } config_var_t;
    static config_var_t config_vars[] = {
        #include <build_config.h>
        {NULL, NULL}
    };
    config_var_t *var;

    for (var = config_vars; var->name != NULL; ++var) {
        printf("#define %-25s %s\n", var->name, var->value);
    }
}
