/**
* Copyright (C) Tactical Computing Labs, LLC. 2022. ALL RIGHTS RESERVED.
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#if defined(__riscv)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/global_opts.h>
#include <ucs/config/parser.h>

ucs_config_field_t ucs_arch_global_opts_table[] = {
    {NULL}
};

void ucs_arch_print_memcpy_limits(ucs_arch_global_opts_t *config)
{
}

#endif
