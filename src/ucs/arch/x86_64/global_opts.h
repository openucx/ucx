/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_GLOBAL_OPTS_H_
#define UCS_X86_64_GLOBAL_OPTS_H_

#include <stddef.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#define UCS_ARCH_GLOBAL_OPTS_INITALIZER { \
    .builtin_memcpy_min     = UCS_MEMUNITS_AUTO, \
    .builtin_memcpy_max     = UCS_MEMUNITS_AUTO, \
    .nt_buffer_transfer_min = UCS_MEMUNITS_AUTO, \
    .nt_dest_threshold      = UCS_MEMUNITS_AUTO  \
}

/* built-in memcpy & nt-buffer-transfer config */
typedef struct ucs_arch_global_opts {
    size_t builtin_memcpy_min;
    size_t builtin_memcpy_max;
    size_t nt_buffer_transfer_min;
    size_t nt_dest_threshold;
} ucs_arch_global_opts_t;

END_C_DECLS

#endif
