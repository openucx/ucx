/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
*
* See file LICENSE for terms.
*/


#ifndef UCS_PPC64_GLOBAL_OPTS_H_
#define UCS_PPC64_GLOBAL_OPTS_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#define UCS_ARCH_GLOBAL_OPTS_INITALIZER {}

/* built-in memcpy config */
typedef struct ucs_arch_global_opts {
    char dummy;
} ucs_arch_global_opts_t;

END_C_DECLS

#endif

