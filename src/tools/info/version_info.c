/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucx_info.h"

#include <ucs/sys/lib.h>
#include <ucs/sys/string.h>

void print_version()
{
    const char *path = ucs_sys_get_lib_path();

    printf("# Library version: %s\n", ucp_get_version_string());
    printf("# Library path: %s\n", (path != NULL) ? path : "<unknown>");
    printf("# API headers version: %s\n", UCT_VERNO_STRING);
    printf("# Git branch '%s', revision %s\n", UCT_SCM_BRANCH, UCT_SCM_VERSION);
    printf("# Configured with: %s\n", UCX_CONFIGURE_FLAGS);
}
