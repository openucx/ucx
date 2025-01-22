/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_MALLOC_HOOK_H_
#define UCM_MALLOC_HOOK_H_

#include <ucs/type/status.h>

ucs_status_t ucm_malloc_install(int events);
void ucm_init_malloc_hook();
void ucm_malloc_state_reset(int default_mmap_thresh, int default_trim_thresh);

#endif
