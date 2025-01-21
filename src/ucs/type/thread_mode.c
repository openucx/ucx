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

#include "thread_mode.h"


const char *ucs_thread_mode_names[] = {
    [UCS_THREAD_MODE_SINGLE]     = "single",
    [UCS_THREAD_MODE_SERIALIZED] = "serialized",
    [UCS_THREAD_MODE_MULTI]      = "multi"
};
