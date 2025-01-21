/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */
#include "ucs_constants.h"

#include <ucs/sys/string.h>
#include <stdlib.h>

const char* ucxgo_get_ucs_mem_type_name(ucs_memory_type_t idx) {
    return ucs_memory_type_names[idx];
}

ssize_t ucxgo_parse_ucs_mem_type_name(void* value) {
    ssize_t idx = ucs_string_find_in_list((const char *)value, ucs_memory_type_names, 0);
    free(value);
    return idx;
}
