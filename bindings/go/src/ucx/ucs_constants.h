/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */
#ifndef GO_UCS_CONSTANTS_H_
#define GO_UCS_CONSTANTS_H_

#include <ucs/memory/memory_type.h>
#include <sys/types.h>

const char* ucxgo_get_ucs_mem_type_name(ucs_memory_type_t idx);
ssize_t ucxgo_parse_ucs_mem_type_name(void* value);

#endif
