/*
 * Copyright (C) Advanced Micro Devices, Inc. 2023. ALL RIGHTS RESERVED.
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ucs/arch/cpu.h>
#include <ucs/type/class.h>
#include <ucs/sys/string.h>


typedef struct uct_rocm_base_signal_desc {
    hsa_signal_t     signal;
    void             *mapped_addr;
    uct_completion_t *comp;
    ucs_queue_elem_t queue;
} uct_rocm_base_signal_desc_t;

extern ucs_mpool_ops_t uct_rocm_base_signal_desc_mpool_ops;

unsigned uct_rocm_base_progress(ucs_queue_head_t *signal_queue);
