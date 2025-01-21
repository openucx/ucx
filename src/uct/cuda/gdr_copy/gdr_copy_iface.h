/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2017-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

#ifndef UCT_GDR_COPY_IFACE_H
#define UCT_GDR_COPY_IFACE_H

#include <uct/base/uct_iface.h>


typedef uint64_t uct_gdr_copy_iface_addr_t;


typedef struct uct_gdr_copy_iface {
    uct_base_iface_t          super;
    uct_gdr_copy_iface_addr_t id;
    struct {
        uct_ppn_bandwidth_t   get_bw;
        uct_ppn_bandwidth_t   put_bw;
        ucs_linear_func_t     get_latency;
        ucs_linear_func_t     put_latency;
    } config;
} uct_gdr_copy_iface_t;


typedef struct uct_gdr_copy_iface_config {
    uct_iface_config_t  super;
    uct_ppn_bandwidth_t get_bw;
    uct_ppn_bandwidth_t put_bw;
    double              get_latency;
    double              put_latency;
} uct_gdr_copy_iface_config_t;

#endif
