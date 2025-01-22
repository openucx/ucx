/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2001-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#ifndef UD_IFACE_COMMON_H
#define UD_IFACE_COMMON_H

#include <ucs/config/parser.h>


/**
 * Common configuration for IB non peer-to-peer transports (UD and DC).
 */
typedef struct uct_ud_iface_common_config {
    unsigned      rx_queue_len_init;
} uct_ud_iface_common_config_t;


extern ucs_config_field_t uct_ud_iface_common_config_table[];


#endif
