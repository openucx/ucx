/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_COPY_IFACE_H
#define UCT_MUSA_COPY_IFACE_H

#include <uct/base/uct_iface.h>

#define UCT_MUSA_COPY_TL_NAME "musa_copy"

typedef uint64_t uct_musa_copy_iface_addr_t;

typedef struct {
    uct_base_iface_t           super;
    uct_musa_copy_iface_addr_t id;
    int                        device;
    struct {
        double                 latency;
    } config;
} uct_musa_copy_iface_t;

typedef struct {
    uct_iface_config_t         super;
    double                     latency;
} uct_musa_copy_iface_config_t;

#endif
