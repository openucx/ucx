/*
 * Copyright (C) Intel Corporation, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GAUDI_GDR_MD_H
#define UCT_GAUDI_GDR_MD_H

#include <stdbool.h>
#include <uct/base/uct_md.h>
#include <ucs/config/types.h>

extern uct_component_t uct_gaudi_gdr_component;

typedef struct uct_gaudi_md {
    uct_md_t         super;
    int              fd;
    bool             fd_created;
    uint64_t         device_base_allocated_address;
    uint64_t         device_base_address;
    uint64_t         totalSize;
    int              dmabuf_fd;
    ucs_sys_device_t sys_dev;
} uct_gaudi_md_t;

typedef struct uct_gaudi_md_config {
    uct_md_config_t super;
    int             device_id;
} uct_gaudi_md_config_t;

#endif
