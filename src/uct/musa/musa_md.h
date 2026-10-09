/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_MD_H
#define UCT_MUSA_MD_H

#include <uct/base/uct_md.h>
#include <uct/musa/musa_utils.h>

extern uct_component_t uct_musa_component;

typedef struct {
    uct_md_t              super;
    uct_musa_reg_mode_t   mode;
    int                   have_dmabuf;
} uct_musa_md_t;

typedef struct {
    void                 *address;
    void                 *base_address;
    size_t                length;
    size_t                alloc_length;
    int                   device;
} uct_musa_mem_t;

typedef struct {
    uct_md_config_t       super;
    uct_musa_reg_mode_t   mode;
} uct_musa_md_config_t;

ucs_status_t uct_musa_md_mem_query(uct_md_h md, const void *address,
                                    size_t length,
                                    uct_md_mem_attr_v2_t *mem_attr);

#endif
