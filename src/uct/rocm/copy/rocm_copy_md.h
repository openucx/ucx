/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_COPY_MD_H
#define UCT_ROCM_COPY_MD_H

#include <uct/base/uct_md.h>


extern uct_component_t uct_rocm_copy_component;

typedef struct uct_rocm_copy_md {
    struct uct_md super;
} uct_rocm_copy_md_t;

typedef struct uct_rocm_copy_md_config {
    uct_md_config_t super;
} uct_rocm_copy_md_config_t;

#endif
