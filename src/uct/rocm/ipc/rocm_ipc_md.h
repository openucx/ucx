/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_IPC_MD_H
#define ROCM_IPC_MD_H

#include <uct/base/uct_md.h>
#include <hsa_ext_amd.h>

#define UCT_ROCM_IPC_MD_NAME    "rocm_ipc"

extern uct_md_component_t uct_rocm_ipc_md_component;

typedef struct uct_rocm_ipc_md {
    struct uct_md super;
} uct_rocm_ipc_md_t;

typedef struct uct_rocm_ipc_md_config {
    uct_md_config_t super;
} uct_rocm_ipc_md_config_t;

typedef struct uct_rocm_ipc_key {
    hsa_amd_ipc_memory_t ipc;
    int ipc_valid;
    uintptr_t address;
    uintptr_t lock_address;
    size_t length;
    int dev_num;
} uct_rocm_ipc_key_t;

#endif
