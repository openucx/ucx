/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_IPC_MD_H
#define ROCM_IPC_MD_H

#include <uct/base/uct_md.h>
#include <ucs/sys/sys.h>
#include <hsa_ext_amd.h>
#include <pthread.h>


typedef struct uct_rocm_ipc_cache uct_rocm_ipc_cache_t;

typedef struct uct_rocm_ipc_component {
    uct_component_t super;
} uct_rocm_ipc_component_t;

extern uct_rocm_ipc_component_t uct_rocm_ipc_component;

typedef struct uct_rocm_ipc_md {
    struct uct_md super;
} uct_rocm_ipc_md_t;

typedef struct uct_rocm_ipc_md_config {
    uct_md_config_t super;
} uct_rocm_ipc_md_config_t;

typedef struct uct_rocm_ipc_key {
    hsa_amd_ipc_memory_t ipc;
    uintptr_t            address;
    size_t               length;
    int                  dev_num;
    pid_t                pid;    /* PID of the process that owns the memory */
    ucs_sys_ns_t         pid_ns; /* PID namespace of the owner process */
} uct_rocm_ipc_key_t;

#endif
