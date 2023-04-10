/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019-2023. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */


#ifndef ROCM_IPC_IFACE_H
#define ROCM_IPC_IFACE_H

#include <uct/base/uct_iface.h>

#include <hsa.h>

#define UCT_ROCM_IPC_TL_NAME "rocm_ipc"

typedef struct uct_rocm_ipc_iface {
    uct_base_iface_t super;
    ucs_mpool_t signal_pool;
    ucs_queue_head_t signal_queue;
} uct_rocm_ipc_iface_t;

typedef struct uct_rocm_ipc_iface_config {
    uct_iface_config_t super;
} uct_rocm_ipc_iface_config_t;

#endif
