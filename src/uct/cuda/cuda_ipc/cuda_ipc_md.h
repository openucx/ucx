/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_MD_H
#define UCT_CUDA_IPC_MD_H

#include <uct/base/uct_md.h>

#define UCT_CUDA_IPC_MD_NAME           "cuda_ipc"

extern uct_md_component_t uct_cuda_ipc_md_component;

/**
 * @brief cuda ipc MD descriptor
 */
typedef struct uct_cuda_ipc_md {
    struct uct_md super;   /**< Domain info */
} uct_cuda_ipc_md_t;

/**
 * @brief cuda ipc domain configuration.
 */
typedef struct uct_cuda_ipc_md_config {
    uct_md_config_t super;
} uct_cuda_ipc_md_config_t;

#endif
