/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_EP_DEVICE_H
#define UCT_CUDA_IPC_EP_DEVICE_H

#include <stddef.h>

typedef struct {
    ptrdiff_t mapped_offset;
} uct_cuda_ipc_device_mem_element_t;

#endif
