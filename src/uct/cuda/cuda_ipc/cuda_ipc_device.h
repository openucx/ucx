/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_EP_DEVICE_H
#define UCT_CUDA_IPC_EP_DEVICE_H

#include <stdint.h>

typedef struct {
    uintptr_t dst_bptr;
    void      *mapped_addr;
} uct_cuda_ipc_device_mem_element_t;

#endif
