/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_ADDRESS_H
#define UCT_CUDA_IPC_IFACE_ADDRESS_H

#include "ucs/sys/sys.h"
#include "uct/api/uct_def.h"

#include <sys/types.h>

typedef struct {
    pid_t        pid;
    ucs_sys_ns_t pid_ns;
} UCS_S_PACKED uct_cuda_ipc_iface_address_t;


/**
 * Pack the CUDA IPC interface address into the given interface address.
 */
void uct_cuda_ipc_iface_address_pack(uct_iface_addr_t *iface_addr);


/**
 * Get the length of the packed CUDA IPC interface address.
 */
size_t uct_cuda_ipc_iface_address_length(void);


/**
 * Unpack the PID from the given interface address.
 */
pid_t uct_cuda_ipc_iface_address_unpack_pid(const uct_iface_addr_t *iface_addr);


/**
 * Unpack the CUDA IPC interface address from the given interface address.
 */
uct_cuda_ipc_iface_address_t
uct_cuda_ipc_iface_address_unpack(const uct_iface_addr_t *iface_addr,
                                  size_t iface_addr_length);

#endif
