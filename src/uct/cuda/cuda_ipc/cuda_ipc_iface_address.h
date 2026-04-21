/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_IFACE_ADDRESS_H
#define UCT_CUDA_IPC_IFACE_ADDRESS_H

#include "ucs/sys/sys.h"
#include "uct/api/uct_def.h"

#include <sys/types.h>

/**
 * Pack CUDA IPC interface address to the given interface address buffer.
 */
void uct_cuda_ipc_iface_address_pack(uct_iface_addr_t *iface_addr);


/**
 * Get the length of the packed CUDA IPC interface address.
 */
size_t uct_cuda_ipc_iface_address_length(void);


/**
 * Unpack the PID from the CUDA IPC interface address.
 */
pid_t uct_cuda_ipc_iface_address_pid(const uct_iface_addr_t *iface_addr);


/**
 * Unpack the PID namespace from the CUDA IPC interface address.
 */
ucs_sys_ns_t
uct_cuda_ipc_iface_address_pid_ns(const uct_iface_addr_t *iface_addr,
                                  size_t iface_addr_length);

#endif
