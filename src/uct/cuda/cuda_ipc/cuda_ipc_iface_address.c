/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "cuda_ipc_iface_address.h"

#include "ucs/sys/compiler_def.h"
#include "ucs/sys/sys.h"

#include <unistd.h>

static ucs_sys_ns_t
uct_cuda_ipc_iface_address_unpack_pid_ns(const uct_iface_addr_t *iface_addr,
                                         size_t iface_addr_length)
{
    const uct_cuda_ipc_iface_address_t *cuda_ipc_iface_address;

    if (iface_addr_length != sizeof(uct_cuda_ipc_iface_address_t)) {
        return ucs_sys_get_default_ns(UCS_SYS_NS_TYPE_PID);
    }

    cuda_ipc_iface_address = (const uct_cuda_ipc_iface_address_t*)iface_addr;
    return cuda_ipc_iface_address->pid_ns;
}

void uct_cuda_ipc_iface_address_pack(uct_iface_addr_t *iface_addr)
{
    uct_cuda_ipc_iface_address_t *cuda_ipc_iface_address;

    *(pid_t*)iface_addr = getpid();
    if (ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID)) {
        return;
    }

    cuda_ipc_iface_address         = (uct_cuda_ipc_iface_address_t*)iface_addr;
    cuda_ipc_iface_address->pid_ns = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);
}

size_t uct_cuda_ipc_iface_address_length(void)
{
    return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID) ?
                   sizeof(pid_t) :
                   sizeof(uct_cuda_ipc_iface_address_t);
}

pid_t uct_cuda_ipc_iface_address_unpack_pid(const uct_iface_addr_t *iface_addr)
{
    return *(const pid_t*)iface_addr;
}

uct_cuda_ipc_iface_address_t
uct_cuda_ipc_iface_address_unpack(const uct_iface_addr_t *iface_addr,
                                  size_t iface_addr_length)
{
    return (uct_cuda_ipc_iface_address_t){
        .pid = uct_cuda_ipc_iface_address_unpack_pid(iface_addr),
        .pid_ns = uct_cuda_ipc_iface_address_unpack_pid_ns(iface_addr,
                                                           iface_addr_length)
    };
}
