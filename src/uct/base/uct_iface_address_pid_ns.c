/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "uct_iface_address_pid_ns.h"

#include <unistd.h>

typedef struct {
    pid_t pid;
} uct_iface_address_pid_t;

typedef struct {
    uct_iface_address_pid_t super;
    ucs_sys_ns_t            ns;
} uct_iface_address_pid_ns_t;

ucs_status_t
uct_iface_get_address_pid_ns(uct_iface_t*, uct_iface_addr_t *iface_addr)
{
    uct_iface_address_pid_ns_t *iface_address_pid_ns =
            (uct_iface_address_pid_ns_t*)iface_addr;

    ucs_assert(!(getpid() & UCT_IFACE_ADDRESS_PID_NS_FLAG));

    iface_address_pid_ns->super.pid = getpid();
    if (!ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID)) {
        iface_address_pid_ns->super.pid |= UCT_IFACE_ADDRESS_PID_NS_FLAG;
        iface_address_pid_ns->ns         = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);
    }

    return UCS_OK;
}

size_t uct_iface_address_pid_ns_length(void)
{
    return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID) ?
                   sizeof(uct_iface_address_pid_t) :
                   sizeof(uct_iface_address_pid_ns_t);
}

ucs_sys_ns_t uct_iface_address_pid_ns_get_ns(const uct_iface_addr_t *iface_addr)
{
    const uct_iface_address_pid_ns_t *iface_address_pid_ns =
            (const uct_iface_address_pid_ns_t*)iface_addr;

    if (iface_address_pid_ns->super.pid & UCT_IFACE_ADDRESS_PID_NS_FLAG) {
        return iface_address_pid_ns->ns;
    }

    return ucs_sys_get_default_ns(UCS_SYS_NS_TYPE_PID);
}
