/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "uct_iface_address_pid_ns.h"

#include <unistd.h>

ucs_status_t
uct_iface_get_address_pid_ns(uct_iface_t *iface, uct_iface_addr_t *iface_addr,
                             unsigned long flags)
{
    uct_iface_address_pid_ns_t *iface_address_pid_ns;

    ucs_assert(!(getpid() & flags));

    iface_address_pid_ns           = (uct_iface_address_pid_ns_t*)iface_addr;
    iface_address_pid_ns->super.id = getpid();
    if (!ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID)) {
        iface_address_pid_ns->super.id |= flags;
        iface_address_pid_ns->pid_ns    = ucs_sys_get_ns(UCS_SYS_NS_TYPE_PID);
    }

    return UCS_OK;
}

size_t uct_iface_address_pid_ns_length(void)
{
    return ucs_sys_ns_is_default(UCS_SYS_NS_TYPE_PID) ?
                   sizeof(uct_iface_address_pid_t) :
                   sizeof(uct_iface_address_pid_ns_t);
}
