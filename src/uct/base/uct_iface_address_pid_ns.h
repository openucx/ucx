/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_IFACE_ADDRESS_PID_NS_H
#define UCT_IFACE_ADDRESS_PID_NS_H

#include "ucs/sys/sys.h"
#include "uct/api/uct_def.h"
#include "uct/api/uct.h"

#include <sys/types.h>

#define UCT_IFACE_ADDRESS_PID_NS_FLAG UCS_BIT(31) /* use PID NS in address */


/**
  * Pack PID and PID namespace into the interface address.
  */
ucs_status_t
uct_iface_get_address_pid_ns(uct_iface_t *iface, uct_iface_addr_t *iface_addr);


/**
  * Get the length of the interface address packed with PID and PID namespace.
  */
size_t uct_iface_address_pid_ns_length(void);


/**
  * Get the PID namespace from the interface address.
  */
ucs_sys_ns_t
uct_iface_address_pid_ns_get_ns(const uct_iface_addr_t *iface_addr);


/**
  * Get the PID from the interface address.
  */
static UCS_F_ALWAYS_INLINE pid_t
uct_iface_address_pid_ns_get_pid(const uct_iface_addr_t *iface_addr)
{
    return *(const pid_t*)iface_addr & ~UCT_IFACE_ADDRESS_PID_NS_FLAG;
}

#endif
