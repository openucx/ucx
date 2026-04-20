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

typedef struct {
    pid_t id;
} uct_iface_address_pid_t;


typedef struct {
    uct_iface_address_pid_t super;
    ucs_sys_ns_t            pid_ns;
} uct_iface_address_pid_ns_t;


/**
  * Pack PID and PID namespace into the interface address.
  */
ucs_status_t uct_iface_get_address_pid_ns(uct_iface_t *iface,
                                          uct_iface_addr_t *iface_addr,
                                          unsigned long flags);


/**
  * Get the length of the interface address packed with PID and PID namespace.
  */
size_t uct_iface_address_pid_ns_length(void);

#endif
