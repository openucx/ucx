/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SM_IFACE_H_
#define SM_IFACE_H_

#include <uct/api/uct.h>


#define UCT_SM_IFACE_DEVICE_ADDR_LEN    sizeof(uint64_t)


ucs_status_t uct_sm_iface_get_device_address(uct_iface_t *tl_iface,
                                             uct_device_addr_t *addr);

#endif
