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

int uct_sm_iface_is_reachable(uct_iface_t *tl_iface,
                              const uct_device_addr_t *addr);

ucs_status_t uct_sm_iface_fence(uct_iface_t *tl_iface, unsigned flags);

ucs_status_t uct_sm_ep_fence(uct_ep_t *tl_ep, unsigned flags);

#endif
