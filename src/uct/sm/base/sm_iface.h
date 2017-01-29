/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SM_IFACE_H_
#define SM_IFACE_H_

#include <uct/api/uct.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>

#define UCT_SM_IFACE_DEVICE_ADDR_LEN    sizeof(uint64_t)
#define UCT_SM_MAX_IOV                  16

ucs_status_t uct_sm_iface_get_device_address(uct_iface_t *tl_iface,
                                             uct_device_addr_t *addr);

int uct_sm_iface_is_reachable(const uct_iface_h tl_iface, const uct_device_addr_t *dev_addr,
                              const uct_iface_addr_t *iface_addr);

ucs_status_t uct_sm_iface_fence(uct_iface_t *tl_iface, unsigned flags);

ucs_status_t uct_sm_ep_fence(uct_ep_t *tl_ep, unsigned flags);

static UCS_F_ALWAYS_INLINE size_t uct_sm_get_max_iov() {
    return ucs_min(UCT_SM_MAX_IOV, ucs_get_max_iov());
}


#endif
