/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_SELF_EP_H
#define UCT_SELF_EP_H

#include <uct/base/uct_iface.h>


typedef struct uct_self_ep {
    uct_base_ep_t super;
} uct_self_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_self_ep_t, uct_ep_t, uct_iface_t *,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_self_ep_t, uct_ep_t);

ucs_status_t uct_self_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length);
ssize_t uct_self_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                             uct_pack_callback_t pack_cb, void *arg,
                             unsigned flags);

ucs_status_t uct_self_ep_check(uct_ep_h ep, unsigned flags, uct_completion_t *comp);

#endif
