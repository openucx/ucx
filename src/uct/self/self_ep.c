/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "self_ep.h"
#include "self_iface.h"

static UCS_CLASS_INIT_FUNC(uct_self_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_self_iface_t *local_iface = 0;

    ucs_trace_func("Creating an EP for loop-back transport self=%p", self);
    local_iface = ucs_derived_of(tl_iface, uct_self_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &local_iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_self_ep_t)
{
    ucs_trace_func("self=%p", self);
}

UCS_CLASS_DEFINE(uct_self_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_self_ep_t, uct_ep_t, uct_iface_t *,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_self_ep_t, uct_ep_t);

ucs_status_t uct_self_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    uct_base_iface_t *local_iface = 0;

    ucs_trace_data("TX: self AM [%d] buf=%p len=%u", id, payload, length);
    local_iface = ucs_derived_of(ep->iface, uct_base_iface_t);
    /* stub for testing */
    uct_iface_invoke_am(local_iface, id, (void *)payload, length, (void *)payload);
    return UCS_OK;
}
