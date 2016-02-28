/**
 * Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "cuda_ep.h"
#include "cuda_iface.h"

#include <uct/base/uct_log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/type/class.h>


static UCS_CLASS_INIT_FUNC(uct_cuda_ep_t, uct_iface_t *tl_iface,
                           const uct_device_addr_t *dev_addr,
                           const uct_iface_addr_t *iface_addr)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ep_t)
{
}

UCS_CLASS_DEFINE(uct_cuda_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ep_t, uct_ep_t, uct_iface_t*,
                          const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ep_t, uct_ep_t);


ucs_status_t uct_cuda_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                   unsigned length, uint64_t remote_addr,
                                   uct_rkey_t rkey)
{
    /* Code for PUT here */
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_cuda_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}

