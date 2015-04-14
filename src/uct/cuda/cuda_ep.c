/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>
#include <uct/tl/tl_log.h>

#include "cuda_ep.h"
#include "cuda_iface.h"

unsigned cuda_ep_global_counter = 0;

static UCS_CLASS_INIT_FUNC(uct_cuda_ep_t, uct_iface_t *tl_iface)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_iface, uct_cuda_iface_t);
    UCS_CLASS_CALL_SUPER_INIT(uct_base_ep_t, &iface->super)
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_cuda_ep_t)
{
    /* No op */
}
UCS_CLASS_DEFINE(uct_cuda_ep_t, uct_base_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_cuda_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_cuda_ep_t, uct_ep_t);

ucs_status_t uct_cuda_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_iface_t);
    ((uct_cuda_ep_addr_t*)ep_addr)->ep_id = iface->addr.nic_addr;
    return UCS_OK;
}

ucs_status_t uct_cuda_ep_connect_to_ep(uct_ep_h tl_ep, 
                                       uct_iface_addr_t *tl_iface_addr, 
                                       uct_ep_addr_t *tl_ep_addr)
{
    /* No op */
    return UCS_OK;
}

ucs_status_t uct_cuda_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                   unsigned length, uint64_t remote_addr,
                                   uct_rkey_t rkey)
{
    uct_cuda_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_cuda_iface_t); 

    if (0 == length) {
        ucs_trace_data("Zero length request: skip it");
        return UCS_OK;
    }
    /* FIXME make this user-configurable */
    UCT_CHECK_LENGTH(length <= iface->config.max_put, "put_short");

    /* Code for PUT here */

    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t uct_cuda_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}

