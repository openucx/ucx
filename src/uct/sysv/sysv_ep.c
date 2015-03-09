/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/
#include <ucs/datastruct/sglib_wrapper.h>
#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/type/class.h>

#include "sysv_ep.h"
#include "sysv_iface.h"
#include "sysv_device.h"

static inline ptrdiff_t uct_sysv_ep_compare(uct_sysv_ep_t *ep1, 
                                            uct_sysv_ep_t *ep2)
{
    return ep1->hash_key - ep2->hash_key;
}

static inline unsigned uct_sysv_ep_hash(uct_sysv_ep_t *ep)
{
    return ep->hash_key;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_sysv_ep_t, uct_sysv_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_sysv_ep_t, UCT_SYSV_HASH_SIZE, 
                                         uct_sysv_ep_hash);
SGLIB_DEFINE_LIST_FUNCTIONS(uct_sysv_ep_t, uct_sysv_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_FUNCTIONS(uct_sysv_ep_t, UCT_SYSV_HASH_SIZE, 
                                        uct_sysv_ep_hash);

static UCS_CLASS_INIT_FUNC(uct_sysv_ep_t, uct_iface_t *tl_iface)
{
    uct_sysv_iface_t *iface = ucs_derived_of(tl_iface, uct_sysv_iface_t);

    UCS_CLASS_CALL_SUPER_INIT(tl_iface)

    /* FIXME create ep based from iface */

    /* add this ep to list of iface's eps */
    sglib_hashed_uct_sysv_ep_t_add(iface->eps, self);
    return UCS_OK;
}

static UCS_CLASS_CLEANUP_FUNC(uct_sysv_ep_t)
{
    uct_sysv_iface_t *iface = ucs_derived_of(self->super.iface, uct_sysv_iface_t);

    /* FIXME destroy ep */

    /* remove this ep from list of iface's eps */
    sglib_hashed_uct_sysv_ep_t_delete(iface->eps, self);
}
UCS_CLASS_DEFINE(uct_sysv_ep_t, uct_ep_t)
UCS_CLASS_DEFINE_NEW_FUNC(uct_sysv_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DEFINE_DELETE_FUNC(uct_sysv_ep_t, uct_ep_t);

uct_sysv_ep_t *uct_sysv_iface_lookup_ep(uct_sysv_iface_t *iface, 
                                        uintptr_t hash_key)
{
    uct_sysv_ep_t tmp;
    tmp.hash_key = hash_key;
    return sglib_hashed_uct_sysv_ep_t_find_member(iface->eps, &tmp);
}

ucs_status_t uct_sysv_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr)
{
    uct_sysv_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sysv_iface_t);
    ((uct_sysv_ep_addr_t*)ep_addr)->ep_id = iface->addr.nic_addr;
    return UCS_OK;
}

ucs_status_t uct_sysv_ep_connect_to_ep(uct_ep_h tl_ep, 
                                       uct_iface_addr_t *tl_iface_addr, 
                                       uct_ep_addr_t *tl_ep_addr)
{
    uct_sysv_ep_t *ep = ucs_derived_of(tl_ep, uct_sysv_ep_t);
    uct_sysv_iface_addr_t *iface_addr = ucs_derived_of(tl_iface_addr, 
                                                       uct_sysv_iface_addr_t);
    uct_sysv_ep_addr_t *ep_addr = ucs_derived_of(tl_ep_addr, uct_sysv_ep_addr_t);

    /* FIXME bind the ep using the iface addr */

    ucs_debug("Binding ep %p to address (%d %d)",
              ep, iface_addr->nic_addr,
              ep_addr->ep_id);
    return UCS_OK;
}

ucs_status_t uct_sysv_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                  unsigned length, uint64_t remote_addr,
                                  uct_rkey_t rkey)
{
    //uct_sysv_ep_t *ep = ucs_derived_of(tl_ep, uct_sysv_ep_t);
    //uct_sysv_iface_t *iface = ucs_derived_of(tl_ep->iface, uct_sysv_iface_t);
    uintptr_t *mem_hndl = (void *)rkey;
    ptrdiff_t remote_offset;

    /* FIXME add debug mode to check remote_addr within attached region */

    /* debug
    printf("put_short mem_hndl[1]x = %" PRIxPTR "\n", mem_hndl[2]);
    printf("put_short mem_hndl[1]p = %p\n", (void *)  mem_hndl[2]);
    printf("put_short mem_hndl[2]x = %" PRIxPTR "\n", mem_hndl[2]);
    printf("put_short mem_hndl[2]p = %p\n", (void *)  mem_hndl[2]);
    */

    remote_offset = (void *)mem_hndl[1] - (void *)remote_addr;
    memcpy((void *)(mem_hndl[2] + remote_offset), buffer, length);

    ucs_trace_data("Posting PUT Short, memcpy of size %u from %p to %p",
            length,
            (void *)(mem_hndl[1]+remote_addr),
            (void *)mem_hndl[2]);

    return UCS_OK;
}

ucs_status_t uct_sysv_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                            void *payload, unsigned length)
{
    return UCS_ERR_UNSUPPORTED;
}

