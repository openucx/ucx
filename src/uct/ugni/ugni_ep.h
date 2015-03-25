/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_UGNI_EP_H
#define UCT_UGNI_EP_H

#include <gni_pub.h>
#include <uct/api/uct.h>
#include <uct/tl/tl_base.h>
#include <ucs/type/class.h>


typedef struct uct_ugni_ep_addr {
    uct_ep_addr_t     super;
    int               ep_id;
} uct_ugni_ep_addr_t;

typedef struct uct_ugni_ep {
    uct_base_ep_t     super;
    gni_ep_handle_t   ep;
    unsigned          outstanding;
    uintptr_t         hash_key;
    struct uct_ugni_ep *next;
} uct_ugni_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

ucs_status_t uct_ugni_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *ep_addr);
ucs_status_t uct_ugni_ep_connect_to_ep(uct_ep_h tl_ep, uct_iface_addr_t *tl_iface_addr,
                                       uct_ep_addr_t *tl_ep_addr);
ucs_status_t uct_ugni_ep_put_short(uct_ep_h tl_ep, void *buffer,
                                   unsigned length, uint64_t remote_addr,
                                   uct_rkey_t rkey);
ucs_status_t uct_ugni_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                   void *arg, size_t length, uint64_t remote_addr,
                                   uct_rkey_t rkey);
ucs_status_t uct_ugni_ep_put_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_ugni_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  void *payload, unsigned length);
ucs_status_t uct_ugni_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                      uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_ugni_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);
ucs_status_t uct_ugni_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);
ucs_status_t uct_ugni_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp);
ucs_status_t uct_ugni_ep_get_bcopy(uct_ep_h tl_ep, size_t length, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_ugni_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
#endif
