/**
 * Copyright (c) UT-Battelle, LLC. 2014-2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_UGNI_EP_H
#define UCT_UGNI_EP_H

#include "ugni_def.h"
#include "ugni_types.h"
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/sglib_wrapper.h>

static inline int32_t uct_ugni_ep_compare(uct_ugni_ep_t *ep1, uct_ugni_ep_t *ep2)
{
    return ep1->hash_key - ep2->hash_key;
}

static inline unsigned uct_ugni_ep_hash(uct_ugni_ep_t *ep)
{
    return ep->hash_key;
}

SGLIB_DEFINE_LIST_PROTOTYPES(uct_ugni_ep_t, uct_ugni_ep_compare, next);
SGLIB_DEFINE_HASHED_CONTAINER_PROTOTYPES(uct_ugni_ep_t, UCT_UGNI_HASH_SIZE, uct_ugni_ep_hash);

UCS_CLASS_DECLARE(uct_ugni_ep_t, uct_iface_t*, const uct_device_addr_t*,
                  const uct_iface_addr_t*);
UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*,
                           const uct_device_addr_t*, const uct_iface_addr_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

uct_ugni_ep_t *uct_ugni_iface_lookup_ep(uct_ugni_iface_t *iface, uintptr_t hash_key);
ucs_status_t ugni_connect_ep(uct_ugni_iface_t *iface, const uct_devaddr_ugni_t *dev_addr,
                             const uct_sockaddr_ugni_t *iface_addr, uct_ugni_ep_t *ep);
ucs_status_t uct_ugni_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n);
void uct_ugni_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                               void *arg);
ucs_arbiter_cb_result_t uct_ugni_ep_process_pending(ucs_arbiter_t *arbiter,
                                                    ucs_arbiter_elem_t *elem,
                                                    void *arg);
ucs_arbiter_cb_result_t uct_ugni_ep_abriter_purge_cb(ucs_arbiter_t *arbiter,
                                                     ucs_arbiter_elem_t *elem,
                                                     void *arg);

static inline void uct_ugni_ep_check_flush(uct_ugni_ep_t *ep)
{
    if (!ep->outstanding && ep->flush_flag && !ep->arb_flush) {
        ep->flush_flag = 0;
    }
}

static inline int uct_ugni_can_send(uct_ugni_ep_t *ep)
{

    return (((ep->arb_size > 0) || ep->flush_flag) && !ep->arb_sched) ? 0 : 1;
}

static inline int uct_ugni_can_flush(uct_ugni_ep_t *ep)
{
    return ((ep->arb_flush <= 0) || ep->arb_sched) && !ep->outstanding ? 1 : 0;
}

#endif
