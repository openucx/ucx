/**
* Copyright (C) UT-Battelle, LLC. 2015-2017. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_UGNI_UDT_EP_H
#define UCT_UGNI_UDT_EP_H

#include <uct/ugni/base/ugni_types.h>
#include <uct/ugni/base/ugni_ep.h>
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>

#define UCT_UGNI_UDT_ANY    0
#define UCT_UGNI_UDT_CANCEL 1

struct uct_ugni_udt_desc;

typedef struct uct_ugni_udt_ep {
    uct_ugni_ep_t super;
    struct uct_ugni_udt_desc *posted_desc;
    uct_ugni_flush_group_t *desc_flush_group;
} uct_ugni_udt_ep_t;

ucs_status_t uct_ugni_udt_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                      const void *payload, unsigned length);
ssize_t uct_ugni_udt_ep_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                 uct_pack_callback_t pack_cb, void *arg,
                                 unsigned flags);
ucs_status_t uct_ugni_udt_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n);
ucs_arbiter_cb_result_t uct_ugni_udt_ep_process_pending(ucs_arbiter_t *arbiter,
                                                        ucs_arbiter_elem_t *elem,
                                                        void *arg);
void uct_ugni_udt_ep_pending_purge(uct_ep_h tl_ep,
                                   uct_pending_purge_callback_t cb,
                                   void *arg);
UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_udt_ep_t, uct_ep_t, uct_iface_t*,
                           const uct_device_addr_t *, const uct_iface_addr_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_udt_ep_t, uct_ep_t);

#endif
