/**
 * Copyright (C) UT-Battelle, LLC. 2022. ALL RIGHTS
 */

#ifndef UCT_OFI_EP_H
#define UCT_OFI_EP_H

#include "ofi_types.h"
#include "ofi_ep.h"
#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>

UCS_CLASS_DECLARE_NEW_FUNC(uct_ofi_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ofi_ep_t, uct_ep_t);

ucs_status_t uct_ofi_ep_am_short(uct_ep_h tl_ep, uint8_t id, uint64_t header,
                                       const void *payload, unsigned length);
ssize_t uct_ofi_am_bcopy(uct_ep_h tl_ep, uint8_t id,
                                  uct_pack_callback_t pack_cb, void *arg,
                                  unsigned flags);
ucs_status_t uct_ofi_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);

ucs_status_t uct_ofi_ep_pending_add(uct_ep_h tl_ep, uct_pending_req_t *n,
                                     unsigned flags);
void uct_ofi_ep_pending_purge(uct_ep_h tl_ep, uct_pending_purge_callback_t cb,
                               void *arg);
ucs_arbiter_cb_result_t uct_ofi_ep_process_pending(ucs_arbiter_t *arbiter,
                                                    ucs_arbiter_group_t *group,
                                                    ucs_arbiter_elem_t *elem,
                                                    void *arg);
ucs_arbiter_cb_result_t uct_ofi_ep_arbiter_purge_cb(ucs_arbiter_t *arbiter,
                                                     ucs_arbiter_group_t *group,
                                                     ucs_arbiter_elem_t *elem,
                                                     void *arg);
ucs_status_t uct_ofi_ep_flush(uct_ep_h tl_ep, unsigned flags,
                               uct_completion_t *comp);

#endif
