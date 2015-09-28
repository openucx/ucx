/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_UGNI_RDMA_EP_H
#define UCT_UGNI_RDMA_EP_H

#include <gni_pub.h>
#include <uct/api/uct.h>
#include <uct/tl/tl_base.h>
#include <ucs/type/class.h>
#include <uct/ugni/base/ugni_ep.h>

UCS_CLASS_DECLARE_NEW_FUNC(uct_ugni_ep_t, uct_ep_t, uct_iface_t*, const struct sockaddr*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ugni_ep_t, uct_ep_t);

ucs_status_t uct_ugni_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                   unsigned length, uint64_t remote_addr,
                                   uct_rkey_t rkey);
ssize_t uct_ugni_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                              void *arg, uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_ugni_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_ugni_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                  const void *payload, unsigned length);
ucs_status_t uct_ugni_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                      uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_ugni_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uint64_t *result, uct_completion_t *comp);
ucs_status_t uct_ugni_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uint64_t *result, uct_completion_t *comp);
ucs_status_t uct_ugni_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare, uint64_t swap,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uint64_t *result, uct_completion_t *comp);
ucs_status_t uct_ugni_ep_get_bcopy(uct_ep_h tl_ep,
                                   uct_unpack_callback_t unpack_cb,
                                   void *arg, size_t length,
                                   uint64_t remote_addr, uct_rkey_t rkey,
                                   uct_completion_t *comp);
ucs_status_t uct_ugni_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                   uct_mem_h memh, uint64_t remote_addr,
                                   uct_rkey_t rkey, uct_completion_t *comp);
#endif
