/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_SM_EP_H
#define UCT_SM_EP_H

#include "uct/base/uct_iface.h"


ucs_status_t uct_sm_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr,
                                 uct_rkey_t rkey);
ssize_t uct_sm_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                            void *arg, uint64_t remote_addr, uct_rkey_t rkey);

ucs_status_t uct_sm_ep_get_bcopy(uct_ep_h ep, uct_unpack_callback_t unpack_cb,
                                 void *arg, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey,
                                 uct_completion_t *comp);

ucs_status_t uct_sm_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_sm_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp);
ucs_status_t uct_sm_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint64_t *result, uct_completion_t *comp);
ucs_status_t uct_sm_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint64_t *result,
                                      uct_completion_t *comp);

ucs_status_t uct_sm_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_sm_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp);
ucs_status_t uct_sm_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uint32_t *result, uct_completion_t *comp);
ucs_status_t uct_sm_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr,
                                      uct_rkey_t rkey, uint32_t *result,
                                      uct_completion_t *comp);

#endif
