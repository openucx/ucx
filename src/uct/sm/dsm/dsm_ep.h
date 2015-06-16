/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_DSM_EP_H
#define UCT_DSM_EP_H

#include "dsm_iface.h"

#include <uct/tl/tl_log.h>


typedef struct uct_dsm_ep {
    uct_base_ep_t      super;
    struct uct_dsm_ep *next;
} uct_dsm_ep_t;
UCS_CLASS_DECLARE(uct_dsm_ep_t, uct_dsm_iface_t *)

UCS_CLASS_DECLARE_NEW_FUNC(uct_dsm_ep_t, uct_dsm_iface_t, uct_dsm_iface_t*);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_dsm_ep_t, uct_ep_t);


ucs_status_t uct_dsm_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                 unsigned length, uint64_t remote_addr, 
                                 uct_rkey_t rkey);
ucs_status_t uct_dsm_ep_put_bcopy(uct_ep_h ep, uct_pack_callback_t pack_cb,
                                 void *arg, size_t length, 
                                 uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_dsm_ep_put_zcopy(uct_ep_h tl_ep, const void *buffer,
                                 size_t length, uct_mem_h memh, 
                                 uint64_t remote_addr, uct_rkey_t rkey, 
                                 uct_completion_t *comp);
ucs_status_t uct_dsm_ep_am_short(uct_ep_h ep, uint8_t id, uint64_t header,
                                const void *payload, unsigned length);
ucs_status_t uct_dsm_ep_atomic_add64(uct_ep_h tl_ep, uint64_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_dsm_ep_atomic_fadd64(uct_ep_h tl_ep, uint64_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_completion_t *comp);
ucs_status_t uct_dsm_ep_atomic_swap64(uct_ep_h tl_ep, uint64_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_completion_t *comp);
ucs_status_t uct_dsm_ep_atomic_cswap64(uct_ep_h tl_ep, uint64_t compare,
                                      uint64_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_dsm_ep_atomic_add32(uct_ep_h tl_ep, uint32_t add,
                                    uint64_t remote_addr, uct_rkey_t rkey);
ucs_status_t uct_dsm_ep_atomic_fadd32(uct_ep_h tl_ep, uint32_t add,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_completion_t *comp);
ucs_status_t uct_dsm_ep_atomic_swap32(uct_ep_h tl_ep, uint32_t swap,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_completion_t *comp);
ucs_status_t uct_dsm_ep_atomic_cswap32(uct_ep_h tl_ep, uint32_t compare,
                                      uint32_t swap, uint64_t remote_addr, 
                                      uct_rkey_t rkey, uct_completion_t *comp);
ucs_status_t uct_dsm_ep_get_bcopy(uct_ep_h tl_ep, size_t length,
                                 uint64_t remote_addr, uct_rkey_t rkey, 
                                 uct_completion_t *comp);
ucs_status_t uct_dsm_ep_get_zcopy(uct_ep_h tl_ep, void *buffer, size_t length,
                                 uct_mem_h memh, uint64_t remote_addr,
                                 uct_rkey_t rkey, uct_completion_t *comp);
#endif
