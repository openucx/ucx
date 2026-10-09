/*
 * Copyright (C) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_MUSA_COPY_EP_H
#define UCT_MUSA_COPY_EP_H

#include <uct/base/uct_iface.h>

typedef struct {
    uct_base_ep_t super;
} uct_musa_copy_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_musa_copy_ep_t, uct_ep_t,
                           const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_musa_copy_ep_t, uct_ep_t);

ucs_status_t uct_musa_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                         unsigned length, uint64_t remote_addr,
                                         uct_rkey_t rkey);

ucs_status_t uct_musa_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                         unsigned length, uint64_t remote_addr,
                                         uct_rkey_t rkey);

ucs_status_t uct_musa_copy_ep_get_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                         size_t iovcnt, uint64_t remote_addr,
                                         uct_rkey_t rkey,
                                         uct_completion_t *comp);

ucs_status_t uct_musa_copy_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov,
                                         size_t iovcnt, uint64_t remote_addr,
                                         uct_rkey_t rkey,
                                         uct_completion_t *comp);

#endif
