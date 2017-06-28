/*
 * Copyright (C) Advanced Micro Devices, Inc. 2016 - 2017. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_CMA_EP_H
#define ROCM_CMA_EP_H

#include "rocm_cma_iface.h"

typedef struct uct_rocm_cma_ep {
    uct_base_ep_t   super;
    pid_t           remote_pid;
} uct_rocm_cma_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_rocm_cma_ep_t, uct_ep_t, uct_iface_t*,
                           const uct_device_addr_t *, const uct_iface_addr_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rocm_cma_ep_t, uct_ep_t);

ucs_status_t uct_rocm_cma_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr,  uct_rkey_t rkey,
                                       uct_completion_t *comp);
ucs_status_t uct_rocm_cma_ep_get_zcopy(uct_ep_h tl_ep,  const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

#endif
