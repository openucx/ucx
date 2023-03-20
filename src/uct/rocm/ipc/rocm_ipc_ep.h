/*
 * Copyright (C) Advanced Micro Devices, Inc. 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef ROCM_IPC_EP_H
#define ROCM_IPC_EP_H

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>

#include "rocm_ipc_cache.h"

typedef struct uct_rocm_ipc_ep {
    uct_base_ep_t   super;
    pid_t           remote_pid;
} uct_rocm_ipc_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_rocm_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_rocm_ipc_ep_t, uct_ep_t);

ucs_status_t uct_rocm_ipc_ep_put_zcopy(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr,  uct_rkey_t rkey,
                                       uct_completion_t *comp);
ucs_status_t uct_rocm_ipc_ep_get_zcopy(uct_ep_h tl_ep,  const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

#endif
