/*
 * Copyright (C) Intel Corporation, 2023-2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ZE_IPC_EP_H
#define UCT_ZE_IPC_EP_H

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>

#include <sys/types.h>

/* Forward declaration */
typedef struct uct_ze_ipc_iface uct_ze_ipc_iface_t;


typedef struct uct_ze_ipc_ep {
    uct_base_ep_t   super;
    pid_t           remote_pid;
} uct_ze_ipc_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_ze_ipc_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_ze_ipc_ep_t, uct_ep_t);

ucs_status_t uct_ze_ipc_ep_get_zcopy(uct_ep_h tl_ep,
                                     const uct_iov_t *iov, size_t iovcnt,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_completion_t *comp);

ucs_status_t uct_ze_ipc_ep_put_zcopy(uct_ep_h tl_ep,
                                     const uct_iov_t *iov, size_t iovcnt,
                                     uint64_t remote_addr, uct_rkey_t rkey,
                                     uct_completion_t *comp);

int uct_ze_ipc_ep_is_connected(const uct_ep_h tl_ep,
                               const uct_ep_is_connected_params_t *params);

int uct_ze_ipc_dup_fd_from_pid(pid_t remote_pid, int remote_fd);

#endif
