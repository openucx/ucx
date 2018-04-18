/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_EP_H
#define UCT_CUDA_IPC_EP_H

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>
#include "cuda_ipc_md.h"


#define UCT_CUDA_IPC_HASH_SIZE 256
#define UCT_CUDA_IPC_PUT 0
#define UCT_CUDA_IPC_GET 1

typedef struct uct_cuda_ipc_rem_seg  uct_cuda_ipc_rem_seg_t;


typedef struct uct_cuda_ipc_rem_seg {
    uct_cuda_ipc_rem_seg_t *next;
    CUipcMemHandle         ph;         /* Memory handle of GPU memory */
    CUdeviceptr            d_bptr;     /* Allocation base address */
    size_t                 b_len;      /* Allocation size */
    int                    dev_num;    /* GPU Device number */
} uct_cuda_ipc_rem_seg_t;


typedef struct uct_cuda_ipc_ep_addr {
    int                ep_id;
} uct_cuda_ipc_ep_addr_t;


typedef struct uct_cuda_ipc_ep {
    uct_base_ep_t          super;
    uct_cuda_ipc_rem_seg_t *rem_segments_hash[UCT_CUDA_IPC_HASH_SIZE];
} uct_cuda_ipc_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_cuda_ipc_ep_t, uct_ep_t, uct_iface_t*,
                           const uct_device_addr_t *, const uct_iface_addr_t *);

UCS_CLASS_DECLARE_DELETE_FUNC(uct_cuda_ipc_ep_t, uct_ep_t);


ucs_status_t uct_cuda_ipc_ep_get_zcopy(uct_ep_h tl_ep,
                                       const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);


ucs_status_t uct_cuda_ipc_ep_put_zcopy(uct_ep_h tl_ep,
                                       const uct_iov_t *iov, size_t iovcnt,
                                       uint64_t remote_addr, uct_rkey_t rkey,
                                       uct_completion_t *comp);

#endif
