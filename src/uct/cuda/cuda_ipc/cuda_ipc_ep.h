/**
* Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_IPC_EP_H
#define UCT_CUDA_IPC_EP_H

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>
#include <ucs/datastruct/khash.h>
#include "cuda_ipc_md.h"

typedef struct uct_cuda_ipc_rem_seg uct_cuda_ipc_rem_seg_t;

struct uct_cuda_ipc_rem_seg {
    uct_cuda_ipc_rem_seg_t *next;
    CUipcMemHandle         ph;         /* Memory handle of GPU memory */
    CUdeviceptr            d_bptr;     /* Allocation base address */
    size_t                 b_len;      /* Allocation size */
    int                    dev_num;    /* GPU Device number */
};

static inline khint_t uct_cuda_ipc_memh_hash_func(CUipcMemHandle seg)
{
    int hash_val = 7;
    int i;

    for (i = 0; i < sizeof(seg); i++) {
        hash_val = hash_val*31 + seg.reserved[i];
    }

    return (khint_t) (hash_val);
}

#define uct_cuda_ipc_memh_hash_equal(sg1, sg2)  \
    strncmp((const char *) &sg1, (const char *) &sg2, sizeof(CUipcMemHandle))

KHASH_INIT(uct_cuda_ipc_memh_hash, CUipcMemHandle, CUdeviceptr, 1,
           uct_cuda_ipc_memh_hash_func, uct_cuda_ipc_memh_hash_equal);

typedef struct uct_cuda_ipc_ep_addr {
    int                ep_id;
} uct_cuda_ipc_ep_addr_t;

typedef struct uct_cuda_ipc_ep {
    uct_base_ep_t                   super;
    khash_t(uct_cuda_ipc_memh_hash) memh_hash;
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
