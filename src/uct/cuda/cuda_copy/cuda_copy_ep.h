/**
* Copyright (C) Mellanox Technologies Ltd. 2017-2019.  ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_COPY_EP_H
#define UCT_CUDA_COPY_EP_H

#include <uct/api/uct.h>
#include <uct/base/uct_iface.h>
#include <ucs/type/class.h>


typedef struct uct_cuda_copy_ep_addr {
    int                ep_id;
} uct_cuda_copy_ep_addr_t;

typedef struct uct_cuda_copy_ep {
    uct_base_ep_t           super;
    struct uct_cuda_copy_ep *next;
} uct_cuda_copy_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_cuda_copy_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cuda_copy_ep_t, uct_ep_t);

ucs_status_t uct_cuda_copy_ep_get_zcopy(uct_ep_h tl_ep,
                                        const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp);

ucs_status_t uct_cuda_copy_ep_put_zcopy(uct_ep_h tl_ep,
                                        const uct_iov_t *iov, size_t iovcnt,
                                        uint64_t remote_addr, uct_rkey_t rkey,
                                        uct_completion_t *comp);

ucs_status_t uct_cuda_copy_ep_put_short(uct_ep_h tl_ep, const void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey);

ucs_status_t uct_cuda_copy_ep_get_short(uct_ep_h tl_ep, void *buffer,
                                        unsigned length, uint64_t remote_addr,
                                        uct_rkey_t rkey);

#endif
