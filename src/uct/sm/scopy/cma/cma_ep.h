/**
* Copyright (C) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2019. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifndef UCT_CMA_EP_H
#define UCT_CMA_EP_H

#include "cma_iface.h"

#include <uct/sm/scopy/base/scopy_ep.h>


typedef struct uct_cma_ep {
    uct_scopy_ep_t       super;
    pid_t                remote_pid;
    uct_keepalive_info_t keepalive;
} uct_cma_ep_t;


UCS_CLASS_DECLARE_NEW_FUNC(uct_cma_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_cma_ep_t, uct_ep_t);

ucs_status_t uct_cma_ep_tx(uct_ep_h tl_ep, const uct_iov_t *iov, size_t iov_cnt,
                           ucs_iov_iter_t *iov_iter, size_t *length_p,
                           uint64_t remote_addr, uct_rkey_t rkey,
                           uct_scopy_tx_op_t tx_op);

ucs_status_t uct_cma_ep_check(const uct_ep_h tl_ep, unsigned flags,
                              uct_completion_t *comp);

#endif
