/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2024. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_EP_H
#define UCT_GDAKI_EP_H

#include "gdaki_ep_dev.h"
#include <uct/cuda/base/cuda_iface.h>
#include <uct/ib/mlx5/ib_mlx5.h>

typedef struct {
    struct doca_gpu_verbs_qp *qp_cpu;
    struct doca_gpu_dev_verbs_qp *qp_gpu;
    uct_gdaki_dev_ep_t *ep_gpu;
} uct_gdaki_export_ep_t;

typedef struct uct_gdaki_ep {
    uct_base_ep_t super;
    uct_ib_mlx5_cq_t cq;
    uct_ib_mlx5_txwq_t qp;
    uct_gdaki_export_ep_t e;
} uct_gdaki_ep_t;

UCS_CLASS_DECLARE_NEW_FUNC(uct_gdaki_ep_t, uct_ep_t, const uct_ep_params_t *);
UCS_CLASS_DECLARE_DELETE_FUNC(uct_gdaki_ep_t, uct_ep_t);

ucs_status_t
uct_gdaki_ep_connect_to_ep_v2(uct_ep_h ep, const uct_device_addr_t *device_addr,
                             const uct_ep_addr_t *ep_addr,
                             const uct_ep_connect_to_ep_params_t *params);
int
uct_gdaki_base_ep_is_connected(uct_ep_h ep, const uct_ep_is_connected_params_t *params);
ucs_status_t uct_gdaki_ep_get_address(uct_ep_h tl_ep, uct_ep_addr_t *addr);

ucs_status_t uct_gdaki_ep_batch_prepare(uct_ep_h tl_ep, const uct_rma_iov_t *iov,
                                        size_t iovcnt, uint64_t signal_var,
                                        uct_rkey_t signal_rkey, uct_batch_h *batch_p);
void uct_gdaki_ep_batch_release(uct_ep_h tl_ep, uct_batch_h batch);

ucs_status_t uct_gdaki_ep_export_dev(uct_ep_h ep, uct_dev_ep_h *dev_ep_p);

#endif /* UCT_GDAKI_EP_H */
