/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_EFA_DV_H_
#define UCT_IB_EFA_DV_H_

#ifndef UCT_IB_EFA_H_
# error "Never include <uct/ib/efa/ib_efa_dv.h> directly; use <uct/ib/efa/ib_efa.h> instead."
#endif

#include <ucs/type/status.h>
#include <uct/ib/base/ib_device.h>
#include <infiniband/efadv.h>


typedef struct efadv_device_attr uct_ib_efadv_attr_t;

typedef struct uct_ib_efadv {
    uct_ib_efadv_attr_t efadv_attr;
} uct_ib_efadv_t;


ucs_status_t uct_ib_efadv_query(struct ibv_context *ctx,
                                uct_ib_efadv_attr_t *efadv_attr);

#endif
