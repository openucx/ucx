/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_EFA_H_
#define UCT_IB_EFA_H_


#include <ucs/type/status.h>
#include <uct/ib/base/ib_device.h>
#include <uct/ib/base/ib_md.h>
#include <infiniband/efadv.h>


typedef struct uct_ib_efadv {
    struct efadv_device_attr efadv_attr;
} uct_ib_efadv_t;


typedef struct uct_ib_efadv_md {
    uct_ib_md_t     super;
    uct_ib_efadv_t  efadv; /* EFA-specific cached device attributes */
} uct_ib_efadv_md_t;


ucs_status_t uct_ib_efadv_query(struct ibv_context *ctx,
                                struct efadv_device_attr *efadv_attr);
#endif
