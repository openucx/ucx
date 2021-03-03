/**
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_IB_EFA_H_
#define UCT_IB_EFA_H_

#include <uct/ib/efa/ib_efa_dv.h>
#include <uct/ib/base/ib_md.h>


typedef struct uct_ib_efadv_md {
    uct_ib_md_t super;
    uct_ib_efadv_t efadv;
} uct_ib_efadv_md_t;

#endif
