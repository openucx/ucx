/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_DEV_H
#define UCT_GDAKI_DEV_H

#include <uct/api/device/uct_device_types.h>

typedef struct {
    uct_device_completion_t *comp;
} uct_rc_gdaki_op_t;


typedef struct {
    uct_device_ep_t              super;
    struct doca_gpu_dev_verbs_qp *qp;
    void                         *atomic_va;
    uint32_t                     cq_dbrec[2];
    uint32_t                     qp_dbrec[2];
    uct_rc_gdaki_op_t            ops[0];
} uct_rc_gdaki_dev_ep_t;


typedef struct uct_rc_gdaki_device_mem_element {
    uint32_t lkey;
    uint32_t rkey;
} uct_rc_gdaki_device_mem_element_t;

#endif /* UCT_GDAKI_DEV_H */
