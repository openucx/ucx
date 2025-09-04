/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_DEV_H
#define UCT_GDAKI_DEV_H

typedef struct {
    /* TODO add uct completion */
} uct_rc_gdaki_op_t;


typedef struct {
    struct doca_gpu_dev_verbs_qp *qp;
    uint32_t                     cq_dbrec[2];
    uint32_t                     qp_dbrec[2];
    uct_rc_gdaki_op_t            ops[0];
} uct_rc_gdaki_dev_ep_t;

#endif /* UCT_GDAKI_DEV_H */
