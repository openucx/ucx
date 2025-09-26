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
    void                         *atomic_va;
    uint32_t                     atomic_lkey;
    uint32_t                     pad[1];
    uint32_t                     cq_dbrec[2];
    uint32_t                     qp_dbrec[2];

    uint64_t                     sq_rsvd_index;
    uint64_t                     sq_ready_index;
    uint64_t                     sq_wqe_pi;
    uint64_t                     cqe_ci;
    int                          sq_lock;

    uint8_t                      *sq_wqe_daddr;
    uint32_t                     *sq_dbrec;
    uint64_t                     *sq_db;
    uint8_t                      *cqe_daddr;
    uint32_t                     cqe_num;
    uint16_t                     sq_wqe_num;
    uint32_t                     sq_num;

    uct_rc_gdaki_op_t            ops[0];
} uct_rc_gdaki_dev_ep_t;


typedef struct uct_rc_gdaki_device_mem_element {
    uint32_t lkey;
    uint32_t rkey;
} uct_rc_gdaki_device_mem_element_t;

#endif /* UCT_GDAKI_DEV_H */
