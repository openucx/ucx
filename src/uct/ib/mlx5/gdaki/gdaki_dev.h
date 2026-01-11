/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_DEV_H
#define UCT_GDAKI_DEV_H

#include <uct/api/device/uct_device_types.h>

typedef struct {
    uint8_t                      cq_buff[64];

    uint32_t                     cq_dbrec[2];
    uint32_t                     qp_dbrec[2];

    uint64_t                     sq_rsvd_index;
    uint64_t                     sq_ready_index;
    int                          sq_lock;
    uint32_t                     sq_num;

    uint64_t                     *sq_db;
    uint8_t                      pad[16];
} uct_rc_gdaki_dev_qp_t;


typedef struct {
    uct_device_ep_t              super;
    void                         *atomic_va;
    uint32_t                     atomic_lkey;

    uint8_t                      *sq_wqe_daddr;
    uint16_t                     sq_wqe_num;
    uint16_t                     sq_fc_mask;
    uint8_t                      channel_mask;

    uint8_t                      pad[23];

    uct_rc_gdaki_dev_qp_t        qps[0];
} uct_rc_gdaki_dev_ep_t;


typedef struct uct_rc_gdaki_device_mem_element {
    uint32_t lkey;
    uint32_t rkey;
} uct_rc_gdaki_device_mem_element_t;

typedef struct {
    uint64_t wqe_idx;
    unsigned channel_id;
} uct_rc_gda_completion_t;

#endif /* UCT_GDAKI_DEV_H */
