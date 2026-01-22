/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_DEVICE_TYPES_H
#define UCT_GDAKI_DEVICE_TYPES_H

#include <stdint.h>

typedef struct uct_rc_gdaki_device_mem_element {
    uint32_t lkey;
    uint32_t rkey;
} uct_rc_gdaki_device_mem_element_t;

typedef struct {
    uint64_t wqe_idx;
    unsigned channel_id;
} uct_rc_gda_completion_t;

#endif /* UCT_GDAKI_DEVICE_TYPES_H */
