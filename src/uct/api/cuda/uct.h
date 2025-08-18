/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_H
#define UCT_CUDA_H

typedef struct uct_batch {
    unsigned tl_id;
} uct_batch_t;


typedef struct uct_dev_ep {
    unsigned tl_id;
} uct_dev_ep_t;

typedef enum {
    UCT_DEV_SCALE_THREAD,
    UCT_DEV_SCALE_WARP,
    UCT_DEV_SCALE_BLOCK,
    /* TODO add
    UCT_DEV_SCALE_GRID */
} uct_dev_scale_t;

#endif
