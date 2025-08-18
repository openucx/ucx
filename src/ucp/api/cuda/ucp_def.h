/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef CUDA_UCP_DEF_H
#define CUDA_UCP_DEF_H

#include <uct/api/uct.h>
#include <uct/api/cuda/uct.h>

typedef enum {
    UCP_DEV_SCALE_THREAD = UCT_DEV_SCALE_THREAD,
    UCP_DEV_SCALE_WARP = UCT_DEV_SCALE_WARP,
    UCP_DEV_SCALE_BLOCK = UCT_DEV_SCALE_BLOCK,
    /* TODO add
    UCP_DEV_SCALE_GRID */
} ucp_dev_scale_t;

/**
 * @ingroup UCP_COMM
 * @brief Opaque batch descriptor to be used by GPU code.
 *
 * This currently mirrors the UCT batch handle, but it seems there might
 * eventually be multiple uct_batch_h, each one of then incrementing the
 * same signal memory area.
 *
 * This ucp_batch_t is stored on GPU.
 */
typedef struct ucp_batch {
    /* Default are GPU pointers */
    uct_batch_t  *uct_batch;
    uct_dev_ep_h exported_uct_ep;

    struct {
        /* Host pointer usable after copying from GPU */
        void    *uct_ep;

        /* Host handle to release */
        uct_allocated_memory_t mem;
    } host;
} ucp_batch_t;

#endif /* CUDA_UCP_DEF_H */
