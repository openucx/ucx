/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_DEV_H
#define UCT_DEV_H

extern "C" {
#include <ucs/type/status.h>
}

#include <uct/api/cuda/uct.h>
#include <uct/cuda/gdaki/gdaki.cuh>

/* execute prepared batch */
template<uct_dev_scale_t scale = UCT_DEV_SCALE_BLOCK>
__device__ static inline ucs_status_t
uct_dev_batch_execute(uct_batch_h batch, uint64_t flags,
                      uint64_t signal_inc, uct_dev_completion_t *comp)
{
    assert(batch->tl_id == UCT_DEV_TL_GDAKI);
    return uct_gdaki_batch_execute<scale>(batch, flags, signal_inc, comp);
}

/* execute prepared batch */
template<uct_dev_scale_t scale = UCT_DEV_SCALE_BLOCK>
__device__ static inline ucs_status_t
uct_dev_batch_execute_part(uct_batch_h batch, uint64_t flags,
                           uint64_t signal_inc, size_t count,
                           const int *indices, const size_t *src_offs,
                           const size_t *dst_offs, size_t *sizes,
                           uct_dev_completion_t *comp)
{
    assert(batch->tl_id == UCT_DEV_TL_GDAKI);
    return uct_gdaki_batch_execute_part<scale>(batch, flags, signal_inc, count,
                                               indices, src_offs, dst_offs, sizes,
                                               comp);
}

template<uct_dev_scale_t scale = UCT_DEV_SCALE_WARP, bool res_ctrl = false>
__device__ static inline ucs_status_t
uct_dev_batch_execute_single(uct_batch_h batch, uint64_t flags,
                           const size_t src_off, const size_t dst_off, size_t size,
                           uct_dev_completion_t *comp)
{
    return uct_gdaki_batch_execute_single<scale, res_ctrl>(
            batch, flags, src_off, dst_off, size, comp);
}

template<uct_dev_scale_t scale = UCT_DEV_SCALE_WARP, bool res_ctrl = false>
__device__ static inline ucs_status_t
uct_dev_batch_execute_atomic(uct_batch_h batch, uint64_t flags,
                           const int signal_inc, const size_t dst_off,
                           uct_dev_completion_t *comp)
{
    return uct_gdaki_batch_execute_atomic<scale, res_ctrl>(
            batch, flags, signal_inc, dst_off, comp);
}

/* progress ep */
template<uct_dev_scale_t scale = UCT_DEV_SCALE_BLOCK>
__device__ static inline ucs_status_t uct_dev_ep_progress(uct_dev_ep_h ep)
{
    assert(ep->tl_id == UCT_DEV_TL_GDAKI);
    return uct_gdaki_progress<scale>(ep);
}

#endif /* UCT_DEV_H */
