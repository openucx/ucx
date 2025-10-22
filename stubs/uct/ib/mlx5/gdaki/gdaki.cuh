/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCT_GDAKI_CUH_H
#define UCT_GDAKI_CUH_H

/*
 * Stub implementation for GDAKI (GPU Direct Async Kernel Interface).
 * This file provides stub functions when DOCA GPUNetIO package is not available.
 * If DOCA GPUNetIO is installed, the real implementation from that package
 * will be used instead (via higher priority include path).
 */

#include "gdaki_dev.h"
#include <ucs/sys/device_code.h>
#include <ucs/type/status.h>

/**
 * Stub implementation: put_single operation
 * Returns UCS_ERR_UNSUPPORTED as GDAKI is not available
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_single(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_elem,
        const void *address, uint64_t remote_address, size_t length,
        uint64_t flags, uct_device_completion_t *comp)
{
    return UCS_ERR_UNSUPPORTED;
}

/**
 * Stub implementation: atomic_add operation
 * Returns UCS_ERR_UNSUPPORTED as GDAKI is not available
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_atomic_add(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_elem,
        uint64_t value, uint64_t remote_address, uint64_t flags,
        uct_device_completion_t *comp)
{
    return UCS_ERR_UNSUPPORTED;
}

/**
 * Stub implementation: put_multi operation
 * Returns UCS_ERR_UNSUPPORTED as GDAKI is not available
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_multi(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_list,
        unsigned mem_list_count, void *const *addresses,
        const uint64_t *remote_addresses, const size_t *lengths,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *tl_comp)
{
    return UCS_ERR_UNSUPPORTED;
}

/**
 * Stub implementation: put_multi_partial operation
 * Returns UCS_ERR_UNSUPPORTED as GDAKI is not available
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_multi_partial(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_list,
        const unsigned *mem_list_indices, unsigned mem_list_count,
        void *const *addresses, const uint64_t *remote_addresses,
        const size_t *local_offsets, const size_t *remote_offsets,
        const size_t *lengths, unsigned counter_index,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *tl_comp)
{
    return UCS_ERR_UNSUPPORTED;
}

/**
 * Stub implementation: endpoint progress
 * No-op as GDAKI is not available
 */
template<ucs_device_level_t level>
UCS_F_DEVICE void uct_rc_mlx5_gda_ep_progress(uct_device_ep_h tl_ep)
{
    /* No-op stub */
}

/**
 * Stub implementation: check completion
 * Returns UCS_ERR_UNSUPPORTED as GDAKI is not available
 */
template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_check_completion(
        uct_device_ep_h tl_ep, uct_device_completion_t *tl_comp)
{
    return UCS_ERR_UNSUPPORTED;
}

#endif /* UCT_GDAKI_CUH_H */

