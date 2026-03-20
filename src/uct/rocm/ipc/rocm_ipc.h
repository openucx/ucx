/**
 * Copyright (c) Advanced Micro Devices, Inc. 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_ROCM_IPC_H
#define UCT_ROCM_IPC_H

#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>
#include <ucs/sys/device_code.h>
#include <ucs/type/status.h>

#define UCT_ROCM_IPC_IS_ALIGNED_POW2(_n, _p) (!((_n) & ((_p)-1)))

/* Dynamically detect wavefront size using compiler builtin. */
#if __has_builtin(__builtin_amdgcn_wavefrontsize)
#define UCT_ROCM_IPC_WAVEFRONT_SIZE __builtin_amdgcn_wavefrontsize()
#else
#define UCT_ROCM_IPC_WAVEFRONT_SIZE __AMDGCN_WAVEFRONT_SIZE
#endif

#define UCT_ROCM_IPC_COPY_LOOP_UNROLL        8

/* Vectorized load with cache coherency - using direct memory access */
__device__ static inline int4 uct_rocm_ipc_ld_global_cg(const int4 *p)
{
    return *p;
}

__device__ static inline void uct_rocm_ipc_st_global_cg(int4 *p, const int4 &v)
{
    *p = v;
}

__device__ static inline int2 uct_rocm_ipc_ld_global_cg(const int2 *p)
{
    return *p;
}

__device__ static inline void uct_rocm_ipc_st_global_cg(int2 *p, const int2 &v)
{
    *p = v;
}

/* Get lane ID and number of lanes based on parallelism level */
template<ucs_device_level_t level>
__device__ static inline void
uct_rocm_ipc_get_lane(unsigned &lane_id, unsigned &num_lanes)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        lane_id   = 0;
        num_lanes = 1;
        break;
    case UCS_DEVICE_LEVEL_WARP:
        lane_id   = threadIdx.x % UCT_ROCM_IPC_WAVEFRONT_SIZE;
        num_lanes = UCT_ROCM_IPC_WAVEFRONT_SIZE;
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        lane_id   = threadIdx.x;
        num_lanes = blockDim.x;
        break;
    case UCS_DEVICE_LEVEL_GRID:
        lane_id   = threadIdx.x + blockIdx.x * blockDim.x;
        num_lanes = blockDim.x * gridDim.x;
        break;
    }
}

/* Map remote address using IPC handle */
__device__ static inline void *
uct_rocm_ipc_map_remote(const uct_rocm_ipc_device_mem_element_t *elem,
                        uint64_t remote_address)
{
    return reinterpret_cast<void*>((uintptr_t)remote_address +
                                   elem->mapped_offset);
}

/* System-wide atomic increment */
__device__ static inline void
uct_rocm_ipc_atomic_inc(uint64_t *dst, uint64_t inc_value)
{
    atomicAdd_system((unsigned long long*)dst, (unsigned long long)inc_value);
    __threadfence_system();
}

/* Level-appropriate synchronization */
template<ucs_device_level_t level>
__device__ static inline void uct_rocm_ipc_level_sync()
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        break;
    case UCS_DEVICE_LEVEL_WARP:
    case UCS_DEVICE_LEVEL_BLOCK:
        __syncthreads();
        break;
    case UCS_DEVICE_LEVEL_GRID:
        /* Not implemented */
        break;
    }
}

/* Copy routines for different parallelism levels */
template<ucs_device_level_t level>
__device__ void uct_rocm_ipc_copy_level(void *dst, const void *src, size_t len);

/* Thread-level copy */
template<>
__device__ inline void
uct_rocm_ipc_copy_level<UCS_DEVICE_LEVEL_THREAD>(void *dst, const void *src,
                                                 size_t len)
{
    memcpy(dst, src, len);
}

/* Wavefront-level copy (64 threads) */
template<>
__device__ inline void
uct_rocm_ipc_copy_level<UCS_DEVICE_LEVEL_WARP>(void *dst, const void *src,
                                               size_t len)
{
    using vec4 = int4;
    using vec2 = int2;
    unsigned int lane_id, num_lanes;

    uct_rocm_ipc_get_lane<UCS_DEVICE_LEVEL_WARP>(lane_id, num_lanes);
    auto s1 = reinterpret_cast<const char*>(src);
    auto d1 = reinterpret_cast<char*>(dst);

    /* 16B-aligned fast path using vec4 */
    if (UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)s1, sizeof(vec4)) &&
        UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)d1, sizeof(vec4))) {
        const vec4 *s4 = reinterpret_cast<const vec4*>(s1);
        vec4 *d4       = reinterpret_cast<vec4*>(d1);
        size_t n4      = len / sizeof(vec4);

        for (size_t i = lane_id; i < n4; i += num_lanes) {
            vec4 v = uct_rocm_ipc_ld_global_cg(s4 + i);
            uct_rocm_ipc_st_global_cg(d4 + i, v);
        }

        len = len - n4 * sizeof(vec4);
        if (len == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s4 + n4);
        d1 = reinterpret_cast<char*>(d4 + n4);
    }

    /* 8B-aligned fast path using vec2 */
    if (UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)s1, sizeof(vec2)) &&
        UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)d1, sizeof(vec2))) {
        const vec2 *s2 = reinterpret_cast<const vec2*>(s1);
        vec2 *d2       = reinterpret_cast<vec2*>(d1);
        size_t n2      = len / sizeof(vec2);

        for (size_t i = lane_id; i < n2; i += num_lanes) {
            vec2 v2 = uct_rocm_ipc_ld_global_cg(s2 + i);
            uct_rocm_ipc_st_global_cg(d2 + i, v2);
        }

        len = len - n2 * sizeof(vec2);
        if (len == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s2 + n2);
        d1 = reinterpret_cast<char*>(d2 + n2);
    }

    /* Byte tail */
    for (size_t i = lane_id; i < len; i += num_lanes) {
        d1[i] = s1[i];
    }
}

template<>
__device__ inline void
uct_rocm_ipc_copy_level<UCS_DEVICE_LEVEL_BLOCK>(void *dst, const void *src,
                                                size_t len)
{
    using vec4 = int4;
    using vec2 = int2;
    auto s1    = reinterpret_cast<const char*>(src);
    auto d1    = reinterpret_cast<char*>(dst);

    if (UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)s1, sizeof(vec4)) &&
        UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)d1, sizeof(vec4))) {
        const vec4 *s4   = reinterpret_cast<const vec4*>(s1);
        vec4 *d4         = reinterpret_cast<vec4*>(d1);
        size_t num_lines = len / sizeof(vec4);

        for (size_t line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vec4 v = uct_rocm_ipc_ld_global_cg(s4 + line);
            uct_rocm_ipc_st_global_cg(d4 + line, v);
        }

        len = len - num_lines * sizeof(vec4);
        if (len == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s4 + num_lines);
        d1 = reinterpret_cast<char*>(d4 + num_lines);
    }

    /* 8B-aligned fast path using vec2 */
    if (UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)s1, sizeof(vec2)) &&
        UCT_ROCM_IPC_IS_ALIGNED_POW2((intptr_t)d1, sizeof(vec2))) {
        const vec2 *s2   = reinterpret_cast<const vec2*>(s1);
        vec2 *d2         = reinterpret_cast<vec2*>(d1);
        size_t num_lines = len / sizeof(vec2);

        for (size_t line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vec2 v2 = uct_rocm_ipc_ld_global_cg(s2 + line);
            uct_rocm_ipc_st_global_cg(d2 + line, v2);
        }

        len = len - num_lines * sizeof(vec2);
        if (len == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s2 + num_lines);
        d1 = reinterpret_cast<char*>(d2 + num_lines);
    }

    /* Byte tail */
    for (size_t line = threadIdx.x; line < len; line += blockDim.x) {
        d1[line] = s1[line];
    }
}

/* Grid-level copy - not implemented */
template<>
__device__ inline void
uct_rocm_ipc_copy_level<UCS_DEVICE_LEVEL_GRID>(void *dst, const void *src,
                                               size_t len)
{
    /* Not implemented */
}

template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK>
__device__ ucs_status_t uct_rocm_ipc_ep_put(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        const void *address, uint64_t remote_address, size_t length,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto rocm_ipc_mem_element =
            reinterpret_cast<const uct_rocm_ipc_device_mem_element_t*>(
                    mem_elem);
    void *mapped_rem_addr;

    mapped_rem_addr = uct_rocm_ipc_map_remote(rocm_ipc_mem_element,
                                              remote_address);
    uct_rocm_ipc_copy_level<level>(mapped_rem_addr, address, length);
    uct_rocm_ipc_level_sync<level>();

    return UCS_OK;
}

/* Atomic add operation */
template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK>
__device__ ucs_status_t uct_rocm_ipc_ep_atomic_add(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_elem,
        uint64_t inc_value, uint64_t remote_address, uint64_t flags,
        uct_device_completion_t *comp)
{
    auto rocm_ipc_mem_element =
            reinterpret_cast<const uct_rocm_ipc_device_mem_element_t*>(
                    mem_elem);
    uint64_t *mapped_rem_addr;
    unsigned int lane_id, num_lanes;

    uct_rocm_ipc_get_lane<level>(lane_id, num_lanes);
    if (lane_id == 0) {
        mapped_rem_addr = reinterpret_cast<uint64_t*>(
                uct_rocm_ipc_map_remote(rocm_ipc_mem_element, remote_address));
        uct_rocm_ipc_atomic_inc(mapped_rem_addr, inc_value);
    }

    uct_rocm_ipc_level_sync<level>();
    return UCS_OK;
}

__device__ static inline ucs_status_t
uct_rocm_ipc_ep_get_ptr(uct_device_ep_h device_ep,
                        const uct_device_mem_element_t *mem_elem,
                        uint64_t address, void **addr_p)
{
    auto rocm_ipc_mem_element =
            reinterpret_cast<const uct_rocm_ipc_device_mem_element_t*>(
                    mem_elem);
    *addr_p = uct_rocm_ipc_map_remote(rocm_ipc_mem_element, address);
    return UCS_OK;
}

#endif /* UCT_ROCM_IPC_H */
