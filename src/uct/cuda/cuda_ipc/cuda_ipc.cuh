/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_IPC_CUH
#define UCT_CUDA_IPC_CUH

#include "ucs/type/status.h"
#include "uct/api/uct_def.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <uct/api/device/uct_device_types.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_device.h>

#define UCT_CUDA_IPC_IS_ALIGNED_POW2(_n, _p) (!((_n) & ((_p) - 1)))
#define UCT_CUDA_IPC_WARP_SIZE 32
#define UCT_CUDA_IPC_COPY_LOOP_UNROLL 8

UCS_F_DEVICE int4 uct_cuda_ipc_ld_global_cg(const int4* p) {
    int4 v;
    asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
}

UCS_F_DEVICE void uct_cuda_ipc_st_global_cg(int4* p, const int4& v) {
    asm volatile ("st.global.cg.v4.s32 [%0], {%1,%2,%3,%4};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

UCS_F_DEVICE int2 uct_cuda_ipc_ld_global_cg(const int2* p) {
    int2 v;
    asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];"
                  : "=r"(v.x), "=r"(v.y)
                  : "l"(p));
    return v;
}

UCS_F_DEVICE void uct_cuda_ipc_st_global_cg(int2* p, const int2& v) {
    asm volatile ("st.global.cg.v2.s32 [%0], {%1,%2};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y));
}

UCS_F_DEVICE void uct_cuda_ipc_copy_single_nv(void *dst, const void *src, size_t size)
{
    /* TODO: add vectorized version*/
    auto s1 = reinterpret_cast<const char*>(src);
    auto d1 = reinterpret_cast<char *>(dst);

    for (size_t i = threadIdx.x; i < size; i += blockDim.x) {
        d1[i] = s1[i];
    }
}

template<int UNROLL>
UCS_F_DEVICE void uct_cuda_ipc_copy_single(void *dst, const void *src, size_t size)
{
    using vec4 = int4;
    using vec2 = int2;
    auto s1  = reinterpret_cast<const char*>(src);
    auto d1  = reinterpret_cast<char *>(dst);
    const vec4 *s4;
    vec4 *d4;
    int warp, num_warps, idx;
    size_t num_lines;

    if (UCT_CUDA_IPC_IS_ALIGNED_POW2((intptr_t)s1, sizeof(vec4)) &&
        UCT_CUDA_IPC_IS_ALIGNED_POW2((intptr_t)d1, sizeof(vec4))) {
        vec4 tmp[UNROLL];
        warp      = threadIdx.x / UCT_CUDA_IPC_WARP_SIZE;
        num_warps = blockDim.x / UCT_CUDA_IPC_WARP_SIZE;
        idx       = threadIdx.x % UCT_CUDA_IPC_WARP_SIZE;
        s4        = reinterpret_cast<const vec4*>(s1);
        d4        = reinterpret_cast<vec4*>(d1);
        num_lines = (size / (UCT_CUDA_IPC_WARP_SIZE * UNROLL * sizeof(vec4))) *
                    (UCT_CUDA_IPC_WARP_SIZE * UNROLL);

        for (size_t line = warp * UCT_CUDA_IPC_WARP_SIZE * UNROLL + idx; line < num_lines;
             line += num_warps * UCT_CUDA_IPC_WARP_SIZE * UNROLL) {
#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                tmp[i] = uct_cuda_ipc_ld_global_cg(s4 + (line + UCT_CUDA_IPC_WARP_SIZE * i));
            }

#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                uct_cuda_ipc_st_global_cg(d4 + (line + UCT_CUDA_IPC_WARP_SIZE * i), tmp[i]);
            }
        }
        size = size - num_lines * sizeof(vec4);
        if (size == 0) {
            return;
        }

        s4 = s4 + num_lines;
        d4 = d4 + num_lines;
        num_lines = size / sizeof(vec4);
        for (size_t line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vec4 v = uct_cuda_ipc_ld_global_cg(s4 + line);
            uct_cuda_ipc_st_global_cg(d4 + line, v);
        }

        size = size - num_lines * sizeof(vec4);
        if (size == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s4 + num_lines);
        d1 = reinterpret_cast<char*>(d4 + num_lines);
    }

    /* If not 16B-aligned, try 8B-aligned fast path using vec2 */
    if (UCT_CUDA_IPC_IS_ALIGNED_POW2((intptr_t)s1, sizeof(vec2)) &&
        UCT_CUDA_IPC_IS_ALIGNED_POW2((intptr_t)d1, sizeof(vec2))) {
        const vec2 *s2;
        vec2 *d2;
        vec2 tmp2[UNROLL];

        warp      = threadIdx.x / UCT_CUDA_IPC_WARP_SIZE;
        num_warps = blockDim.x / UCT_CUDA_IPC_WARP_SIZE;
        idx       = threadIdx.x % UCT_CUDA_IPC_WARP_SIZE;
        s2        = reinterpret_cast<const vec2*>(s1);
        d2        = reinterpret_cast<vec2*>(d1);
        num_lines = (size / (UCT_CUDA_IPC_WARP_SIZE * UNROLL * sizeof(vec2))) *
                    (UCT_CUDA_IPC_WARP_SIZE * UNROLL);

        for (size_t line = warp * UCT_CUDA_IPC_WARP_SIZE * UNROLL + idx; line < num_lines;
             line += num_warps * UCT_CUDA_IPC_WARP_SIZE * UNROLL) {
#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                tmp2[i] = uct_cuda_ipc_ld_global_cg(s2 + (line + UCT_CUDA_IPC_WARP_SIZE * i));
            }

#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                uct_cuda_ipc_st_global_cg(d2 + (line + UCT_CUDA_IPC_WARP_SIZE * i), tmp2[i]);
            }
        }

        size = size - num_lines * sizeof(vec2);
        if (size == 0) {
            return;
        }

        s2 = s2 + num_lines;
        d2 = d2 + num_lines;
        num_lines = size / sizeof(vec2);
        for (size_t line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vec2 v2 = uct_cuda_ipc_ld_global_cg(s2 + line);
            uct_cuda_ipc_st_global_cg(d2 + line, v2);
        }

        size = size - num_lines * sizeof(vec2);
        if (size == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s2 + num_lines);
        d1 = reinterpret_cast<char*>(d2 + num_lines);
    }

    for (size_t line = threadIdx.x; line < size; line += blockDim.x) {
        d1[line] = s1[line];
    }
}

template<uct_device_level_t level = UCT_DEVICE_LEVEL_BLOCK>
UCS_F_DEVICE ucs_status_t
uct_cuda_ipc_ep_put_single(uct_device_ep_h device_ep,
                           const uct_device_mem_element_t *mem_elem,
                           const void *address, uint64_t remote_address,
                           size_t length, uint64_t flags,
                           uct_device_completion_t *comp)
{
    auto cuda_ipc_mem_element =
        reinterpret_cast<const uct_cuda_ipc_device_mem_element_t *>(mem_elem);
    void *mapped_rem_addr;

    mapped_rem_addr = (void *)((uintptr_t)(remote_address) + cuda_ipc_mem_element->mapped_offset);

    switch (level) {
        case UCT_DEVICE_LEVEL_THREAD:
            /* TODO: add vectorized version*/
            memcpy(mapped_rem_addr, address, length);
            break;
        case UCT_DEVICE_LEVEL_WARP:
            /* TODO: check if we can use uct_cuda_ipc_copy_single, need to see perf impact */
            uct_cuda_ipc_copy_single_nv(mapped_rem_addr, address, length);
            break;
        case UCT_DEVICE_LEVEL_BLOCK:
            uct_cuda_ipc_copy_single<UCT_CUDA_IPC_COPY_LOOP_UNROLL>(mapped_rem_addr, address, length);
            break;
        case UCT_DEVICE_LEVEL_GRID:
            return UCS_ERR_UNSUPPORTED;
        default:
            return UCS_ERR_INVALID_PARAM;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        comp->count  = 0;
    }
    return UCS_OK;
}

#endif /* UCT_CUDA_IPC_CUH */
