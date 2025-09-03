/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_IPC_CUH
#define UCT_CUDA_IPC_CUH

#include "uct/api/uct_def.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <uct/api/device/uct_device_types.h>

extern "C" {
#include <uct/cuda/cuda_ipc/cuda_ipc_device.h>
}

#define align_pow2(_n, _p) (!((_n) & ((_p) - 1)))
#define WARP_SIZE 32
#define COPY_LOOP_UNROLL 8

__device__ static inline void uct_cuda_ipc_copy_single_nv(void *dst,
                                                          const void *src,
                                                          size_t size)
{
    size_t i;
    char *s1 = (char *)src;
    char *d1 = (char *)dst;

    for (i = threadIdx.x; i <size; i += blockDim.x) {
        d1[i] = s1[i];
    }
}

static __device__ __forceinline__ int4 ld_global_cg(const int4* p) {
    int4 v;
    asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cg(int4* p, const int4& v) {
    asm volatile ("st.global.cg.v4.s32 [%0], {%1,%2,%3,%4};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

static __device__ __forceinline__ int2 ld_global_cg(const int2* p) {
    int2 v;
    asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];"
                  : "=r"(v.x), "=r"(v.y)
                  : "l"(p));
    return v;
}

static __device__ __forceinline__ void st_global_cg(int2* p, const int2& v) {
    asm volatile ("st.global.cg.v2.s32 [%0], {%1,%2};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y));
}

#include <stdio.h>
template<int UNROLL>
__device__ static void uct_cuda_ipc_copy_single(void *dst,
                                                const void *src,
                                                size_t size)
{
    typedef int4 vec4;
    typedef int2 vec2;
    const char *s1  = reinterpret_cast<const char*>(src);
    char       *d1  = reinterpret_cast<char *>(dst);
    const vec4 *s4;
    vec4 *d4;
    int warp, num_warps, idx;
    size_t line, num_lines;

    if (align_pow2((intptr_t)s1, sizeof(vec4)) && align_pow2((intptr_t)d1, sizeof(vec4))) {
        vec4 tmp[UNROLL];
        warp      = threadIdx.x / WARP_SIZE;
        num_warps = blockDim.x / WARP_SIZE;
        idx       = threadIdx.x % WARP_SIZE;
        s4        = reinterpret_cast<const vec4*>(s1);
        d4        = reinterpret_cast<vec4*>(d1);
        num_lines = (size / (WARP_SIZE * UNROLL * sizeof(vec4))) *
                    (WARP_SIZE * UNROLL);

        for (line = warp * WARP_SIZE * UNROLL + idx; line < num_lines;
             line += num_warps * WARP_SIZE * UNROLL) {
#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                tmp[i] = ld_global_cg(s4 + (line + WARP_SIZE * i));
            }

#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                st_global_cg(d4 + (line + WARP_SIZE * i), tmp[i]);
            }
        }
        size = size - num_lines * sizeof(vec4);
        if (size == 0) {
            return;
        }

        s4 = s4 + num_lines;
        d4 = d4 + num_lines;
        num_lines = size / sizeof(vec4);
        for (line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vec4 v = ld_global_cg(s4 + line);
            st_global_cg(d4 + line, v);
        }

        size = size - num_lines * sizeof(vec4);
        if (size == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s4 + num_lines);
        d1 = reinterpret_cast<char*>(d4 + num_lines);
    }

    /* If not 16B-aligned, try 8B-aligned fast path using vec2 */
    if (align_pow2((intptr_t)s1, sizeof(vec2)) && align_pow2((intptr_t)d1, sizeof(vec2))) {
        const vec2 *s2;
        vec2 *d2;
        vec2 tmp2[UNROLL];

        warp      = threadIdx.x / WARP_SIZE;
        num_warps = blockDim.x / WARP_SIZE;
        idx       = threadIdx.x % WARP_SIZE;
        s2        = reinterpret_cast<const vec2*>(s1);
        d2        = reinterpret_cast<vec2*>(d1);
        num_lines = (size / (WARP_SIZE * UNROLL * sizeof(vec2))) *
                    (WARP_SIZE * UNROLL);

        for (line = warp * WARP_SIZE * UNROLL + idx; line < num_lines;
             line += num_warps * WARP_SIZE * UNROLL) {
#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                tmp2[i] = ld_global_cg(s2 + (line + WARP_SIZE * i));
            }

#pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                st_global_cg(d2 + (line + WARP_SIZE * i), tmp2[i]);
            }
        }

        size = size - num_lines * sizeof(vec2);
        if (size == 0) {
            return;
        }

        s2 = s2 + num_lines;
        d2 = d2 + num_lines;
        num_lines = size / sizeof(vec2);
        for (line = threadIdx.x; line < num_lines; line += blockDim.x) {
            vec2 v2 = ld_global_cg(s2 + line);
            st_global_cg(d2 + line, v2);
        }

        size = size - num_lines * sizeof(vec2);
        if (size == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s2 + num_lines);
        d1 = reinterpret_cast<char*>(d2 + num_lines);
    }

    for (line = threadIdx.x; line < size; line += blockDim.x) {
        d1[line] = s1[line];
    }
}

template<uct_device_level_t level = UCT_DEVICE_LEVEL_BLOCK>
__device__ static inline ucs_status_t
uct_cuda_ipc_ep_put_single(uct_device_ep_h device_ep,
                           const uct_device_mem_element_t *mem_elem,
                           const void *address, uint64_t remote_address,
                           size_t length, uint64_t flags,
                           uct_device_completion_t *comp)
{
    uct_cuda_ipc_device_mem_element_t *cuda_ipc_mem_element =
        (uct_cuda_ipc_device_mem_element_t *)mem_elem;
    size_t offset;
    void *mapped_rem_addr;

    offset = (uintptr_t)remote_address - cuda_ipc_mem_element->dst_bptr;
    mapped_rem_addr = (void *)((uintptr_t)cuda_ipc_mem_element->mapped_addr + offset);

    switch (level) {
        case UCT_DEVICE_LEVEL_THREAD:
            memcpy(mapped_rem_addr, address, length);
            break;
        case UCT_DEVICE_LEVEL_WARP:
            uct_cuda_ipc_copy_single_nv(mapped_rem_addr, address, length);
            break;
        case UCT_DEVICE_LEVEL_BLOCK:
            uct_cuda_ipc_copy_single<COPY_LOOP_UNROLL>(mapped_rem_addr, address, length);
            break;
        case UCT_DEVICE_LEVEL_GRID:
            return UCS_ERR_UNSUPPORTED;
        default:
            __builtin_unreachable();
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        comp->count  = 0;
        comp->status = UCS_OK;
    }
    return UCS_OK;
}

#endif /* UCT_CUDA_IPC_CUH */
