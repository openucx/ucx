/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_CUDA_IPC_CUH
#define UCT_CUDA_IPC_CUH

#include <uct/api/uct_def.h>
#include <uct/api/device/uct_device_types.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_device.h>
#include <ucs/sys/device_code.h>
#include <ucs/type/status.h>
#include <cuda/atomic>

#define UCT_CUDA_IPC_IS_ALIGNED_POW2(_n, _p) (!((_n) & ((_p) - 1)))
#define UCT_CUDA_IPC_COPY_LOOP_UNROLL 8

UCS_F_DEVICE int4 uct_cuda_ipc_ld_global_cg(const int4* p)
{
    int4 v;
    asm volatile ("ld.global.cg.v4.s32 {%0,%1,%2,%3}, [%4];"
                  : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                  : "l"(p));
    return v;
}

UCS_F_DEVICE void uct_cuda_ipc_st_global_cg(int4* p, const int4& v)
{
    asm volatile ("st.global.cg.v4.s32 [%0], {%1,%2,%3,%4};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

UCS_F_DEVICE int2 uct_cuda_ipc_ld_global_cg(const int2* p)
{
    int2 v;
    asm volatile ("ld.global.cg.v2.s32 {%0,%1}, [%2];"
                  : "=r"(v.x), "=r"(v.y)
                  : "l"(p));
    return v;
}

UCS_F_DEVICE void uct_cuda_ipc_st_global_cg(int2* p, const int2& v)
{
    asm volatile ("st.global.cg.v2.s32 [%0], {%1,%2};"
                  :
                  : "l"(p), "r"(v.x), "r"(v.y));
}

template<ucs_device_level_t level>
UCS_F_DEVICE void
uct_cuda_ipc_get_lane(unsigned &lane_id, unsigned &num_lanes)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        lane_id   = 0;
        num_lanes = 1;
        break;
    case UCS_DEVICE_LEVEL_WARP:
        lane_id   = threadIdx.x % UCS_DEVICE_NUM_THREADS_IN_WARP;
        num_lanes = UCS_DEVICE_NUM_THREADS_IN_WARP;
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

template<ucs_device_level_t level>
UCS_F_DEVICE void uct_cuda_ipc_level_sync()
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        break;
    case UCS_DEVICE_LEVEL_WARP:
        __syncwarp();
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        __syncthreads();
        break;
    case UCS_DEVICE_LEVEL_GRID:
        assert(false);
        /* not implemented */
        break;
    }
    return;
}

template<typename VecT>
UCS_F_DEVICE void uct_cuda_ipc_try_copy_aligned(const char* &src, char* &dst,
                                                size_t &len,
                                                unsigned warp_id,
                                                unsigned num_warps,
                                                unsigned lane_id,
                                                unsigned num_lanes)
{
    if (!(UCT_CUDA_IPC_IS_ALIGNED_POW2((intptr_t)src, sizeof(VecT)) &&
          UCT_CUDA_IPC_IS_ALIGNED_POW2((intptr_t)dst, sizeof(VecT)))) {
        return;
    }

    auto src_vec                    = reinterpret_cast<const VecT*>(src);
    auto dst_vec                    = reinterpret_cast<VecT*>(dst);
    constexpr unsigned lanes_unroll = UCS_DEVICE_NUM_THREADS_IN_WARP *
                                      UCT_CUDA_IPC_COPY_LOOP_UNROLL;
    size_t num_lines                = (len / (lanes_unroll * sizeof(VecT))) *
                                      lanes_unroll;
    VecT tmp[UCT_CUDA_IPC_COPY_LOOP_UNROLL];

    for (size_t line = warp_id * lanes_unroll + lane_id % UCS_DEVICE_NUM_THREADS_IN_WARP;
         line < num_lines;
         line += num_warps * lanes_unroll) {
#pragma unroll
        for (int i = 0; i < UCT_CUDA_IPC_COPY_LOOP_UNROLL; i++) {
            tmp[i] = uct_cuda_ipc_ld_global_cg(
                src_vec + (line + UCS_DEVICE_NUM_THREADS_IN_WARP * i));
        }

#pragma unroll
        for (int i = 0; i < UCT_CUDA_IPC_COPY_LOOP_UNROLL; i++) {
            uct_cuda_ipc_st_global_cg(
                dst_vec + (line + UCS_DEVICE_NUM_THREADS_IN_WARP * i), tmp[i]);
        }
    }

    src_vec += num_lines;
    dst_vec += num_lines;
    len = len - num_lines * sizeof(VecT);

    num_lines = len / sizeof(VecT);
    for (size_t line = lane_id; line < num_lines; line += num_lanes) {
        VecT v = uct_cuda_ipc_ld_global_cg(src_vec + line);
        uct_cuda_ipc_st_global_cg(dst_vec + line, v);
    }

    len -= num_lines * sizeof(VecT);
    src = reinterpret_cast<const char*>(src_vec + num_lines);
    dst = reinterpret_cast<char*>(dst_vec + num_lines);
}

UCS_F_DEVICE void*
uct_cuda_ipc_map_remote(const uct_cuda_ipc_device_mem_element_t* elem,
                        uint64_t remote_address)
{
    return reinterpret_cast<void*>((uintptr_t)remote_address + elem->mapped_offset);
}

UCS_F_DEVICE void
uct_cuda_ipc_atomic_inc(uint64_t *dst, uint64_t inc_value)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_system> dst_ref{*dst};
    dst_ref.fetch_add(inc_value, cuda::memory_order_relaxed);
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
}

template<ucs_device_level_t level>
UCS_F_DEVICE void uct_cuda_ipc_copy_level(void *dst, const void *src, size_t len)
{
    auto s1 = reinterpret_cast<const char*>(src);
    auto d1 = reinterpret_cast<char *>(dst);
    unsigned int lane_id, num_lanes, warp_id, num_warps;

    uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
    warp_id = lane_id / UCS_DEVICE_NUM_THREADS_IN_WARP;
    num_warps = num_lanes / UCS_DEVICE_NUM_THREADS_IN_WARP;

    uct_cuda_ipc_try_copy_aligned<int4>(s1, d1, len, warp_id, num_warps,
                                        lane_id, num_lanes);
    uct_cuda_ipc_try_copy_aligned<int2>(s1, d1, len, warp_id, num_warps,
                                        lane_id, num_lanes);

    for (size_t line = lane_id; line < len; line += num_lanes) {
        d1[line] = s1[line];
    }
}

template<>
__device__ __forceinline__ void
uct_cuda_ipc_copy_level<UCS_DEVICE_LEVEL_THREAD>(void *dst, const void *src,
                                                 size_t len)
{
    memcpy(dst, src, len);
}

template<>
__device__ __forceinline__ void
uct_cuda_ipc_copy_level<UCS_DEVICE_LEVEL_GRID>(void *dst, const void *src,
                                               size_t len)
{/* not implemented */}

template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK>
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

    mapped_rem_addr = uct_cuda_ipc_map_remote(cuda_ipc_mem_element, remote_address);
    uct_cuda_ipc_copy_level<level>(mapped_rem_addr, address, length);
    uct_cuda_ipc_level_sync<level>();
    return UCS_OK;
}

template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK>
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_ep_put_multi(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_list,
        unsigned mem_list_count, void *const *addresses,
        const uint64_t *remote_addresses, const size_t *lengths,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *comp)
{
    unsigned int num_put_ops = (counter_remote_address != 0) ? mem_list_count - 1 : mem_list_count;
    unsigned int lane_id, num_lanes;

    uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
    for (int i = 0; i < num_put_ops; i++) {
        auto cuda_ipc_mem_element = reinterpret_cast<const uct_cuda_ipc_device_mem_element_t *>(
                UCS_PTR_BYTE_OFFSET(mem_list, sizeof(uct_cuda_ipc_device_mem_element_t) * i));
        auto mapped_rem_addr = uct_cuda_ipc_map_remote(
                cuda_ipc_mem_element, remote_addresses[i]);
        uct_cuda_ipc_copy_level<level>(mapped_rem_addr, addresses[i], lengths[i]);
    }

    if ((counter_remote_address != 0) && (lane_id == 0)) {
        auto cuda_ipc_mem_element = reinterpret_cast<const uct_cuda_ipc_device_mem_element_t *>(
                UCS_PTR_BYTE_OFFSET(mem_list, sizeof(uct_cuda_ipc_device_mem_element_t) * num_put_ops));
        auto mapped_counter_rem_addr = reinterpret_cast<uint64_t *>(uct_cuda_ipc_map_remote(cuda_ipc_mem_element,
                                                                                            counter_remote_address));
        uct_cuda_ipc_atomic_inc(mapped_counter_rem_addr, counter_inc_value);
    }

    uct_cuda_ipc_level_sync<level>();
    return UCS_OK;
}

template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK>
UCS_F_DEVICE ucs_status_t uct_cuda_ipc_ep_put_multi_partial(
        uct_device_ep_h device_ep, const uct_device_mem_element_t *mem_list,
        const unsigned *mem_list_indices, unsigned mem_list_count,
        void *const *addresses, const uint64_t *remote_addresses,
        const size_t *local_offsets, const size_t *remote_offsets,
        const size_t *lengths, unsigned counter_index,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *comp)
{
    unsigned int lane_id, num_lanes;

    uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
    for (int i = 0; i < mem_list_count; i++) {
        unsigned idx = mem_list_indices[i];
        auto cuda_ipc_mem_element =
                reinterpret_cast<const uct_cuda_ipc_device_mem_element_t*>(
                        UCS_PTR_BYTE_OFFSET(
                                mem_list,
                                sizeof(uct_cuda_ipc_device_mem_element_t) *
                                        idx));
        auto src_addr = UCS_PTR_BYTE_OFFSET(addresses[idx], local_offsets[i]);
        auto mapped_rem_addr = uct_cuda_ipc_map_remote(
                cuda_ipc_mem_element,
                remote_addresses[idx] + remote_offsets[i]);
        uct_cuda_ipc_copy_level<level>(mapped_rem_addr, src_addr, lengths[i]);
    }


    if ((counter_inc_value != 0) && (lane_id == 0)) {
        auto cuda_ipc_mem_element =
                reinterpret_cast<const uct_cuda_ipc_device_mem_element_t*>(
                        UCS_PTR_BYTE_OFFSET(
                                mem_list,
                                sizeof(uct_cuda_ipc_device_mem_element_t) *
                                        counter_index));
        auto mapped_counter_rem_addr = reinterpret_cast<uint64_t*>(
                uct_cuda_ipc_map_remote(cuda_ipc_mem_element,
                                        counter_remote_address));
        uct_cuda_ipc_atomic_inc(mapped_counter_rem_addr, counter_inc_value);
    }

    uct_cuda_ipc_level_sync<level>();
    return UCS_OK;
}

template<ucs_device_level_t level = UCS_DEVICE_LEVEL_BLOCK>
UCS_F_DEVICE ucs_status_t
uct_cuda_ipc_ep_atomic_add(uct_device_ep_h device_ep,
                           const uct_device_mem_element_t *mem_elem,
                           uint64_t inc_value, uint64_t remote_address,
                           uint64_t flags, uct_device_completion_t *comp)
{
    auto cuda_ipc_mem_element =
        reinterpret_cast<const uct_cuda_ipc_device_mem_element_t *>(mem_elem);
    uint64_t *mapped_rem_addr;
    unsigned int lane_id, num_lanes;

    uct_cuda_ipc_get_lane<level>(lane_id, num_lanes);
    if (lane_id == 0) {
        mapped_rem_addr = reinterpret_cast<uint64_t *>(uct_cuda_ipc_map_remote(cuda_ipc_mem_element,
                                                                               remote_address));
        uct_cuda_ipc_atomic_inc(mapped_rem_addr, inc_value);
    }

    uct_cuda_ipc_level_sync<level>();
    return UCS_OK;
}

#endif /* UCT_CUDA_IPC_CUH */
