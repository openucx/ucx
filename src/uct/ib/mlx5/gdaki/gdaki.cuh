/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_GDAKI_CUH_H
#define UCT_GDAKI_CUH_H

#include "gdaki_dev.h"

#include <doca_gpunetio_dev_verbs_qp.cuh>
#include <cooperative_groups.h>

#define UCT_RC_GDA_RESV_WQE_NO_RESOURCE -1ULL
#define UCT_RC_GDA_WQE_ERR              UCS_BIT(63)
#define UCT_RC_GDA_WQE_MASK             UCS_MASK(63)


UCS_F_DEVICE uct_rc_gdaki_dev_qp_t *
uct_rc_mlx5_gda_get_qp(uct_rc_gdaki_dev_ep_t *ep, unsigned cid)
{
    return ep->qps + cid;
}

UCS_F_DEVICE void *uct_rc_mlx5_gda_get_wqe_ptr(uct_rc_gdaki_dev_ep_t *ep,
                                               unsigned cid, uint16_t wqe_idx)
{
    const uint16_t wqe_num    = __ldg(&ep->sq_wqe_num);
    const uintptr_t wqe_addr  = __ldg((uintptr_t*)&ep->sq_wqe_daddr);
    const uint32_t idx        = wqe_idx & (wqe_num - 1);
    const uint32_t full_idx   = idx + cid * wqe_num;
    return (void*)(wqe_addr + (full_idx << DOCA_GPUNETIO_MLX5_WQE_SQ_SHIFT));
}

UCS_F_DEVICE void uct_rc_mlx5_gda_ring_db(uct_rc_gdaki_dev_ep_t *ep,
                                          unsigned cid, uint64_t prod_index)
{
    uct_rc_gdaki_dev_qp_t *qp = uct_rc_mlx5_gda_get_qp(ep, cid);
    struct doca_gpu_dev_verbs_wqe_ctrl_seg ctrl_seg = {0};
    __be64 *db_ptr = (__be64*)__ldg((uintptr_t*)&qp->sq_db);

    ctrl_seg.qpn_ds = doca_gpu_dev_verbs_bswap32(__ldg(&qp->sq_num) << 8);
    ctrl_seg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
            (prod_index << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT));

    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
    doca_gpu_dev_verbs_store_relaxed_mmio((uint64_t*)db_ptr,
                                          *(uint64_t*)&ctrl_seg);
}

UCS_F_DEVICE void uct_rc_mlx5_gda_update_dbr(uct_rc_gdaki_dev_ep_t *ep,
                                             unsigned cid, uint32_t prod_index)
{
    __be32 dbrec_val  = doca_gpu_dev_verbs_prepare_dbr(prod_index);
    __be32 *dbrec_ptr = &ep->qps[cid].qp_dbrec[MLX5_SND_DBR];

    cuda::atomic_ref<__be32, cuda::thread_scope_system> dbrec_ptr_aref(
            *dbrec_ptr);
    dbrec_ptr_aref.store(dbrec_val, cuda::std::memory_order_relaxed);
}

template<ucs_device_level_t level>
UCS_F_DEVICE void
uct_rc_mlx5_gda_exec_init(unsigned &lane_id, unsigned &num_lanes)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        lane_id   = 0;
        num_lanes = 1;
        break;
    case UCS_DEVICE_LEVEL_WARP:
        lane_id   = doca_gpu_dev_verbs_get_lane_id();
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

template<ucs_device_level_t level> UCS_F_DEVICE void uct_rc_mlx5_gda_sync(void)
{
    switch (level) {
    case UCS_DEVICE_LEVEL_WARP:
        __syncwarp();
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        __syncthreads();
        break;
    case UCS_DEVICE_LEVEL_THREAD:
        break;
    case UCS_DEVICE_LEVEL_GRID:
        auto g = cooperative_groups::this_grid();
        g.sync();
    }
}

UCS_F_DEVICE uint16_t uct_rc_mlx5_gda_bswap16(uint16_t x)
{
    uint32_t ret;
    asm volatile("{\n\t"
                 ".reg .b32 mask;\n\t"
                 ".reg .b32 ign;\n\t"
                 "mov.b32 mask, 0x1;\n\t"
                 "prmt.b32 %0, %1, ign, mask;\n\t"
                 "}"
                 : "=r"(ret)
                 : "r"((uint32_t)x));
    return ret;
}

UCS_F_DEVICE uint64_t uct_rc_mlx5_gda_parse_cqe(uct_rc_gdaki_dev_ep_t *ep,
                                                unsigned cid, uint16_t *wqe_cnt,
                                                uint8_t *opcode)
{
    uct_rc_gdaki_dev_qp_t *qp = uct_rc_mlx5_gda_get_qp(ep, cid);
    auto *cqe64               = reinterpret_cast<mlx5_cqe64*>(qp->cq_buff);
    uint32_t *data_ptr        = (uint32_t*)&cqe64->wqe_counter;
    uint32_t data             = READ_ONCE(*data_ptr);
    uint64_t rsvd_idx         = READ_ONCE(qp->sq_rsvd_index);

    *wqe_cnt = uct_rc_mlx5_gda_bswap16(data);
    if (opcode != nullptr) {
        *opcode = data >> 28;
    }

    return rsvd_idx - ((rsvd_idx - *wqe_cnt) & 0xffff);
}

UCS_F_DEVICE uint64_t uct_rc_mlx5_gda_max_alloc_wqe_base(
        uct_rc_gdaki_dev_ep_t *ep, unsigned cid, unsigned count)
{
    uint16_t wqe_cnt;
    uint64_t pi;

    pi = uct_rc_mlx5_gda_parse_cqe(ep, cid, &wqe_cnt, nullptr);
    return pi + ep->sq_wqe_num + 1 - count;
}

UCS_F_DEVICE uint64_t uct_rc_mlx5_gda_reserv_wqe_thread(
        uct_rc_gdaki_dev_ep_t *ep, unsigned cid, unsigned count)
{
    uct_rc_gdaki_dev_qp_t *qp = uct_rc_mlx5_gda_get_qp(ep, cid);
    /* Do not attempt to reserve if the available space is less than the
     * requested count, to avoid starvation of threads trying to rollback the
     * reservation with atomicCAS. */
    uint64_t max_wqe_base = uct_rc_mlx5_gda_max_alloc_wqe_base(ep, cid, count);
    if (qp->sq_rsvd_index > max_wqe_base) {
        return UCT_RC_GDA_RESV_WQE_NO_RESOURCE;
    }

    uint64_t wqe_base = atomicAdd(reinterpret_cast<unsigned long long*>(
                                          &qp->sq_rsvd_index),
                                  static_cast<unsigned long long>(count));

    /*
     *  Attempt to reserve 'count' WQEs by atomically incrementing the reserved
     *  index. If the reservation exceeds the available space in the work queue,
     *  enter a rollback loop.
     *
     *  Rollback Logic:
     *  - Calculate the next potential index (wqe_next) after attempting the
     *    reservation.
     *  - Use atomic CAS to check if the current reserved index matches wqe_next.
     *    If it does, revert the reservation by resetting the reserved index to
     *    wqe_base.
     *  - A successful CAS indicates no other thread has modified the reserved
     *    index, allowing the rollback to complete, and the function returns
     *    UCT_RC_GDA_RESV_WQE_NO_RESOURCE to signal insufficient resources.
     *  - If CAS fails, it means another thread has modified the reserved index.
     *    The loop continues to reevaluate resource availability to determine if
     *    the reservation can now be satisfied, possibly due to other operations
     *    freeing up resources.
     */
    while (wqe_base > max_wqe_base) {
        uint64_t wqe_next = wqe_base + count;
        if (atomicCAS(reinterpret_cast<unsigned long long*>(&qp->sq_rsvd_index),
                      wqe_next, wqe_base) == wqe_next) {
            return UCT_RC_GDA_RESV_WQE_NO_RESOURCE;
        }

        max_wqe_base = uct_rc_mlx5_gda_max_alloc_wqe_base(ep, cid, count);
    }

    return wqe_base;
}

template<ucs_device_level_t level>
UCS_F_DEVICE void
uct_rc_mlx5_gda_reserv_wqe(uct_rc_gdaki_dev_ep_t *ep, unsigned cid,
                           unsigned count, unsigned lane_id, uint64_t &wqe_base)
{
    if (lane_id == 0) {
        wqe_base = uct_rc_mlx5_gda_reserv_wqe_thread(ep, cid, count);
    }

    if (level == UCS_DEVICE_LEVEL_WARP) {
        wqe_base = __shfl_sync(0xffffffff, wqe_base, 0);
    } else if (level == UCS_DEVICE_LEVEL_BLOCK) {
        __syncthreads();
    }
}

UCS_F_DEVICE void uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
        uct_rc_gdaki_dev_ep_t *ep, void *wqe_ptr, uint16_t wqe_idx,
        uint32_t opcode, unsigned ctrl_flags, uint64_t raddr, uint32_t rkey,
        uint64_t laddr, uint32_t lkey, uint32_t bytes, bool is_atomic,
        uint64_t add, unsigned cid)
{
    uct_rc_gdaki_dev_qp_t *qp = uct_rc_mlx5_gda_get_qp(ep, cid);
    uint64_t *dseg_ptr  = (uint64_t*)wqe_ptr + 4 + 2 * is_atomic;
    uint64_t *cseg_ptr  = (uint64_t*)wqe_ptr;
    uint64_t *rseg_ptr  = (uint64_t*)wqe_ptr + 2;
    uint64_t *atseg_ptr = (uint64_t*)wqe_ptr + 4;
    int ds              = 3 + is_atomic;
    struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
    struct mlx5_wqe_raddr_seg rseg;
    struct mlx5_wqe_data_seg dseg;
    struct mlx5_wqe_atomic_seg atseg;

    cseg.opmod_idx_opcode = doca_gpu_dev_verbs_bswap32(
            ((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
    cseg.qpn_ds           = doca_gpu_dev_verbs_bswap32((qp->sq_num << 8) | ds);
    cseg.fm_ce_se         = ctrl_flags;

    rseg.raddr = doca_gpu_dev_verbs_bswap64(raddr);
    rseg.rkey  = rkey;

    dseg.byte_count = doca_gpu_dev_verbs_bswap32(bytes);
    dseg.lkey       = lkey;
    dseg.addr       = doca_gpu_dev_verbs_bswap64(laddr);

    if (is_atomic) {
        atseg.swap_add = doca_gpu_dev_verbs_bswap64(add);
        doca_gpu_dev_verbs_store_wqe_seg(atseg_ptr, (uint64_t*)&(atseg));
    }

    doca_gpu_dev_verbs_store_wqe_seg(cseg_ptr, (uint64_t*)&(cseg));
    doca_gpu_dev_verbs_store_wqe_seg(rseg_ptr, (uint64_t*)&(rseg));
    doca_gpu_dev_verbs_store_wqe_seg(dseg_ptr, (uint64_t*)&(dseg));
}

UCS_F_DEVICE void uct_rc_mlx5_gda_lock(int *lock) {
    while (atomicCAS(lock, 0, 1) != 0)
        ;
#ifdef DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX
    asm volatile("fence.acquire.gpu;");
#else
    uint32_t dummy;
    uint32_t UCS_V_UNUSED val;
    asm volatile("ld.acquire.gpu.b32 %0, [%1];" : "=r"(val) : "l"(&dummy));
#endif
}

UCS_F_DEVICE void uct_rc_mlx5_gda_unlock(int *lock) {
    cuda::atomic_ref<int, cuda::thread_scope_device> lock_aref(*lock);
    lock_aref.store(0, cuda::std::memory_order_release);
}

UCS_F_DEVICE void uct_rc_mlx5_gda_db(uct_rc_gdaki_dev_ep_t *ep, unsigned cid,
                                     uint64_t wqe_base, unsigned count,
                                     uint64_t flags)
{
    uct_rc_gdaki_dev_qp_t *qp = uct_rc_mlx5_gda_get_qp(ep, cid);
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(
            qp->sq_ready_index);
    const uint64_t wqe_next = wqe_base + count;
    const bool skip_db      = !(flags & UCT_DEVICE_FLAG_NODELAY) &&
                              !((wqe_base ^ wqe_next) & 128);

    __threadfence();
    if (skip_db) {
        const uint64_t wqe_base_orig = wqe_base;
        while (!ref.compare_exchange_strong(wqe_base, wqe_next,
                                            cuda::std::memory_order_relaxed)) {
            wqe_base = wqe_base_orig;
        }
    } else {
        while (READ_ONCE(qp->sq_ready_index) != wqe_base) {
        }
        uct_rc_mlx5_gda_ring_db(ep, cid, wqe_next);
        uct_rc_mlx5_gda_update_dbr(ep, cid, wqe_next);
        uct_rc_mlx5_gda_ring_db(ep, cid, wqe_next);
        ref.store(wqe_next, cuda::std::memory_order_release);
    }
}

UCS_F_DEVICE bool
uct_rc_mlx5_gda_fc(const uct_rc_gdaki_dev_ep_t *ep, uint16_t wqe_idx)
{
    return !(wqe_idx & ep->sq_fc_mask);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_single(
        uct_rc_gdaki_dev_ep_t *ep, const uct_device_mem_element_t *tl_mem_elem,
        const void *address, uint32_t lkey, uint64_t remote_address,
        uint32_t rkey, size_t length, unsigned cid, uint64_t flags,
        uct_device_completion_t *tl_comp, uint32_t opcode, bool is_atomic,
        uint64_t add)
{
    uct_rc_gda_completion_t *comp = &tl_comp->rc_gda;
    unsigned cflag                = 0;
    uint64_t wqe_base;
    unsigned lane_id;
    unsigned num_lanes;

    uct_rc_mlx5_gda_exec_init<level>(lane_id, num_lanes);
    uct_rc_mlx5_gda_reserv_wqe<level>(ep, cid, 1, lane_id, wqe_base);
    if (wqe_base == UCT_RC_GDA_RESV_WQE_NO_RESOURCE) {
        return UCS_ERR_NO_RESOURCE;
    }

    if (lane_id == 0) {
        uint16_t wqe_idx = (uint16_t)wqe_base;
        if ((comp != nullptr) || uct_rc_mlx5_gda_fc(ep, wqe_idx)) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            if (comp != nullptr) {
                comp->wqe_idx = wqe_base;
                comp->channel_id = cid;
            }
        }

        uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
                ep, uct_rc_mlx5_gda_get_wqe_ptr(ep, cid, wqe_idx), wqe_idx,
                opcode, cflag, remote_address, rkey,
                reinterpret_cast<uint64_t>(address), lkey, length, is_atomic,
                add, cid);
    }

    uct_rc_mlx5_gda_sync<level>();

    if (lane_id == 0) {
        uct_rc_mlx5_gda_db(ep, cid, wqe_base, 1, flags);
    }

    uct_rc_mlx5_gda_sync<level>();
    return UCS_INPROGRESS;
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_single(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_elem,
        const void *address, uint64_t remote_address, size_t length,
        unsigned channel_id, uint64_t flags, uct_device_completion_t *comp)
{
    auto ep       = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    auto mem_elem = reinterpret_cast<const uct_rc_gdaki_device_mem_element_t*>(
            tl_mem_elem);
    auto cid      = channel_id & ep->channel_mask;

    return uct_rc_mlx5_gda_ep_single<level>(ep, tl_mem_elem, address,
                                            mem_elem->lkey, remote_address,
                                            mem_elem->rkey, length, cid, flags,
                                            comp, MLX5_OPCODE_RDMA_WRITE, false,
                                            0);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_atomic_add(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_elem,
        uint64_t value, uint64_t remote_address, unsigned channel_id,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep       = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    auto mem_elem = reinterpret_cast<const uct_rc_gdaki_device_mem_element_t*>(
            tl_mem_elem);
    auto cid      = channel_id & ep->channel_mask;

    return uct_rc_mlx5_gda_ep_single<level>(ep, tl_mem_elem, ep->atomic_va,
                                            ep->atomic_lkey, remote_address,
                                            mem_elem->rkey, sizeof(uint64_t),
                                            cid, flags, comp,
                                            MLX5_OPCODE_ATOMIC_FA, true, value);
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_multi(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_list,
        unsigned mem_list_count, void *const *addresses,
        const uint64_t *remote_addresses, const size_t *lengths,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        unsigned channel_id, uint64_t flags, uct_device_completion_t *tl_comp)
{
    auto ep       = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    auto mem_list = reinterpret_cast<const uct_rc_gdaki_device_mem_element_t*>(
            tl_mem_list);
    auto cid      = channel_id & ep->channel_mask;
    uct_rc_gda_completion_t *comp = &tl_comp->rc_gda;
    int count                     = mem_list_count;
    int counter_index             = count - 1;
    bool atomic                   = false;
    uint64_t wqe_idx;
    unsigned cflag;
    unsigned lane_id;
    unsigned num_lanes;
    uint64_t wqe_base;
    size_t length;
    void *address;
    uint32_t lkey;
    uint64_t remote_address;
    uint32_t rkey;
    int opcode;

    if ((level != UCS_DEVICE_LEVEL_THREAD) &&
        (level != UCS_DEVICE_LEVEL_WARP)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (counter_inc_value == 0) {
        count--;
    }

    uct_rc_mlx5_gda_exec_init<level>(lane_id, num_lanes);
    uct_rc_mlx5_gda_reserv_wqe<level>(ep, cid, count, lane_id, wqe_base);
    if (wqe_base == UCT_RC_GDA_RESV_WQE_NO_RESOURCE) {
        return UCS_ERR_NO_RESOURCE;
    }

    wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_base, lane_id);
    for (uint32_t i = lane_id; i < count; i += num_lanes) {
        if (i == counter_index) {
            atomic         = true;
            address        = ep->atomic_va;
            lkey           = ep->atomic_lkey;
            remote_address = counter_remote_address;
            length         = 8;
            opcode         = MLX5_OPCODE_ATOMIC_FA;
        } else if (i < counter_index) {
            address        = addresses[i];
            lkey           = mem_list[i].lkey;
            remote_address = remote_addresses[i];
            length         = lengths[i];
            opcode         = MLX5_OPCODE_RDMA_WRITE;
        } else {
            continue;
        }

        cflag = 0;
        if (((comp != nullptr) && (i == count - 1)) ||
            ((comp == nullptr) && uct_rc_mlx5_gda_fc(ep, wqe_idx))) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            if (comp != nullptr) {
                comp->wqe_idx = wqe_base;
                comp->channel_id = cid;
            }
        }

        auto wqe_ptr = uct_rc_mlx5_gda_get_wqe_ptr(ep, cid, wqe_idx);
        rkey         = mem_list[i].rkey;

        uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
                ep, wqe_ptr, wqe_idx, opcode, cflag, remote_address, rkey,
                reinterpret_cast<uint64_t>(address), lkey, length, atomic,
                counter_inc_value, cid);
        wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_idx, num_lanes);
    }

    uct_rc_mlx5_gda_sync<level>();

    if (lane_id == 0) {
        uct_rc_mlx5_gda_db(ep, cid, wqe_base, count, flags);
    }

    uct_rc_mlx5_gda_sync<level>();
    return UCS_INPROGRESS;
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_multi_partial(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_list,
        const unsigned *mem_list_indices, unsigned mem_list_count,
        void *const *addresses, const uint64_t *remote_addresses,
        const size_t *local_offsets, const size_t *remote_offsets,
        const size_t *lengths, unsigned counter_index,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        unsigned channel_id, uint64_t flags, uct_device_completion_t *tl_comp)
{
    auto ep       = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    auto mem_list = reinterpret_cast<const uct_rc_gdaki_device_mem_element_t*>(
            tl_mem_list);
    auto cid      = channel_id & ep->channel_mask;
    uct_rc_gda_completion_t *comp = &tl_comp->rc_gda;
    unsigned count                = mem_list_count;
    bool atomic                   = false;
    uint64_t wqe_idx;
    unsigned lane_id;
    unsigned num_lanes;
    unsigned cflag;
    uint64_t wqe_base;
    size_t length;
    void *address;
    uint32_t lkey;
    uint64_t remote_address;
    uint32_t rkey;
    int opcode;
    uint32_t idx;

    if ((level != UCS_DEVICE_LEVEL_THREAD) &&
        (level != UCS_DEVICE_LEVEL_WARP)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (counter_inc_value != 0) {
        count++;
    }

    uct_rc_mlx5_gda_exec_init<level>(lane_id, num_lanes);
    uct_rc_mlx5_gda_reserv_wqe<level>(ep, cid, count, lane_id, wqe_base);
    if (wqe_base == UCT_RC_GDA_RESV_WQE_NO_RESOURCE) {
        return UCS_ERR_NO_RESOURCE;
    }

    wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_base, lane_id);
    for (uint32_t i = lane_id; i < count; i += num_lanes) {
        if (i == mem_list_count) {
            idx            = counter_index;
            atomic         = true;
            address        = ep->atomic_va;
            lkey           = ep->atomic_lkey;
            remote_address = counter_remote_address;
            length         = 8;
            opcode         = MLX5_OPCODE_ATOMIC_FA;
        } else if (i < mem_list_count) {
            idx     = mem_list_indices[i];
            address = UCS_PTR_BYTE_OFFSET(addresses[idx], local_offsets[i]);
            lkey    = mem_list[idx].lkey;
            remote_address = remote_addresses[idx] + remote_offsets[i];
            length         = lengths[i];
            opcode         = MLX5_OPCODE_RDMA_WRITE;
        } else {
            continue;
        }

        cflag = 0;
        if (((comp != nullptr) && (i == count - 1)) ||
            ((comp == nullptr) && uct_rc_mlx5_gda_fc(ep, wqe_idx))) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            if (comp != nullptr) {
                comp->wqe_idx = wqe_base;
                comp->channel_id = cid;
            }
        }

        auto wqe_ptr = uct_rc_mlx5_gda_get_wqe_ptr(ep, cid, wqe_idx);
        rkey         = mem_list[idx].rkey;

        uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
                ep, wqe_ptr, wqe_idx, opcode, cflag, remote_address, rkey,
                reinterpret_cast<uint64_t>(address), lkey, length, atomic,
                counter_inc_value, cid);
        wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_idx, num_lanes);
    }

    uct_rc_mlx5_gda_sync<level>();

    if (lane_id == 0) {
        uct_rc_mlx5_gda_db(ep, cid, wqe_base, count, flags);
    }

    uct_rc_mlx5_gda_sync<level>();
    return UCS_INPROGRESS;
}

UCS_F_DEVICE void
uct_rc_mlx5_gda_qedump(const char *pfx, void *buff, ssize_t len)
{
    uint32_t *p = (uint32_t*)buff;
    size_t c;

    while (len > 0) {
        printf("%4s %#lx+%04lx:", pfx, (intptr_t)buff, p - (uint32_t*)buff);
        for (c = 0; c < 4; c++) {
            printf(" %08x", doca_gpu_dev_verbs_bswap32(p[c]));
        }
        printf("\n");
        p   += 4;
        len -= 16;
    }
}

template<ucs_device_level_t level>
UCS_F_DEVICE void uct_rc_mlx5_gda_ep_progress(uct_device_ep_h tl_ep)
{
}

template<ucs_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_check_completion(
        uct_device_ep_h tl_ep, uct_device_completion_t *tl_comp)
{
    uct_rc_gdaki_dev_ep_t *ep = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    uct_rc_gda_completion_t *comp = &tl_comp->rc_gda;
    unsigned cid                  = comp->channel_id;
    uint16_t wqe_cnt;
    uint8_t opcode;
    uint64_t pi;

    pi = uct_rc_mlx5_gda_parse_cqe(ep, cid, &wqe_cnt, &opcode);

    /* since first message wqe_idx is 0 and initial pi is -1
       we need to cast to signed */
    if ((int64_t)pi < (int64_t)comp->wqe_idx) {
        return UCS_INPROGRESS;
    }

    if (opcode == MLX5_CQE_REQ_ERR) {
        uint16_t wqe_idx = wqe_cnt & (ep->sq_wqe_num - 1);
        auto wqe_ptr     = uct_rc_mlx5_gda_get_wqe_ptr(ep, cid, wqe_idx);
        uct_rc_mlx5_gda_qedump("WQE", wqe_ptr, 64);
        uct_rc_mlx5_gda_qedump("CQE", ep->qps[cid].cq_buff, 64);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

#endif
