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

constexpr unsigned uct_rc_mlx5_gda_warp_size = 32;

template<uct_device_level_t level>
UCS_F_DEVICE void
uct_rc_mlx5_gda_exec_init(unsigned &lane_id, unsigned &num_lanes)
{
    switch (level) {
    case UCT_DEVICE_LEVEL_THREAD:
        lane_id   = 0;
        num_lanes = 1;
        break;
    case UCT_DEVICE_LEVEL_WARP:
        lane_id   = doca_gpu_dev_verbs_get_lane_id();
        num_lanes = uct_rc_mlx5_gda_warp_size;
        break;
    case UCT_DEVICE_LEVEL_BLOCK:
        lane_id   = threadIdx.x;
        num_lanes = blockDim.x;
        break;
    case UCT_DEVICE_LEVEL_GRID:
        lane_id   = threadIdx.x + blockIdx.x * blockDim.x;
        num_lanes = blockDim.x * gridDim.x;
        break;
    }
}

template<uct_device_level_t level> UCS_F_DEVICE void uct_rc_mlx5_gda_sync(void)
{
    switch (level) {
    case UCT_DEVICE_LEVEL_WARP:
        __syncwarp();
        break;
    case UCT_DEVICE_LEVEL_BLOCK:
        __syncthreads();
        break;
    case UCT_DEVICE_LEVEL_THREAD:
        break;
    case UCT_DEVICE_LEVEL_GRID:
        auto g = cooperative_groups::this_grid();
        g.sync();
    }
}

UCS_F_DEVICE uint64_t uct_rc_mlx5_gda_reserv_wqe_thread(
        struct doca_gpu_dev_verbs_qp *qp, unsigned count)
{
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
     *    -1ULL to signal insufficient resources.
     *  - If CAS fails, it means another thread has modified the reserved index.
     *    The loop continues to reevaluate resource availability to determine if
     *    the reservation can now be satisfied, possibly due to other operations
     *    freeing up resources.
     */
    while (wqe_base + count > qp->sq_wqe_pi + qp->sq_wqe_num) {
        uint64_t wqe_next = wqe_base + count;
        if (atomicCAS(reinterpret_cast<unsigned long long*>(&qp->sq_rsvd_index),
                      wqe_next, wqe_base) == wqe_next) {
            return -1ULL;
        }
    }

    return wqe_base;
}

template<uct_device_level_t level>
UCS_F_DEVICE void
uct_rc_mlx5_gda_reserv_wqe(struct doca_gpu_dev_verbs_qp *qp, unsigned count,
                           unsigned lane_id, uint64_t &wqe_base)
{
    if (lane_id == 0) {
        wqe_base = uct_rc_mlx5_gda_reserv_wqe_thread(qp, count);
    }
    if (level == UCT_DEVICE_LEVEL_WARP) {
        wqe_base = __shfl_sync(0xffffffff, wqe_base, 0);
    } else if (level == UCT_DEVICE_LEVEL_BLOCK) {
        __syncthreads();
    }
}

UCS_F_DEVICE void uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
        doca_gpu_dev_verbs_qp *qp, doca_gpu_dev_verbs_wqe *wqe_ptr,
        uint32_t wqe_idx, uint32_t opcode, unsigned ctrl_flags, uint64_t raddr,
        uint32_t rkey, uint64_t laddr, uint32_t lkey, uint32_t bytes,
        bool is_atomic, uint64_t add)
{
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
    cseg.qpn_ds           = doca_gpu_dev_verbs_bswap32(qp->sq_num_shift8 | ds);
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

UCS_F_DEVICE void uct_rc_mlx5_gda_db(doca_gpu_dev_verbs_qp *qp,
                                     uint64_t wqe_base, unsigned count,
                                     uint64_t flags)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(
            qp->sq_ready_index);
    uint64_t wqe_base_orig = wqe_base;
    while (!ref.compare_exchange_strong(wqe_base, wqe_base + count,
                                        cuda::std::memory_order_relaxed)) {
        wqe_base = wqe_base_orig;
    }

    if (!(flags & UCT_DEVICE_FLAG_NODELAY)) {
        return;
    }

    doca_gpu_dev_verbs_lock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->sq_lock);
    doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                               DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>(
            qp, qp->sq_ready_index);
    doca_priv_gpu_dev_verbs_update_dbr<DOCA_GPUNETIO_VERBS_QP_SQ>(
            qp, qp->sq_ready_index);
    doca_gpu_dev_verbs_ring_db<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
                               DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>(
            qp, qp->sq_ready_index);
    doca_gpu_dev_verbs_unlock<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
            &qp->sq_lock);
}

template<uct_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_single(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_elem,
        const void *address, uint64_t remote_address, size_t length,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep       = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    auto mem_elem = reinterpret_cast<const uct_rc_gdaki_device_mem_element_t*>(
            tl_mem_elem);
    doca_gpu_dev_verbs_qp *qp = ep->qp;
    unsigned cflag            = 0;
    uint64_t wqe_idx;
    unsigned lane_id;
    unsigned num_lanes;
    uint32_t fc;

    uct_rc_mlx5_gda_exec_init<level>(lane_id, num_lanes);
    uct_rc_mlx5_gda_reserv_wqe<level>(qp, 1, lane_id, wqe_idx);
    if (wqe_idx == -1ULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    fc = doca_gpu_dev_verbs_wqe_idx_inc_mask(qp->sq_wqe_pi, qp->sq_wqe_num / 2);
    if (lane_id == 0) {
        if ((comp != nullptr) || (wqe_idx == fc)) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            ep->ops[wqe_idx & qp->sq_wqe_mask].comp = comp;
        }

        uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
                qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx,
                MLX5_OPCODE_RDMA_WRITE, cflag, remote_address, mem_elem->rkey,
                reinterpret_cast<uint64_t>(address), mem_elem->lkey, length, 0,
                0);
    }

    uct_rc_mlx5_gda_sync<level>();

    if (lane_id == 0) {
        uct_rc_mlx5_gda_db(qp, wqe_idx, 1, flags);
    }

    uct_rc_mlx5_gda_sync<level>();
    return UCS_OK;
}

template<uct_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_put_multi_partial(
        uct_device_ep_h tl_ep, const uct_device_mem_element_t *tl_mem_list,
        const unsigned *mem_list_indices, unsigned mem_list_count,
        void *const *addresses, const uint64_t *remote_addresses,
        const size_t *lengths, unsigned counter_index,
        uint64_t counter_inc_value, uint64_t counter_remote_address,
        uint64_t flags, uct_device_completion_t *comp)
{
    auto ep       = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(tl_ep);
    auto mem_list = reinterpret_cast<const uct_rc_gdaki_device_mem_element_t*>(
            tl_mem_list);
    doca_gpu_dev_verbs_qp *qp = ep->qp;
    unsigned cflag            = 0;
    unsigned count            = mem_list_count;
    int atomic                = 0;
    uint64_t wqe_idx;
    unsigned lane_id;
    unsigned num_lanes;
    uint32_t fc;
    uint64_t wqe_base;
    size_t length;
    void *address;
    uint32_t lkey;
    uint64_t remote_address;
    uint32_t rkey;
    int opcode;
    uint32_t idx;

    if ((level != UCT_DEVICE_LEVEL_THREAD) &&
        (level != UCT_DEVICE_LEVEL_WARP)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if (counter_remote_address != 0) {
        count++;
    }

    uct_rc_mlx5_gda_exec_init<level>(lane_id, num_lanes);
    uct_rc_mlx5_gda_reserv_wqe<level>(qp, count, lane_id, wqe_base);
    if (wqe_base == -1ULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    fc = doca_gpu_dev_verbs_wqe_idx_inc_mask(qp->sq_wqe_pi, qp->sq_wqe_num / 2);
    wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_base, lane_id);
    for (uint32_t i = lane_id; i < count; i += num_lanes) {
        if (i == mem_list_count) {
            idx            = counter_index;
            atomic         = 1;
            address        = ep->atomic_va;
            remote_address = counter_remote_address;
            length         = 8;
            opcode         = MLX5_OPCODE_ATOMIC_FA;
        } else if (i < mem_list_count) {
            idx            = mem_list_indices[i];
            address        = addresses[idx];
            remote_address = remote_addresses[idx];
            length         = lengths[idx];
            opcode         = MLX5_OPCODE_RDMA_WRITE;
        } else {
            continue;
        }

        if (((comp != nullptr) && (i == count - 1)) ||
            ((comp == nullptr) && (wqe_idx == fc))) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            ep->ops[wqe_idx & qp->sq_wqe_mask].comp = comp;
        }

        auto wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
        lkey         = mem_list[idx].lkey;
        rkey         = mem_list[idx].rkey;

        uct_rc_mlx5_gda_wqe_prepare_put_or_atomic(
                qp, wqe_ptr, wqe_idx, opcode, cflag, remote_address, rkey,
                reinterpret_cast<uint64_t>(address), lkey, length, atomic,
                counter_inc_value);
        wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_idx, num_lanes);
    }

    uct_rc_mlx5_gda_sync<level>();

    if (lane_id == 0) {
        uct_rc_mlx5_gda_db(qp, wqe_base, count, flags);
    }

    uct_rc_mlx5_gda_sync<level>();
    return UCS_OK;
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

UCS_F_DEVICE ucs_status_t
uct_rc_mlx5_gda_progress_thread(uct_rc_gdaki_dev_ep_t *ep)
{
    doca_gpu_dev_verbs_qp *qp = ep->qp;
    doca_gpu_dev_verbs_cq *cq = &qp->cq_sq;

    void *cqe                = cq->cqe_daddr;
    size_t cqe_num           = cq->cqe_num;
    uint64_t cqe_idx         = cq->cqe_ci;
    const size_t cqe_sz      = DOCA_GPUNETIO_VERBS_CQE_SIZE;
    uint32_t idx             = cqe_idx & (cqe_num - 1);
    void *curr_cqe           = (uint8_t*)cqe + idx * cqe_sz;
    auto *cqe64              = reinterpret_cast<mlx5_cqe64*>(curr_cqe);
    uint8_t op_owner;

    op_owner = READ_ONCE(cqe64->op_own);
    if ((op_owner & MLX5_CQE_OWNER_MASK) ^ !!(cqe_idx & cqe_num)) {
        return UCS_INPROGRESS;
    }

    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(cq->cqe_ci);
    if (!ref.compare_exchange_strong(cqe_idx, cqe_idx + 1,
                                     cuda::std::memory_order_relaxed)) {
        return UCS_OK;
    }

    uint8_t opcode   = op_owner >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
    uint16_t wqe_cnt = uct_rc_mlx5_gda_bswap16(cqe64->wqe_counter);
    uint16_t wqe_idx = wqe_cnt & qp->sq_wqe_mask;

    if (opcode == MLX5_CQE_REQ_ERR) {
        auto err_cqe = reinterpret_cast<mlx5_err_cqe_ex*>(cqe64);
        auto wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
        printf("thread%d:block%d: CQE[%d] with syndrome:%x vendor:%x hw:%x "
               "wqe_idx:0x%x qp:0x%x\n",
               threadIdx.x, blockIdx.x, idx, err_cqe->syndrome,
               err_cqe->vendor_err_synd, err_cqe->hw_err_synd, wqe_idx,
               doca_gpu_dev_verbs_bswap32(err_cqe->s_wqe_opcode_qpn) &
                       0xffffff);
        uct_rc_mlx5_gda_qedump("wQE", wqe_ptr, 64);
        uct_rc_mlx5_gda_qedump("CQE", cqe64, 64);
        return UCS_ERR_IO_ERROR;
    }

    if (ep->ops[wqe_idx].comp != nullptr) {
        ep->ops[wqe_idx].comp->count--; // TODO maybe atomic?
    }

    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> pi_ref(qp->sq_wqe_pi);
    uint64_t sq_wqe_pi = qp->sq_wqe_pi;
    pi_ref.fetch_max(((wqe_cnt - sq_wqe_pi) & 0xffff) + sq_wqe_pi + 1);

    doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
    return UCS_OK;
}

template<uct_device_level_t level>
UCS_F_DEVICE ucs_status_t uct_rc_mlx5_gda_ep_progress(uct_device_ep_h tl_ep)
{
    uct_rc_gdaki_dev_ep_t *ep = (uct_rc_gdaki_dev_ep_t*)tl_ep;

    if (level == UCT_DEVICE_LEVEL_BLOCK) {
        __shared__ ucs_status_t status;

        if (threadIdx.x == 0) {
            status = uct_rc_mlx5_gda_progress_thread(ep);
        }

        __syncthreads();
        return status;
    } else if (level == UCT_DEVICE_LEVEL_WARP) {
        unsigned lane_id = doca_gpu_dev_verbs_get_lane_id();
        ucs_status_t status;

        if (lane_id == 0) {
            status = uct_rc_mlx5_gda_progress_thread(ep);
        }

        status = (ucs_status_t)__shfl_sync(0xffffffff, status, 0);
        __syncwarp();
        return status;
    } else if (level == UCT_DEVICE_LEVEL_THREAD) {
        return uct_rc_mlx5_gda_progress_thread(ep);
    } else {
        return UCS_ERR_UNSUPPORTED;
    }
}

#endif
