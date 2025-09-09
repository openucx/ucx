/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCT_GDAKI_H
#define UCT_GDAKI_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdint.h>

#include <cooperative_groups.h>

#include <doca_gpunetio_verbs_def.h>
#include <doca_gpunetio_dev_verbs_qp.cuh>

#include <uct/api/cuda/uct.h>

extern "C" {
#include "gdaki_ep_dev.h"
}

#define WARP_THREADS 32

__device__ static __forceinline__
uint64_t HtoBE64(uint64_t x) {
    uint64_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, 0x0123;\n\t"
        "prmt.b32 new_lo, hi, ign, 0x0123;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}" : "=l"(ret) : "l"(x));
    return ret;
}

__device__ static __forceinline__
uint32_t HtoBE32(uint32_t x) {
    uint32_t ret;
    asm("{\n\t"
        ".reg .b32 ign;\n\t"
        "prmt.b32 %0, %1, ign, 0x0123;\n\t"
        "}" : "=r"(ret) : "r"(x));
    return ret;
}

__device__ static inline uint16_t uct_gdaki_bswap16(uint16_t x)
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

__device__ static void uct_gdaki_qedump(const char *pfx, void *buff, ssize_t len)
{
    uint32_t *p = (uint32_t *)buff;
    size_t c;

    while (len > 0) {
        printf("%4s %#lx+%04lx:", pfx, (intptr_t)buff, p - (uint32_t *)buff);
        for (c = 0; c < 4; c++) {
            printf(" %08x", doca_gpu_dev_verbs_bswap32(p[c]));
        }
        printf("\n");
        p += 4;
        len -= 16;
    }
}

__device__ static inline ucs_status_t uct_gdaki_progress_thread(uct_gdaki_dev_ep_t *ep)
{
    struct doca_gpu_dev_verbs_qp *qp = ep->qp;
    struct doca_gpu_dev_verbs_cq *cq = &qp->cq_sq;

    void *cqe = cq->cqe_daddr;
    size_t cqe_num = cq->cqe_num;
    uint64_t cqe_idx = cq->cqe_ci;
    const size_t cqe_sz = DOCA_GPUNETIO_VERBS_CQE_SIZE;
    uint32_t idx = cqe_idx & (cqe_num - 1);
    void *curr_cqe = (uint8_t *)cqe + idx * cqe_sz;
    struct mlx5_cqe64 *cqe64 = (struct mlx5_cqe64 *)curr_cqe;
    uint8_t opown = READ_ONCE(cqe64->op_own);
    if ((opown & MLX5_CQE_OWNER_MASK) ^ !!(cqe_idx & cqe_num)) {
        return UCS_INPROGRESS;
    }

    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(cq->cqe_ci);
    if (ref.compare_exchange_strong(cqe_idx, cqe_idx + 1,
                                    cuda::std::memory_order_relaxed)) {
        uint8_t opcode = opown >> DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT;
        uint16_t wqe_cnt = uct_gdaki_bswap16(cqe64->wqe_counter);
        uint16_t wqe_idx = wqe_cnt & qp->sq_wqe_mask;

        if (opcode == MLX5_CQE_REQ_ERR) {
            struct mlx5_err_cqe_ex *err_cqe = (struct mlx5_err_cqe_ex *)cqe64;
            struct doca_gpu_dev_verbs_wqe *wqe_ptr;
            wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
            printf("thread%d:block%d: CQE[%d] with syndrom:%x vendor:%x hw:%x wqe_idx:0x%x qp:0x%x",
                   threadIdx.x, blockIdx.x, idx, err_cqe->syndrome, err_cqe->vendor_err_synd,
                   err_cqe->hw_err_synd, wqe_idx,
                   doca_gpu_dev_verbs_bswap32(err_cqe->s_wqe_opcode_qpn) & 0xffffff);
            uct_gdaki_qedump("wQE", wqe_ptr, 64);
            uct_gdaki_qedump("CQE", cqe64, 64);
            return UCS_ERR_IO_ERROR;
        }

        if (ep->ops[wqe_idx].comp != NULL) {
            ep->ops[wqe_idx].comp->count--; // TODO maybe atomic?
        }

        cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(qp->sq_wqe_pi);
        uint64_t sq_wqe_pi = qp->sq_wqe_pi;
        ref.fetch_max(((wqe_cnt - sq_wqe_pi) & 0xffff) + sq_wqe_pi + 1);

        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
    }

    return UCS_OK;
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t uct_gdaki_progress(uct_dev_ep_h tl_ep)
{
    uct_gdaki_dev_ep_t *ep = (uct_gdaki_dev_ep_t *)tl_ep;

    if (scale == UCT_DEV_SCALE_BLOCK) {
        __shared__ ucs_status_t status;

        if (threadIdx.x == 0) {
            status = uct_gdaki_progress_thread(ep);
        }

        __syncthreads();
        return status;
    } else if (scale == UCT_DEV_SCALE_WARP) {
        unsigned lane_id = doca_gpu_dev_verbs_get_lane_id();
        ucs_status_t status;

        if (lane_id == 0) {
            status = uct_gdaki_progress_thread(ep);
        }

        status = (ucs_status_t)__shfl_sync(0xffffffff, status, 0);
        __syncwarp();
        return status;
    } else if (scale == UCT_DEV_SCALE_THREAD) {
        return uct_gdaki_progress_thread(ep);
    } else {
        return UCS_ERR_UNSUPPORTED;
    }
}

__device__ __forceinline__ void st_na_relaxed(const int4 *ptr, int4 val) {
    asm volatile("st.relaxed.gpu.global.L1::no_allocate.v4.s32 [%0], {%1, %2, %3, %4};"
            : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
}

__device__ static inline void
uct_gdaki_wqe_prepare_put_or_atomic(struct doca_gpu_dev_verbs_qp *qp,
                                    struct doca_gpu_dev_verbs_wqe *wqe_ptr,
                                    const uint32_t wqe_idx,
                                    const uint32_t opcode,
                                    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrl_flags,
                                    const uint64_t raddr,
                                    const uint32_t rkey,
                                    const uint64_t laddr0,
                                    const uint32_t lkey0,
                                    const uint32_t bytes0,
                                    const int is_atomic,
                                    const uint64_t add)
{
    uint64_t *dseg_ptr = (uint64_t *)wqe_ptr + 4 + 2 * is_atomic;
    uint64_t *cseg_ptr = (uint64_t *)wqe_ptr;
    uint64_t *rseg_ptr = (uint64_t *)wqe_ptr + 2;
    uint64_t *atseg_ptr = (uint64_t *)wqe_ptr + 4;
    int ds = 3 + is_atomic;
    struct doca_gpu_dev_verbs_wqe_ctrl_seg cseg;
    struct mlx5_wqe_raddr_seg rseg;
    struct mlx5_wqe_data_seg dseg0;
    struct mlx5_wqe_atomic_seg atseg;

    cseg.opmod_idx_opcode = HtoBE32(
        ((uint32_t)wqe_idx << DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT) | opcode);
    cseg.qpn_ds = HtoBE32(qp->sq_num_shift8 | ds);
    cseg.fm_ce_se = ctrl_flags;

    rseg.raddr = HtoBE64(raddr);
    rseg.rkey = rkey;

    dseg0.byte_count = HtoBE32(bytes0);
    dseg0.lkey = lkey0;
    dseg0.addr = HtoBE64(laddr0);

    if (is_atomic) {
        atseg.swap_add = HtoBE64(add);
        doca_gpu_dev_verbs_store_wqe_seg(atseg_ptr, (uint64_t *)&(atseg));
    }

    st_na_relaxed(reinterpret_cast<int4*>(cseg_ptr), *reinterpret_cast<const int4*>(&cseg));
    st_na_relaxed(reinterpret_cast<int4*>(rseg_ptr), *reinterpret_cast<const int4*>(&rseg));
    st_na_relaxed(reinterpret_cast<int4*>(dseg_ptr), *reinterpret_cast<const int4*>(&dseg0));
}

template<bool res_ctrl>
__device__ static inline uint64_t
uct_gdaki_reserv_wqe_thread(struct doca_gpu_dev_verbs_qp *qp, unsigned count)
{
    uint64_t wqe_base = atomicAdd(reinterpret_cast<unsigned long long*>(&qp->sq_rsvd_index), static_cast<unsigned long long>(count));

    while (res_ctrl && (wqe_base + count > qp->sq_wqe_pi + qp->sq_wqe_num)) {
        uint64_t wqe_next = wqe_base + count;
        if (atomicCAS(reinterpret_cast<unsigned long long*>(&qp->sq_rsvd_index), wqe_next, wqe_base) == wqe_next) {
            return -1ULL;
        }
    }

    return wqe_base;
}

template<uct_dev_scale_t scale>
__device__ static inline void
uct_gdaki_reserv_wqe(struct doca_gpu_dev_verbs_qp *qp, unsigned count,
                     unsigned lane_id, uint64_t &wqe_base)
{
    if (lane_id == 0) {
        wqe_base = uct_gdaki_reserv_wqe_thread<true>(qp, count);
    }
    if (scale == UCT_DEV_SCALE_WARP) {
        wqe_base = __shfl_sync(0xffffffff, wqe_base, 0);
    } else if (scale == UCT_DEV_SCALE_BLOCK) {
        __syncthreads();
    }
}

__device__ static inline void
uct_gdaki_db(struct doca_gpu_dev_verbs_qp *qp, uint64_t wqe_base, unsigned count, uint64_t flags)
{
    cuda::atomic_ref<uint64_t, cuda::thread_scope_device> ref(qp->sq_ready_index);
    uint64_t wqe_base_orig = wqe_base;
    while (!ref.compare_exchange_strong(wqe_base, wqe_base + count,
                                        cuda::std::memory_order_relaxed)) {
        wqe_base = wqe_base_orig;
    }

    if (!(flags & UCT_DEV_BATCH_FLAG_NODELAY)) {
        return;
    }

    // TODO advance with atomicMAX, doorbell accordingly
    doca_gpu_dev_verbs_lock<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_lock);
    doca_gpu_dev_verbs_ring_db<
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>(qp, qp->sq_ready_index);
    doca_priv_gpu_dev_verbs_update_dbr<
        DOCA_GPUNETIO_VERBS_QP_SQ>(qp, qp->sq_ready_index);
    doca_gpu_dev_verbs_ring_db<
        DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU,
        DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT>(qp, qp->sq_ready_index);
    doca_gpu_dev_verbs_unlock<
        DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qp->sq_lock);
}

template<uct_dev_scale_t scale>
__device__ static inline void
uct_gdaki_exec_init(unsigned *lane_id, unsigned *num_lanes)
{
    switch (scale) {
    case UCT_DEV_SCALE_WARP:
        *lane_id = doca_gpu_dev_verbs_get_lane_id();
        *num_lanes = warpSize;
        break;
    case UCT_DEV_SCALE_BLOCK:
        *lane_id = threadIdx.x;
        *num_lanes = blockDim.x;
        break;
    case UCT_DEV_SCALE_THREAD:
        *lane_id = 0;
        *num_lanes = 1;
        break;
    /* TODO
    case UCT_DEV_SCALE_KERNEL:
        *lane_id = threadIdx.x + blockIdx.x * blockDim.x;
        *num_lanes = blockDim.x * gridDim.x;
        break; */
    }
}

template<uct_dev_scale_t scale>
__device__ static inline void
uct_gdaki_sync(void) {
    switch (scale) {
    case UCT_DEV_SCALE_WARP:
        __syncwarp();
        break;
    case UCT_DEV_SCALE_BLOCK:
        __syncthreads();
        break;
    case UCT_DEV_SCALE_THREAD:
        break;
    /* TODO
    case UCT_DEV_SCALE_KERNEL:
        auto g = cooperative_groups::this_grid();
        g.sync();
        break; */
    }
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t
uct_gdaki_put_batch(uct_gdaki_batch_t *batch, uint64_t flags,
                    int signal_inc, uct_dev_completion_t *comp) = delete;

__device__ static inline int
uct_gdaki_batch_has_atomic(const uct_gdaki_batch_t *batch)
{
    return batch->list[batch->num - 1].e_op == MLX5_OPCODE_ATOMIC_FA;
}

__device__ static inline int
uct_gdaki_batch_has_iov(const uct_gdaki_batch_t *batch)
{
    return batch->list[0].e_op != MLX5_OPCODE_ATOMIC_FA;
}

#if ENABLE_PARAMS_CHECK
__device__ static inline ucs_status_t
uct_gdaki_batch_params_check(const uct_gdaki_batch_t *batch, const uint64_t flags,
                             const int has_iov, const int has_atomic,
                             uct_dev_completion_t *comp)
{
    if ((flags & UCT_DEV_BATCH_FLAG_ATOMIC) && !has_atomic) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((flags & UCT_DEV_BATCH_FLAG_RMA_IOV) && !has_iov) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((flags & UCT_DEV_BATCH_FLAG_COMP) && (comp == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}
#endif

template<>
__device__ inline ucs_status_t
uct_gdaki_put_batch<UCT_DEV_SCALE_BLOCK>(uct_gdaki_batch_t *batch,
                                         uint64_t flags,
                                         int signal_inc,
                                         uct_dev_completion_t *comp)
{
    const int has_atomic = uct_gdaki_batch_has_atomic(batch);
    uct_gdaki_dev_ep_t *ep = batch->ep;
    struct doca_gpu_dev_verbs_qp *qp = ep->qp;
    int atomic = 0;
    __shared__ uint64_t wqe_base;
    int opcode;
    uint32_t wqe_idx;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = (doca_gpu_dev_verbs_wqe_ctrl_flags)0;
    size_t size;
    uint64_t src;
    uint32_t lkey;
    uint64_t dst;
    uint32_t rkey;
    uint32_t fc;
    unsigned lane_id;
    unsigned num_lanes;
#if ENABLE_PARAMS_CHECK
    const int has_iov = uct_gdaki_batch_has_iov(batch);
    ucs_status_t status;
#endif
    unsigned count = batch->num;

#if ENABLE_PARAMS_CHECK
    status = uct_gdaki_batch_params_check(batch, flags, has_iov, has_atomic, comp);
    if (status != UCS_OK) {
        return status;
    }
#endif

    if (has_atomic && !(flags & UCT_DEV_BATCH_FLAG_ATOMIC)) {
        count--;
    }

    uct_gdaki_exec_init<UCT_DEV_SCALE_BLOCK>(&lane_id, &num_lanes);
    uct_gdaki_reserv_wqe<UCT_DEV_SCALE_BLOCK>(qp, count, lane_id, wqe_base);
    if (wqe_base == -1ULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    fc = doca_gpu_dev_verbs_wqe_idx_inc_mask(qp->sq_wqe_pi, qp->sq_wqe_num / 2);
    wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_base, lane_id);
    for (uint32_t idx = lane_id; idx < count; idx += num_lanes) {
        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
        size    = batch->list[idx].size;
        src     = batch->list[idx].src;
        lkey    = batch->list[idx].lkey;
        dst     = batch->list[idx].dst;
        rkey    = batch->list[idx].rkey;
        opcode  = batch->list[idx].e_op;

        if ((idx == batch->num - 1) && (flags & UCT_DEV_BATCH_FLAG_ATOMIC)) {
            atomic = 1;
        }

        if (((flags & UCT_DEV_BATCH_FLAG_COMP) && (idx == count - 1)) ||
            (!(flags & UCT_DEV_BATCH_FLAG_COMP) && (wqe_idx == fc))) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            ep->ops[wqe_idx & qp->sq_wqe_mask].comp = comp;
        }

        uct_gdaki_wqe_prepare_put_or_atomic(qp, wqe_ptr, wqe_idx, opcode,
                                  cflag, dst, rkey, src, lkey, size, atomic, signal_inc);
        wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_idx, num_lanes);
    }

    uct_gdaki_sync<UCT_DEV_SCALE_BLOCK>();
    if (lane_id == 0) {
        uct_gdaki_db(qp, wqe_base, batch->num, flags);
    }

    return UCS_OK;
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t
uct_gdaki_batch_execute(uct_batch_h tl_batch, uint64_t flags,
                        uint64_t signal_inc, uct_dev_completion_t *comp)
{
    uct_gdaki_batch_t *batch = (uct_gdaki_batch_t *)tl_batch;

    return uct_gdaki_put_batch<scale>(batch, flags, signal_inc, comp);
}

template<uct_dev_scale_t scale, bool res_ctrl>
__device__ static inline ucs_status_t
uct_gdaki_put_batch_single(uct_gdaki_batch_t *batch, uint64_t flags,
                           const size_t src_off, const size_t dst_off,
                           const size_t size, uct_dev_completion_t *comp)
{
    uint64_t wqe_idx;
    uct_gdaki_dev_ep_t *ep = batch->ep;
    struct doca_gpu_dev_verbs_qp *qp = ep->qp;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = (doca_gpu_dev_verbs_wqe_ctrl_flags)0;
    unsigned lane_id;
    unsigned num_lanes;

    uct_gdaki_exec_init<scale>(&lane_id, &num_lanes);
    if (lane_id == 0) {
        wqe_idx = uct_gdaki_reserv_wqe_thread<res_ctrl>(qp, 1);
        uct_gdaki_wqe_prepare_put_or_atomic(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx, batch->list[0].e_op,
                cflag, batch->list[0].dst + dst_off, batch->list[0].rkey, batch->list[0].src + src_off, batch->list[0].lkey, size, 0, 0);
    }

    uct_gdaki_sync<scale>();

    if (lane_id == 0) {
        uct_gdaki_db(qp, wqe_idx, 1, flags);
    }

    uct_gdaki_sync<scale>();
    return UCS_OK;
}

template<uct_dev_scale_t scale, bool res_ctrl>
__device__ static inline ucs_status_t
uct_gdaki_batch_execute_single(uct_batch_t *tl_batch, uint64_t flags,
                           const size_t src_off, const size_t dst_off,
                           const size_t size, uct_dev_completion_t *comp)
{
    uct_gdaki_batch_t *batch = (uct_gdaki_batch_t *)tl_batch;
    return uct_gdaki_put_batch_single<scale, res_ctrl>(batch, flags, src_off, dst_off, size, comp);
}


template<uct_dev_scale_t scale, bool res_ctrl>
__device__ static inline ucs_status_t
uct_gdaki_atomic(uct_gdaki_batch_t *batch, uint64_t flags, int signal_inc, size_t dst_off,
		 uct_dev_completion_t *comp)
{
    uint64_t wqe_idx;
    uct_gdaki_dev_ep_t *ep = batch->ep;
    struct doca_gpu_dev_verbs_qp *qp = ep->qp;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = (doca_gpu_dev_verbs_wqe_ctrl_flags)0;
    unsigned lane_id;
    unsigned num_lanes;

    uct_gdaki_exec_init<scale>(&lane_id, &num_lanes);
    if (lane_id == 0) {
        wqe_idx = uct_gdaki_reserv_wqe_thread<res_ctrl>(qp, 1);
        uct_gdaki_wqe_prepare_put_or_atomic(qp, doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx), wqe_idx, batch->list[0].e_op,
                cflag, batch->list[0].dst + dst_off, batch->list[0].rkey, batch->list[0].src, batch->list[0].lkey, 8, 1, signal_inc);
    }

    uct_gdaki_sync<scale>();

    if (lane_id == 0) {
        uct_gdaki_db(qp, wqe_idx, 1, flags);
    }

    uct_gdaki_sync<scale>();
    return UCS_OK;
}

template<uct_dev_scale_t scale, bool res_ctrl>
__device__ static inline ucs_status_t
uct_gdaki_batch_execute_atomic(uct_batch_t *tl_batch, uint64_t flags,
                           const int signal_inc, const size_t dst_off,
                           uct_dev_completion_t *comp)
{
    uct_gdaki_batch_t *batch = (uct_gdaki_batch_t *)tl_batch;
    return uct_gdaki_atomic<scale, res_ctrl>(batch, flags, signal_inc, dst_off, comp);
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t
uct_gdaki_put_batch_part_impl(uct_gdaki_batch_t *batch, uint64_t flags,
                              int signal_inc, size_t put_count, const int *indices,
                              const size_t *src_offs, const size_t *dst_offs,
                              const size_t *sizes, uct_dev_completion_t *comp,
                              uint64_t &wqe_base)
{
    uct_gdaki_dev_ep_t *ep = batch->ep;
    struct doca_gpu_dev_verbs_qp *qp = ep->qp;
    int atomic = 0;
    uint64_t wqe_idx;
    struct doca_gpu_dev_verbs_wqe *wqe_ptr;
    enum doca_gpu_dev_verbs_wqe_ctrl_flags cflag = (doca_gpu_dev_verbs_wqe_ctrl_flags)0;
    size_t size;
    uint64_t src;
    uint32_t lkey;
    uint64_t dst;
    uint32_t rkey;
    int opcode;
    uint32_t fc;
    unsigned lane_id;
    unsigned num_lanes;
#if ENABLE_PARAMS_CHECK
    const int has_atomic = uct_gdaki_batch_has_atomic(batch);
    const int has_iov = put_count > 0;
    ucs_status_t status;
#endif
    unsigned count = put_count;

#if ENABLE_PARAMS_CHECK
    status = uct_gdaki_batch_params_check(batch, flags, has_iov, has_atomic, comp);
    if (status != UCS_OK) {
        return status;
    }
#endif

    if (flags & UCT_DEV_BATCH_FLAG_ATOMIC) {
        count++;
    }

    uct_gdaki_exec_init<scale>(&lane_id, &num_lanes);
    uct_gdaki_reserv_wqe<scale>(qp, count, lane_id, wqe_base);
    if (wqe_base == -1ULL) {
        return UCS_ERR_NO_RESOURCE;
    }

    fc = doca_gpu_dev_verbs_wqe_idx_inc_mask(qp->sq_wqe_pi, qp->sq_wqe_num / 2);
    wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_base, lane_id);
    for (uint32_t i = lane_id; i < count; i += num_lanes) {
        uint32_t idx;
        size_t src_off, dst_off;

        if (i == put_count) {
            idx = batch->num - 1;
            atomic = 1;
            src_off = 0;
            dst_off = 0;
            size = 8;
        } else if (i < put_count) {
            idx = indices[i];
            src_off = src_offs[i];
            dst_off = dst_offs[i];
            size    = sizes[i];
        } else {
            continue;
        }

        if (((flags & UCT_DEV_BATCH_FLAG_COMP) && (i == count - 1)) ||
            (!(flags & UCT_DEV_BATCH_FLAG_COMP) && (wqe_idx == fc))) {
            cflag = DOCA_GPUNETIO_MLX5_WQE_CTRL_CQ_UPDATE;
            ep->ops[wqe_idx & qp->sq_wqe_mask].comp = comp;
        }

        wqe_ptr = doca_gpu_dev_verbs_get_wqe_ptr(qp, wqe_idx);
        src     = batch->list[idx].src + src_off;
        lkey    = batch->list[idx].lkey;
        dst     = batch->list[idx].dst + dst_off;
        rkey    = batch->list[idx].rkey;
        opcode  = batch->list[idx].e_op;
        uct_gdaki_wqe_prepare_put_or_atomic(qp, wqe_ptr, wqe_idx, opcode,
                                  cflag, dst, rkey, src, lkey, size, atomic, signal_inc);
        wqe_idx = doca_gpu_dev_verbs_wqe_idx_inc_mask(wqe_idx, num_lanes);
    }

    uct_gdaki_sync<scale>();
    if (lane_id == 0) {
        uct_gdaki_db(qp, wqe_base, count, flags);
    }

    uct_gdaki_sync<scale>();
    return UCS_OK;
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t
uct_gdaki_put_batch_part(uct_gdaki_batch_t *batch, uint64_t flags,
        int signal_inc, size_t count, const int *indices, const size_t *src_offs,
        const size_t *dst_offs, const size_t *sizes, uct_dev_completion_t *comp) = delete;

template<>
__device__ inline ucs_status_t
uct_gdaki_put_batch_part<UCT_DEV_SCALE_THREAD>(uct_gdaki_batch_t *batch,
        uint64_t flags, int signal_inc, size_t count,
        const int *indices, const size_t *src_offs, const size_t *dst_offs,
        const size_t *sizes, uct_dev_completion_t *comp)
{
    uint64_t wqe_base;
    return uct_gdaki_put_batch_part_impl<UCT_DEV_SCALE_THREAD>(batch, flags,
            signal_inc, count, indices, src_offs, dst_offs, sizes, comp,
            wqe_base);
}

template<>
__device__ inline ucs_status_t
uct_gdaki_put_batch_part<UCT_DEV_SCALE_WARP>(uct_gdaki_batch_t *batch,
        uint64_t flags, int signal_inc, size_t count,
        const int *indices, const size_t *src_offs, const size_t *dst_offs,
        const size_t *sizes, uct_dev_completion_t *comp)
{
    uint64_t wqe_base;
    return uct_gdaki_put_batch_part_impl<UCT_DEV_SCALE_WARP>(batch, flags,
            signal_inc, count, indices, src_offs, dst_offs, sizes, comp,
            wqe_base);
}

template<>
__device__ inline ucs_status_t
uct_gdaki_put_batch_part<UCT_DEV_SCALE_BLOCK>(uct_gdaki_batch_t *batch,
        uint64_t flags, int signal_inc, size_t count,
        const int *indices, const size_t *src_offs, const size_t *dst_offs,
        const size_t *sizes, uct_dev_completion_t *comp)
{
    __shared__ uint64_t wqe_base;
    return uct_gdaki_put_batch_part_impl<UCT_DEV_SCALE_BLOCK>(batch, flags,
            signal_inc, count, indices, src_offs, dst_offs, sizes, comp,
            wqe_base);
}

template<uct_dev_scale_t scale>
__device__ static inline ucs_status_t
uct_gdaki_batch_execute_part(uct_batch_h tl_batch, uint64_t flags,
                             uint64_t signal_inc, size_t count,
                             const int *indices, const size_t *src_offs,
                             const size_t *dst_offs, size_t *sizes,
                             uct_dev_completion_t *comp)
{
    uct_gdaki_batch_t *batch = (uct_gdaki_batch_t *)tl_batch;

    return uct_gdaki_put_batch_part<scale>(batch, flags, signal_inc, count,
                                           indices, src_offs, dst_offs, sizes,
                                           comp);
}

#endif /* UCT_GDAKI_H */
