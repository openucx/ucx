/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "test_kernels.h"

#include <uct/api/device/uct_device_impl.h>
#include <common/cuda.h>


namespace ucx_cuda {

static __global__ void
uct_put_kernel(uct_device_ep_h ep, const uct_device_mem_elem_t *src_elem,
               const uct_device_mem_elem_t *mem_elem, const void *va,
               uint64_t rva, size_t length, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    ucs_status_t status = uct_device_ep_put<UCS_DEVICE_LEVEL_THREAD>(
            ep, src_elem, mem_elem, va, rva, length, 0, UCT_DEVICE_FLAG_NODELAY,
            &comp);
    if (status != UCS_INPROGRESS) {
        *status_p = status;
        return;
    }

    while ((status = uct_device_ep_check_completion<UCS_DEVICE_LEVEL_THREAD>(
                    ep, &comp)) == UCS_INPROGRESS) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = status;
}

/**
 * Basic single element put operation.
 */
ucs_status_t launch_uct_put(uct_device_ep_h ep,
                            const uct_device_mem_elem_t *src_elem,
                            const uct_device_mem_elem_t *mem_elem,
                            const void *va, uint64_t rva, size_t length)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_put_kernel<<<1, 1>>>(ep, src_elem, mem_elem, va, rva, length,
                             status.device_ptr());
    synchronize();
    return *status;
}

static __global__ void
uct_atomic_kernel(uct_device_ep_h ep, uct_device_mem_elem_t *mem_elem,
                  uint64_t rva, uint64_t add, ucs_status_t *status_p)
{
    uct_device_completion_t comp;

    ucs_status_t status = uct_device_ep_atomic_add<UCS_DEVICE_LEVEL_THREAD>(
            ep, mem_elem, add, rva, 0, UCT_DEVICE_FLAG_NODELAY, &comp);
    if (status != UCS_INPROGRESS) {
        *status_p = status;
        return;
    }

    while ((status = uct_device_ep_check_completion<UCS_DEVICE_LEVEL_THREAD>(
                    ep, &comp)) == UCS_INPROGRESS) {
        uct_device_ep_progress<UCS_DEVICE_LEVEL_THREAD>(ep);
    }
    *status_p = status;
}

/**
 * Atomic operation.
 */
ucs_status_t launch_uct_atomic(uct_device_ep_h ep,
                               uct_device_mem_elem_t *mem_elem, uint64_t rva,
                               uint64_t add)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    uct_atomic_kernel<<<1, 1>>>(ep, mem_elem, rva, add, status.device_ptr());
    synchronize();
    return *status;
}

#if HAVE_GDA
template<ucs_device_level_t level>
static __global__ void gda_reserve_wqe_kernel(unsigned *errors_p)
{
    /* One queue per cooperative group; WARP uses exactly one full warp. */
    constexpr size_t storage_size = sizeof(uct_rc_gdaki_dev_ep_t) +
                                    sizeof(uct_rc_gdaki_dev_qp_t);
    __shared__ uint64_t storage[(storage_size + sizeof(uint64_t) - 1) /
                                 sizeof(uint64_t)];
    auto *ep            = reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(storage);
    const uint64_t base = UCS_BIT(32);
    unsigned lane_id, num_lanes;

    uct_dev_exec_init<level>(lane_id, num_lanes);

    if (threadIdx.x == 0) {
        ep->qps[0]               = {};
        ep->qps[0].sq_rsvd_index = base;
        ep->sq_wqe_num           = 2;
        memset(ep->qps[0].cq_buff, 0xff, sizeof(ep->qps[0].cq_buff));
    }

    __syncthreads();

    /* The driver's initial CQE counter (0xffff) represents base - 1.
     * Two queue entries allow base and base + 1, then no resource. */
    for (unsigned i = 0; i < 3; ++i) {
        uint64_t expected = (i < 2) ? base + i : UCT_RC_GDA_RESV_WQE_NO_RESOURCE;
        uint64_t wqe_base = ~expected;
        unsigned errors;

        uct_rc_mlx5_gda_reserv_wqe<level>(ep, 0, 1, lane_id, wqe_base);
        errors = __syncthreads_count(wqe_base != expected);
        if (threadIdx.x == 0) {
            *errors_p += errors;
        }
    }

    if (threadIdx.x == 0) {
        *errors_p += ep->qps[0].sq_rsvd_index != base + 2;
    }
}

ucs_status_t launch_gda_reserve_wqe(ucs_device_level_t level,
                                    unsigned num_threads, unsigned &num_errors)
{
    device_result_ptr<unsigned> errors = 0;
    ucs_status_t status;

    switch (level) {
    case UCS_DEVICE_LEVEL_THREAD:
        gda_reserve_wqe_kernel<UCS_DEVICE_LEVEL_THREAD>
                <<<1, 1>>>(errors.device_ptr());
        break;
    case UCS_DEVICE_LEVEL_WARP:
        gda_reserve_wqe_kernel<UCS_DEVICE_LEVEL_WARP>
                <<<1, num_threads>>>(errors.device_ptr());
        break;
    case UCS_DEVICE_LEVEL_BLOCK:
        gda_reserve_wqe_kernel<UCS_DEVICE_LEVEL_BLOCK>
                <<<1, num_threads>>>(errors.device_ptr());
        break;
    default:
        return UCS_ERR_UNSUPPORTED;
    }
    status = synchronize();
    if (status != UCS_OK) {
        return status;
    }

    num_errors = *errors;
    return UCS_OK;
}

static __global__ void gda_grid_unsupported_kernel(ucs_status_t *statuses)
{
    unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

    /* Unsupported levels must not access endpoint or completion state. */
    statuses[index] =
            uct_rc_mlx5_gda_ep_single<UCS_DEVICE_LEVEL_GRID>(
                    nullptr, nullptr, nullptr, 0, 0, 0, 0, 0, 0, nullptr,
                    MLX5_OPCODE_RDMA_WRITE, false, 0);
    statuses[gridDim.x * blockDim.x + index] =
            uct_ib_d2p_post_desc<UCS_DEVICE_LEVEL_GRID>(
                    nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

ucs_status_t launch_gda_grid_unsupported(unsigned &num_errors)
{
    constexpr unsigned num_blocks  = 2;
    constexpr unsigned num_threads = 33;
    std::vector<ucs_status_t> statuses(2 * num_blocks * num_threads,
                                      UCS_INPROGRESS);
    auto device_statuses = make_device_vector(statuses);

    gda_grid_unsupported_kernel<<<num_blocks, num_threads>>>(
            device_statuses.ptr());
    ucs_status_t status = synchronize();
    if (status != UCS_OK) {
        return status;
    }

    if (cudaMemcpy(statuses.data(), device_statuses.ptr(),
                   statuses.size() * sizeof(statuses[0]),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return UCS_ERR_IO_ERROR;
    }

    num_errors = 0;
    for (auto result : statuses) {
        num_errors += result != UCS_ERR_UNSUPPORTED;
    }
    return UCS_OK;
}

#endif

} // namespace ucx_cuda
