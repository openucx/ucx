/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_kernels.h"

#include <ucp/api/device/ucp_device_impl.h>
#include <ucs/debug/log.h>
#include <common/cuda.h>


template<ucs_device_level_t level>
ucs_status_t UCS_F_DEVICE
ucp_test_kernel_do_operation(const test_ucp_device_kernel_params_t &params,
                             uint64_t flags, ucp_device_request_t *req_ptr)
{
    ucs_status_t status;
    unsigned channel_id;

    if (level == UCS_DEVICE_LEVEL_THREAD) {
        channel_id = threadIdx.x % params.num_channels;
    } else if (level == UCS_DEVICE_LEVEL_WARP) {
        channel_id = (threadIdx.x / UCS_DEVICE_NUM_THREADS_IN_WARP) %
                     params.num_channels;
    }

    switch (params.operation) {
    case TEST_UCP_DEVICE_KERNEL_PUT_SINGLE:
        status = ucp_device_put_single<level>(params.mem_list,
                                              params.single.mem_list_index, 0,
                                              0, params.single.length,
                                              channel_id, flags, req_ptr);
        break;
    case TEST_UCP_DEVICE_KERNEL_PUT_MULTI:
        status = ucp_device_put_multi<level>(params.mem_list,
                                             params.multi.counter_inc_value,
                                             channel_id, flags, req_ptr);
        break;
    case TEST_UCP_DEVICE_KERNEL_PUT_MULTI_PARTIAL:
        status = ucp_device_put_multi_partial<level>(
                params.mem_list, params.partial.mem_list_indices,
                params.partial.mem_list_count,
                (size_t*)params.partial.local_offsets,
                (size_t*)params.partial.remote_offsets, params.partial.lengths,
                params.partial.counter_index, params.partial.counter_inc_value,
                params.partial.counter_remote_offset, channel_id, flags,
                req_ptr);
        break;
    case TEST_UCP_DEVICE_KERNEL_COUNTER_INC:
        status = ucp_device_counter_inc<level>(
                params.mem_list, params.counter_inc.mem_list_index,
                params.counter_inc.inc_value, 0, 0, flags, req_ptr);
        break;
    case TEST_UCP_DEVICE_KERNEL_COUNTER_WRITE:
        ucp_device_counter_write(params.local_counter.address,
                                 params.local_counter.value);
        /* req_ptr is not used in this case */
        return UCS_OK;
    case TEST_UCP_DEVICE_KERNEL_COUNTER_READ:
        uint64_t value = ucp_device_counter_read(params.local_counter.address);
        if (value != params.local_counter.value) {
            ucs_device_error("counter value mismatch: expected %lu, got %lu",
                             params.local_counter.value, value);
            return UCS_ERR_IO_ERROR;
        }
        /* req_ptr is not used in this case */
        return UCS_OK;
    }

    if (UCS_STATUS_IS_ERR(status)) {
        return status;
    }

    if (!(flags & UCT_DEVICE_FLAG_NODELAY) || (req_ptr == nullptr)) {
        return UCS_OK;
    }

    do {
        status = ucp_device_progress_req<level>(req_ptr);
    } while (status == UCS_INPROGRESS);
    return status;
}

template<ucs_device_level_t level> class device_request {
public:
    static constexpr size_t MAX_THREADS = 256;

    __device__ device_request(ucp_device_request_t *shared_reqs) :
        m_ptr(&shared_reqs[threadIdx.x / threads_per_req()])
    {
    }

    __device__ static constexpr size_t num_shared_reqs()
    {
        return MAX_THREADS / threads_per_req();
    }

    __device__ ucp_device_request_t *ptr() const
    {
        return m_ptr;
    }

private:
    __device__ static constexpr size_t threads_per_req()
    {
        switch (level) {
        case UCS_DEVICE_LEVEL_THREAD:
            return 1;
        case UCS_DEVICE_LEVEL_WARP:
            return UCS_DEVICE_NUM_THREADS_IN_WARP;
        default:
            return MAX_THREADS;
        }
    }

    ucp_device_request_t *m_ptr;
};

UCS_F_DEVICE ucs_status_t
ucp_test_kernel_get_state(const test_ucp_device_kernel_params_t &params,
                          test_ucp_device_kernel_result_t &result)
{
    uct_device_ep_t *device_ep;
    const uct_device_mem_element_t *uct_elem;
    uct_device_completion_t *comp;
    ucs_status_t status = UCS_OK;

    if (nullptr == params.mem_list) {
        return UCS_OK;
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        status = ucp_device_prepare_send(params.mem_list, 0, nullptr, device_ep,
                                         uct_elem, comp);
        result.producer_index = 0;
        result.ready_index    = 0;
        if ((status == UCS_OK) &&
            (device_ep->uct_tl_id == UCT_DEVICE_TL_RC_MLX5_GDA)) {
            uint16_t wqe_cnt;
            uct_rc_gdaki_dev_ep_t *ep =
                        reinterpret_cast<uct_rc_gdaki_dev_ep_t*>(device_ep);
            unsigned i;

            for (i = 0; i < params.num_channels; i++) {
                result.producer_index +=
                        uct_rc_mlx5_gda_parse_cqe(ep, i, &wqe_cnt, nullptr) + 1;
                result.ready_index += ep->qps[i].sq_ready_index;
            }
        }
    }

    __syncthreads();
    return status;
}

template<ucs_device_level_t level>
UCS_F_DEVICE void
ucp_test_kernel_job(const test_ucp_device_kernel_params_t &params,
                    test_ucp_device_kernel_result_t *result_ptr)
{
    ucs_status_t &status = result_ptr->status;

    if (blockDim.x > device_request<level>::MAX_THREADS) {
        ucs_device_error("blockDim.x > MAX_THREADS");
        status = UCS_ERR_INVALID_PARAM;
        return;
    }

    __shared__ ucp_device_request_t
            shared_reqs[device_request<level>::num_shared_reqs()];
    device_request<level> req(shared_reqs);

    status = ucp_test_kernel_get_state(params, *result_ptr);
    if (status != UCS_OK) {
        return;
    }

    ucp_device_request_t *req_ptr = params.with_request ? req.ptr() : nullptr;
    uint64_t flags                = params.with_no_delay ?
                                                UCT_DEVICE_FLAG_NODELAY : 0;

    for (size_t i = 0; i < params.num_iters - 1; i++) {
        status = ucp_test_kernel_do_operation<level>(params, flags, req_ptr);
        if (status != UCS_OK) {
            return;
        }
    }

    // Last iteration must use no-delay flag and request, to be able to wait
    // properly for completion. Alternatively, we could add a device flush
    // function to the API.
    status = ucp_test_kernel_do_operation<level>(params, UCT_DEVICE_FLAG_NODELAY,
                                                 req.ptr());
    if (status != UCS_OK) {
        return;
    }

    status = ucp_test_kernel_get_state(params, *result_ptr);
}

template<ucs_device_level_t level>
static __global__ void
ucp_test_kernel(const test_ucp_device_kernel_params_t params,
                test_ucp_device_kernel_result_t *result_ptr)
{
    ucp_test_kernel_job<level>(params, result_ptr);
    /* Execute fence on any return, to ensure result is visible to the host */
    __threadfence_system();
}

static ucs_status_t check_warp_size()
{
    CUdevice cuda_device;
    CUresult result;
    int warp_size;

    result = cuCtxGetDevice(&cuda_device);
    if (result != CUDA_SUCCESS) {
        ucs_error("cuCtxGetDevice failed: %d", result);
        return UCS_ERR_NO_DEVICE;
    }

    result = cuDeviceGetAttribute(&warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
                                  cuda_device);
    if (result != CUDA_SUCCESS) {
        ucs_error("cuDeviceGetAttribute failed: %d", result);
        return UCS_ERR_IO_ERROR;
    }

    if (UCS_DEVICE_NUM_THREADS_IN_WARP != warp_size) {
        ucs_error("Warp size mismatch: expected %d, got %d",
                  UCS_DEVICE_NUM_THREADS_IN_WARP, warp_size);
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

/**
 * Basic single element put operation.
 */
test_ucp_device_kernel_result_t
launch_test_ucp_device_kernel(const test_ucp_device_kernel_params_t &params)
{
    ucs_status_t check_status;

    check_status = check_warp_size();
    if (check_status != UCS_OK) {
        return {check_status};
    }

    ucx_cuda::device_result_ptr<test_ucp_device_kernel_result_t> result;
    result->status         = UCS_ERR_NOT_IMPLEMENTED;
    result->producer_index = 0;
    result->ready_index    = 0;

    switch (params.level) {
    case UCS_DEVICE_LEVEL_THREAD:
        ucp_test_kernel<UCS_DEVICE_LEVEL_THREAD>
                <<<params.num_blocks, params.num_threads>>>(
                        params, result.device_ptr());
        break;
    case UCS_DEVICE_LEVEL_WARP:
        ucp_test_kernel<UCS_DEVICE_LEVEL_WARP>
                <<<params.num_blocks, params.num_threads>>>(
                        params, result.device_ptr());
        break;
    default:
        return {UCS_ERR_INVALID_PARAM};
    }

    ucs_status_t sync_status = ucx_cuda::synchronize();
    if (sync_status != UCS_OK) {
        return {sync_status};
    }

    return *result;
}
