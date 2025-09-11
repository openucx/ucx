/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_kernels.h"

#include <ucp/api/device/ucp_device_impl.h>
#include <common/cuda.h>


namespace ucx_cuda {

static __global__ void memcmp_kernel(const void* s1, const void* s2,
                                     int* result, size_t size)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    *result = 0;
    for (size_t i = idx; i < size; i += blockDim.x * gridDim.x) {
        if (reinterpret_cast<const uint8_t*>(s1)[i]
            != reinterpret_cast<const uint8_t*>(s2)[i]) {
            *result = 1;
            break;
        }
    }
}

UCS_F_DEVICE ucs_status_t ucp_device_wait_req(ucp_device_request_t *req)
{
    ucs_status_t status;
    do {
        status = ucp_device_progress_req(req);
    } while (status == UCS_INPROGRESS);
    return status;
}

static __global__ void
ucp_put_single_kernel(const kernel_params params, ucs_status_t *status)
{
    ucp_device_request_t req;
    ucs_status_t req_status;

    req_status = ucp_device_put_single(params.mem_list,
                                       params.single.mem_list_index,
                                       params.single.address,
                                       params.single.remote_address,
                                       params.single.length,
                                       UCT_DEVICE_FLAG_NODELAY, &req);
    if (req_status != UCS_OK) {
        *status = req_status;
        return;
    }

    *status = ucp_device_wait_req(&req);
}

static __global__ void
ucp_put_multi_kernel(const kernel_params params, ucs_status_t *status)
{
    ucp_device_request_t req;
    ucs_status_t req_status;

    req_status = ucp_device_put_multi(params.mem_list, params.multi.addresses,
                                      params.multi.remote_addresses,
                                      params.multi.lengths,
                                      params.multi.counter_inc_value,
                                      params.multi.counter_remote_address,
                                      UCT_DEVICE_FLAG_NODELAY, &req);
    if (req_status != UCS_OK) {
        *status = req_status;
        return;
    }

    *status = ucp_device_wait_req(&req);
}

static __global__ void
ucp_put_multi_partial_kernel(const kernel_params params, ucs_status_t *status)
{
    ucp_device_request_t req;
    ucs_status_t req_status;

    req_status = ucp_device_put_multi_partial(
            params.mem_list, params.partial.mem_list_indices,
            params.partial.mem_list_count, params.partial.addresses,
            params.partial.remote_addresses, params.partial.lengths,
            params.partial.counter_index, params.partial.counter_inc_value,
            params.partial.counter_remote_address, UCT_DEVICE_FLAG_NODELAY,
            &req);
    if (req_status != UCS_OK) {
        *status = req_status;
        return;
    }

    *status = ucp_device_wait_req(&req);
}

static __global__ void
ucp_counter_inc_kernel(const kernel_params params, ucs_status_t *status)
{
    ucp_device_request_t req;
    ucs_status_t req_status;

    req_status = ucp_device_counter_inc(params.mem_list,
                                        params.counter.mem_list_index,
                                        params.counter.inc_value,
                                        params.counter.remote_address,
                                        UCT_DEVICE_FLAG_NODELAY, &req);
    if (req_status != UCS_OK) {
        *status = req_status;
        return;
    }

    *status = ucp_device_wait_req(&req);
}

/**
 * @brief Compares two blocks of device memory.
 *
 * Compares @a size bytes of the memory areas pointed to by @a s1 and @a s2,
 * which must both point to device memory.
 *
 * @param s1   Pointer to the first block of device memory.
 * @param s2   Pointer to the second block of device memory.
 * @param size Number of bytes to compare.
 *
 * @return int Returns 0 only if the memory blocks are equal.
 */
int launch_memcmp(const void *s1, const void *s2, size_t size)
{
    device_result_ptr<int> result = 0;

    memcmp_kernel<<<16, 64>>>(s1, s2, result.device_ptr(), size);
    synchronize();

    return *result;
}

class device_status_result_ptr : public device_result_ptr<ucs_status_t> {
public:
    device_status_result_ptr() :
        device_result_ptr<ucs_status_t>(UCS_ERR_NOT_IMPLEMENTED)
    {
    }

    ucs_status_t sync_read() const
    {
        synchronize();
        return device_result_ptr<ucs_status_t>::operator*();
    }
};

/**
 * Basic single element put operation.
 */
ucs_status_t launch_ucp_put_single(const kernel_params &params)
{
    device_status_result_ptr status;
    ucp_put_single_kernel<<<1, 1>>>(params, status.device_ptr());
    return status.sync_read();
}

ucs_status_t launch_ucp_put_multi(const kernel_params &params)
{
    device_status_result_ptr status;
    ucp_put_multi_kernel<<<1, 1>>>(params, status.device_ptr());
    return status.sync_read();
}

ucs_status_t launch_ucp_put_multi_partial(const kernel_params &params)
{
    device_status_result_ptr status;
    ucp_put_multi_partial_kernel<<<1, 1>>>(params, status.device_ptr());
    return status.sync_read();
}

ucs_status_t launch_ucp_counter_inc(const kernel_params &params)
{
    device_status_result_ptr status;
    ucp_counter_inc_kernel<<<1, 1>>>(params, status.device_ptr());
    return status.sync_read();
}

} // namespace ucx_cuda
