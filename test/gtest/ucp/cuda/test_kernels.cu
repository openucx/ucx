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

static __global__ void
ucp_put_single_kernel(ucp_device_mem_list_handle_h mem_list,
                      unsigned mem_list_index,
                      const void *address, uint64_t remote_address,
                      size_t length, ucs_status_t *status)
{
    ucp_device_request_t req;
    ucs_status_t req_status;

    ucp_device_request_init(&req);
    req_status = ucp_device_put_single(mem_list, mem_list_index,
                                       address, remote_address,
                                       length, 0, &req);
    if (req_status != UCS_OK) {
        *status = req_status;
        return;
    }

    do {
        req_status = ucp_device_progress_req(&req);
    } while (req_status == UCS_INPROGRESS);
    *status = req_status;
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

/**
 * Basic single element put operation.
 */
ucs_status_t launch_ucp_put_single(ucp_device_mem_list_handle_h mem_list,
                                   unsigned mem_list_index,
                                   const void *address, uint64_t remote_address,
                                   size_t length)
{
    device_result_ptr<ucs_status_t> status = UCS_ERR_NOT_IMPLEMENTED;

    ucp_put_single_kernel<<<1, 1>>>(mem_list, mem_list_index, address,
                                    remote_address, length,
                                    status.device_ptr());
    synchronize();

    return *status;
}

} // namespace ucx_cuda
