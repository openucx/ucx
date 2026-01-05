/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_INL
#define UCT_CUDA_IPC_INL

#include <uct/cuda/base/cuda_iface.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_md.h>
#include <uct/cuda/cuda_ipc/cuda_ipc_cache.h>

#include <cuda.h>

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_check_and_push_ctx(CUdeviceptr address, CUdevice *cuda_device_p,
                                int *is_ctx_pushed)
{
#define UCT_CUDA_IPC_NUM_ATTRS 2
    CUpointer_attribute attr_type[UCT_CUDA_IPC_NUM_ATTRS];
    void *attr_data[UCT_CUDA_IPC_NUM_ATTRS];
    CUcontext cuda_ctx, cuda_ctx_current;
    int cuda_device_ordinal;
    ucs_status_t status;
    CUdevice cuda_device;

    attr_type[0] = CU_POINTER_ATTRIBUTE_CONTEXT;
    attr_data[0] = &cuda_ctx;
    attr_type[1] = CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL;
    attr_data[1] = &cuda_device_ordinal;

    status = UCT_CUDADRV_FUNC_LOG_ERR(
            cuPointerGetAttributes(UCT_CUDA_IPC_NUM_ATTRS, attr_type, attr_data,
                                   address));
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ucs_assertv(cuda_device_ordinal >= 0, "cuda_device_ordinal=%d",
                cuda_device_ordinal);

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuDeviceGet(&cuda_device,
                                                  cuda_device_ordinal));
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    if (cuda_ctx == NULL) {
        status = uct_cuda_primary_ctx_retain(cuda_device, 0, &cuda_ctx);
        if (ucs_unlikely(status != UCS_OK)) {
           return status;
        }

        UCT_CUDADRV_FUNC_LOG_WARN(cuDevicePrimaryCtxRelease(cuda_device));
    }

    status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxGetCurrent(&cuda_ctx_current));
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    if (cuda_ctx != cuda_ctx_current) {
        status = UCT_CUDADRV_FUNC_LOG_ERR(cuCtxPushCurrent(cuda_ctx));
        if (ucs_unlikely(status != UCS_OK)) {
            return status;
        }

        *is_ctx_pushed = 1;
    } else {
        *is_ctx_pushed = 0;
    }

    *cuda_device_p = cuda_device;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
uct_cuda_ipc_check_and_pop_ctx(int is_ctx_pushed)
{
    if (is_ctx_pushed) {
        UCT_CUDADRV_FUNC_LOG_WARN(cuCtxPopCurrent(NULL));
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t
uct_cuda_ipc_get_remote_address(uct_cuda_ipc_rkey_t *rkey, uint64_t raddr,
                                CUdevice cu_dev, void **laddr_p,
                                void **base_addr_p)
{
    ucs_status_t status;
    ptrdiff_t offset;
    void *mapped_addr;

    status = uct_cuda_ipc_map_memhandle(rkey, cu_dev, &mapped_addr,
                                        UCS_LOG_LEVEL_ERROR);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    offset = UCS_PTR_BYTE_DIFF(rkey->d_bptr, raddr);
    ucs_assertv(offset <= rkey->b_len,
                "offset:%ld b_len:%lu", offset, rkey->b_len);
    *laddr_p = UCS_PTR_BYTE_OFFSET(mapped_addr, offset);
    if (base_addr_p != NULL) {
        *base_addr_p = mapped_addr;
    }

    return UCS_OK;
}

#endif
