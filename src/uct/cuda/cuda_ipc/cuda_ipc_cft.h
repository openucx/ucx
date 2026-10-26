/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_CFT_H
#define UCT_CUDA_IPC_CFT_H

#include <ucs/config/types.h>
#include <ucs/type/status.h>
#include <cuda.h>


/*
 * CUDA fabric transport (CFT) support for cuda_ipc: memory registered with
 * cuda_ipc is bound into a per-device CUDA logical endpoint (LE) instead of
 * being exported with an IPC handle, and the LE fabric handle and offset are
 * carried in the rkey. Peers import the LE and access the memory only by
 * device operations, with fabric operations, since it cannot be mapped.
 *
 * The LE driver API is resolved at runtime, so UCX still loads with drivers
 * that do not provide it.
 */


/**
 * Check CFT support for a memory domain. The first call which does not
 * disable CFT resolves the driver API, and its @a le_size defines the size of
 * the logical endpoints of the process.
 *
 * @param [in]  le_size   Requested size of per-device logical endpoints.
 *
 * @return UCS_OK, or UCS_ERR_UNSUPPORTED if CFT is not supported.
 */
ucs_status_t uct_cuda_ipc_cft_init(size_t le_size);


/**
 * @return Nonzero if CFT is enabled by some memory domain of the process.
 */
int uct_cuda_ipc_cft_is_enabled(void);


#if HAVE_CUDA_LOGICAL_ENDPOINT

/**
 * Bind device memory allocation to the logical endpoint of its device.
 *
 * @param [in]  cu_dev      Device which owns the allocation.
 * @param [in]  d_bptr      Allocation base address.
 * @param [in]  b_len       Allocation length.
 * @param [out] handle      Fabric handle of the logical endpoint.
 * @param [out] le_offset_p Offset of the allocation in the logical endpoint.
 *
 * @return UCS_OK, UCS_ERR_UNSUPPORTED if the allocation cannot be bound.
 */
ucs_status_t uct_cuda_ipc_cft_bind(CUdevice cu_dev, CUdeviceptr d_bptr,
                                   size_t b_len,
                                   CUlogicalEndpointFabricHandle *handle,
                                   uint64_t *le_offset_p);


/**
 * Unbind memory bound by @ref uct_cuda_ipc_cft_bind.
 */
void uct_cuda_ipc_cft_unbind(CUdevice cu_dev, uint64_t le_offset,
                             size_t b_len);


/**
 * Get a local logical endpoint id for a fabric handle, importing it on first
 * use. Handles exported by this process resolve to the owner id.
 *
 * @param [in]  handle  Fabric handle of the logical endpoint.
 * @param [out] le_id_p Local logical endpoint id.
 */
ucs_status_t
uct_cuda_ipc_cft_import(const CUlogicalEndpointFabricHandle *handle,
                        uint32_t *le_id_p);

#endif /* HAVE_CUDA_LOGICAL_ENDPOINT */

#endif
