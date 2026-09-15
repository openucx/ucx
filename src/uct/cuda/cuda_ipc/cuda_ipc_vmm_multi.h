/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_VMM_MULTI_H
#define UCT_CUDA_IPC_VMM_MULTI_H

#include "cuda_ipc_md.h"

#if HAVE_CUDA_FABRIC

/**
 * @brief Descriptor of one chunk in a multi-chunk VMM region
 */
typedef struct uct_cuda_ipc_vmm_chunk_desc {
    CUmemFabricHandle fabric_handle;
    CUdeviceptr       d_bptr;
    size_t            b_len;
    /* Identity of the backing allocation. Remote addresses are recycled, so
     * this is what distinguishes "the same chunk" from "a different allocation
     * that happens to sit at the same address". */
    uint64_t          buffer_id;
} uct_cuda_ipc_vmm_chunk_desc_t;


/**
 * @brief Wire format version of the multi-chunk VMM metadata
 *
 * Bump when the meaning of the inline metadata information or chunk descriptor
 * fields changes.
 */
#define UCT_CUDA_IPC_VMM_MULTI_VERSION 1


/**
 * @brief Release all exporter metadata associated with a local key
 */
void uct_cuda_ipc_vmm_multi_meta_cleanup(uct_cuda_ipc_lkey_t *key);


/**
 * @brief Pack a VMM range spanning multiple physical allocations
 *
 * @param memh     CUDA IPC registration handle
 * @param key      Local key containing the first allocation
 * @param address  Requested range start
 * @param length   Requested range length
 * @param meta_p   Metadata record used to pack the key
 *
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED for a single allocation, or
 *         another error status on failure
 */
ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(
        uct_cuda_ipc_memh_t *memh, uct_cuda_ipc_lkey_t *key, void *address,
        size_t length, const uct_cuda_ipc_vmm_multi_meta_t **meta_p);


/**
 * @brief Import multichunk descriptors from exporter metadata
 *
 * @param rkey       Unpacked CUDA IPC remote key
 * @param cu_dev     CUDA device where metadata is imported
 * @param log_level  Log level for CUDA failures
 *
 * @return UCS_OK on success, or an error status on failure
 */
ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_chunks(uct_cuda_ipc_unpacked_rkey_t *rkey,
                                    CUdevice cu_dev, ucs_log_level_t log_level);

#endif

#endif
