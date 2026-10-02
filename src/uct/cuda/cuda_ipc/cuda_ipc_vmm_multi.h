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
 * @brief multi-chunk VMM registration metadata
 */
typedef struct {
    ucs_list_link_t               link;          /* Entry in metadata list */
    CUdeviceptr                   d_bptr;        /* Base of all chunks */
    size_t                        b_len;         /* Length of all chunks */
    CUdeviceptr                   dev_ptr;       /* Metadata buffer VA */
    CUmemFabricHandle             fabric_handle; /* Metadata fabric handle */
    uct_cuda_ipc_vmm_multi_info_t info;          /* Inline metadata layout */
} uct_cuda_ipc_vmm_multi_meta_t;


/**
 * @brief Release all exporter metadata associated with a local key
 */
void uct_cuda_ipc_vmm_multi_meta_cleanup(uct_cuda_ipc_lkey_t *key);


/**
 * @brief Pack a VMM range spanning multiple physical allocations
 *
 * @param key      Local key whose allocation contains @a address
 * @param address  Requested range start
 * @param length   Requested range length
 * @param meta_p   Metadata record used to pack the key
 *
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED for a single allocation, or
 *         another error status on failure
 */
ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(
        uct_cuda_ipc_lkey_t *key, void *address, size_t length,
        const uct_cuda_ipc_vmm_multi_meta_t **meta_p);


#endif

#endif
