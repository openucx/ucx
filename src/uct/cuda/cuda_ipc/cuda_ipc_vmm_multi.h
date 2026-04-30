/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_VMM_MULTI_H
#define UCT_CUDA_IPC_VMM_MULTI_H

#include "cuda_ipc_md.h"

#if HAVE_CUDA_FABRIC

/**
 * @brief tagged fabric handle for a single VMM chunk
 */
typedef struct {
    uct_cuda_ipc_key_handle_t handle_type;
    union {
        CUmemFabricHandle fabric;
    } handle;
} uct_cuda_ipc_vmm_handle_t;


/**
 * @brief descriptor of one chunk in a multi-chunk VMM region
 */
typedef struct uct_cuda_ipc_vmm_chunk_desc {
    uct_cuda_ipc_vmm_handle_t vmm_handle;
    CUdeviceptr               d_bptr;
    size_t                    b_len;
    unsigned long long        buffer_id;
} uct_cuda_ipc_vmm_chunk_desc_t;


/**
 * @brief header of the on-GPU metadata buffer published to peers
 */
typedef struct {
    uct_cuda_ipc_vmm_handle_t chunks_handle;
    uint16_t                  num_chunks;
} uct_cuda_ipc_vmm_meta_header_t;


void uct_cuda_ipc_vmm_multi_meta_cleanup(uct_cuda_ipc_lkey_t *key);

ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(uct_cuda_ipc_memh_t *memh,
                                                    uct_cuda_ipc_lkey_t *key,
                                                    void *address,
                                                    size_t length);

ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_chunks(uct_cuda_ipc_unpacked_rkey_t *rkey,
                                    CUdevice cu_dev, ucs_log_level_t log_level);

#endif

#endif
