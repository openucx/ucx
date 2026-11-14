/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCT_CUDA_IPC_VMM_MULTI_H
#define UCT_CUDA_IPC_VMM_MULTI_H

#include "cuda_ipc_md.h"

#include <ucs/datastruct/dynamic_bitmap.h>

#if HAVE_CUDA_FABRIC

/**
 * @brief Descriptor of one chunk in a multi-chunk VMM region
 */
typedef struct uct_cuda_ipc_vmm_chunk_desc {
    CUmemFabricHandle fabric_handle;
    CUdeviceptr       d_bptr;
    size_t            b_len;
} uct_cuda_ipc_vmm_chunk_desc_t;


/**
 * @brief Wire format version of the multi-chunk VMM metadata
 *
 * Bump when the meaning of the inline metadata information or chunk descriptor
 * fields changes.
 */
#define UCT_CUDA_IPC_VMM_MULTI_VERSION 1


/**
 * @brief Check whether the metadata format of a multi-chunk key is supported
 *
 * @param info  Inline metadata information of the key
 *
 * @return Nonzero if the version and handle type of @a info are supported
 */
static UCS_F_ALWAYS_INLINE int
uct_cuda_ipc_vmm_multi_is_supported(const uct_cuda_ipc_vmm_multi_info_t *info)
{
    return (info->version == UCT_CUDA_IPC_VMM_MULTI_VERSION) &&
           (info->handle_type == UCT_CUDA_IPC_VMM_MULTI_HANDLE_FABRIC);
}


/**
 * @brief multi-chunk VMM registration metadata
 *
 * The metadata buffer has a descriptor slot for every chunk of the registered
 * range. A slot is written when its chunk is first packed, and never changes
 * after that, so peers may read the slots of their keys at any time.
 */
struct uct_cuda_ipc_vmm_multi_meta {
    CUdeviceptr                   dev_ptr;       /* Metadata buffer VA */
    size_t                        alloc_size;    /* Metadata buffer size */
    /* Identity of the metadata buffer. Each publication allocates a new
     * buffer and CUDA never reuses an ID, so it names this exact layout. */
    unsigned long long            buffer_id;
    CUmemFabricHandle             fabric_handle; /* Metadata fabric handle */
    uint16_t                      num_chunks;    /* Registered chunks */
    uct_cuda_ipc_vmm_chunk_desc_t *chunks;       /* Host copy of the slots */
    ucs_dynamic_bitmap_t          exported;      /* Written slots */
};


/**
 * @brief Chunks of the metadata covered by one multi-chunk key
 */
typedef struct {
    CUdeviceptr                   d_bptr; /* Base of the covered chunks */
    size_t                        b_len;  /* Length of the covered chunks */
    uct_cuda_ipc_vmm_multi_info_t info;   /* Slots of the covered chunks */
} uct_cuda_ipc_vmm_multi_range_t;


/**
 * @brief Release the exporter metadata associated with a memory handle
 */
void uct_cuda_ipc_vmm_multi_meta_destroy(uct_cuda_ipc_memh_t *memh);


/**
 * @brief Pack a VMM range spanning multiple physical allocations
 *
 * The first call on @a memh publishes a metadata buffer with a slot for every
 * chunk of the registered range. Each call exports the chunks of the requested
 * range which were not packed before, and returns the chunk range covering the
 * request with its slots in the metadata buffer.
 *
 * @param memh     Memory handle registering the range
 * @param address  Requested range start
 * @param length   Requested range length
 * @param meta_p   Metadata record used to pack the key
 * @param range_p  Filled with the chunks and slots covering the request
 *
 * @return UCS_OK on success, UCS_ERR_UNSUPPORTED if the registered range is not
 *         fully mapped or a chunk of it does not allow fabric handles, or
 *         another error status on failure
 */
ucs_status_t uct_cuda_ipc_mkey_pack_vmm_multi_chunk(
        uct_cuda_ipc_memh_t *memh, void *address, size_t length,
        const uct_cuda_ipc_vmm_multi_meta_t **meta_p,
        uct_cuda_ipc_vmm_multi_range_t *range_p);


/**
 * @brief Fetch the chunk descriptors a peer published for a multi-chunk key
 *
 * @param key        Remote multi-chunk key
 * @param cu_dev     CUDA device where the metadata is imported
 * @param chunks_p   Filled with the key's descriptors in address order, which
 *                   cover the key range. Release with ucs_free().
 * @param log_level  Log level for failures
 *
 * @return UCS_OK on success, or an error status on failure
 */
ucs_status_t
uct_cuda_ipc_vmm_multi_fetch_chunks(const uct_cuda_ipc_rkey_t *key,
                                    CUdevice cu_dev,
                                    uct_cuda_ipc_vmm_chunk_desc_t **chunks_p,
                                    ucs_log_level_t log_level);


#endif

#endif
