/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */


#ifndef UCP_MM_H_
#define UCP_MM_H_

#include <ucp/api/ucp_def.h>
#include <ucp/core/ucp_ep.h>
#include <uct/api/uct.h>
#include <ucs/arch/bitops.h>
#include <ucs/debug/log.h>
#include <ucs/datastruct/mpool.h>

#include <inttypes.h>


/**
 * Memory handle.
 * Contains general information, and a list of UCT handles.
 * md_map specifies which MDs from the current context are present in the array.
 * The array itself contains only the MDs specified in md_map, without gaps.
 */
typedef struct ucp_mem {
    void                          *address;     /* Region start address */
    size_t                        length;       /* Region length */
    uct_alloc_method_t            alloc_method; /* Method used to allocate the memory */
    ucs_memory_type_t             mem_type;     /**< type of allocated memory */
    uct_md_h                      alloc_md;     /* MD used to allocated the memory */
    ucp_md_map_t                  md_map;       /* Which MDs have valid memory handles */
    uct_mem_h                     uct[0];       /* Valid memory handles, as popcount(md_map) */
} ucp_mem_t;


/**
 * Memory descriptor.
 * Contains a memory handle of the chunk it belongs to.
 */
typedef struct ucp_mem_desc {
    ucp_mem_h                     memh;
} ucp_mem_desc_t;


ucs_status_t ucp_reg_mpool_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);

void ucp_reg_mpool_free(ucs_mpool_t *mp, void *chunk);

void ucp_mpool_obj_init(ucs_mpool_t *mp, void *obj, void *chunk);

ucs_status_t ucp_frag_mpool_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);

void ucp_frag_mpool_free(ucs_mpool_t *mp, void *chunk);

/**
 * Update memory registration to a specified set of memory domains.
 *
 * @param [in] context     UCP context with MDs to use for registration.
 * @param [in] reg_md_map  Map of memory domains to update the registration to.
 *                         MDs which are present in reg_md_map, but not yet
 *                         registered will be registered.
 *                         MDs which were registered, but not present in r
 *                         eg_md_map, will be de-registered.
 * @param [in] address     Address to register, unused if reg_md_map == 0
 * @param [in] length      Length to register, unused if reg_md_map == 0
 * @param [in] uct_flags   Flags for UCT registration, unused if reg_md_map == 0
 * @param [in] alloc_md    If != NULL, MD that was used to register the memory.
 *                         This MD will not be used to register the memory again;
 *                         rather, the memh will be taken from *alloc_md_memh.
 * @param [inout] alloc_md_memh_p  If non-NULL, specifies/filled with the memory
 *                                 handle on alloc_md.
 * @param [inout] uct_memh Array of memory handles to update.
 * @param [inout] md_map_p Current map of registered MDs, updated by the function
 *                         to the new map o
 *
 * In case alloc_md != NULL, alloc_md_memh will hold the memory key obtained from
 * allocation. It will be put in the array of keys in the proper index.
 */
ucs_status_t ucp_mem_rereg_mds(ucp_context_h context, ucp_md_map_t reg_md_map,
                               void *address, size_t length, unsigned uct_flags,
                               uct_md_h alloc_md, ucs_memory_type_t mem_type,
                               uct_mem_h *alloc_md_memh_p, uct_mem_h *uct_memh,
                               ucp_md_map_t *md_map_p);

ucs_status_t ucp_mem_type_reg_buffers(ucp_worker_h worker, void *remote_addr,
                                      size_t length, ucs_memory_type_t mem_type,
                                      ucp_md_index_t md_index, uct_mem_h *memh,
                                      ucp_md_map_t *md_map,
                                      uct_rkey_bundle_t *rkey_bundle);

void ucp_mem_type_unreg_buffers(ucp_worker_h worker, ucs_memory_type_t mem_type,
                                ucp_md_index_t md_index, uct_mem_h *memh,
                                ucp_md_map_t *md_map,
                                uct_rkey_bundle_t *rkey_bundle);

static UCS_F_ALWAYS_INLINE ucp_md_map_t
ucp_rkey_packed_md_map(const void *rkey_buffer)
{
    return *(const ucp_md_map_t*)rkey_buffer;
}

static UCS_F_ALWAYS_INLINE ucs_memory_type_t
ucp_rkey_packed_mem_type(const void *rkey_buffer)
{
    return (ucs_memory_type_t)(*(uint8_t *)((const ucp_md_map_t*)rkey_buffer + 1));
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_memh_map2uct(const uct_mem_h *uct, ucp_md_map_t md_map, ucp_md_index_t md_idx)
{
    if (!(md_map & UCS_BIT(md_idx))) {
        return NULL;
    }

    return uct[ucs_bitmap2idx(md_map, md_idx)];
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_memh2uct(ucp_mem_h memh, ucp_md_index_t md_idx)
{
    return ucp_memh_map2uct(memh->uct, memh->md_map, md_idx);
}


#define UCP_MEM_IS_HOST(_mem_type) ((_mem_type) == UCS_MEMORY_TYPE_HOST)
#define UCP_MEM_IS_ROCM(_mem_type) ((_mem_type) == UCS_MEMORY_TYPE_ROCM)
#define UCP_MEM_IS_CUDA(_mem_type) ((_mem_type) == UCS_MEMORY_TYPE_CUDA)
#define UCP_MEM_IS_CUDA_MANAGED(_mem_type) ((_mem_type) == UCS_MEMORY_TYPE_CUDA_MANAGED)
#define UCP_MEM_IS_ROCM_MANAGED(_mem_type) ((_mem_type) == UCS_MEMORY_TYPE_ROCM_MANAGED)
#define UCP_MEM_IS_ACCESSIBLE_FROM_CPU(_mem_type) \
    (UCS_BIT(_mem_type) & UCS_MEMORY_TYPES_CPU_ACCESSIBLE)

#endif
