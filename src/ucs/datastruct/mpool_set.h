/**
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCS_MPOOL_SET_H_
#define UCS_MPOOL_SET_H_

#include <ucs/datastruct/mpool.h>

BEGIN_C_DECLS


#define UCS_MPOOL_SET_SIZE     32

/* All element sizes are power of two, except the cutoff by maximal allowed size
 * (max_mp_entry_size in ucs_mpool_set_init). MSB is reserved to indicate
 * presence of this cutoff.
 */
#define UCS_MPOOL_SET_MAX_SIZE UCS_BIT(30)


/* The structure represents a set of memory pools with different element sizes.
 * The elements must be power of two. Requested memory size needs to be
 * specified when fetching element from this set. Then it is actually fetched
 * from the memory pool with the smallest element size which can fit the
 * requested amount of data. This approach reduces wasting of memory when mpool
 * elements are taken for different size objects. Due to performance constrains
 * there are some restrictions for this structure:
 * - memory pool sizes have to be power of two
 * - maximal supported memory pool size is limited by UCS_MPOOL_SET_MAX_SIZE
 */
typedef struct ucs_mpool_set {
    /* Bitmap of mpool sizes. */
    uint32_t            bitmap;

    /* Array of pointers to mpools, where index is log2 of maximal supported
     * element size. Different elements may point to the same mpool, depending
     * on the number of mpool sizes requested during this set initialization.
     */
    ucs_mpool_t         *map[UCS_MPOOL_SET_SIZE];

    /* Auxiliary data containing:
     * - array of mpools used by this set
     * - private data area specified during mpool set initialization  */
    void                *data;
} ucs_mpool_set_t;


/**
 * Initialize a set of memory pools.
 *
 * @param mp_set            Memory pool set structure.
 * @param sizes             Array of requested memory pool sizes. All values
 *                          must be power of 2.
 * @param sizes_count       Length of @a sizes array.
 * @param max_mp_entry_size Maximal size which needs to be supported by this
 *                          set. Can be non power of 2.
 * @param priv_size         Size of user-defined private data area.
 * @param priv_elem_size    Size of auxiliary data which needs to be stored with
 *                          every element of this set. Every created mpool will
 *                          be capable of storing elements of requested size
 *                          plus this value.
 * @param align_offset      Offset in the element which should be aligned to the
 *                          given boundary.
 * @param alignment         Boundary to which align the given offset within the
 *                          element.
 * @param elems_per_chunk   Number of elements in a single chunk of every memory
 *                          pool.
 * @param max_elems         Maximal number of elements which can be allocated by
 *                          every mpool in the current set. -1 or UINT_MAX means
 *                          no limit.
 * @param ops               Memory pool operations.
 * @param name              Name of this memory pool set.
 *
 * @return UCS status code.
 */
ucs_status_t
ucs_mpool_set_init(ucs_mpool_set_t *mp_set, size_t *sizes, unsigned sizes_count,
                   size_t max_mp_entry_size, size_t priv_size,
                   size_t priv_elem_size, size_t align_offset, size_t alignment,
                   unsigned elems_per_chunk, unsigned max_elems,
                   ucs_mpool_ops_t *ops, const char *name);


/**
 * Cleanup a memory pool set and release all its memory.
 *
 * @param mp_set           Memory pool set structure.
 * @param leak_check       Whether to check for leaks (object which were not
 *                         returned to the set).
 */
void ucs_mpool_set_cleanup(ucs_mpool_set_t *mp_set, int leak_check);


/**
 * @param mp_set           Memory pool set structure.
 *
 * @return Pointer to the private area of size specified in @a
 * ucs_mpool_set_init().
 */
void *ucs_mpool_set_priv(ucs_mpool_set_t *mp_set);


/**
 * @param mp_set           Memory pool set structure.
 *
 * @return Memory pool set name.
 */
const char *ucs_mpool_set_name(ucs_mpool_set_t *mp_set);


END_C_DECLS

#endif /* UCS_MPOOL_SET_H_ */
