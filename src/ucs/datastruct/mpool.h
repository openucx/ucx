/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_MPOOL_H_
#define UCS_MPOOL_H_

#include <ucs/debug/memtrack.h>
#include <ucs/type/status.h>


#define UCS_MPOOL_INFINITE     ((unsigned)-1)
#define UCS_MPOOL_HEADER_SIZE  (sizeof(void*))


/**
 * @param context    Mempool context as passed to ucs_mpool_create().
 */
typedef void  (*ucs_mpool_non_empty_cb_t)(void *context);

/**
 * @param size       Minimal size to allocate. The function may modify it to
 *                   the actual allocated size.
 * @param mp_context User-defined argument.
 * @return           Pointer to the allocated chunk.
 */
typedef void* (*ucs_mpool_alloc_chunk_cb)(size_t *size, void *mp_context UCS_MEMTRACK_ARG);

/**
 * @param chunk      Pointer to the chunk to free.
 * @param mp_context User-defined argument.
 */
typedef void  (*ucs_mpool_free_chunk_cb)(void *chunk, void *mp_context);

/**
 * @param obj        Object to initialize.
 * @param chunk      Chunk this object belongs to.
 * @param mp_context User-defined argument.
 * @param mp_arf     User-defined argument.
 */
typedef void  (*ucs_mpool_init_obj_cb)(void *obj, void *chunk, void *mp_context, void *arg);


typedef struct ucs_mpool *ucs_mpool_h;

/**
 * Create a memory pool, which returns elements consisting of header and data.
 * The data is guaranteed to be aligned to the specified value.
 *
 * @param elem_size        Size of an element.
 * @param align_offset     Offset in the element which should be aligned to the given boundary..
 * @param alignment        Boundary to which align the given offset within the element.
 * @param elems_per_chunk  Number of elements in a single chunk.
 * @param max_elems        Maximal number of elements which can be allocated by the pool.
 * @param mp_context       Mempool context, passed to all callbacks
 * @param alloc_chunk_cb   Called to allocate a chunk of objects.
 * @param free_chunk_cb    Called to free a previously allocated chunk of objects.
 * @param init_obj_cb      Called to first-time initialize an object.
 * @param init_obj_arg     Additional rgument for init_obj_cb.
 * @param mpp              Upon success, filled with a handle to the memory pool.
 *
 * @return UCS status code.
 */
ucs_status_t
ucs_mpool_create(const char *name, size_t elem_size, size_t align_offset,
                 size_t alignment, unsigned elems_per_chunk, unsigned max_elems,
                 void *mp_context,
                 ucs_mpool_alloc_chunk_cb alloc_chunk_cb,
                 ucs_mpool_free_chunk_cb free_chunk_cb,
                 ucs_mpool_init_obj_cb init_obj_cb, void *init_obj_arg,
                 ucs_mpool_h *mpp);

void ucs_mpool_destroy(ucs_mpool_h mp);
void ucs_mpool_destroy_unchecked(ucs_mpool_h mp);

/**
 * Get an element from the memory pool.
 */
void *ucs_mpool_get(ucs_mpool_h mp);


/**
 * Return an element to its memory pool.
 */
void ucs_mpool_put(void *obj);

/**
 * Simple chunk allocator (default).
 */
void *ucs_mpool_chunk_malloc(size_t *size, void *mp_context UCS_MEMTRACK_ARG);
void ucs_mpool_chunk_free(void *chunk, void *mp_context);


/*
 * mmap chunk allocator.
 */
void *ucs_mpool_chunk_mmap(size_t *size, void *mp_context UCS_MEMTRACK_ARG);
void ucs_mpool_chunk_munmap(void *chunk, void *mp_context);

/**
 * Hugetlb chunk allocator.
 */
void* ucs_mpool_hugetlb_malloc(size_t *size, void *mp_context UCS_MEMTRACK_ARG);
void ucs_mpool_hugetlb_free(void *ptr, void *mp_context);


#endif /* MPOOL_H_ */
