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
 * @param mp_context Mempool context as passed to ucs_mpool_create().
 */
typedef void  (*ucs_mpool_non_empty_cb_t)(void *mp_context);

/**
 * @param size         Minimal size to allocate. The function may modify it to
 *                     the actual allocated size.
 * @param chunk_p      Filled with a pointer to the allocated chunk.
 * @param mp_context   Mempool context as passed to ucs_mpool_create().
 *
 * @return             Error status.
 */
typedef ucs_status_t (*ucs_mpool_chunk_alloc_cb_t)(void *mp_context, size_t *size,
                                                   void **chunk_p UCS_MEMTRACK_ARG);


/**
 * @param mp_context Mempool context as passed to ucs_mpool_create().
 * @param chunk      Pointer to the chunk to free.
 */
typedef void  (*ucs_mpool_chunk_free_cb_t)(void *mp_context, void *chunk);


/**
 * @param mp_context  Mempool context as passed to ucs_mpool_create().
 * @param
 * @param obj         Object to initialize.
 * @param arg         User-defined argument to init function.
 */
typedef void  (*ucs_mpool_init_obj_cb_t)(void *mp_context, void *obj, void *chunk,
                                         void *arg);


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
 *                         -1 or UINT_MAX means no limit.
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
                 ucs_mpool_chunk_alloc_cb_t alloc_chunk_cb,
                 ucs_mpool_chunk_free_cb_t free_chunk_cb,
                 ucs_mpool_init_obj_cb_t init_obj_cb, void *init_obj_arg,
                 ucs_mpool_h *mpp);

void ucs_mpool_destroy(ucs_mpool_h mp);
void ucs_mpool_destroy_unchecked(ucs_mpool_h mp);

/**
 * Check that memory pool is empty. Also called from ucs_mpool_destroy().
 */
void ucs_mpool_check_empty(ucs_mpool_h mp);

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
ucs_status_t ucs_mpool_chunk_malloc(void *mp_context, size_t *size, void **chunk_p
                                    UCS_MEMTRACK_ARG);
void ucs_mpool_chunk_free(void *mp_context, void *chunk);


/*
 * mmap chunk allocator.
 */
ucs_status_t ucs_mpool_chunk_mmap(void *mp_context, size_t *size, void **chunk_p
                                  UCS_MEMTRACK_ARG);
void ucs_mpool_chunk_munmap(void *mp_context, void *chunk);

/**
 * Hugetlb chunk allocator.
 */
ucs_status_t ucs_mpool_hugetlb_malloc(void *mp_context, size_t *size, void **chunk_p
                                      UCS_MEMTRACK_ARG);
void ucs_mpool_hugetlb_free(void *mp_context, void *chunk);


#endif /* MPOOL_H_ */
