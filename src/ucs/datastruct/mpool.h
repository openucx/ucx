/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_MPOOL_H_
#define UCS_MPOOL_H_

#include <stddef.h>
#include <ucs/type/status.h>


typedef struct ucs_mpool_chunk   ucs_mpool_chunk_t;
typedef union  ucs_mpool_elem    ucs_mpool_elem_t;
typedef struct ucs_mpool         ucs_mpool_t;
typedef struct ucs_mpool_data    ucs_mpool_data_t;
typedef struct ucs_mpool_ops     ucs_mpool_ops_t;


/**
 * Manages memory allocations of same-size objects.
 *
 * A chunk of elements looks like this:
 * +-----------+-------+----------+-------+----------+------+---------+
 * | <padding> | elem0 | padding0 | elem1 | padding1 | .... | elemN-1 |
 * +-----------+-------+----------+-------+----------+------+---------+
 *
 * An element looks like this:
 * +------------+--------+------+
 * | mpool_elem | header | data |
 * +------------+--------+------+
 *                       |
 *                       This location is aligned.
 */


/**
 * Memory pool element header.
 */
union ucs_mpool_elem {
    ucs_mpool_elem_t       *next;   /* Next free elem - when elem is in the pool */
    ucs_mpool_t            *mpool;  /* Used when elem is allocated */
};


/**
 * Memory pool chunk, which contains many elements.
 */
struct ucs_mpool_chunk {
    ucs_mpool_chunk_t      *next;      /* Next chunk */
    void                   *elems;     /* Array of elements */
    unsigned               num_elems;  /* How many elements */
};


/**
 * Memory pool structure.
 */
struct ucs_mpool {
    ucs_mpool_elem_t       *freelist;  /* List of available elements */
    ucs_mpool_data_t       *data;      /* Slow-path data */
};


/**
 * Memory pool slow-path data.
 */
struct ucs_mpool_data {
    unsigned               elem_size;       /* Size of element in the chunk */
    unsigned               alignment;       /* Element alignment */
    unsigned               align_offset;    /* Offset to alignment point */
    unsigned               elems_per_chunk; /* Number of elements per chunk */
    unsigned               quota;           /* How many more elements can be allocated */
    ucs_mpool_elem_t       *tail;           /* Free list tail */
    ucs_mpool_chunk_t      *chunks;         /* List of allocated chunks */
    ucs_mpool_ops_t        *ops;            /* Memory pool operations */
    char                   *name;           /* Name - used for debugging */
};


/**
 * Defines callbacks for memory pool operations.
 */
struct ucs_mpool_ops {
    /**
     * Allocate a chunk of memory to be used by the mpool.
     *
     * @param mp           Memory pool structure.
     * @param size_p       Points to minimal size to allocate. The function may
     *                      modify it to the actual allocated size. which must be
     *                      larger or equal.
     * @param chunk_p      Filled with a pointer to the allocated chunk.
     *
     * @return             Error status.
     */
    ucs_status_t (*chunk_alloc)(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);

    /**
     * Release previously allocated chunk of memory.
     *
     * @param mp           Memory pool structure.
     * @param chunk        Chunk to release.
     */
    void         (*chunk_release)(ucs_mpool_t *mp, void *chunk);

    /**
     * Initialize an object in the memory pool on the first time it's allocated.
     * May be NULL.
     *
     * @param mp           Memory pool structure.
     * @param obj          Object to initialize.
     * @param chunk        The chunk on which the object was allocated, as returned
     *                      from chunk_alloc().
     */
    void         (*obj_init)(ucs_mpool_t *mp, void *obj, void *chunk);

    /**
     * Cleanup an object in the memory pool just before its memory is released.
     * May be NULL.
     *
     * @param mp           Memory pool structure.
     * @param obj          Object to initialize.
     */
    void         (*obj_cleanup)(ucs_mpool_t *mp, void *obj);
};


/**
 * Initialize a memory pool.
 *
 * @param mp               Memory pool structure.
 * @param priv_size        Size of user-defined private data area.
 * @param elem_size        Size of an element.
 * @param align_offset     Offset in the element which should be aligned to the given boundary.
 * @param alignment        Boundary to which align the given offset within the element.
 * @param elems_per_chunk  Number of elements in a single chunk.
 * @param max_elems        Maximal number of elements which can be allocated by the pool.
 *                         -1 or UINT_MAX means no limit.
 * @param ops              Memory pool operations.
 * @param name             Memory pool name.
 *
 * @return UCS status code.
 */
ucs_status_t ucs_mpool_init(ucs_mpool_t *mp, size_t priv_size,
                            size_t elem_size, size_t align_offset, size_t alignment,
                            unsigned elems_per_chunk, unsigned max_elems,
                            ucs_mpool_ops_t *ops, const char *name);


/**
 * Cleanup a memory pool and release all its memory.
 *
 * @param mp               Memory pool structure.
 * @param leak_check       Whether to check for leaks (object which were not
 *                          returned to the pool).
 */
void ucs_mpool_cleanup(ucs_mpool_t *mp, int leak_check);


/**
 * @param mp               Memory pool structure.
 *
 * @return Memory pool name.
 */
const char *ucs_mpool_name(ucs_mpool_t *mp);


/**
 * @param mp               Memory pool structure.
 *
 * @return User-defined context, as passed to mpool_init().
 */
void *ucs_mpool_priv(ucs_mpool_t *mp);


/**
 * Check if a memory pool is empty (cannot allocate more objects).
 *
 * @param mp               Memory pool structure.
 *
 * @return Whether a memory pool is empty.
 */
int ucs_mpool_is_empty(ucs_mpool_t *mp);


/**
 * Get an element from the memory pool.
 *
 * @param mp               Memory pool structure.
 *
 * @return New allocated object, or NULL if cannot allocate.
 */
void *ucs_mpool_get(ucs_mpool_t *mp);


/**
 * Return an object to the memory pool.
 *
 * @param obj              Object to return.
 */
void ucs_mpool_put(void *obj);


/**
 * Grow the memory pool by a specified amount of elements.
 *
 * @param mp               Memory pool structure.
 * @param num_elems        By how many elements to grow.
 */
void ucs_mpool_grow(ucs_mpool_t *mp, unsigned num_elems);


/**
 * Allocate and object and grow the memory pool if necessary.
 * Used internally by ucs_mpool_get().
 *
 * @param mp               Memory pool structure.
 *
 * @return New allocated object, or NULL if cannot allocate.
 */
void *ucs_mpool_get_grow(ucs_mpool_t *mp);


/**
 * heap-based chunk allocator.
 */
ucs_status_t ucs_mpool_chunk_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);
void ucs_mpool_chunk_free(ucs_mpool_t *mp, void *chunk);


/*
 * mmap chunk allocator.
 */
ucs_status_t ucs_mpool_chunk_mmap(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);
void ucs_mpool_chunk_munmap(ucs_mpool_t *mp, void *chunk);


/**
 * hugetlb chunk allocator.
 */
ucs_status_t ucs_mpool_hugetlb_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p);
void ucs_mpool_hugetlb_free(ucs_mpool_t *mp, void *chunk);


#endif /* MPOOL_H_ */
