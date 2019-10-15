/*
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCS_STRIDED_ALLOC_H_
#define UCS_STRIDED_ALLOC_H_

#include "queue_types.h"

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/math.h>
#include <stddef.h>


BEGIN_C_DECLS

/** @file strided_alloc.h */

/* the distance between allocated elements */
#define UCS_STRIDED_ALLOC_STRIDE (128 * UCS_KBYTE)


/**
 * Get a pointer to another element in the strided object
 *
 * Example with stride_count=3:
 *
 * chunk
 * start -+
 *        |
 *        |
 *        | <--    128 kB    --> . <--    128 kB    --> .
 *        |                      .                      .
 *        \/                     .                      .
 *        +--------+--   ...   --+--------+--   ...   --+--------+
 *        | stride |             | stride |             | stride |
 * obj0:  | elem 0 |             | elem 1 |             | elem 2 |
 *        | (base) |             |        |             |        |
 *        +--------+--   ...   --+--------+--   ...   --+--------+
 *                 +--------+--  ...    --+--------+--  ...    --+--------+
 *                 | stride |             | stride |             | stride |
 * obj1:           | elem 0 |             | elem 1 |             | elem 2 |
 *                 | (base) |             |        |             |        |
 *                 +--------+--  ...     -+--------+--  ...    --+--------+
 *                          +--------+--  ...    --+--------+--   ...   --+--------+
 *                          | stride |             | stride |             | stride |
 * obj2:                    | elem 0 |             | elem 1 |             | elem 2 |
 *                          | (base) |             |        |             |        |
 *                          +--------+--  ...     -+--------+--   ...   --+--------+
 *
 * ...
 *
 * @param _elem        Pointer to the current element
 * @param _stride_idx  Stride index of the current element
 * @param _wanted_idx  Stride index of the desired element
 *
 * @return Pointer to the desired element
 */
#define ucs_strided_elem_get(_elem, _stride_idx, _wanted_idx) \
    UCS_PTR_BYTE_OFFSET(_elem, (ptrdiff_t)UCS_STRIDED_ALLOC_STRIDE * \
                        ((ptrdiff_t)(_wanted_idx) - (ptrdiff_t)(_stride_idx)))


/* Forward declaration, used internally */
typedef struct ucs_strided_alloc_elem ucs_strided_alloc_elem_t;


/**
 * Strided allocator - allows allocating objects which are split to several
 * memory areas with a constant stride (gap) in-between.
 * This improves the cache locality when the first memory area is used mostly.
 */
typedef struct ucs_strided_alloc {
    ucs_strided_alloc_elem_t  *freelist;    /* LIFO of free elements */
    ucs_queue_head_t          chunks;       /* Queue of allocated chunks */
    size_t                    elem_size;    /* Size of a single memory area */
    unsigned                  stride_count; /* Number of strides */
    unsigned                  inuse_count;  /* Number of allocated elements */
} ucs_strided_alloc_t;


/**
 * Initialize the split allocator context
 *
 * @param [in] sa            Strided allocator structure to initialize
 * @param [in] elem_size     Size of a single stride element
 * @param [in] stride_count  How many memory strides per object
 */
void ucs_strided_alloc_init(ucs_strided_alloc_t *sa, size_t elem_size,
                            unsigned stride_count);


/**
 * Cleanup the split allocator context
 *
 * @param [in] sa           Strided allocator structure to cleanup
 */
void ucs_strided_alloc_cleanup(ucs_strided_alloc_t *sa);


/**
 * Allocate an object
 *
 * @param [in] sa            Strided allocator to allocate on
 * @param [in] alloc_name    Debug name of the allocation
 *
 * @return Pointer to the first stride of the allocated object.
 */
void* ucs_strided_alloc_get(ucs_strided_alloc_t *sa, const char *alloc_name);


/**
 * Release an object
 *
 * @param [in] sa            Strided allocator to release the object to
 * @param [in] base          Pointer to the first stride of the object to release
 */
void ucs_strided_alloc_put(ucs_strided_alloc_t *sa, void *base);


/**
 * Get the number of currently allocated objects
 *
 * @param [in] sa            Strided allocator to get the information for
*
 * @return Number of currently allocated objects
 */
unsigned ucs_strided_alloc_inuse_count(ucs_strided_alloc_t *sa);


END_C_DECLS

#endif
