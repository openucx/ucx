/*
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "strided_alloc.h"
#include "queue.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/sys.h>


#define ucs_strided_alloc_chunk_to_mem(_chunk) \
    UCS_PTR_BYTE_OFFSET(_chunk, + sizeof(ucs_strided_alloc_chunk_t)  \
                                - UCS_STRIDED_ALLOC_STRIDE)

#define ucs_strided_alloc_mem_to_chunk(_mem) \
    UCS_PTR_BYTE_OFFSET(_mem,   - sizeof(ucs_strided_alloc_chunk_t)  \
                                + UCS_STRIDED_ALLOC_STRIDE)

typedef struct ucs_splitalloc_chunk {
    ucs_queue_elem_t         queue;
} ucs_strided_alloc_chunk_t;

struct ucs_strided_alloc_elem {
    ucs_strided_alloc_elem_t *next;
};

static ucs_strided_alloc_chunk_t *
ucs_strided_alloc_chunk_alloc(ucs_strided_alloc_t *sa, size_t chunk_size,
                              const char *alloc_name)
{
    ucs_status_t status;
    size_t size;
    void *ptr;

    size   = chunk_size;
    ptr    = NULL;
    status = ucs_mmap_alloc(&size, &ptr, 0, alloc_name);
    if (status != UCS_OK) {
        ucs_error("failed to allocate a chunk of %zu bytes", chunk_size);
        return NULL;
    }

    return ucs_strided_alloc_mem_to_chunk(ptr);
}

static void ucs_strided_alloc_chunk_free(ucs_strided_alloc_t *sa,
                                         ucs_strided_alloc_chunk_t *chunk,
                                         size_t chunk_size)
{
    /* coverity[offset_free] */
    ucs_mmap_free(ucs_strided_alloc_chunk_to_mem(chunk), chunk_size);
}

static void ucs_strided_alloc_push_to_freelist(ucs_strided_alloc_t *sa,
                                               ucs_strided_alloc_elem_t *elem)
{
    elem->next   = sa->freelist;
    sa->freelist = elem;
}

static void ucs_strided_alloc_calc(ucs_strided_alloc_t *sa, size_t *chunk_size,
                                   size_t *elems_per_chunk)
{
    *chunk_size      = ucs_align_up_pow2(UCS_STRIDED_ALLOC_STRIDE * sa->stride_count,
                                         ucs_get_page_size());
    *elems_per_chunk = (UCS_STRIDED_ALLOC_STRIDE -
                        sizeof(ucs_strided_alloc_chunk_t)) / sa->elem_size;
}

static ucs_status_t
ucs_strided_alloc_grow(ucs_strided_alloc_t *sa, const char *alloc_name)
{
    size_t chunk_size, elems_per_chunk;
    ucs_strided_alloc_chunk_t *chunk;
    ucs_strided_alloc_elem_t *elem;
    void *chunk_mem;
    ssize_t i;

    ucs_strided_alloc_calc(sa, &chunk_size, &elems_per_chunk);

    chunk = ucs_strided_alloc_chunk_alloc(sa, chunk_size, alloc_name);
    if (chunk == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    chunk_mem = ucs_strided_alloc_chunk_to_mem(chunk);
    for (i = elems_per_chunk - 1; i >= 0; --i) {
        elem = UCS_PTR_BYTE_OFFSET(chunk_mem, i * sa->elem_size);
        ucs_strided_alloc_push_to_freelist(sa, elem);
    }

    ucs_queue_push(&sa->chunks, &chunk->queue);

    VALGRIND_MAKE_MEM_NOACCESS(chunk_mem, chunk_size);

    return UCS_OK;
}

void ucs_strided_alloc_init(ucs_strided_alloc_t *sa, size_t elem_size,
                            unsigned stride_count)
{
    ucs_assert(elem_size >= sizeof(ucs_strided_alloc_elem_t));
    ucs_assert(elem_size <= (UCS_STRIDED_ALLOC_STRIDE -
                             sizeof(ucs_strided_alloc_chunk_t)));
    ucs_assert(stride_count >= 1);

    ucs_queue_head_init(&sa->chunks);

    sa->freelist     = NULL;
    sa->elem_size    = elem_size;
    sa->stride_count = stride_count;
    sa->inuse_count  = 0;
    VALGRIND_CREATE_MEMPOOL(sa, 0, 0);
}

void ucs_strided_alloc_cleanup(ucs_strided_alloc_t *sa)
{
    size_t chunk_size, elems_per_chunk;
    ucs_strided_alloc_chunk_t *chunk;

    VALGRIND_DESTROY_MEMPOOL(sa);

    ucs_strided_alloc_calc(sa, &chunk_size, &elems_per_chunk);

    while (!ucs_queue_is_empty(&sa->chunks)) {
        chunk = ucs_queue_head_elem_non_empty(&sa->chunks, ucs_strided_alloc_chunk_t,
                                              queue);
        VALGRIND_MAKE_MEM_DEFINED(chunk, sizeof(*chunk));
        ucs_queue_pull_non_empty(&sa->chunks);
        ucs_strided_alloc_chunk_free(sa, chunk, chunk_size);
    }
}

void* ucs_strided_alloc_get(ucs_strided_alloc_t *sa, const char *alloc_name)
{
    ucs_strided_alloc_elem_t *elem;
    ucs_status_t status;
    unsigned i;

    if (sa->freelist == NULL) {
        status = ucs_strided_alloc_grow(sa, alloc_name);
        if (status != UCS_OK) {
            return NULL;
        }
    }

    ucs_assert(sa->freelist != NULL);

    elem         = sa->freelist;
    VALGRIND_MAKE_MEM_DEFINED(elem, sizeof(*elem));
    sa->freelist = elem->next;
    VALGRIND_MAKE_MEM_NOACCESS(elem, sizeof(*elem));

    for (i = 0; i < sa->stride_count; ++i) {
        VALGRIND_MEMPOOL_ALLOC(sa, ucs_strided_elem_get(elem, 0, i),
                               sa->elem_size);
    }

    ++sa->inuse_count;

    return elem;
}

void ucs_strided_alloc_put(ucs_strided_alloc_t *sa, void *base)
{
    ucs_strided_alloc_elem_t *elem = base;
    unsigned i;

    ucs_assert(sa->inuse_count > 0);

    ucs_strided_alloc_push_to_freelist(sa, elem);

    for (i = 0; i < sa->stride_count; ++i) {
        VALGRIND_MEMPOOL_FREE(sa, ucs_strided_elem_get(elem, 0, i));
    }

    --sa->inuse_count;
}

unsigned ucs_strided_alloc_inuse_count(ucs_strided_alloc_t *sa)
{
    return sa->inuse_count;
}
