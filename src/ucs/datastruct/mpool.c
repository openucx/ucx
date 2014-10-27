/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "mpool.h"
#include "queue.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>

typedef struct ucs_mpool_elem {
    union {
        struct ucs_mpool_elem       *next;   /* Used when elem is in the pool */
        ucs_mpool_h                 mpool;  /* Used when elem is outside the pool */
    };
} ucs_mpool_elem_t;

typedef struct ucs_mpool_chunk {
    ucs_queue_elem_t                    queue;
} ucs_mpool_chunk_t;


#define ELEM_TO_OBJECT(_elem)   ((void*)( (char*)_elem + sizeof(ucs_mpool_elem_t) ))
#define OBJECT_TO_ELEM(_obj)    ((void*)( (char*)_obj  - sizeof(ucs_mpool_elem_t) ))

#define UCS_MPOOL_CHUNK_GROW     8

/*
 *
 * A chunk of elements looks like this:
 * +-----------+-------+----------+-------+----------+------+---------+
 * | <padding> | elem0 | padding0 | elem1 | padding1 | .... | elemN-1 |
 * +-----------+-------+----------+-------+----------+------+---------+
 *
 * An element loooks like this:
 * +------------+--------+------+
 * | mpool_elem | header | data |
 * +------------+--------+------+
 *                       |
 *                       This location is aligned.
 */

struct ucs_mpool {
    ucs_mpool_elem_t                *freelist;

    size_t                          elem_size; /* Size of element in the chunk */
    size_t                          elem_padding;
    size_t                          align_offset;
    size_t                          alignment;

    unsigned                        num_elems;
    unsigned                        max_elems;
    unsigned                        elems_per_chunk;
    ucs_queue_head_t                    chunks;
#if ENABLE_ASSERT
    unsigned                        num_elems_inuse;
#endif

    void                            *mp_context;
    ucs_mpool_alloc_chunk_cb        alloc_chunk_cb;
    ucs_mpool_free_chunk_cb         free_chunk_cb;
    ucs_mpool_init_obj_cb           init_obj_cb;
    void                            *init_obj_arg;

    /* Used mostly for debugging (memtrack) */
    char                            *name;
    unsigned                        alloc_id;
};



static ucs_status_t ucs_mpool_allocate_chunk(ucs_mpool_h mp);


ucs_status_t
ucs_mpool_create(const char *name, size_t elem_size, size_t align_offset,
                 size_t alignment, unsigned elems_per_chunk, unsigned max_elems,
                 void *mp_context,
                 ucs_mpool_alloc_chunk_cb alloc_chunk_cb,
                 ucs_mpool_free_chunk_cb free_chunk_cb,
                 ucs_mpool_init_obj_cb init_obj_cb, void *init_obj_arg,
                 ucs_mpool_h *mpp)
{
    ucs_mpool_h mp;

    /* Check input values */
    if ((alignment < 1) || (elem_size == 0) || (elems_per_chunk < 1) ||
        (max_elems < elems_per_chunk))
    {
        ucs_error("Invalid memory pool parameter(s)");
        return UCS_ERR_INVALID_PARAM;
    }

    mp = ucs_malloc(sizeof *mp, "mpool context");
    if (mp == NULL) {
        ucs_error("Failed to allocate memory pool");
        return UCS_ERR_NO_MEMORY;
    }

    UCS_STATIC_ASSERT(UCS_MPOOL_HEADER_SIZE == sizeof(ucs_mpool_elem_t));

    /* Initialize the pool */
    mp->freelist = NULL;
    mp->alignment = alignment;
    mp->elems_per_chunk = elems_per_chunk;
    mp->mp_context = mp_context;
    mp->alloc_chunk_cb = alloc_chunk_cb;
    mp->free_chunk_cb = free_chunk_cb;
    mp->init_obj_cb = init_obj_cb;
    mp->init_obj_arg = init_obj_arg;
    mp->name = strdup(name);
#if ENABLE_MEMTRACK
    mp->alloc_id = ucs_calc_crc32(0, name, strlen(name));
#endif
    mp->num_elems = 0;
    mp->max_elems = max_elems;
    ucs_queue_head_init(&mp->chunks);
#if ENABLE_ASSERT
    mp->num_elems_inuse = 0;
#endif

    /* Calculate element size and padding */
    mp->elem_size    = sizeof(ucs_mpool_elem_t) + elem_size;
    mp->align_offset = sizeof(ucs_mpool_elem_t) + align_offset;
    mp->elem_padding = ucs_padding(mp->elem_size, alignment);

    VALGRIND_CREATE_MEMPOOL(mp, 0, 0);

    ucs_debug("mpool %s: align %lu, maxelems %u, elemsize %lu, padding %lu",
              mp->name, mp->alignment, mp->max_elems, mp->elem_size, mp->elem_padding);
    *mpp = mp;
    return UCS_OK;
}

static void __mpool_destroy(ucs_mpool_h mp, unsigned check_inuse)
{
    ucs_mpool_chunk_t *chunk;

    /* Sanity check - all elements should be put back to the mpool */
#if ENABLE_ASSERT
    if (check_inuse && mp->num_elems_inuse > 0) {
        ucs_warn("destroying memory pool %s with %u used elements", mp->name,
                 mp->num_elems_inuse);
        ucs_assert(0);
    }
#endif

    while (!ucs_queue_is_empty(&mp->chunks)) {
        chunk = ucs_queue_pull_elem_non_empty(&mp->chunks, ucs_mpool_chunk_t, queue);
        mp->free_chunk_cb(chunk, mp->mp_context);
    }

    VALGRIND_DESTROY_MEMPOOL(mp);
    ucs_debug("mpool %s destroyed", mp->name);
    free(mp->name);
    ucs_free(mp);
}

void ucs_mpool_destroy(ucs_mpool_h mp)
{
    __mpool_destroy(mp, 1);
}

void ucs_mpool_destroy_unchecked(ucs_mpool_h mp)
{
    __mpool_destroy(mp, 0);
}

void *ucs_mpool_get(ucs_mpool_h mp)
{
    ucs_mpool_elem_t *elem;
    void *obj;

    if (mp->freelist == NULL && ucs_mpool_allocate_chunk(mp) != UCS_OK) {
        return NULL;
    }

    /* Disconnect an element from the pool */
    elem = mp->freelist;
    VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
    mp->freelist = elem->next;
    elem->mpool = mp;
    VALGRIND_MAKE_MEM_NOACCESS(elem, sizeof *elem);

#if ENABLE_ASSERT
    ++mp->num_elems_inuse;
    ucs_assert(mp->num_elems_inuse <= mp->num_elems);
#endif

    obj = ELEM_TO_OBJECT(elem);
    VALGRIND_MEMPOOL_ALLOC(mp, obj, mp->elem_size - sizeof(ucs_mpool_elem_t));
    return obj;
}

void ucs_mpool_put(void *obj)
{
    ucs_mpool_elem_t *elem;
    ucs_mpool_h mp;

    /* Reconnect the element to the pool */
    elem = OBJECT_TO_ELEM(obj);
    VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
    mp = elem->mpool;
    elem->next = mp->freelist;
    VALGRIND_MAKE_MEM_NOACCESS(elem, sizeof *elem);
    mp->freelist = elem;

    VALGRIND_MEMPOOL_FREE(mp, obj);

#if ENABLE_ASSERT
    ucs_assert(mp->num_elems_inuse > 0);
    --mp->num_elems_inuse;
#endif
}

static UCS_F_NOINLINE ucs_status_t ucs_mpool_allocate_chunk(ucs_mpool_h mp)
{
    ucs_mpool_elem_t *elem, *nextelem;
    int elems_in_chunk;
    size_t chunk_size, chunk_padding;
    ucs_mpool_chunk_t *chunk;
    unsigned i;

    if (mp->num_elems >= mp->max_elems) {
        return UCS_ERR_NO_MEMORY;
    }

    chunk_size = sizeof(ucs_mpool_chunk_t) + mp->alignment +
                    mp->elems_per_chunk * (mp->elem_size + mp->elem_padding);
    chunk = mp->alloc_chunk_cb(&chunk_size, mp->mp_context
#if ENABLE_MEMTRACK
                               , mp->name, mp->alloc_id
#endif
                               );
    if (chunk == NULL) {
        ucs_error("Failed to allocate memory pool chunk");
        return UCS_ERR_NO_MEMORY;
    }

    /* Calculate padding, and update element count according to allocated size */
    chunk_padding = ucs_padding((uintptr_t)(chunk + 1) + mp->align_offset, mp->alignment);
    elems_in_chunk = (chunk_size - chunk_padding) /
                    (mp->elem_size + mp->elem_padding);
    ucs_debug("mpool %s: allocated chunk %p of %lu bytes with %u elements",
              mp->name, chunk, chunk_size, elems_in_chunk);

    nextelem = mp->freelist;
    for (i = 0; i < elems_in_chunk; ++i) {
        elem = (ucs_mpool_elem_t*)((char*)(chunk + 1) + chunk_padding +
                        i * (mp->elem_size + mp->elem_padding));
        elem->next = nextelem;
        nextelem = elem;
        if (mp->init_obj_cb) {
            mp->init_obj_cb(ELEM_TO_OBJECT(elem), chunk, mp->mp_context, mp->init_obj_arg);
        }
    }

    mp->freelist = nextelem;
    mp->num_elems += elems_in_chunk;
    ucs_queue_push(&mp->chunks, &chunk->queue);

    VALGRIND_MAKE_MEM_NOACCESS(chunk + 1, chunk_size - sizeof(*chunk));
    return UCS_OK;
}

void *ucs_mpool_chunk_malloc(size_t *size, void *mp_context UCS_MEMTRACK_ARG)
{
    return ucs_calloc_fwd(1, *size UCS_MEMTRACK_VAL);
}

void ucs_mpool_chunk_free(void *chunk, void *mp_context)
{
    ucs_free(chunk);
}

typedef struct ucs_mmap_mpool_chunk_hdr {
    size_t size;
} ucs_mmap_mpool_chunk_hdr_t;

void *ucs_mpool_chunk_mmap(size_t *size, void *mp_context UCS_MEMTRACK_ARG)
{
    ucs_mmap_mpool_chunk_hdr_t *chunk;
    size_t real_size;

    real_size = ucs_align_up(*size + sizeof(*chunk), ucs_get_page_size());
    chunk = ucs_mmap_fwd(NULL, real_size, PROT_READ|PROT_WRITE,
                         MAP_PRIVATE|MAP_ANONYMOUS, -1, 0 UCS_MEMTRACK_VAL);
    if (chunk == MAP_FAILED) {
        return NULL;
    }

    chunk->size = real_size;
    *size = real_size - sizeof(*chunk);
    return chunk + 1;
}

void ucs_mpool_chunk_munmap(void *ptr, void *mp_context)
{
    ucs_mmap_mpool_chunk_hdr_t *chunk = ptr;
    chunk -= 1;
    ucs_munmap(chunk, chunk->size);
}

typedef struct ucs_hugetlb_mpool_chunk_hdr {
    int hugetlb;
} ucs_hugetlb_mpool_chunk_hdr_t;

void* ucs_mpool_hugetlb_malloc(size_t *size, void *mp_context UCS_MEMTRACK_ARG)
{
    ucs_hugetlb_mpool_chunk_hdr_t *chunk;
    void *ptr;
    ucs_status_t status;
    size_t real_size;
    int shmid;

    /* First, try hugetlb */
    real_size = *size;
    status = ucs_sysv_alloc(&real_size, (void**)&ptr, SHM_HUGETLB, &shmid);
    if (status == UCS_OK) {
        chunk = ptr;
        chunk->hugetlb = 1;
        goto out_ok;
    }

    /* Fallback to glibc */
    real_size = *size;
    chunk = ucs_malloc_fwd(real_size UCS_MEMTRACK_VAL);
    if (chunk != NULL) {
        chunk->hugetlb = 0;
        goto out_ok;
    }

    return NULL;

out_ok:
    *size = real_size - sizeof(*chunk);
    return chunk + 1;
}

void ucs_mpool_hugetlb_free(void *ptr, void *mp_context)
{
    ucs_hugetlb_mpool_chunk_hdr_t *chunk;

    chunk = (ucs_hugetlb_mpool_chunk_hdr_t*)ptr - 1;
    if (chunk->hugetlb) {
        ucs_sysv_free(chunk);
    } else {
        ucs_free(chunk);
    }
}
