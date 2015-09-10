/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "mpool.h"
#include "mpool.inl"
#include "queue.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/sys.h>


static inline unsigned ucs_mpool_elem_total_size(ucs_mpool_data_t *data)
{
    return ucs_align_up_pow2(data->elem_size, data->alignment);
}

static inline ucs_mpool_elem_t *ucs_mpool_chunk_elem(ucs_mpool_data_t *data,
                                                     ucs_mpool_chunk_t *chunk,
                                                     unsigned elem_index)
{
    return chunk->elems + elem_index * ucs_mpool_elem_total_size(data);
}

static void ucs_mpool_chunk_leak_check(ucs_mpool_t *mp, ucs_mpool_chunk_t *chunk)
{
    ucs_mpool_elem_t *elem;
    unsigned i;

    for (i = 0; i < chunk->num_elems; ++i) {
        elem = ucs_mpool_chunk_elem(mp->data, chunk, i);
        if (elem->mpool != NULL) {
            ucs_warn("object %p was not returned to mpool %s", elem + 1,
                     ucs_mpool_name(mp));
        }
    }
}

ucs_status_t ucs_mpool_init(ucs_mpool_t *mp, size_t priv_size,
                            size_t elem_size, size_t align_offset, size_t alignment,
                            unsigned elems_per_chunk, unsigned max_elems,
                            ucs_mpool_ops_t *ops, const char *name)
{
    /* Check input values */
    if ((elem_size == 0) || (align_offset > elem_size) ||
        (alignment == 0) || !ucs_is_pow2(alignment) ||
        (elems_per_chunk == 0) || (max_elems < elems_per_chunk))
    {
        ucs_error("Invalid memory pool parameter(s)");
        return UCS_ERR_INVALID_PARAM;
    }

    mp->data = ucs_malloc(sizeof(*mp->data) + priv_size, "mpool_data");
    if (mp->data == NULL) {
        ucs_error("Failed to allocate memory pool slow-path area");
        return UCS_ERR_NO_MEMORY;
    }

    mp->freelist           = NULL;
    mp->data->elem_size    = sizeof(ucs_mpool_elem_t) + elem_size;
    mp->data->alignment    = alignment;
    mp->data->align_offset = sizeof(ucs_mpool_elem_t) + align_offset;
    mp->data->quota        = max_elems;
    mp->data->chunk_size   = sizeof(ucs_mpool_chunk_t) + alignment +
                             elems_per_chunk * ucs_mpool_elem_total_size(mp->data);
    mp->data->chunks       = NULL;
    mp->data->ops          = ops;
    mp->data->name         = strdup(name);

    VALGRIND_CREATE_MEMPOOL(mp, 0, 0);

    ucs_debug("mpool %s: align %u, maxelems %u, elemsize %u",
              ucs_mpool_name(mp), mp->data->alignment, max_elems, mp->data->elem_size);
    return UCS_OK;
}

void ucs_mpool_cleanup(ucs_mpool_t *mp, int leak_check)
{
    ucs_mpool_chunk_t *chunk, *next_chunk;
    ucs_mpool_elem_t *elem, *next_elem;
    ucs_mpool_data_t *data = mp->data;
    void *obj;

    /* Cleanup all elements in the freelist and set their header to NULL to mark
     * them as released for the leak check.
     */
    next_elem = mp->freelist;
    while (next_elem != NULL) {
        elem = next_elem;
        VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
        next_elem = elem->next;
        if (data->ops->obj_cleanup != NULL) {
            obj = elem + 1;
            VALGRIND_MEMPOOL_ALLOC(mp, obj, mp->data->elem_size - sizeof(ucs_mpool_elem_t));
            VALGRIND_MAKE_MEM_DEFINED(obj, mp->data->elem_size - sizeof(ucs_mpool_elem_t));
            data->ops->obj_cleanup(mp, obj);
            VALGRIND_MEMPOOL_FREE(mp, obj);
        }
        elem->mpool = NULL;
    }

    /*
     * Go over all elements in the chunks and make sure they were on the freelist.
     * Then, release the chunk.
     */
    next_chunk = data->chunks;
    while (next_chunk != NULL) {
        chunk      = next_chunk;
        next_chunk = chunk->next;

        if (leak_check) {
            ucs_mpool_chunk_leak_check(mp, chunk);
        }
        data->ops->chunk_release(mp, chunk);
    }

    VALGRIND_DESTROY_MEMPOOL(mp);
    ucs_debug("mpool %s destroyed", ucs_mpool_name(mp));

    free(data->name);
    ucs_free(data);
}

void *ucs_mpool_priv(ucs_mpool_t *mp)
{
    return mp->data + 1;
}

const char *ucs_mpool_name(ucs_mpool_t *mp)
{
    return mp->data->name;
}

int ucs_mpool_is_empty(ucs_mpool_t *mp)
{
    return (mp->freelist == NULL) && (mp->data->quota == 0);
}

void *ucs_mpool_get(ucs_mpool_t *mp)
{
    return ucs_mpool_get_inline(mp);
}

void ucs_mpool_put(void *obj)
{
    ucs_mpool_put_inline(obj);
}

void *ucs_mpool_get_grow(ucs_mpool_t *mp)
{
    size_t chunk_size, chunk_padding;
    ucs_mpool_data_t *data = mp->data;
    ucs_mpool_chunk_t *chunk;
    ucs_mpool_elem_t *elem;
    ucs_status_t status;
    unsigned i;
    void *ptr;

    if (data->quota == 0) {
        return NULL;
    }

    chunk_size = data->chunk_size;
    status = data->ops->chunk_alloc(mp, &chunk_size, &ptr);
    if (status != UCS_OK) {
        ucs_error("Failed to allocate memory pool chunk: %s", ucs_status_string(status));
        return NULL;
    }

    /* Calculate padding, and update element count according to allocated size */
    chunk            = ptr;
    chunk_padding    = ucs_padding((uintptr_t)(chunk + 1) + data->align_offset,
                                   data->alignment);
    chunk->elems     = (void*)(chunk + 1) + chunk_padding;
    chunk->num_elems = (chunk_size - chunk_padding - sizeof(*chunk)) /
                       ucs_mpool_elem_total_size(data);

    ucs_debug("mpool %s: allocated chunk %p of %lu bytes with %u elements",
              ucs_mpool_name(mp), chunk, chunk_size, chunk->num_elems);

    for (i = 0; i < chunk->num_elems; ++i) {
        elem         = ucs_mpool_chunk_elem(data, chunk, i);
        elem->next   = mp->freelist;
        mp->freelist = elem;
        if (data->ops->obj_init != NULL) {
            data->ops->obj_init(mp, elem + 1, chunk);
        }
    }

    chunk->next  = data->chunks;
    data->chunks = chunk;

    if (data->quota == UINT_MAX) {
        /* Infinite memory pool */
    } else if (data->quota >= chunk->num_elems) {
        data->quota -= chunk->num_elems;
    } else {
        data->quota = 0;
    }

    VALGRIND_MAKE_MEM_NOACCESS(chunk + 1, chunk_size - sizeof(*chunk));

    ucs_assert(mp->freelist != NULL); /* Should not recurse */
    return ucs_mpool_get(mp);
}

ucs_status_t ucs_mpool_chunk_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    *chunk_p = ucs_malloc(*size_p, ucs_mpool_name(mp));
    return (*chunk_p == NULL) ? UCS_ERR_NO_MEMORY : UCS_OK;
}

void ucs_mpool_chunk_free(ucs_mpool_t *mp, void *chunk)
{
    ucs_free(chunk);
}


typedef struct ucs_mmap_mpool_chunk_hdr {
    size_t size;
} ucs_mmap_mpool_chunk_hdr_t;

ucs_status_t ucs_mpool_chunk_mmap(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucs_mmap_mpool_chunk_hdr_t *chunk;
    size_t real_size;

    real_size = ucs_align_up(*size_p + sizeof(*chunk), ucs_get_page_size());
    chunk = ucs_mmap(NULL, real_size, PROT_READ|PROT_WRITE,
                     MAP_PRIVATE|MAP_ANONYMOUS, -1, 0, ucs_mpool_name(mp));
    if (chunk == MAP_FAILED) {
        return UCS_ERR_NO_MEMORY;
    }

    chunk->size = real_size;
    *size_p     = real_size - sizeof(*chunk);
    *chunk_p    = chunk + 1;
    return UCS_OK;
}

void ucs_mpool_chunk_munmap(ucs_mpool_t *mp, void *chunk)
{
    ucs_mmap_mpool_chunk_hdr_t *hdr = chunk;
    hdr -= 1;
    ucs_munmap(hdr, hdr->size);
}


typedef struct ucs_hugetlb_mpool_chunk_hdr {
    int hugetlb;
} ucs_hugetlb_mpool_chunk_hdr_t;

ucs_status_t ucs_mpool_hugetlb_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucs_hugetlb_mpool_chunk_hdr_t *chunk;
    void *ptr;
    ucs_status_t status;
    size_t real_size;
    int shmid;

    /* First, try hugetlb */
    real_size = *size_p;
    status = ucs_sysv_alloc(&real_size, (void**)&ptr, SHM_HUGETLB, &shmid
                            UCS_MEMTRACK_NAME(ucs_mpool_name(mp)));
    if (status == UCS_OK) {
        chunk = ptr;
        chunk->hugetlb = 1;
        goto out_ok;
    }

    /* Fallback to glibc */
    real_size = *size_p;
    chunk = ucs_malloc(real_size, ucs_mpool_name(mp));
    if (chunk != NULL) {
        chunk->hugetlb = 0;
        goto out_ok;
    }

    return UCS_ERR_NO_MEMORY;

out_ok:
    *size_p  = real_size - sizeof(*chunk);
    *chunk_p = chunk + 1;
    return UCS_OK;
}

void ucs_mpool_hugetlb_free(ucs_mpool_t *mp, void *chunk)
{
    ucs_hugetlb_mpool_chunk_hdr_t *hdr;

    hdr = (ucs_hugetlb_mpool_chunk_hdr_t*)chunk - 1;
    if (hdr->hugetlb) {
        ucs_sysv_free(hdr);
    } else {
        ucs_free(hdr);
    }
}
