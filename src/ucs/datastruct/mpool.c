/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "mpool.h"
#include "mpool.inl"
#include "queue.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/sys.h>
#include <ucs/arch/cpu.h>


static size_t ucs_mpool_elem_total_size(ucs_mpool_data_t *data)
{
    return ucs_align_up_pow2(data->elem_size, data->alignment);
}

static UCS_F_ALWAYS_INLINE ucs_mpool_elem_t *
ucs_mpool_chunk_elem(ucs_mpool_data_t *data, ucs_mpool_chunk_t *chunk,
                     unsigned elem_index)
{
    return UCS_PTR_BYTE_OFFSET(chunk->elems,
                               elem_index * ucs_mpool_elem_total_size(data));
}

static void ucs_mpool_chunk_leak_check(ucs_mpool_t *mp, ucs_mpool_chunk_t *chunk)
{
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    ucs_mpool_data_t *data = mp->data;
    ucs_mpool_elem_t *elem;
    unsigned i;
    void *obj;

    for (i = 0; i < chunk->num_elems; ++i) {
        elem = ucs_mpool_chunk_elem(mp->data, chunk, i);
        VALGRIND_MAKE_MEM_DEFINED(elem, sizeof *elem);
        if (elem->mpool != NULL) {
            obj = elem + 1;
            ucs_string_buffer_reset(&strb);
            if (data->ops->obj_str != NULL) {
                ucs_string_buffer_appendf(&strb, " {");
                data->ops->obj_str(mp, obj, &strb);
                ucs_string_buffer_appendf(&strb, "}");
            }
            ucs_warn("object %p%s was not returned to mpool %s", obj,
                     ucs_string_buffer_cstr(&strb), ucs_mpool_name(mp));
        }
    }
}

void ucs_mpool_params_reset(ucs_mpool_params_t *params)
{
    params->priv_size       = 0;
    params->elem_size       = 0;
    params->align_offset    = 0;
    params->alignment       = UCS_SYS_CACHE_LINE_SIZE;
    params->malloc_safe     = 0;
    params->elems_per_chunk = 128;
    params->max_chunk_size  = 128 * UCS_MBYTE;
    params->max_elems       = UINT_MAX;
    params->grow_factor     = 1.0;
    params->ops             = NULL;
    params->name            = "";
}

static size_t ucs_mpool_chunk_size(ucs_mpool_t *mp, unsigned num_elems)
{
    return sizeof(ucs_mpool_chunk_t) + mp->data->alignment +
           (num_elems * ucs_mpool_elem_total_size(mp->data));
}

ucs_status_t ucs_mpool_init(const ucs_mpool_params_t *params, ucs_mpool_t *mp)
{
    size_t min_chunk_size;
    ucs_status_t status;

    /* Check input values */
    if ((params->elem_size == 0) ||
        (params->align_offset > params->elem_size) ||
        (params->alignment == 0) || !ucs_is_pow2(params->alignment) ||
        (params->elems_per_chunk == 0) ||
        (params->max_elems < params->elems_per_chunk) ||
        (params->ops == NULL) ||
        (!params->ops->chunk_alloc || !params->ops->chunk_release) ||
        (params->grow_factor < 1))
    {
        ucs_error("Invalid memory pool parameter(s)");
        return UCS_ERR_INVALID_PARAM;
    }

    mp->data = ucs_malloc(sizeof(*mp->data) + params->priv_size, "mpool_data");
    if (mp->data == NULL) {
        ucs_error("Failed to allocate memory pool slow-path area");
        return UCS_ERR_NO_MEMORY;
    }

    mp->freelist              = NULL;
    mp->data->elem_size       = sizeof(ucs_mpool_elem_t) + params->elem_size;
    mp->data->grow_factor     = params->grow_factor;
    mp->data->max_chunk_size  = params->max_chunk_size;
    mp->data->alignment       = params->alignment;
    mp->data->align_offset    = sizeof(ucs_mpool_elem_t) + params->align_offset;
    mp->data->elems_per_chunk = params->elems_per_chunk;
    mp->data->malloc_safe     = params->malloc_safe;
    mp->data->quota           = params->max_elems;
    mp->data->tail            = NULL;
    mp->data->chunks          = NULL;
    mp->data->ops             = params->ops;
    mp->data->name            = ucs_strdup(params->name, "mpool_data_name");

    if (mp->data->name == NULL) {
        ucs_error("Failed to allocate memory pool data name");
        status = UCS_ERR_NO_MEMORY;
        goto err_strdup;
    }

    min_chunk_size = ucs_mpool_chunk_size(mp, 1);
    if (params->max_chunk_size < min_chunk_size) {
        ucs_error("Invalid memory pool parameter: chunk size is too small (%zu)",
                  params->max_chunk_size);
        status = UCS_ERR_INVALID_PARAM;
        goto err_free_name;
    }

    VALGRIND_CREATE_MEMPOOL(mp, 0, 0);

    ucs_debug("mpool %s: align %zu, maxelems %u, elemsize %zu",
              ucs_mpool_name(mp), mp->data->alignment, params->max_elems,
              mp->data->elem_size);
    return UCS_OK;

err_free_name:
    ucs_free(mp->data->name);
err_strdup:
    ucs_free(mp->data);
    mp->data = NULL;
    return status;
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

    /* Check and log leaks before valgrind-destroying the memory pool */
    if (leak_check) {
        for (chunk = data->chunks; chunk != NULL; chunk = chunk->next) {
            ucs_mpool_chunk_leak_check(mp, chunk);
        }
    }

    /* Must be done before chunks are released and other threads could allocated
     * the same memory address
     */
    VALGRIND_DESTROY_MEMPOOL(mp);

    /* Release the chunks */
    next_chunk = data->chunks;
    while (next_chunk != NULL) {
        chunk      = next_chunk;
        next_chunk = chunk->next;
        data->ops->chunk_release(mp, chunk);
    }

    ucs_debug("mpool %s destroyed", ucs_mpool_name(mp));

    ucs_free(data->name);
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

static void *ucs_mpool_chunk_elems(ucs_mpool_t *mp, ucs_mpool_chunk_t *chunk)
{
    ucs_mpool_data_t *data = mp->data;
    size_t chunk_padding;

    chunk_padding = ucs_padding((uintptr_t)(chunk + 1) + data->align_offset,
                                data->alignment);
    return UCS_PTR_BYTE_OFFSET(chunk + 1, chunk_padding);
}

unsigned ucs_mpool_num_elems_per_chunk(ucs_mpool_t *mp,
                                       ucs_mpool_chunk_t *chunk,
                                       size_t chunk_size)
{
    ucs_mpool_data_t *data = mp->data;
    void *chunk_end;
    size_t elem_size;

    chunk_end = UCS_PTR_BYTE_OFFSET(chunk, chunk_size);
    elem_size = UCS_PTR_BYTE_DIFF(ucs_mpool_chunk_elems(mp, chunk), chunk_end);
    return ucs_min(data->quota, elem_size / ucs_mpool_elem_total_size(data));
}

void ucs_mpool_grow(ucs_mpool_t *mp, unsigned num_elems)
{
    ucs_mpool_data_t *data = mp->data;
    size_t chunk_size;
    ucs_mpool_chunk_t *chunk;
    ucs_mpool_elem_t *elem;
    ucs_status_t status;
    unsigned i;
    unsigned allocated_num_elems;
    void *ptr;

    if (data->quota == 0) {
        return;
    }

    allocated_num_elems = ucs_min(data->quota, num_elems);
    chunk_size          = ucs_mpool_chunk_size(mp, allocated_num_elems);
    chunk_size          = ucs_min(chunk_size, data->max_chunk_size);
    status = data->ops->chunk_alloc(mp, &chunk_size, &ptr);
    if (status != UCS_OK) {
        if (!data->malloc_safe) {
            ucs_error("Failed to allocate memory pool (name=%s) chunk: %s",
                      ucs_mpool_name(mp), ucs_status_string(status));
        }
        return;
    }

    /* Calculate padding, and update element count according to allocated size */
    chunk            = ptr;
    chunk->elems     = ucs_mpool_chunk_elems(mp, chunk);
    chunk->num_elems = ucs_mpool_num_elems_per_chunk(mp, chunk, chunk_size);

    if (!data->malloc_safe) {
        ucs_debug("mpool %s: allocated chunk %p of %lu bytes with %u elements",
                  ucs_mpool_name(mp), chunk, chunk_size, chunk->num_elems);
    }

    for (i = 0; i < chunk->num_elems; ++i) {
        elem         = ucs_mpool_chunk_elem(data, chunk, i);
        if (data->ops->obj_init != NULL) {
            data->ops->obj_init(mp, elem + 1, chunk);
        }
        ucs_mpool_add_to_freelist(mp, elem);
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
}

void *ucs_mpool_get_grow(ucs_mpool_t *mp)
{
    ucs_mpool_data_t *data = mp->data;
    unsigned num_elems;

    ucs_mpool_grow(mp, data->elems_per_chunk);
    if (mp->freelist == NULL) {
        return NULL;
    }

    /* Calculate num of elems for next growing */
    ucs_assert(data->chunks != NULL);
    num_elems             = ucs_min(data->elems_per_chunk,
                                    data->chunks->num_elems);
    data->elems_per_chunk = (num_elems * data->grow_factor) + 0.5;

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
    size_t real_size;
#ifdef SHM_HUGETLB
    void *ptr;
    ucs_status_t status;
    int shmid;
#endif

#ifdef SHM_HUGETLB
    ptr = NULL;

    /* First, try hugetlb */
    real_size = *size_p;
    status = ucs_sysv_alloc(&real_size, real_size * 2, (void**)&ptr, SHM_HUGETLB,
                            ucs_mpool_name(mp), &shmid);
    if (status == UCS_OK) {
        chunk = ptr;
        chunk->hugetlb = 1;
        goto out_ok;
    }
#endif

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
