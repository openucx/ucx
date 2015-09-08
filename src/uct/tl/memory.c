/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "context.h"
#include "tl_base.h"


typedef struct {
    uct_alloc_method_t method;
    size_t             length;
    uct_mem_h          memh;
} uct_iface_mp_chunk_hdr_t;


typedef struct {
    uct_base_iface_t               *iface;
    uct_iface_mpool_init_obj_cb_t  init_obj_cb;
} uct_iface_mp_priv_t;


const char *uct_alloc_method_names[] = {
    [UCT_ALLOC_METHOD_PD]   = "pd",
    [UCT_ALLOC_METHOD_HEAP] = "heap",
    [UCT_ALLOC_METHOD_MMAP] = "mmap",
    [UCT_ALLOC_METHOD_HUGE] = "huge",
    [UCT_ALLOC_METHOD_LAST] = NULL
};


ucs_status_t uct_mem_alloc(size_t min_length, uct_alloc_method_t *methods,
                           unsigned num_methods, uct_pd_h *pds, unsigned num_pds,
                           const char *alloc_name, uct_allocated_memory_t *mem)
{
    uct_alloc_method_t *method;
    uct_pd_attr_t pd_attr;
    ucs_status_t status;
    size_t alloc_length;
    unsigned pd_index;
    uct_mem_h memh;
    uct_pd_h pd;
    void *address;
    int shmid;

    if (min_length == 0) {
        ucs_error("Allocation length cannot be 0");
        return UCS_ERR_INVALID_PARAM;
    }

    if (num_methods == 0) {
        ucs_error("No allocation methods provided");
        return UCS_ERR_INVALID_PARAM;
    }

    for (method = methods; method < methods + num_methods; ++method) {
        ucs_debug("trying allocation method %s", uct_alloc_method_names[*method]);

        switch (*method) {
        case UCT_ALLOC_METHOD_PD:
            /* Allocate with one of the specified protection domains */
            for (pd_index = 0; pd_index < num_pds; ++pd_index) {
                pd = pds[pd_index];
                status = uct_pd_query(pd, &pd_attr);
                if (status != UCS_OK) {
                    ucs_error("Failed to query PD");
                    return status;
                }

                /* Check if PD supports allocation */
                if (!(pd_attr.cap.flags & UCT_PD_FLAG_ALLOC)) {
                    continue;
                }

                /* Allocate memory using the PD.
                 * If the allocation fails, it's considered an error and we don't
                 * fall-back, because this PD already exposed support for memory
                 * allocation.
                 */
                alloc_length = min_length;
                status = uct_pd_mem_alloc(pd, &alloc_length, &address,
                                          alloc_name, &memh);
                if (status != UCS_OK) {
                    ucs_error("failed to allocate %zu bytes using pd %s: %s",
                              alloc_length, pd->component->name,
                              ucs_status_string(status));
                    return status;
                }
                mem->pd   = pd;
                mem->memh = memh;
                goto allocated;

            }
            break;

        case UCT_ALLOC_METHOD_HEAP:
            /* Allocate aligned memory using libc allocator */
            alloc_length = min_length;
            address = ucs_memalign(UCS_SYS_CACHE_LINE_SIZE, alloc_length
                                   UCS_MEMTRACK_VAL);
            if (address != NULL) {
                goto allocated_without_pd;
            }

            ucs_debug("failed to allocate %zu bytes from the heap", alloc_length);
            break;

        case UCT_ALLOC_METHOD_MMAP:
            /* Request memory from operating system using mmap() */
            alloc_length = ucs_align_up_pow2(min_length, ucs_get_page_size());
            address = ucs_mmap(NULL, alloc_length, PROT_READ|PROT_WRITE,
                               MAP_PRIVATE|MAP_ANON, -1, 0 UCS_MEMTRACK_VAL);
            if (address != MAP_FAILED) {
                goto allocated_without_pd;
            }

            ucs_debug("failed to mmap %zu bytes: %m", alloc_length);
            break;

        case UCT_ALLOC_METHOD_HUGE:
            /* Allocate huge pages */
            alloc_length = min_length;
            status = ucs_sysv_alloc(&alloc_length, &address, SHM_HUGETLB, &shmid
                                    UCS_MEMTRACK_VAL);
            if (status == UCS_OK) {
                goto allocated_without_pd;
            }

            ucs_debug("failed to allocate %zu bytes from hugetlb: %s",
                      min_length, ucs_status_string(status));
            break;

        default:
            ucs_error("Invalid allocation method %d", *method);
            return UCS_ERR_INVALID_PARAM;
        }
    }

    ucs_debug("Could not allocate memory with any of the provided methods");
    return UCS_ERR_NO_MEMORY;

allocated_without_pd:
    mem->pd      = NULL;
    mem->memh    = UCT_INVALID_MEM_HANDLE;
allocated:
    ucs_debug("allocated %zu bytes at %p using %s", alloc_length, address,
              (mem->pd == NULL) ? uct_alloc_method_names[*method]
                                : mem->pd->component->name);
    mem->address = address;
    mem->length  = alloc_length;
    mem->method  = *method;
    return UCS_OK;
}

ucs_status_t uct_mem_free(const uct_allocated_memory_t *mem)
{
    int ret;

    switch (mem->method) {
    case UCT_ALLOC_METHOD_PD:
        return uct_pd_mem_free(mem->pd, mem->memh);

    case UCT_ALLOC_METHOD_HEAP:
        ucs_free(mem->address);
        return UCS_OK;

    case UCT_ALLOC_METHOD_MMAP:
        ret = ucs_munmap(mem->address, mem->length);
        if (ret != 0) {
            ucs_warn("munmap(address=%p, length=%zu) failed: %m", mem->address,
                     mem->length);
            return UCS_ERR_INVALID_PARAM;
        }
        return UCS_OK;

    case UCT_ALLOC_METHOD_HUGE:
        return ucs_sysv_free(mem->address);

    default:
        ucs_warn("Invalid memory allocation method: %d", mem->method);
        return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t uct_iface_mem_alloc(uct_iface_h tl_iface, size_t length,
                                 const char *name, uct_allocated_memory_t *mem)
{
    uct_base_iface_t *iface = ucs_derived_of(tl_iface, uct_base_iface_t);
    uct_pd_attr_t pd_attr;
    ucs_status_t status;

    status = uct_mem_alloc(length, iface->config.alloc_methods,
                           iface->config.num_alloc_methods, &iface->pd, 1,
                           name, mem);
    if (status != UCS_OK) {
        goto err;
    }

    /* If the memory was not allocated using PD, register it */
    if (mem->method != UCT_ALLOC_METHOD_PD) {

        status = uct_pd_query(iface->pd, &pd_attr);
        if (status != UCS_OK) {
            goto err_free;
        }

        /* If PD does not support registration, allow only the PD method */
        if (!(pd_attr.cap.flags & UCT_PD_FLAG_REG)) {
            ucs_error("%s pd does not supprt registration, so cannot use any allocation "
                      "method except 'pd'", iface->pd->component->name);
            status = UCS_ERR_NO_MEMORY;
            goto err_free;
        }

        status = uct_pd_mem_reg(iface->pd, mem->address, mem->length, &mem->memh);
        if (status != UCS_OK) {
            goto err_free;
        }

        mem->pd = iface->pd;
    }

    return UCS_OK;

err_free:
    uct_mem_free(mem);
err:
    return status;
}

void uct_iface_mem_free(const uct_allocated_memory_t *mem)
{
    if (mem->method != UCT_ALLOC_METHOD_PD) {
        uct_pd_mem_dereg(mem->pd, mem->memh);
    }
    uct_mem_free(mem);
}

static inline uct_iface_mp_priv_t* uct_iface_mp_priv(ucs_mpool_t *mp)
{
    return (uct_iface_mp_priv_t*)ucs_mpool_priv(mp);
}

static ucs_status_t uct_iface_mp_chunk_alloc(ucs_mpool_t *mp, size_t *size_p,
                                             void **chunk_p)
{
    uct_base_iface_t *iface = uct_iface_mp_priv(mp)->iface;
    uct_iface_mp_chunk_hdr_t *hdr;
    uct_allocated_memory_t mem;
    ucs_status_t status;
    size_t length;

    length = sizeof(*hdr) + *size_p;
    status = uct_iface_mem_alloc(&iface->super, length, ucs_mpool_name(mp), &mem);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(mem.memh != UCT_INVALID_MEM_HANDLE);
    ucs_assert(mem.pd == iface->pd);

    hdr         = mem.address;
    hdr->method = mem.method;
    hdr->length = mem.length;
    hdr->memh   = mem.memh;
    *size_p       = mem.length - sizeof(*hdr);
    *chunk_p    = hdr + 1;
    return UCS_OK;
}

static void uct_iface_mp_chunk_release(ucs_mpool_t *mp, void *chunk)
{
    uct_base_iface_t *iface = uct_iface_mp_priv(mp)->iface;
    uct_iface_mp_chunk_hdr_t *hdr;
    uct_allocated_memory_t mem;

    hdr = chunk - sizeof(*hdr);

    mem.address = hdr;
    mem.method  = hdr->method;
    mem.memh    = hdr->memh;
    mem.length  = hdr->length;
    mem.pd      = iface->pd;

    uct_iface_mem_free(&mem);
}

static void uct_iface_mp_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_base_iface_t *iface = uct_iface_mp_priv(mp)->iface;
    uct_iface_mpool_init_obj_cb_t init_obj_cb;
    uct_iface_mp_chunk_hdr_t *hdr;

    init_obj_cb = uct_iface_mp_priv(mp)->init_obj_cb;
    hdr = chunk - sizeof(*hdr);
    if (init_obj_cb != NULL) {
        init_obj_cb(&iface->super, obj, hdr->memh);
    }
}

static ucs_mpool_ops_t uct_iface_mpool_ops = {
    .chunk_alloc   = uct_iface_mp_chunk_alloc,
    .chunk_release = uct_iface_mp_chunk_release,
    .obj_init      = uct_iface_mp_obj_init,
    .obj_cleanup   = NULL
};

ucs_status_t uct_iface_mpool_init(uct_base_iface_t *iface, ucs_mpool_t *mp,
                                  size_t elem_size, size_t align_offset, size_t alignment,
                                  uct_iface_mpool_config_t *config, unsigned grow,
                                  uct_iface_mpool_init_obj_cb_t init_obj_cb,
                                  const char *name)
{
    unsigned elems_per_chunk;
    ucs_status_t status;

    elems_per_chunk = (config->bufs_grow != 0) ? config->bufs_grow : grow;
    status = ucs_mpool_init(mp, sizeof(uct_iface_mp_priv_t),
                            elem_size, align_offset, alignment,
                            elems_per_chunk, config->max_bufs,
                            &uct_iface_mpool_ops, name);
    if (status != UCS_OK) {
        return status;
    }

    uct_iface_mp_priv(mp)->iface       = iface;
    uct_iface_mp_priv(mp)->init_obj_cb = init_obj_cb;
    return UCS_OK;
}

