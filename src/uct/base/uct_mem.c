/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "uct_iface.h"
#include "uct_md.h"

#include <ucs/arch/cpu.h>
#include <ucs/profile/profile.h>
#include <ucs/sys/math.h>


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
    [UCT_ALLOC_METHOD_THP]  = "thp",
    [UCT_ALLOC_METHOD_MD]   = "md",
    [UCT_ALLOC_METHOD_HEAP] = "heap",
    [UCT_ALLOC_METHOD_MMAP] = "mmap",
    [UCT_ALLOC_METHOD_HUGE] = "huge",
    [UCT_ALLOC_METHOD_LAST] = NULL
};


static inline int uct_mem_get_mmap_flags(unsigned uct_mmap_flags)
{
    int mm_flags = 0;

#ifdef MAP_NONBLOCK
    if (uct_mmap_flags & UCT_MD_MEM_FLAG_NONBLOCK) {
        mm_flags |= MAP_NONBLOCK;
    }
#endif

    if (uct_mmap_flags & UCT_MD_MEM_FLAG_FIXED) {
        mm_flags |= MAP_FIXED;
    }

    return mm_flags;
}

static inline void *uct_mem_alloc_params_get_address(const uct_mem_alloc_params_t *params)
{
    return (params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS) ?
            params->address : NULL;
}

ucs_status_t uct_mem_alloc(size_t length, const uct_alloc_method_t *methods,
                           unsigned num_methods,
                           const uct_mem_alloc_params_t *params,
                           uct_allocated_memory_t *mem)
{
    const char *alloc_name;
    const uct_alloc_method_t *method;
    ucs_memory_type_t mem_type;
    uct_md_attr_t md_attr;
    ucs_status_t status;
    unsigned flags;
    size_t alloc_length;
    unsigned md_index;
    uct_mem_h memh;
    uct_md_h md;
    void *address;
    int ret;
#ifdef SHM_HUGETLB
    int shmid;
#endif
#ifdef MADV_HUGEPAGE
    ssize_t huge_page_size;
#endif

    status = uct_mem_alloc_check_params(length, methods, num_methods, params);
    if (status != UCS_OK) {
        return status;
    }

    /* set defaults in case some param fields are not set */
    address      = uct_mem_alloc_params_get_address(params);
    flags        = (params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_FLAGS) ?
                   params->flags : (UCT_MD_MEM_ACCESS_LOCAL_READ |
                                    UCT_MD_MEM_ACCESS_LOCAL_WRITE);
    alloc_name   = (params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_NAME) ?
                   params->name : "anonymous-uct_mem_alloc";
    mem_type     = (params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE) ?
                   params->mem_type : UCS_MEMORY_TYPE_HOST;
    alloc_length = length;

    ucs_trace("allocating %s: %s memory length %zu flags 0x%x", alloc_name,
              ucs_memory_type_names[mem_type], alloc_length, flags);
    ucs_log_indent(1);

    for (method = methods; method < methods + num_methods; ++method) {
        ucs_trace("trying allocation method %s", uct_alloc_method_names[*method]);

        switch (*method) {
        case UCT_ALLOC_METHOD_MD:
            if (!(params->field_mask & UCT_MEM_ALLOC_PARAM_FIELD_MDS)) {
                break;
            }

            /* Allocate with one of the specified memory domains */
            for (md_index = 0; md_index < params->mds.count; ++md_index) {
                alloc_length = length;
                address      = uct_mem_alloc_params_get_address(params);
                md           = params->mds.mds[md_index];
                status = uct_md_query(md, &md_attr);
                if (status != UCS_OK) {
                    ucs_error("Failed to query MD");
                    goto out;
                }

                /* Check if MD supports allocation */
                if (!(md_attr.cap.flags & UCT_MD_FLAG_ALLOC)) {
                    continue;
                }

                /* Check if MD supports allocation with fixed address
                 * if it's requested */
                if ((flags & UCT_MD_MEM_FLAG_FIXED) &&
                    !(md_attr.cap.flags & UCT_MD_FLAG_FIXED)) {
                    continue;
                }

                /* Check if MD supports allocation on requested mem_type */
                if (!(md_attr.cap.alloc_mem_types & UCS_BIT(mem_type))) {
                    continue;
                }

                /* Allocate memory using the MD.
                 * If the allocation fails, it's considered an error and we don't
                 * fall-back, because this MD already exposed support for memory
                 * allocation.
                 */
                status = uct_md_mem_alloc(md, &alloc_length, &address,
                                          mem_type, flags, alloc_name,
                                          &memh);
                if (status != UCS_OK) {
                    ucs_error("failed to allocate %zu bytes using md %s for %s: %s",
                              alloc_length, md->component->name,
                              alloc_name, ucs_status_string(status));
                    goto out;
                }

                ucs_assert(memh != UCT_MEM_HANDLE_NULL);
                mem->md       = md;
                mem->mem_type = mem_type;
                mem->memh     = memh;
                goto allocated;

            }

            if (mem_type != UCS_MEMORY_TYPE_HOST) {
                /* assumes that only MDs are capable of allocating non-host
                 * memory
                 */
                ucs_trace("unable to allocate requested memory type");
                status = UCS_ERR_UNSUPPORTED;
                goto out;
            }

            break;

        case UCT_ALLOC_METHOD_THP:
            if (mem_type != UCS_MEMORY_TYPE_HOST) {
                break;
            }

#ifdef MADV_HUGEPAGE
            /* Fixed option is not supported for "thp" allocation */
            if (flags & UCT_MD_MEM_FLAG_FIXED) {
                break;
            }

            if (!ucs_is_thp_enabled()) {
                break;
            }

            huge_page_size = ucs_get_huge_page_size();
            if (huge_page_size <= 0) {
                break;
            }

            alloc_length = ucs_align_up(length, huge_page_size);
            if (alloc_length >= (2 * length)) {
                break;
            }

            ret = ucs_posix_memalign(&address, huge_page_size, alloc_length,
                                     alloc_name);
            if (ret != 0) {
                ucs_trace("failed to allocate %zu bytes using THP: %m", alloc_length);
            } else {
                ret = madvise(address, alloc_length, MADV_HUGEPAGE);
                if (ret != 0) {
                    ucs_trace("madvise(address=%p, length=%zu, HUGEPAGE) "
                              "returned %d: %m", address, alloc_length, ret);
                    ucs_free(address);
                } else {
                    goto allocated_without_md;
                }
            }
#endif
            break;

        case UCT_ALLOC_METHOD_HEAP:
            if (mem_type != UCS_MEMORY_TYPE_HOST) {
                break;
            }

            /* Allocate aligned memory using libc allocator */

            /* Fixed option is not supported for heap allocation*/
            if (flags & UCT_MD_MEM_FLAG_FIXED) {
                break;
            }

            address = uct_mem_alloc_params_get_address(params);
            ret = ucs_posix_memalign(&address, UCS_SYS_CACHE_LINE_SIZE,
                                     length, alloc_name);
            if (ret == 0) {
                goto allocated_without_md;
            }

            ucs_trace("failed to allocate %zu bytes from the heap", alloc_length);
            break;

        case UCT_ALLOC_METHOD_MMAP:
            if (mem_type != UCS_MEMORY_TYPE_HOST) {
                break;
            }

            /* Request memory from operating system using mmap() */

            alloc_length = length;
            address      = uct_mem_alloc_params_get_address(params);
            status = ucs_mmap_alloc(&alloc_length, &address,
                                    uct_mem_get_mmap_flags(flags), alloc_name);
            if (status== UCS_OK) {
                goto allocated_without_md;
            }

            ucs_trace("failed to mmap %zu bytes: %s", length,
                      ucs_status_string(status));
            break;

        case UCT_ALLOC_METHOD_HUGE:
            if (mem_type != UCS_MEMORY_TYPE_HOST) {
                break;
            }

#ifdef SHM_HUGETLB
            /* Allocate huge pages */
            alloc_length = length;
            address      = (flags & UCT_MD_MEM_FLAG_FIXED) ?
                           params->address : NULL;
            status = ucs_sysv_alloc(&alloc_length, length * 2, &address,
                                    SHM_HUGETLB, alloc_name, &shmid);
            if (status == UCS_OK) {
                goto allocated_without_md;
            }
#else
            status = UCS_ERR_NO_MEMORY;
#endif

            ucs_trace("failed to allocate %zu bytes from hugetlb: %s",
                      length, ucs_status_string(status));
            break;

        default:
            ucs_error("Invalid allocation method %d", *method);
            status = UCS_ERR_INVALID_PARAM;
            goto out;
        }
    }

    ucs_debug("could not allocate memory with any of the provided methods");
    status = UCS_ERR_NO_MEMORY;
    goto out;

allocated_without_md:
    mem->md       = NULL;
    mem->mem_type = UCS_MEMORY_TYPE_HOST;
    mem->memh     = UCT_MEM_HANDLE_NULL;
allocated:
    ucs_trace("allocated %zu bytes at %p using %s", alloc_length, address,
              (mem->md == NULL) ? uct_alloc_method_names[*method]
                                : mem->md->component->name);
    mem->address = address;
    mem->length  = alloc_length;
    mem->method  = *method;
    status       = UCS_OK;
out:
    ucs_log_indent(-1);
    return status;
}

ucs_status_t uct_mem_free(const uct_allocated_memory_t *mem)
{
    switch (mem->method) {
    case UCT_ALLOC_METHOD_MD:
        return uct_md_mem_free(mem->md, mem->memh);

    case UCT_ALLOC_METHOD_THP:
    case UCT_ALLOC_METHOD_HEAP:
        ucs_free(mem->address);
        return UCS_OK;

    case UCT_ALLOC_METHOD_MMAP:
        return ucs_mmap_free(mem->address, mem->length);

    case UCT_ALLOC_METHOD_HUGE:
        return ucs_sysv_free(mem->address);

    default:
        ucs_warn("Invalid memory allocation method: %d", mem->method);
        return UCS_ERR_INVALID_PARAM;
    }
}

static int uct_iface_is_allowed_alloc_method(uct_base_iface_t *iface,
                                             uct_alloc_method_t method)
{
    unsigned i;

    for (i = 0; i < iface->config.num_alloc_methods; i++) {
        if (iface->config.alloc_methods[i] == method) {
            return 1;
        }
    }

    return 0;
}

ucs_status_t uct_iface_mem_alloc(uct_iface_h tl_iface, size_t length, unsigned flags,
                                 const char *name, uct_allocated_memory_t *mem)
{
    static uct_alloc_method_t method_md = UCT_ALLOC_METHOD_MD;
    uct_base_iface_t *iface             = ucs_derived_of(tl_iface,
                                                         uct_base_iface_t);
    void *address                       = NULL;
    uct_md_attr_t md_attr;
    ucs_status_t status;
    uct_mem_alloc_params_t params;
    unsigned num_alloc_methods;
    uct_alloc_method_t *alloc_methods;

    status = uct_md_query(iface->md, &md_attr);
    if (status != UCS_OK) {
        goto err;
    }

    if (!(md_attr.cap.flags & UCT_MD_FLAG_REG) &&
        uct_iface_is_allowed_alloc_method(iface, UCT_ALLOC_METHOD_MD)) {
        /* If MD does not support registration, allow only the MD method */
        alloc_methods     = &method_md;
        num_alloc_methods = 1;
    } else if (!(md_attr.cap.flags & UCT_MD_FLAG_REG)) {
        /* If MD does not support registration and MD method is not in list
         * return error */
        status = UCS_ERR_NO_MEMORY;
        goto err;
    } else {
        alloc_methods     = iface->config.alloc_methods;
        num_alloc_methods = iface->config.num_alloc_methods;
    }

    params.field_mask      = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS    |
                             UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS  |
                             UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                             UCT_MEM_ALLOC_PARAM_FIELD_MDS      |
                             UCT_MEM_ALLOC_PARAM_FIELD_NAME;
    params.flags           = UCT_MD_MEM_ACCESS_ALL;
    params.name            = name;
    params.mem_type        = UCS_MEMORY_TYPE_HOST;
    params.address         = address;
    params.mds.mds         = &iface->md;
    params.mds.count       = 1;

    status = uct_mem_alloc(length, alloc_methods, num_alloc_methods, &params,
                           mem);
    if (status != UCS_OK) {
        goto err;
    }

    /* If the memory was not allocated using MD, register it */
    if (mem->method != UCT_ALLOC_METHOD_MD) {
        if ((md_attr.cap.flags & UCT_MD_FLAG_REG) &&
            (md_attr.cap.reg_mem_types & UCS_BIT(mem->mem_type))) {
            status = uct_md_mem_reg(iface->md, mem->address, mem->length, flags,
                                    &mem->memh);
            if (status != UCS_OK) {
                goto err_free;
            }

            ucs_assert(mem->memh != UCT_MEM_HANDLE_NULL);
        } else {
            mem->memh = UCT_MEM_HANDLE_NULL;
        }

        mem->md = iface->md;
    }

    return UCS_OK;

err_free:
    uct_mem_free(mem);
err:
    return status;
}

void uct_iface_mem_free(const uct_allocated_memory_t *mem)
{
    if ((mem->method != UCT_ALLOC_METHOD_MD) &&
        (mem->memh != UCT_MEM_HANDLE_NULL))
    {
        (void)uct_md_mem_dereg(mem->md, mem->memh);
    }
    uct_mem_free(mem);
}

static inline uct_iface_mp_priv_t* uct_iface_mp_priv(ucs_mpool_t *mp)
{
    return (uct_iface_mp_priv_t*)ucs_mpool_priv(mp);
}

UCS_PROFILE_FUNC(ucs_status_t, uct_iface_mp_chunk_alloc, (mp, size_p, chunk_p),
                 ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    uct_base_iface_t *iface = uct_iface_mp_priv(mp)->iface;
    uct_iface_mp_chunk_hdr_t *hdr;
    uct_allocated_memory_t mem;
    ucs_status_t status;
    size_t length;

    length = sizeof(*hdr) + *size_p;
    status = uct_iface_mem_alloc(&iface->super, length,
                                 UCT_MD_MEM_ACCESS_ALL | UCT_MD_MEM_FLAG_LOCK,
                                 ucs_mpool_name(mp), &mem);
    if (status != UCS_OK) {
        return status;
    }

    ucs_assert(mem.memh != UCT_MEM_HANDLE_NULL);
    ucs_assert(mem.md == iface->md);

    hdr         = mem.address;
    hdr->method = mem.method;
    hdr->length = mem.length;
    hdr->memh   = mem.memh;
    *size_p     = mem.length - sizeof(*hdr);
    *chunk_p    = hdr + 1;
    return UCS_OK;
}

UCS_PROFILE_FUNC_VOID(uct_iface_mp_chunk_release, (mp, chunk),
                      ucs_mpool_t *mp, void *chunk)
{
    uct_base_iface_t *iface = uct_iface_mp_priv(mp)->iface;
    uct_iface_mp_chunk_hdr_t *hdr;
    uct_allocated_memory_t mem;

    hdr = UCS_PTR_BYTE_OFFSET(chunk, -sizeof(*hdr));

    mem.address = hdr;
    mem.method  = hdr->method;
    mem.memh    = hdr->memh;
    mem.length  = hdr->length;
    mem.md      = iface->md;

    uct_iface_mem_free(&mem);
}

static void uct_iface_mp_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    uct_base_iface_t *iface = uct_iface_mp_priv(mp)->iface;
    uct_iface_mpool_init_obj_cb_t init_obj_cb;
    uct_iface_mp_chunk_hdr_t *hdr;

    init_obj_cb = uct_iface_mp_priv(mp)->init_obj_cb;
    hdr = UCS_PTR_BYTE_OFFSET(chunk, -sizeof(*hdr));
    if (init_obj_cb != NULL) {
        init_obj_cb(&iface->super, obj, hdr->memh);
    }
}

static ucs_mpool_ops_t uct_iface_mpool_ops = {
    .chunk_alloc   = uct_iface_mp_chunk_alloc,
    .chunk_release = uct_iface_mp_chunk_release,
    .obj_init      = uct_iface_mp_obj_init,
    .obj_cleanup   = NULL,
    .obj_str       = NULL
};

ucs_status_t uct_iface_mpool_init(uct_base_iface_t *iface, ucs_mpool_t *mp,
                                  size_t elem_size, size_t align_offset, size_t alignment,
                                  const uct_iface_mpool_config_t *config, unsigned grow,
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
