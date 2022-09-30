/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_mm.h"
#include "ucp_context.h"
#include "ucp_worker.h"
#include "ucp_mm.inl"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/type/serialize.h>
#include <ucm/api/ucm.h>
#include <string.h>
#include <inttypes.h>


ucp_mem_dummy_handle_t ucp_mem_dummy_handle = {
    .memh = {
        .alloc_method = UCT_ALLOC_METHOD_LAST,
        .alloc_md_index = UCP_NULL_RESOURCE,
    },
    .uct = { UCT_MEM_HANDLE_NULL }
};

ucs_status_t ucp_mem_rereg_mds(ucp_context_h context, ucp_md_map_t reg_md_map,
                               void *address, size_t length, unsigned uct_flags,
                               uct_md_h alloc_md, ucs_memory_type_t mem_type,
                               uct_mem_h *alloc_md_memh_p, uct_mem_h *uct_memh,
                               ucp_md_map_t *md_map_p)
{
    unsigned memh_index, prev_memh_index;
    uct_mem_h *prev_uct_memh;
    ucp_md_map_t new_md_map;
    const uct_md_attr_v2_t *md_attr;
    void *end_address UCS_V_UNUSED;
    unsigned prev_num_memh;
    unsigned md_index;
    ucs_status_t status;
    ucs_log_level_t level;
    ucs_memory_info_t mem_info;
    size_t reg_length;
    void *base_address;

    if (reg_md_map == *md_map_p) {
        return UCS_OK; /* shortcut - no changes required */
    }

    ucs_assertv(reg_md_map <= UCS_MASK(context->num_mds),
                "reg_md_map=0x%" PRIx64 " num_mds=%u", reg_md_map,
                context->num_mds);

    prev_num_memh = ucs_popcount(*md_map_p & reg_md_map);
    prev_uct_memh = ucs_alloca(prev_num_memh * sizeof(*prev_uct_memh));

    /* Go over previous handles, save only the ones we will need */
    memh_index      = 0;
    prev_memh_index = 0;
    ucs_for_each_bit(md_index, *md_map_p) {
        if (reg_md_map & UCS_BIT(md_index)) {
            /* memh still needed, save it */
            ucs_assert(prev_memh_index < prev_num_memh);
            prev_uct_memh[prev_memh_index++] = uct_memh[memh_index];
        } else if (alloc_md == context->tl_mds[md_index].md) {
            /* memh not needed and allocated, return it */
            if (alloc_md_memh_p != NULL) {
                *alloc_md_memh_p = uct_memh[memh_index];
            }
        } else {
            /* memh not needed and registered, deregister it */
            ucs_trace("de-registering memh[%d]=%p from md[%d]", memh_index,
                      uct_memh[memh_index], md_index);
            status = uct_md_mem_dereg(context->tl_mds[md_index].md,
                                      uct_memh[memh_index]);
            if (status != UCS_OK) {
                ucs_warn("failed to dereg from md[%d]=%s: %s", md_index,
                         context->tl_mds[md_index].rsc.md_name,
                         ucs_status_string(status));
            }
        }

        VALGRIND_MAKE_MEM_UNDEFINED(&uct_memh[memh_index],
                                    sizeof(uct_memh[memh_index]));
        ++memh_index;
    }

    /* prev_uct_memh should contain the handles which should be reused */
    ucs_assert(prev_memh_index == prev_num_memh);

    /* Go over requested MD map, and use / register new handles */
    new_md_map      = 0;
    memh_index      = 0;
    prev_memh_index = 0;
    ucs_for_each_bit(md_index, reg_md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if (*md_map_p & UCS_BIT(md_index)) {
            /* already registered, use previous memh */
            ucs_assert(prev_memh_index < prev_num_memh);
            uct_memh[memh_index++] = prev_uct_memh[prev_memh_index++];
            new_md_map            |= UCS_BIT(md_index);
        } else if (context->tl_mds[md_index].md == alloc_md) {
            /* already allocated, add the memh we got from allocation */
            ucs_assert(alloc_md_memh_p != NULL);
            uct_memh[memh_index++] = *alloc_md_memh_p;
            new_md_map            |= UCS_BIT(md_index);
        } else if (length == 0) {
            /* don't register zero-length regions */
            continue;
        } else if (md_attr->flags & UCT_MD_FLAG_REG) {
            ucs_assert(address != NULL);

            if (!(md_attr->reg_mem_types & UCS_BIT(mem_type))) {
                continue;
            }

            if (context->config.ext.reg_whole_alloc_bitmap & UCS_BIT(mem_type)) {
                ucp_memory_detect_internal(context, address, length, &mem_info);
                base_address = mem_info.base_address;
                reg_length   = mem_info.alloc_length;
                end_address  = UCS_PTR_BYTE_OFFSET(base_address, reg_length);
                ucs_trace("extending %p..%p to %p..%p", address,
                          UCS_PTR_BYTE_OFFSET(address, length), base_address,
                          end_address);
                ucs_assertv(base_address <= address,
                            "base_address=%p address=%p", base_address,
                            address);
                ucs_assertv(end_address >= UCS_PTR_BYTE_OFFSET(address, length),
                            "end_address=%p address+length=%p", end_address,
                            UCS_PTR_BYTE_OFFSET(address, length));
            } else {
                base_address = address;
                reg_length   = length;
            }

            /* MD supports registration, register new memh on it */
            status = uct_md_mem_reg(context->tl_mds[md_index].md, base_address,
                                    reg_length, uct_flags, &uct_memh[memh_index]);
            if (status == UCS_OK) {
                ucs_trace("registered address %p length %zu on md[%d]"
                          " memh[%d]=%p",
                          base_address, reg_length, md_index, memh_index,
                          uct_memh[memh_index]);
                new_md_map |= UCS_BIT(md_index);
                ++memh_index;
                continue;
            }

            level = (uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ?
                    UCS_LOG_LEVEL_DIAG : UCS_LOG_LEVEL_ERROR;

            ucs_log(level,
                    "failed to register address %p mem_type bit 0x%lx length %zu on "
                    "md[%d]=%s: %s (md reg_mem_types 0x%"PRIx64")",
                    base_address, UCS_BIT(mem_type), reg_length, md_index,
                    context->tl_mds[md_index].rsc.md_name,
                    ucs_status_string(status), md_attr->reg_mem_types);

            if (!(uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS)) {
                goto err_dereg;
            }
        }
    }

    /* Update md_map, note that MDs which did not support registration will be
     * missing from the map.*/
    *md_map_p = new_md_map;
    return UCS_OK;

err_dereg:
    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, alloc_md, mem_type,
                      alloc_md_memh_p, uct_memh, md_map_p);
    return status;

}

/**
 * @return Whether MD number 'md_index' is selected by the configuration as part
 *         of allocation method number 'config_method_index'.
 */
static int ucp_is_md_selected_by_config(ucp_context_h context,
                                        unsigned config_method_index,
                                        unsigned md_index)
{
    const char *cfg_cmpt_name;
    const char *cmpt_name;

    cfg_cmpt_name = context->config.alloc_methods[config_method_index].cmpt_name;
    cmpt_name     = context->tl_mds[md_index].attr.component_name;

    return !strncmp(cfg_cmpt_name, "*",      UCT_COMPONENT_NAME_MAX) ||
           !strncmp(cfg_cmpt_name, cmpt_name, UCT_COMPONENT_NAME_MAX);
}

static ucs_status_t
ucp_mem_do_alloc(ucp_context_h context, void *address, size_t length,
                 unsigned uct_flags, ucs_memory_type_t mem_type,
                 const char *name, uct_allocated_memory_t *mem)
{
    uct_alloc_method_t method;
    uct_mem_alloc_params_t params;
    unsigned method_index, md_index, num_mds;
    ucs_status_t status;
    uct_md_h *mds;

    mds = ucs_calloc(context->num_mds, sizeof(*mds), "temp mds");
    if (mds == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (method_index = 0; method_index < context->config.num_alloc_methods;
                    ++method_index)
    {
        method = context->config.alloc_methods[method_index].method;

        /* If we are trying MD method, gather all MDs which match the component
         * name specified in the configuration.
         */
        num_mds = 0;
        if (method == UCT_ALLOC_METHOD_MD) {
            for (md_index = 0; md_index < context->num_mds; ++md_index) {
                if (ucp_is_md_selected_by_config(context, method_index, md_index)) {
                    mds[num_mds++] = context->tl_mds[md_index].md;
                }
            }
        }

        memset(&params, 0, sizeof(params));
        params.field_mask      = UCT_MEM_ALLOC_PARAM_FIELD_FLAGS    |
                                 UCT_MEM_ALLOC_PARAM_FIELD_ADDRESS  |
                                 UCT_MEM_ALLOC_PARAM_FIELD_MEM_TYPE |
                                 UCT_MEM_ALLOC_PARAM_FIELD_MDS      |
                                 UCT_MEM_ALLOC_PARAM_FIELD_NAME;
        params.flags           = uct_flags;
        params.name            = name;
        params.mem_type        = mem_type;
        params.address         = address;
        params.mds.mds         = mds;
        params.mds.count       = num_mds;

        status = uct_mem_alloc(length, &method, 1, &params, mem);
        if (status == UCS_OK) {
            goto out;
        }
    }

    status = UCS_ERR_NO_MEMORY;

out:
    ucs_free(mds);
    return status;
}

static unsigned
ucp_mem_map_params2uct_flags(const ucp_mem_map_params_t *params)
{
    unsigned flags = 0;

    if (params->field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS) {
        if (params->flags & UCP_MEM_MAP_NONBLOCK) {
            flags |= UCT_MD_MEM_FLAG_NONBLOCK;
        }

        if (params->flags & UCP_MEM_MAP_FIXED) {
            flags |= UCT_MD_MEM_FLAG_FIXED;
        }
    }

    flags |= UCT_MD_MEM_ACCESS_ALL;
    /* TODO: disable atomic if ucp context does not have it */

    return flags;
}

static void ucp_memh_dereg(ucp_context_h context, ucp_mem_h memh,
                           ucp_md_map_t md_map)
{
    ucp_md_index_t md_index;
    ucs_status_t status;

    /* Unregister from all memory domains */
    ucs_for_each_bit(md_index, md_map) {
        ucs_assertv(md_index != memh->alloc_md_index,
                    "memh %p: md_index %u alloc_md_index %u", memh, md_index,
                    memh->alloc_md_index);

        ucs_trace("de-registering memh[%d]=%p", md_index, memh->uct[md_index]);
        ucs_assert(context->tl_mds[md_index].attr.flags & UCT_MD_FLAG_REG);
        status = uct_md_mem_dereg(context->tl_mds[md_index].md,
                                  memh->uct[md_index]);
        if (status != UCS_OK) {
            ucs_warn("failed to dereg from md[%d]=%s: %s", md_index,
                     context->tl_mds[md_index].rsc.md_name,
                     ucs_status_string(status));
        }

        memh->uct[md_index] = NULL;
    }
}

void ucp_memh_cleanup(ucp_context_h context, ucp_mem_h memh)
{
    ucp_md_map_t md_map = memh->md_map;
    uct_allocated_memory_t mem;
    ucs_status_t status;

    mem.address = ucp_memh_address(memh);
    mem.length  = ucp_memh_length(memh);
    mem.method  = memh->alloc_method;

    if (mem.method == UCT_ALLOC_METHOD_MD) {
        ucs_assert(memh->alloc_md_index != UCP_NULL_RESOURCE);
        mem.md   = context->tl_mds[memh->alloc_md_index].md;
        mem.memh = memh->uct[memh->alloc_md_index];
        md_map  &= ~UCS_BIT(memh->alloc_md_index);
    }

    /* Have a parent memory handle from rcache */
    if ((memh->parent != NULL) && (memh->parent != memh)) {
        ucp_memh_dereg(context, memh, md_map & ~memh->parent->md_map);
        ucp_memh_put(context, memh->parent);
    } else {
        ucp_memh_dereg(context, memh, md_map);
    }

    /* If the memory was also allocated, release it */
    if (memh->alloc_method != UCT_ALLOC_METHOD_LAST) {
        status = uct_mem_free(&mem);
        if (status != UCS_OK) {
            ucs_warn("failed to free: %s", ucs_status_string(status));
        }
    }
}

static void ucp_memh_register_log_fail(ucs_log_level_t log_level, void *address,
                                       size_t length, ucp_md_index_t md_index,
                                       ucp_context_h context,
                                       ucs_status_t status)
{
    ucs_log(log_level, "failed to register %p length %zu on md[%d]=%s: %s",
            address, length, md_index, context->tl_mds[md_index].rsc.md_name,
            ucs_status_string(status));
}

static ucs_status_t ucp_memh_register(ucp_context_h context, ucp_mem_h memh,
                                      ucp_md_map_t md_map, void *address,
                                      size_t length, unsigned uct_flags)
{
    ucp_md_map_t md_map_registered = 0;
    ucp_md_index_t md_index;
    ucs_status_t status;

    ucs_for_each_bit(md_index, md_map) {
        status = uct_md_mem_reg(context->tl_mds[md_index].md,
                                address, length, uct_flags,
                                &memh->uct[md_index]);
        if (ucs_unlikely(status != UCS_OK)) {
            if (uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) {
                ucp_memh_register_log_fail(UCS_LOG_LEVEL_DIAG, address, length,
                                           md_index, context, status);
                continue;
            }

            ucp_memh_register_log_fail(UCS_LOG_LEVEL_ERROR, address, length,
                                       md_index, context, status);
            ucp_memh_dereg(context, memh, md_map_registered);
            return status;
        }

        ucs_trace("registered address %p length %zu on md[%d]=%s %p",
                  address, length, md_index,
                  context->tl_mds[md_index].rsc.md_name,
                  memh->uct[md_index]);
        md_map_registered |= UCS_BIT(md_index);
    }

    memh->md_map |= md_map_registered;
    return UCS_OK;
}

static size_t ucp_memh_size(ucp_context_h context)
{
    return sizeof(ucp_mem_t) + (sizeof(uct_mem_h) * context->num_mds);
}

static void ucp_memh_set(ucp_mem_h memh, ucp_context_h context, void* address,
                         size_t length, ucs_memory_type_t mem_type,
                         uint8_t memh_flags, uct_alloc_method_t method)
{
    memh->super.super.start = (uintptr_t)address;
    memh->super.super.end   = (uintptr_t)address + length;
    memh->flags             = memh_flags;
    memh->context           = context;
    memh->mem_type          = mem_type;
    memh->alloc_method      = method;
    memh->alloc_md_index    = UCP_NULL_RESOURCE;
}

static ucs_status_t
ucp_memh_create(ucp_context_h context, void *address, size_t length,
                ucs_memory_type_t mem_type, uct_alloc_method_t method,
                uint8_t memh_flags, ucp_mem_h *memh_p)
{
    ucp_mem_h memh;

    memh = ucs_calloc(1, ucp_memh_size(context), "ucp_memh");
    if (memh == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    ucp_memh_set(memh, context, address, length, mem_type, memh_flags, method);

    if (context->rcache == NULL) {
        ucs_assert(context->imported_mem_rcaches == NULL);

        /* Point to self */
        memh->parent = memh;
    } else {
        ucs_assert(context->imported_mem_rcaches != NULL);
    }

    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t
ucp_memh_rcache_get(ucs_rcache_t *rcache, void *address, size_t length,
                    ucs_memory_type_t mem_type, uint8_t memh_flags,
                    ucp_mem_h *memh_p)
{
    ucp_mem_attr_t attr = {
        .mem_type = mem_type
    };
    ucs_rcache_region_t *rregion;
    ucs_status_t status;

    status = ucs_rcache_get(rcache, address, length, PROT_READ | PROT_WRITE,
                            &attr, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    *memh_p          = ucs_derived_of(rregion, ucp_mem_t);
    (*memh_p)->flags = UCP_MEM_FLAG_IN_RCACHE | memh_flags;

    return UCS_OK;
}

static ucs_status_t ucp_memh_create_from_mem(ucp_context_h context,
                                             const uct_allocated_memory_t *mem,
                                             ucp_mem_h *memh_p)
{
    ucp_md_index_t md_index;
    ucs_status_t status;
    ucp_mem_h memh;

    status = ucp_memh_create(context, mem->address, mem->length, mem->mem_type,
                             mem->method, 0, &memh);
    if (status != UCS_OK) {
        return status;
    }

    if (mem->method != UCT_ALLOC_METHOD_MD) {
        goto out;
    }

    for (md_index = 0; md_index < context->num_mds; md_index++) {
        if (mem->md != context->tl_mds[md_index].md) {
            continue;
        }

        memh->alloc_md_index = md_index;
        memh->uct[md_index]  = mem->memh;
        memh->md_map        |= UCS_BIT(md_index);
        ucs_trace("allocated address %p length %zu on md[%d]=%s %p",
                  mem->address, mem->length, md_index,
                  context->tl_mds[md_index].rsc.md_name, memh->uct[md_index]);
        break;
    }

    ucs_assert(memh->alloc_md_index != UCP_NULL_RESOURCE);

out:
    *memh_p = memh;
    return UCS_OK;
}

/**
 * Updates a memory handle by a parent's memory handle which stored in RCACHE.
 *
 * @param [in]  memh          Memory handle that should be updated.
 * @param [in]  cache_md_map  MD map of registered UCT memory keys stored in a
 *                            parent's RCACHE memory handle.
 * @param [out] memh_md_map_p Memory handle's MD map that should be updated by
 *                            MDs of UCT memory keys that copied from
 *                            @a cache_md_map.
 * @param [out] md_map_p      MD map that should be updated and it specifies
 *                            which MDs were not copied from a parent's RCACHE
 *                            memory handle.
 */
static UCS_F_ALWAYS_INLINE void
ucp_memh_update_from_parent_memh(ucp_mem_h memh,
                                 ucp_md_map_t cache_md_map,
                                 ucp_md_map_t *memh_md_map_p,
                                 ucp_md_map_t *md_map_p)
{
    ucp_md_index_t md_index;

    ucs_for_each_bit(md_index, cache_md_map) {
        memh->uct[md_index] = memh->parent->uct[md_index];
        *memh_md_map_p     |= UCS_BIT(md_index);
        *md_map_p          &= ~UCS_BIT(md_index);
    }
}

static ucs_status_t
ucp_memh_init_uct_reg(ucp_context_h context, ucp_mem_h memh, unsigned uct_flags)
{
    ucs_memory_type_t mem_type = memh->mem_type;
    ucp_md_map_t reg_md_map    = context->reg_md_map[mem_type] & ~memh->md_map;
    ucp_md_map_t cache_md_map  = context->cache_md_map[mem_type] & reg_md_map;
    void *address              = ucp_memh_address(memh);
    size_t length              = ucp_memh_length(memh);
    ucp_md_index_t md_index;
    ucs_status_t status;

    if (context->rcache == NULL) {
        status = ucp_memh_register(context, memh, reg_md_map, address, length,
                                   uct_flags);
        if (status != UCS_OK) {
            goto err;
        }

        memh->reg_id = context->next_memh_reg_id++;
    } else {
        status = ucp_memh_get(context, address, length, mem_type, cache_md_map,
                              uct_flags, memh->flags, &memh->parent);
        if (status != UCS_OK) {
            goto err;
        }

        ucs_for_each_bit(md_index, cache_md_map) {
            memh->uct[md_index] = memh->parent->uct[md_index];
            memh->md_map       |= UCS_BIT(md_index);
            reg_md_map         &= ~UCS_BIT(md_index);
        }

        status = ucp_memh_register(context, memh, reg_md_map, address, length,
                                   uct_flags);
        if (status != UCS_OK) {
            goto err_put;
        }
    }

    return UCS_OK;

err_put:
    ucp_memh_put(context, memh->parent);
err:
    return status;
}

ucs_status_t
ucp_memh_get_slow(ucp_context_h context, void *address, size_t length,
                  ucs_memory_type_t mem_type, ucp_md_map_t reg_md_map,
                  unsigned uct_flags, ucp_mem_h *memh_p)
{
    ucp_mem_h memh = NULL; /* To suppress compiler warning */
    void *reg_address;
    size_t reg_length;
    ucs_status_t status;
    ucs_memory_info_t mem_info;

    if (context->config.ext.reg_whole_alloc_bitmap & UCS_BIT(mem_type)) {
        ucp_memory_detect_internal(context, address, length, &mem_info);
        reg_address = mem_info.base_address;
        reg_length  = mem_info.alloc_length;
    } else {
        reg_address = address;
        reg_length  = length;
    }

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    if (context->rcache == NULL) {
        status = ucp_memh_create(context, reg_address, reg_length, mem_type,
                                 UCT_ALLOC_METHOD_LAST, 0, &memh);
    } else {
        status = ucp_memh_rcache_get(context->rcache, reg_address, reg_length,
                                     mem_type, 0, &memh);
    }

    if (status != UCS_OK) {
        goto out;
    }

    /* Reinitialize address and length, because they could be greater in case
     * a region which includes the searched region was found in the rcache */
    reg_address = ucp_memh_address(memh);
    reg_length  = ucp_memh_length(memh);

    ucs_assert(memh->mem_type == mem_type);

    status = ucp_memh_register(context, memh, ~memh->md_map & reg_md_map,
                               reg_address, reg_length, uct_flags);
    if (status != UCS_OK) {
        goto err_free_memh;
    }

    memh->reg_id = context->next_memh_reg_id++;

    ucs_trace("memh %p: registered address %p/%p length %zu/%zu md_map %"
              PRIx64 "/%" PRIx64, memh, reg_address, ucp_memh_address(memh),
              reg_length, ucp_memh_length(memh), reg_md_map, memh->md_map);
    *memh_p = memh;

out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;

err_free_memh:
    if (context->rcache == NULL) {
        ucs_free(memh);
    } else {
        ucs_rcache_region_put_unsafe(context->rcache, &memh->super);
    }

    goto out;
}

static ucs_status_t
ucp_memh_alloc(ucp_context_h context, void *address, size_t length,
               ucs_memory_type_t mem_type, unsigned uct_flags,
               const char *alloc_name, ucp_mem_h *memh_p)
{
    uct_allocated_memory_t mem;
    ucs_status_t status;
    ucp_mem_h memh;

    status = ucp_mem_do_alloc(context, address, length, uct_flags, mem_type,
                              alloc_name, &mem);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_memh_create_from_mem(context, &mem, &memh);
    if (status != UCS_OK) {
        goto err_dealloc;
    }

    status = ucp_memh_init_uct_reg(context, memh, UCT_MD_MEM_ACCESS_ALL);
    if (status != UCS_OK) {
        goto err_free_memh;
    }

    *memh_p = memh;
    return UCS_OK;

err_free_memh:
    ucs_free(memh);
err_dealloc:
    uct_mem_free(&mem);
out:
    return status;
}

static ucs_status_t
ucp_memh_normal_reg(ucp_context_h context, void *address, size_t length,
                    ucs_memory_type_t mem_type, unsigned uct_flags,
                    ucp_mem_h *memh_p)
{
    ucs_status_t status;
    ucp_mem_h memh;

    status = ucp_memh_create(context, address, length, mem_type,
                             UCT_ALLOC_METHOD_LAST, 0, &memh);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_memh_init_uct_reg(context, memh, uct_flags);
    if (status != UCS_OK) {
        goto err_free_memh;
    }

    *memh_p = memh;

    return UCS_OK;

err_free_memh:
    ucs_free(memh);
out:
    return status;
}

static ucs_status_t
ucp_memh_import(ucp_context_h context, const void *export_mkey_buffer,
                ucp_mem_h *memh_p);

static UCS_F_ALWAYS_INLINE int
ucp_mem_map_is_zero(uint8_t memh_flags, size_t length,
                    const void *exported_memh_buffer)
{
    if (!(memh_flags & UCP_MEM_FLAG_IMPORTED)) {
        return length == 0;
    }

    return *(uint16_t*)exported_memh_buffer == sizeof(ucp_memh_dummy_buffer);
}

/* Matrix of behavior
 * |--------------------------------------------------------------------------------|
 * | parameter |                             value                                  |
 * |-----------|--------------------------------------------------------------------|
 * | ALLOCATE  |  0     |     1     |  0  |  0  |  1  |     1     |  0  |     1     |
 * | FIXED     |  0     |     0     |  1  |  0  |  1  |     0     |  1  |     1     |
 * | addr      |  0     |     0     |  0  |  1  |  0  |     1     |  1  |     1     |
 * |-----------|--------|-----------|-----|-----|-----|-----------|-----|-----------|
 * | result    | err if | alloc/reg | err | reg | err | alloc/reg | err | alloc/reg |
 * |           | len >0 |           |     |     |     |  (hint)   |     | (fixed)   |
 * |--------------------------------------------------------------------------------|
 */
ucs_status_t ucp_mem_map(ucp_context_h context, const ucp_mem_map_params_t *params,
                         ucp_mem_h *memh_p)
{
    uint8_t memh_flags = 0;
    ucs_memory_type_t mem_type;
    ucp_memory_info_t mem_info;
    ucs_status_t status;
    unsigned uct_flags;
    unsigned flags;
    void *address;
    const void *exported_memh_buffer;
    size_t length;
    ucp_mem_h memh;

    if (!(params->field_mask &
          (UCP_MEM_MAP_PARAM_FIELD_LENGTH |
           UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER))) {
        ucs_error("the length value or exported_memh_buffer for mapping memory"
                  " aren't set: %s", ucs_status_string(UCS_ERR_INVALID_PARAM));
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    address              = UCP_PARAM_VALUE(MEM_MAP, params, address, ADDRESS,
                                           NULL);
    length               = UCP_PARAM_VALUE(MEM_MAP, params, length, LENGTH, 0);
    flags                = UCP_PARAM_VALUE(MEM_MAP, params, flags, FLAGS, 0);
    exported_memh_buffer = UCP_PARAM_VALUE(MEM_MAP, params, exported_memh_buffer,
                                           EXPORTED_MEMH_BUFFER, NULL);

    if ((flags & UCP_MEM_MAP_FIXED) &&
        ((uintptr_t)address % ucs_get_page_size())) {
        ucs_error("UCP_MEM_MAP_FIXED flag requires page aligned address");
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (flags & UCP_MEM_MAP_ALLOCATE) {
        mem_type = UCP_PARAM_VALUE(MEM_MAP, params, memory_type, MEMORY_TYPE,
                                   UCS_MEMORY_TYPE_HOST);
    } else if (!(params->field_mask & UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE) ||
               (params->memory_type == UCS_MEMORY_TYPE_UNKNOWN)) {
        ucp_memory_detect(context, address, length, &mem_info);
        mem_type = mem_info.type;
    } else {
        if (params->memory_type > UCS_MEMORY_TYPE_LAST) {
            ucs_error("invalid memory type %d", params->memory_type);
            status = UCS_ERR_INVALID_PARAM;
            goto out;
        }
        mem_type = params->memory_type;
    }

    if (exported_memh_buffer != NULL) {
        memh_flags |= UCP_MEM_FLAG_IMPORTED;
    }

    if (address == NULL) {
        if (!(flags & UCP_MEM_MAP_ALLOCATE) && (length > 0)) {
            ucs_error("undefined address with nonzero length requires "
                      "UCP_MEM_MAP_ALLOCATE flag");
            status = UCS_ERR_INVALID_PARAM;
            goto out;
        }
    } else if ((!(flags & UCP_MEM_MAP_ALLOCATE) &&
                (flags & UCP_MEM_MAP_FIXED))) {
        ucs_error("wrong combination of flags when address is defined");
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (ucp_mem_map_is_zero(memh_flags, length, exported_memh_buffer)) {
        ucs_assert(ucp_memh_address(&ucp_mem_dummy_handle.memh) == NULL);
        ucs_assert(ucp_memh_length(&ucp_mem_dummy_handle.memh) == 0);

        ucs_debug("mapping zero length buffer, return dummy memh");
        *memh_p = &ucp_mem_dummy_handle.memh;
        status  = UCS_OK;
        goto out;
    }

    uct_flags = ucp_mem_map_params2uct_flags(params);

    if (memh_flags & UCP_MEM_FLAG_IMPORTED) {
        status = ucp_memh_import(context, exported_memh_buffer, &memh);
    } else if (flags & UCP_MEM_MAP_ALLOCATE) {
        status = ucp_memh_alloc(context, address, length, mem_type,
                                uct_flags, "user memory", &memh);
    } else {
        status = ucp_memh_normal_reg(context, address, length, mem_type,
                                     uct_flags, &memh);
    }

    if (status != UCS_OK) {
        goto out;
    }

    ucs_assert(memh->md_map != 0);
    ucs_assert(memh->parent != NULL);

    *memh_p = memh;
    return UCS_OK;

out:
    return status;
}

ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh)
{
    ucp_memh_put(context, memh);
    return UCS_OK;
}

ucs_status_t ucp_mem_type_reg_buffers(ucp_worker_h worker, void *remote_addr,
                                      size_t length, ucs_memory_type_t mem_type,
                                      ucp_md_index_t md_index, uct_mem_h *memh,
                                      ucp_md_map_t *md_map,
                                      uct_rkey_bundle_t *rkey_bundle)
{
    ucp_context_h context           = worker->context;
    const uct_md_attr_v2_t *md_attr = &context->tl_mds[md_index].attr;
    uct_component_h cmpt;
    ucp_tl_md_t *tl_md;
    ucs_status_t status;
    char *rkey_buffer;

    if (!(md_attr->flags & UCT_MD_FLAG_NEED_RKEY)) {
        rkey_bundle->handle = NULL;
        rkey_bundle->rkey   = UCT_INVALID_RKEY;
        status              = UCS_OK;
        goto out;
    }

    tl_md  = &context->tl_mds[md_index];
    cmpt   = context->tl_cmpts[tl_md->cmpt_index].cmpt;

    status = ucp_mem_rereg_mds(context, UCS_BIT(md_index), remote_addr, length,
                               UCT_MD_MEM_ACCESS_ALL,
                               NULL, mem_type, NULL, memh, md_map);
    if (status != UCS_OK) {
        goto out;
    }

    rkey_buffer = ucs_alloca(md_attr->rkey_packed_size);
    status      = uct_md_mkey_pack(tl_md->md, memh[0], rkey_buffer);
    if (status != UCS_OK) {
        ucs_error("failed to pack key from md[%d]: %s",
                  md_index, ucs_status_string(status));
        goto out_dereg_mem;
    }

    status = uct_rkey_unpack(cmpt, rkey_buffer, rkey_bundle);
    if (status != UCS_OK) {
        ucs_error("failed to unpack key from md[%d]: %s",
                  md_index, ucs_status_string(status));
        goto out_dereg_mem;
    }

    return UCS_OK;

out_dereg_mem:
    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL, mem_type, NULL,
                      memh, md_map);
out:
    *memh = UCT_MEM_HANDLE_NULL;
    return status;
}

void ucp_mem_type_unreg_buffers(ucp_worker_h worker, ucs_memory_type_t mem_type,
                                ucp_md_index_t md_index, uct_mem_h *memh,
                                ucp_md_map_t *md_map,
                                uct_rkey_bundle_t *rkey_bundle)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t cmpt_index;

    if (rkey_bundle->rkey != UCT_INVALID_RKEY) {
        cmpt_index = context->tl_mds[md_index].cmpt_index;
        uct_rkey_release(context->tl_cmpts[cmpt_index].cmpt, rkey_bundle);
    }

    ucp_mem_rereg_mds(context, 0, NULL, 0, 0, NULL, mem_type, NULL,
                      memh, md_map);
}

ucs_status_t ucp_mem_query(const ucp_mem_h memh, ucp_mem_attr_t *attr)
{
    if (attr->field_mask & UCP_MEM_ATTR_FIELD_ADDRESS) {
        attr->address = ucp_memh_address(memh);
    }

    if (attr->field_mask & UCP_MEM_ATTR_FIELD_LENGTH) {
        attr->length = ucp_memh_length(memh);
    }

    if (attr->field_mask & UCP_MEM_ATTR_FIELD_MEM_TYPE) {
        attr->mem_type = memh->mem_type;
    }

    return UCS_OK;
}

static ucs_status_t ucp_advice2uct(unsigned ucp_advice, uct_mem_advice_t *uct_advice)
{
    switch(ucp_advice) {
    case UCP_MADV_NORMAL:
        *uct_advice = UCT_MADV_NORMAL;
        return UCS_OK;
    case UCP_MADV_WILLNEED:
        *uct_advice = UCT_MADV_WILLNEED;
        return UCS_OK;
    }
    return UCS_ERR_INVALID_PARAM;
}

ucs_status_t
ucp_mem_advise(ucp_context_h context, ucp_mem_h memh,
               ucp_mem_advise_params_t *params)
{
    ucs_status_t status, tmp_status;
    int md_index;
    uct_mem_advice_t uct_advice;
    uct_mem_h uct_memh;

    if (!ucs_test_all_flags(params->field_mask,
                            UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS|
                            UCP_MEM_ADVISE_PARAM_FIELD_LENGTH|
                            UCP_MEM_ADVISE_PARAM_FIELD_ADVICE)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((params->address < ucp_memh_address(memh)) ||
        (UCS_PTR_BYTE_OFFSET(params->address, params->length) >
         UCS_PTR_BYTE_OFFSET(ucp_memh_address(memh), ucp_memh_length(memh)))) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucp_advice2uct(params->advice, &uct_advice);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("advice buffer %p length %llu memh %p flags %x",
               params->address, (unsigned long long)params->length, memh,
               params->advice);

    if (ucp_memh_is_zero_length(memh)) {
        return UCS_OK;
    }

    UCP_THREAD_CS_ENTER(&context->mt_lock);

    status = UCS_OK;
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        uct_memh = memh->uct[md_index];
        if (!(context->tl_mds[md_index].attr.flags & UCT_MD_FLAG_ADVISE) ||
            (uct_memh == NULL)) {
            continue;
        }
        tmp_status = uct_md_mem_advise(context->tl_mds[md_index].md, uct_memh,
                                       params->address, params->length, uct_advice);
        if (tmp_status != UCS_OK) {
            status = tmp_status;
        }
    }

    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

static ucs_status_t
ucp_mpool_malloc(ucp_worker_h worker, ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    /* Need to get default flags from ucp_mem_map_params2uct_flags() */
    ucp_mem_map_params_t mem_params = {};
    ucp_mem_desc_t *chunk_hdr;
    ucp_mem_h memh;
    ucs_status_t status;

    status = ucp_memh_alloc(worker->context, NULL,
                            *size_p + sizeof(*chunk_hdr), UCS_MEMORY_TYPE_HOST,
                            ucp_mem_map_params2uct_flags(&mem_params),
                            ucs_mpool_name(mp), &memh);
    if (status != UCS_OK) {
        goto out;
    }

    chunk_hdr       = ucp_memh_address(memh);
    chunk_hdr->memh = memh;
    *chunk_p        = chunk_hdr + 1;
    *size_p         = ucp_memh_length(memh) - sizeof(*chunk_hdr);
out:
    return status;
}

static void
ucp_mpool_free(ucp_worker_h worker, ucs_mpool_t *mp, void *chunk)
{
    ucp_mem_desc_t *chunk_hdr;

    chunk_hdr = (ucp_mem_desc_t*)chunk - 1;
    ucp_memh_put(worker->context, chunk_hdr->memh);
}

void ucp_mpool_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucp_mem_desc_t *elem_hdr  = obj;
    ucp_mem_desc_t *chunk_hdr = (ucp_mem_desc_t*)((ucp_mem_desc_t*)chunk - 1);
    elem_hdr->memh = chunk_hdr->memh;
}

static ucs_status_t
ucp_rndv_frag_malloc_mpools(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucp_rndv_mpool_priv_t *mpriv = ucs_mpool_priv(mp);
    ucp_context_h context        = mpriv->worker->context;
    ucs_memory_type_t mem_type   = mpriv->mem_type;
    size_t frag_size             = context->config.ext.rndv_frag_size[mem_type];
    ucp_rndv_frag_mp_chunk_hdr_t *chunk_hdr;
    ucs_status_t status;
    unsigned num_elems;

    /* metadata */
    chunk_hdr = ucs_malloc(sizeof(*chunk_hdr) + *size_p, "chunk_hdr");
    if (chunk_hdr == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    num_elems = ucs_mpool_num_elems_per_chunk(
            mp, (ucs_mpool_chunk_t*)(chunk_hdr + 1), *size_p);

    /* payload; need to get default flags from ucp_mem_map_params2uct_flags() */
    status = ucp_memh_alloc(context, NULL, frag_size * num_elems, mem_type,
                            UCT_MD_MEM_ACCESS_RMA, ucs_mpool_name(mp),
                            &chunk_hdr->memh);
    if (status != UCS_OK) {
        return status;
    }

    chunk_hdr->next_frag_ptr = ucp_memh_address(chunk_hdr->memh);
    *chunk_p                 = chunk_hdr + 1;
    return UCS_OK;
}

static void
ucp_rndv_frag_free_mpools(ucs_mpool_t *mp, void *chunk)
{
    ucp_rndv_mpool_priv_t *mpriv = ucs_mpool_priv(mp);
    ucp_rndv_frag_mp_chunk_hdr_t *chunk_hdr;

    chunk_hdr = (ucp_rndv_frag_mp_chunk_hdr_t*)chunk - 1;
    ucp_memh_put(mpriv->worker->context, chunk_hdr->memh);
    ucs_free(chunk_hdr);
}

void ucp_frag_mpool_obj_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucp_rndv_frag_mp_chunk_hdr_t *chunk_hdr = (ucp_rndv_frag_mp_chunk_hdr_t*)chunk - 1;
    void *next_frag_ptr                     = chunk_hdr->next_frag_ptr;
    ucp_rndv_mpool_priv_t *mpriv            = ucs_mpool_priv(mp);
    ucs_memory_type_t mem_type              = mpriv->mem_type;
    ucp_context_h context                   = mpriv->worker->context;
    ucp_mem_desc_t *elem_hdr                = obj;
    size_t frag_size;

    frag_size                = context->config.ext.rndv_frag_size[mem_type];
    elem_hdr->memh           = chunk_hdr->memh;
    elem_hdr->ptr            = next_frag_ptr;
    chunk_hdr->next_frag_ptr = UCS_PTR_BYTE_OFFSET(next_frag_ptr, frag_size);
}

ucs_status_t ucp_reg_mpool_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, reg_mp);

    return ucp_mpool_malloc(worker, mp, size_p, chunk_p);
}

void ucp_reg_mpool_free(ucs_mpool_t *mp, void *chunk)
{
    ucp_worker_h worker = ucs_container_of(mp, ucp_worker_t, reg_mp);

    ucp_mpool_free(worker, mp, chunk);
}

ucs_status_t ucp_frag_mpool_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    return ucp_rndv_frag_malloc_mpools(mp, size_p, chunk_p);
}

void ucp_frag_mpool_free(ucs_mpool_t *mp, void *chunk)
{
    ucp_rndv_frag_free_mpools(mp, chunk);
}

void ucp_mem_print_info(const char *mem_size, ucp_context_h context, FILE *stream)
{
    size_t min_page_size, max_page_size;
    ucp_mem_map_params_t mem_params;
    size_t mem_size_value;
    char memunits_str[32];
    ucs_status_t status;
    unsigned md_index;
    ucp_mem_h memh;

    status = ucs_str_to_memunits(mem_size, &mem_size_value);
    if (status != UCS_OK) {
        printf("<Failed to convert a memunits string>\n");
        return;
    }

    mem_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                            UCP_MEM_MAP_PARAM_FIELD_LENGTH  |
                            UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_params.address    = NULL;
    mem_params.length     = mem_size_value;
    mem_params.flags      = UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(context, &mem_params, &memh);
    if (status != UCS_OK) {
        printf("<Failed to map memory of size %s>\n", mem_size);
        return;
    }

    fprintf(stream, "#\n");
    fprintf(stream, "# UCP memory allocation\n");
    fprintf(stream, "#\n");

    ucs_memunits_to_str(ucp_memh_length(memh), memunits_str,
                        sizeof(memunits_str));
    fprintf(stream, "#  allocated %s at address %p with ", memunits_str,
            ucp_memh_address(memh));

    if (memh->alloc_md_index == UCP_NULL_RESOURCE) {
        fprintf(stream, "%s", uct_alloc_method_names[memh->alloc_method]);
    } else {
        fprintf(stream, "%s", context->tl_mds[memh->alloc_md_index].rsc.md_name);
    }

    ucs_get_mem_page_size(ucp_memh_address(memh), ucp_memh_length(memh),
                          &min_page_size, &max_page_size);
    ucs_memunits_to_str(min_page_size, memunits_str, sizeof(memunits_str));
    fprintf(stream, ", pagesize: %s", memunits_str);
    if (min_page_size != max_page_size) {
        ucs_memunits_to_str(max_page_size, memunits_str, sizeof(memunits_str));
        fprintf(stream, "-%s", memunits_str);
    }

    fprintf(stream, "\n");
    fprintf(stream, "#  registered on: ");
    ucs_for_each_bit(md_index, memh->md_map) {
        fprintf(stream, "%s ", context->tl_mds[md_index].rsc.md_name);
    }
    fprintf(stream, "\n");
    fprintf(stream, "#\n");

    status = ucp_mem_unmap(context, memh);
    if (status != UCS_OK) {
        printf("<Failed to unmap memory of size %s>\n", mem_size);
    }
}

static ucs_status_t
ucp_mem_rcache_mem_reg_cb(void *ctx, ucs_rcache_t *rcache, void *arg,
                          ucs_rcache_region_t *rregion,
                          uint16_t rcache_mem_reg_flags)
{
    ucp_context_h context = (ucp_context_h)ctx;
    ucp_mem_h memh        = ucs_derived_of(rregion, ucp_mem_t);
    ucp_mem_attr_t *attr  = arg;

    memh->context        = context;
    memh->md_map         = 0;
    memh->alloc_md_index = UCP_NULL_RESOURCE;
    memh->alloc_method   = UCT_ALLOC_METHOD_LAST;
    memh->mem_type       = attr->mem_type;

    return UCS_OK;
}

static void ucp_mem_rcache_mem_dereg_cb(void *ctx, ucs_rcache_t *rcache,
                                        ucs_rcache_region_t *rregion)
{
    ucp_mem_h memh = ucs_derived_of(rregion, ucp_mem_t);

    ucp_memh_cleanup((ucp_context_h)ctx, memh);
}

static void ucp_mem_rcache_dump_region_cb(void *rcontext, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion, char *buf,
                                         size_t max)
{
    UCS_STRING_BUFFER_FIXED(strb, buf, max);
    ucp_mem_h memh        = ucs_derived_of(rregion, ucp_mem_t);
    ucp_context_h context = rcontext;
    unsigned md_index;

    ucs_for_each_bit(md_index, memh->md_map) {
        ucs_string_buffer_appendf(&strb, " md[%d]=%s", md_index,
                                  context->tl_mds[md_index].rsc.md_name);
        if (memh->alloc_md_index == md_index) {
            ucs_string_buffer_appendf(&strb, "(alloc)");
        }
    }
}

static ucs_rcache_ops_t ucp_mem_rcache_ops = {
    .mem_reg     = ucp_mem_rcache_mem_reg_cb,
    .mem_dereg   = ucp_mem_rcache_mem_dereg_cb,
    .dump_region = ucp_mem_rcache_dump_region_cb
};

static ucs_status_t
ucp_mem_rcache_create(ucp_context_h context, const char *name,
                      ucs_rcache_t **rcache_p)
{
    ucs_rcache_params_t rcache_params;

    rcache_params.region_struct_size = ucp_memh_size(context);
    rcache_params.max_alignment      = ucs_get_page_size();
    rcache_params.max_unreleased     = SIZE_MAX;
    rcache_params.max_regions        = -1;
    rcache_params.max_size           = -1;
    rcache_params.ucm_event_priority = 500; /* Default UCT pri - 1000 */
    rcache_params.ucm_events         = UCM_EVENT_VM_UNMAPPED |
                                       UCM_EVENT_MEM_TYPE_FREE;
    rcache_params.context            = context;
    rcache_params.ops                = &ucp_mem_rcache_ops;
    rcache_params.flags              = UCS_RCACHE_FLAG_PURGE_ON_FORK;
    rcache_params.alignment          = UCS_RCACHE_MIN_ALIGNMENT;

    return ucs_rcache_create(&rcache_params, name, ucs_stats_get_root(),
                             rcache_p);
}

ucs_status_t ucp_mem_rcache_init(ucp_context_h context)
{
    ucs_status_t status;

    status = ucp_mem_rcache_create(context, "ucp_rcache", &context->rcache);
    if (status != UCS_OK) {
        goto err;
    }

    context->imported_mem_rcaches =
            kh_init(ucp_context_imported_mem_rcaches_hash);
    if (context->imported_mem_rcaches == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_rcache_destroy;
    }

    return UCS_OK;

err_rcache_destroy:
    ucs_rcache_destroy(context->rcache);
err:
    return status;
}

void ucp_mem_rcache_cleanup(ucp_context_h context)
{
    ucs_rcache_t *rcache;

    if (context->rcache != NULL) {
        ucs_rcache_destroy(context->rcache);
    }

    if (context->imported_mem_rcaches != NULL) {
        kh_foreach_value(context->imported_mem_rcaches, rcache, {
            ucs_rcache_destroy(rcache);
        })
        kh_destroy(ucp_context_imported_mem_rcaches_hash,
                   context->imported_mem_rcaches);
    }
}

ucs_status_t ucp_mem_reg_md_map_update(ucp_context_h context)
{
    ucp_mem_dummy_handle_t memh = {
        .memh.alloc_md_index = UCP_NULL_RESOURCE,
    };
    ucs_status_t status;
    uint8_t buff;

    status = ucp_memh_register(context, &memh.memh,
                               context->reg_md_map[UCS_MEMORY_TYPE_HOST],
                               &buff, sizeof(buff),
                               UCT_MD_MEM_ACCESS_ALL |
                               UCT_MD_MEM_FLAG_HIDE_ERRORS);
    if (status != UCS_OK) {
        return status;
    }

    context->reg_md_map[UCS_MEMORY_TYPE_HOST] = memh.memh.md_map;
    ucp_memh_dereg(context, &memh.memh, memh.memh.md_map);
    return UCS_OK;
}

static ucp_md_map_t ucp_find_md_map(ucp_context_h context,
                                    const void *component_name,
                                    size_t component_name_size,
                                    const void *global_id_buf,
                                    size_t global_id_size)
{
    ucp_md_map_t md_map = 0;
    ucp_md_index_t md_index;
    uct_md_attr_v2_t *md_attr;
    size_t component_name_cmp_size, global_id_cmp_size;

    for (md_index = 0; md_index < context->num_mds; md_index++) {
        md_attr                 = &context->tl_mds[md_index].attr;
        component_name_cmp_size = ucs_min(strlen(md_attr->component_name),
                                          component_name_size);
        global_id_cmp_size      =
                ucs_min(ucp_memh_global_id_packed_size(md_attr),
                        global_id_size);

        if ((memcmp(md_attr->component_name, component_name,
                    component_name_cmp_size) == 0) &&
            (memcmp(md_attr->global_id, global_id_buf,
                    global_id_cmp_size) == 0)) {
            md_map |= UCS_BIT(md_index);
            continue;
        }
    }

    return md_map;
}

static void
ucp_memh_import_parse_tl_mkey_data(ucp_context_h context, 
                                   const void **start_p,
                                   const void **tl_mkey_buf_p,
                                   ucp_md_map_t *md_map_p)
{
    const void *p = *start_p;
    const void *next_tl_md_p, *end;
    size_t tl_mkey_data_size;
    size_t tl_mkey_size, component_name_size, global_id_size;
    const void *tl_mkey_buf, *component_name_buf, *global_id_buf;
    ucp_md_map_t md_map;

    /* TL mkey data size */
    tl_mkey_data_size = *ucs_serialize_next(&p, uint16_t);
    end               = UCS_PTR_BYTE_OFFSET(*start_p, tl_mkey_data_size);
    ucs_assertv((tl_mkey_data_size <= UINT16_MAX) && (tl_mkey_data_size != 0),
                "tl_mkey_data_size %zu", tl_mkey_data_size);

    /* TL mkey size */
    tl_mkey_size = *ucp_memh_serialize_next(&p, uint8_t, end, 0u);
    ucs_assertv((tl_mkey_size <= UINT8_MAX) && (tl_mkey_size != 0),
                "tl_mkey_size %zu", tl_mkey_size);

    /* TL mkey */
    tl_mkey_buf = ucs_serialize_next_raw(&p, void, tl_mkey_size);

    /* Size of component name */
    component_name_size = *ucp_memh_serialize_next(&p, uint8_t, end, 0u);

    /* Component name */
    component_name_buf = ucs_serialize_next_raw(&p, void,
                                                component_name_size);

    /* Size of global MD identifier */
    global_id_size = *ucp_memh_serialize_next(&p, uint8_t, end, 0u);

    /* Global MD identifier */
    global_id_buf = ucs_serialize_next_raw(&p, void, global_id_size);

    /* Get local MD index which corresponds to the remote one */
    md_map = ucp_find_md_map(context, component_name_buf,
                             component_name_size, global_id_buf,
                             global_id_size);

    next_tl_md_p = end;
    ucs_assertv(p <= next_tl_md_p, "p=%p, next_tl_md_p=%p", p, next_tl_md_p);

    *start_p       = next_tl_md_p;
    *tl_mkey_buf_p = tl_mkey_buf;
    *md_map_p      = md_map;
}

typedef struct {
    ucp_md_index_t md_index;
    const void     *tl_mkey_buf;
} ucp_memh_import_attach_params_t;

static ucs_status_t
ucp_memh_import_attach(ucp_context_h context, ucp_mem_h memh,
                       ucp_memh_import_attach_params_t *attach_params_array,
                       unsigned attach_params_num)
{
    ucp_md_index_t md_index = UCP_NULL_RESOURCE;
    unsigned attach_params_iter;
    const void *tl_mkey_buf;
    uct_md_mem_attach_params_t attach_params;
    uct_md_attr_v2_t *md_attr;
    ucs_status_t status;
    uct_mem_h uct_memh;

    for (attach_params_iter = 0; attach_params_iter < attach_params_num;
         ++attach_params_iter) {
        md_index    = attach_params_array[attach_params_iter].md_index;
        tl_mkey_buf = attach_params_array[attach_params_iter].tl_mkey_buf;
        md_attr     = &context->tl_mds[md_index].attr;
        ucs_assert_always(md_attr->flags & UCT_MD_FLAG_EXPORTED_MKEY);

        attach_params.field_mask = UCT_MD_MEM_ATTACH_FIELD_FLAGS;
        attach_params.flags      = UCT_MD_MEM_ATTACH_FLAG_HIDE_ERRORS;

        status = uct_md_mem_attach(context->tl_mds[md_index].md, tl_mkey_buf,
                                   &attach_params, &uct_memh);
        if (ucs_unlikely(status != UCS_OK)) {
            ucs_trace("failed to attach memory on '%s': %s",
                      md_attr->component_name, ucs_status_string(status));
            continue;
        }

        memh->uct[md_index] = uct_memh;

        ucs_trace("imported address %p length %zu on md[%d]=%s: uct_memh %p",
                  ucp_memh_address(memh), ucp_memh_length(memh), md_index,
                  context->tl_mds[md_index].rsc.md_name,
                  memh->uct[md_index]);
        memh->md_map |= UCS_BIT(md_index);
    }

    if (memh->md_map == 0) {
        ucs_error("no suitable UCT memory domains to perform importing on");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_memh_import_slow(ucp_context_h context, ucs_rcache_t *existing_rcache,
                     ucp_mem_h user_memh, ucp_md_map_t remote_md_map,
                     ucp_memh_import_attach_params_t *attach_params_array,
                     unsigned attach_params_num, void *address, size_t length,
                     uint64_t remote_uuid, ucs_memory_type_t mem_type,
                     uint64_t reg_id)
{
    ucs_rcache_t *rcache;
    ucs_status_t status;
    ucp_mem_h memh;
    char rcache_name[128];
    khiter_t iter;
    int ret;

    ucs_assert(user_memh != NULL);

    UCP_THREAD_CS_ENTER(&context->mt_lock);

    if (ucs_likely(context->imported_mem_rcaches != NULL)) {
        if (existing_rcache == NULL) {
            ucs_snprintf_safe(rcache_name, sizeof(rcache_name),
                              "ucp rcache[%zu] of imported memory buffer",
                              remote_uuid);
            status = ucp_mem_rcache_create(context, rcache_name, &rcache);
            if (status != UCS_OK) {
                goto err;
            }

            iter = kh_put(ucp_context_imported_mem_rcaches_hash,
                          context->imported_mem_rcaches, remote_uuid, &ret);
            ucs_assertv((ret != UCS_KH_PUT_FAILED) &&
                        (ret != UCS_KH_PUT_KEY_PRESENT), "ret %d", ret);

            kh_value(context->imported_mem_rcaches, iter) = rcache;
        } else {
            rcache = existing_rcache;
        }

        status = ucp_memh_rcache_get(rcache, address, length, mem_type,
                                     UCP_MEM_FLAG_IMPORTED, &memh);
        if (status != UCS_OK) {
            goto err_rcache_destroy;
        }

        user_memh->parent = memh;
    } else {
        memh   = user_memh;
        rcache = NULL;
    }

    memh->reg_id               = reg_id;
    memh->remote_import_md_map = remote_md_map;
    memh->remote_uuid          = remote_uuid;
    status                     = ucp_memh_import_attach(context, memh,
                                                        attach_params_array,
                                                        attach_params_num);
    if (status != UCS_OK) {
        goto err_memh_free;
    }

    UCP_THREAD_CS_EXIT(&context->mt_lock);

    return UCS_OK;

err_memh_free:
    if (rcache != NULL) {
        ucs_rcache_region_put_unsafe(rcache, &memh->super);
    }
err_rcache_destroy:
    if ((rcache != NULL) && (rcache != existing_rcache)) { 
        /* Rcache was allocated here - remove it from hash and destroy */
        iter = kh_get(ucp_context_imported_mem_rcaches_hash,
                      context->imported_mem_rcaches, remote_uuid);
        ucs_assert(iter != kh_end(context->imported_mem_rcaches));
        kh_del(ucp_context_imported_mem_rcaches_hash,
               context->imported_mem_rcaches, iter);
        ucs_rcache_destroy(rcache);
    }
err:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

static ucs_status_t
ucp_memh_import(ucp_context_h context, const void *export_mkey_buffer,
                ucp_mem_h *memh_p)
{
    ucs_rcache_t *rcache = NULL;
    const void *p        = export_mkey_buffer;
    const void *end;
    ucp_md_index_t md_index;
    ucp_mem_h memh, rcache_memh;
    ucp_md_index_t remote_md_index;
    ucs_status_t status;
    ucp_md_map_t remote_md_map, local_md_map, reg_md_map;
    ucp_memh_import_attach_params_t *attach_params_array;
    unsigned attach_params_num;
    ucs_memory_type_t mem_type;
    uint64_t UCS_V_UNUSED flags;
    uint64_t UCS_V_UNUSED mem_info_parsed_size;
    uint64_t remote_uuid;
    const void *tl_mkey_buf;
    ucs_rcache_region_t *rregion;
    khiter_t iter;
    void *address;
    size_t length;
    uint64_t memh_info_size;
    uint64_t reg_id;

    ucs_assert(export_mkey_buffer != NULL);

    /* Common memory handle information */
    memh_info_size = *ucs_serialize_next(&p, uint16_t);
    end            = UCS_PTR_BYTE_OFFSET(export_mkey_buffer, memh_info_size);

    flags          = *ucp_memh_serialize_next(&p, uint64_t, end, 0lu);
    remote_md_map  = *ucp_memh_serialize_next(&p, ucp_md_map_t, end, 0u);
    mem_type       = *ucp_memh_serialize_next(&p, uint8_t, end,
                                              UCS_MEMORY_TYPE_HOST);

    /* Exported memory handle stuff */
    address     = (void*)*ucp_memh_serialize_next(&p, uint64_t, end, 0lu);
    length      = *ucp_memh_serialize_next(&p, uint64_t, end, 0lu);
    remote_uuid = *ucp_memh_serialize_next(&p, uint64_t, end, 0lu);
    reg_id      = *ucp_memh_serialize_next(&p, uint64_t, end, UINT64_MAX);

    if (ucs_unlikely(!(flags & UCP_MEMH_BUFFER_FLAG_EXPORTED))) {
        ucs_error("passed memory handle buffer which does not contain exported"
                  " memory handle: flags 0x%lx, address %p, length %" PRIu64,
                  flags, (void*)address, length);
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_assert(length != 0);

    mem_info_parsed_size = UCS_PTR_BYTE_DIFF(export_mkey_buffer, p);
    ucs_assertv(mem_info_parsed_size <= memh_info_size,
                "mem_info: parsed_size=%" PRIu64 " packed_size=%" PRIu64,
                mem_info_parsed_size, memh_info_size);
    p = end;

    attach_params_num   = 0;
    attach_params_array = ucs_alloca(sizeof(*attach_params_array) *
                                     ucs_popcount(remote_md_map) *
                                     context->num_mds);

    ucs_for_each_bit(remote_md_index, remote_md_map) {
        ucp_memh_import_parse_tl_mkey_data(context, &p, &tl_mkey_buf,
                                           &local_md_map);

        if (local_md_map == 0) {
            continue;
        }

        ucs_for_each_bit(md_index, local_md_map) {
            attach_params_array[attach_params_num].md_index    = md_index;
            attach_params_array[attach_params_num].tl_mkey_buf = tl_mkey_buf;
            ++attach_params_num;
        }
    }

    if (attach_params_num == 0) {
        ucs_error("couldn't find local MDs which correspond to remote MDs");
        return UCS_ERR_UNREACHABLE;
    }

    status = ucp_memh_create(context, address, length, mem_type,
                             UCT_ALLOC_METHOD_LAST, UCP_MEM_FLAG_IMPORTED,
                             &memh);
    if (status != UCS_OK) {
        goto out;
    }

    if (ucs_likely(context->imported_mem_rcaches != NULL)) {
        /* Try to find rcache of imported memory buffers for the specific peer with
         * UUID packed in the exported_mkey_buffer */
        UCP_THREAD_CS_ENTER(&context->mt_lock);

        iter = kh_get(ucp_context_imported_mem_rcaches_hash,
                      context->imported_mem_rcaches, remote_uuid);
        if (iter != kh_end(context->imported_mem_rcaches)) {
            /* Found rcache for the specific peer */
            rcache = kh_value(context->imported_mem_rcaches, iter);
            ucs_assert(rcache != NULL);

            rregion = ucs_rcache_lookup_unsafe(rcache, address, length,
                                               PROT_READ | PROT_WRITE);
            if (rregion != NULL) {
                rcache_memh = ucs_derived_of(rregion, ucp_mem_t);
                if (ucs_likely(ucp_memh_rcache_is_suitable(
                            rcache_memh, address, length,
                            UCP_MEM_FLAG_IMPORTED, remote_md_map) &&
                               (rcache_memh->reg_id == reg_id))) {
                    memh->parent = rcache_memh;
                    status       = UCS_OK;
                    goto out_memh_update;
                }

                ucs_rcache_region_put_unsafe(rcache, rregion);
            }
        }

        UCP_THREAD_CS_EXIT(&context->mt_lock);
    }

    status = ucp_memh_import_slow(context, rcache, memh, remote_md_map,
                                  attach_params_array, attach_params_num,
                                  address, length, remote_uuid, mem_type,
                                  reg_id);
    if (status != UCS_OK) {
        goto err_memh_free;
    }

out_memh_update:
    if (memh->parent != memh) {
        memh->reg_id = memh->parent->reg_id;
        reg_md_map   = memh->parent->md_map;
        ucp_memh_update_from_parent_memh(memh, memh->parent->md_map,
                                         &memh->md_map, &reg_md_map);
        ucs_assert(reg_md_map == 0);
    }

    *memh_p = memh;
out:
    return status;

err_memh_free:
    ucs_free(memh);
    goto out;
}
