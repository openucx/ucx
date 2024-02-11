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
#include <ucs/sys/sock.h>
#include <ucs/sys/sys.h>
#include <ucs/type/serialize.h>
#include <ucs/type/param.h>
#include <ucm/api/ucm.h>
#include <string.h>
#include <inttypes.h>

/* Context for rcache memory registration callback */
typedef struct {
    ucs_memory_type_t mem_type;    /* Memory type */
    ucp_md_map_t      reg_md_map;  /* Map of memory domains to be registered */
    unsigned          uct_flags;   /* UCT memory registration flags */
    const char        *alloc_name; /* Memory allocation name */
} ucp_mem_rcache_reg_ctx_t;

ucp_mem_dummy_handle_t ucp_mem_dummy_handle = {
    .memh = {
        .alloc_method = UCT_ALLOC_METHOD_LAST,
        .alloc_md_index = UCP_NULL_RESOURCE,
        .parent = &ucp_mem_dummy_handle.memh,
    },
    .uct = { UCT_MEM_HANDLE_NULL }
};

static void
ucp_memh_register_log_fail(ucs_log_level_t log_level, void *address,
                           size_t length, ucs_memory_type_t mem_type,
                           int dmabuf_fd, ucp_md_index_t md_index,
                           ucp_context_h context, ucs_status_t status)
{
    UCS_STRING_BUFFER_ONSTACK(err_str, 256);

    ucs_string_buffer_appendf(&err_str,
                              "failed to register address %p (%s) length %zu",
                              address, ucs_memory_type_names[mem_type], length);

    if (dmabuf_fd != UCT_DMABUF_FD_INVALID) {
        ucs_string_buffer_appendf(&err_str, " dmabuf_fd %d", dmabuf_fd);
    }

    ucs_string_buffer_appendf(&err_str,
                              " on md[%d]=%s: %s (md supports: ", md_index,
                              context->tl_mds[md_index].rsc.md_name,
                              ucs_status_string(status));
    ucs_string_buffer_append_flags(&err_str,
                                   context->tl_mds[md_index].attr.reg_mem_types,
                                   ucs_memory_type_names);
    ucs_string_buffer_appendf(&err_str, ")");

    ucs_log(log_level, "%s", ucs_string_buffer_cstr(&err_str));
}

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
    void *end_address;
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
            ucp_memh_register_log_fail(level, base_address, reg_length,
                                       mem_type, UCT_DMABUF_FD_INVALID,
                                       md_index, context, status);

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
    uct_md_h mds[UCP_MAX_MDS];

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
    return status;
}

static unsigned
ucp_mem_map_params2uct_flags(const ucp_context_h context,
                             const ucp_mem_map_params_t *params)
{
    unsigned flags = 0;

    if (context->config.features & UCP_FEATURE_RMA) {
        flags |= UCT_MD_MEM_ACCESS_RMA;
    }

    if (context->config.features & UCP_FEATURE_AMO) {
        flags |= UCT_MD_MEM_ACCESS_REMOTE_ATOMIC;
    }

    if (params->field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS) {
        if (params->flags & UCP_MEM_MAP_NONBLOCK) {
            flags |= UCT_MD_MEM_FLAG_NONBLOCK;
        }

        if (params->flags & UCP_MEM_MAP_FIXED) {
            flags |= UCT_MD_MEM_FLAG_FIXED;
        }

        if (params->flags & UCP_MEM_MAP_SYMMETRIC_RKEY) {
            flags |= UCT_MD_MEM_SYMMETRIC_RKEY;
        }
    }

    return flags;
}

static void ucp_memh_dereg(ucp_context_h context, ucp_mem_h memh,
                           ucp_md_map_t md_map)
{
    uct_completion_t comp            = {
        .count = 1,
        .func  = (uct_completion_callback_t)ucs_empty_function_do_assert_void
    };
    uct_md_mem_dereg_params_t params = {
        .field_mask = UCT_MD_MEM_DEREG_FIELD_MEMH |
                      UCT_MD_MEM_DEREG_FIELD_FLAGS |
                      UCT_MD_MEM_DEREG_FIELD_COMPLETION,
        .comp       = &comp
    };
    ucp_md_index_t md_index;
    ucs_status_t status;

    /* Unregister from all memory domains */
    ucs_for_each_bit(md_index, md_map) {
        ucs_assertv(md_index != memh->alloc_md_index,
                    "memh %p: md_index %u alloc_md_index %u", memh, md_index,
                    memh->alloc_md_index);
        ucs_trace("de-registering memh[%d]=%p", md_index, memh->uct[md_index]);
        ucs_assert(context->tl_mds[md_index].attr.flags & UCT_MD_FLAG_REG);

        params.memh = memh->uct[md_index];
        if (memh->inv_md_map & UCS_BIT(md_index)) {
            params.flags = UCT_MD_MEM_DEREG_FLAG_INVALIDATE;
            comp.count++;
        } else {
            params.flags = 0;
        }

        status = uct_md_mem_dereg_v2(context->tl_mds[md_index].md, &params);
        if (status != UCS_OK) {
            ucs_warn("failed to dereg from md[%d]=%s: %s", md_index,
                     context->tl_mds[md_index].rsc.md_name,
                     ucs_status_string(status));
            if (params.flags & UCT_MD_MEM_DEREG_FLAG_INVALIDATE) {
                comp.count--;
            }
        }

        memh->uct[md_index] = NULL;
    }

    ucs_assert(comp.count == 1);
}

void ucp_memh_invalidate(ucp_context_h context, ucp_mem_h memh,
                         ucs_rcache_invalidate_comp_func_t cb, void *arg,
                         ucp_md_map_t inv_md_map)
{
    ucs_trace("memh %p: invalidate address %p length %zu md_map %" PRIx64
              " inv_md_map %" PRIx64,
              memh, ucp_memh_address(memh), ucp_memh_length(memh), memh->md_map,
              inv_md_map);

    ucs_assert(memh->parent == NULL);
    ucs_assert(!(memh->flags & UCP_MEMH_FLAG_IMPORTED));

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    memh->inv_md_map |= inv_md_map;
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    ucs_rcache_region_invalidate(context->rcache, &memh->super, cb, arg);
}

static void ucp_memh_put_rcache(ucp_context_h context, ucp_mem_h memh)
{
    ucs_rcache_t *rcache;
    khiter_t iter;

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    if (memh->flags & UCP_MEMH_FLAG_IMPORTED) {
        iter = kh_get(ucp_context_imported_mem_hash, context->imported_mem_hash,
                      memh->remote_uuid);
        ucs_assert(iter != kh_end(context->imported_mem_hash));
        rcache = kh_value(context->imported_mem_hash, iter);
    } else {
        rcache = context->rcache;
    }

    ucs_assert(rcache != NULL);
    ucs_rcache_region_put_unsafe(rcache, &memh->super);
    UCP_THREAD_CS_EXIT(&context->mt_lock);
}

static void ucp_memh_cleanup(ucp_context_h context, ucp_mem_h memh)
{
    ucp_md_map_t md_map = memh->md_map;
    uct_allocated_memory_t mem;
    ucs_status_t status;

    ucs_trace("memh %p: cleanup", memh);

    ucs_assert(ucp_memh_is_user_memh(memh));

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
    if (memh->parent != memh) {
        ucp_memh_dereg(context, memh, md_map & ~memh->parent->md_map);
        ucp_memh_put_rcache(context, memh->parent);
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

    ucs_free(memh);
}

static ucs_status_t
ucp_memh_register_internal(ucp_context_h context, ucp_mem_h memh,
                           ucp_md_map_t md_map, unsigned uct_flags,
                           const char *alloc_name, ucs_log_level_t err_level,
                           int allow_partial_reg)
{
    ucs_memory_type_t mem_type          = memh->mem_type;
    ucp_md_index_t dmabuf_prov_md_index = context->dmabuf_mds[mem_type];
    void *address                       = ucp_memh_address(memh);
    size_t length                       = ucp_memh_length(memh);
    ucp_md_map_t reg_md_map             = ~memh->md_map & md_map;
    ucp_md_map_t md_map_registered      = 0;
    ucp_md_map_t dmabuf_md_map          = 0;
    uct_md_mem_reg_params_t reg_params;
    uct_md_mem_attr_t mem_attr;
    ucp_md_index_t md_index;
    ucs_status_t status;
    void *reg_address;
    size_t reg_length;
    size_t reg_align;

    if (reg_md_map == 0) {
        return UCS_OK;
    }

    if (context->config.ext.reg_nb_mem_types & UCS_BIT(mem_type)) {
        uct_flags |= UCT_MD_MEM_FLAG_NONBLOCK;
    }

    reg_params.flags         = uct_flags;
    reg_params.dmabuf_fd     = UCT_DMABUF_FD_INVALID;
    reg_params.dmabuf_offset = 0;

    if ((dmabuf_prov_md_index != UCP_NULL_RESOURCE) &&
        (reg_md_map & context->dmabuf_reg_md_map)) {
        /* Query dmabuf file descriptor and offset */
        mem_attr.field_mask = UCT_MD_MEM_ATTR_FIELD_DMABUF_FD |
                              UCT_MD_MEM_ATTR_FIELD_DMABUF_OFFSET;
        status = uct_md_mem_query(context->tl_mds[dmabuf_prov_md_index].md,
                                  address, length, &mem_attr);
        if (status != UCS_OK) {
            ucs_log(err_level,
                    "uct_md_mem_query(dmabuf address %p length %zu) failed: %s",
                    address, length, ucs_status_string(status));
        } else {
            ucs_trace("uct_md_mem_query(dmabuf address %p length %zu) returned "
                      "fd %d offset %zu",
                      address, length, mem_attr.dmabuf_fd,
                      mem_attr.dmabuf_offset);
            dmabuf_md_map            = context->dmabuf_reg_md_map;
            reg_params.dmabuf_fd     = mem_attr.dmabuf_fd;
            reg_params.dmabuf_offset = mem_attr.dmabuf_offset;
        }
    }

    ucs_for_each_bit(md_index, reg_md_map) {
        ucs_assertv(context->reg_md_map[mem_type] & UCS_BIT(md_index),
                    "mem_type=%s md[%d]=%s reg_md_map=0x%" PRIx64,
                    ucs_memory_type_names[mem_type], md_index,
                    context->tl_mds[md_index].rsc.md_name,
                    context->reg_md_map[mem_type]);

        reg_params.field_mask = UCT_MD_MEM_REG_FIELD_FLAGS;
        if (dmabuf_md_map & UCS_BIT(md_index)) {
            /* If this MD can consume a dmabuf and we have it - provide it */
            reg_params.field_mask |= UCT_MD_MEM_REG_FIELD_DMABUF_FD |
                                     UCT_MD_MEM_REG_FIELD_DMABUF_OFFSET;
        }

        reg_address = address;
        reg_length  = length;

        if (context->rcache == NULL) {
            reg_align = ucs_max(context->tl_mds[md_index].attr.reg_alignment, 1);
            ucs_align_ptr_range(&reg_address, &reg_length, reg_align);
        }

        status = uct_md_mem_reg_v2(context->tl_mds[md_index].md, reg_address,
                                   reg_length, &reg_params, &memh->uct[md_index]);
        if (ucs_unlikely(status != UCS_OK)) {
            ucp_memh_register_log_fail(err_level, reg_address, reg_length,
                                       mem_type, reg_params.dmabuf_fd, md_index,
                                       context, status);
            if (allow_partial_reg &&
                (uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS)) {
                continue;
            }

            ucp_memh_dereg(context, memh, md_map_registered);
            goto out_close_dmabuf_fd;
        }

        ucs_trace("register address %p length %zu dmabuf-fd %d flags %ld "
                  "on md[%d]=%s %p",
                  reg_address, reg_length,
                  (dmabuf_md_map & UCS_BIT(md_index)) ? reg_params.dmabuf_fd :
                                                        UCT_DMABUF_FD_INVALID,
                  reg_params.flags,
                  md_index, context->tl_mds[md_index].rsc.md_name,
                  memh->uct[md_index]);
        md_map_registered |= UCS_BIT(md_index);
    }

    memh->md_map |= md_map_registered;
    status        = UCS_OK;

    ucs_trace("memh %p: registered %s address %p length %zu md_map %" PRIx64,
              memh, alloc_name, address, length, memh->md_map);

out_close_dmabuf_fd:
    UCS_STATIC_ASSERT(UCT_DMABUF_FD_INVALID == -1);
    ucs_close_fd(&reg_params.dmabuf_fd);
    return status;
}

ucs_status_t ucp_memh_register(ucp_context_h context, ucp_mem_h memh,
                               ucp_md_map_t md_map, unsigned uct_flags,
                               const char *alloc_name)
{
    ucs_log_level_t err_level = (uct_flags & UCT_MD_MEM_FLAG_HIDE_ERRORS) ?
                                        UCS_LOG_LEVEL_DIAG :
                                        UCS_LOG_LEVEL_ERROR;

    return ucp_memh_register_internal(context, memh, md_map, uct_flags,
                                      alloc_name, err_level, 1);
}

static size_t ucp_memh_size(ucp_context_h context)
{
    return sizeof(ucp_mem_t) + (sizeof(uct_mem_h) * context->num_mds);
}

static void ucp_memh_set(ucp_mem_h memh, ucp_context_h context, void* address,
                         size_t length, ucs_memory_type_t mem_type,
                         uint8_t memh_flags, uct_alloc_method_t method)
{
    ucp_memory_info_t info;

    ucp_memory_detect(context, address, length, &info);
    memh->super.super.start = (uintptr_t)address;
    memh->super.super.end   = (uintptr_t)address + length;
    memh->flags             = memh_flags;
    memh->context           = context;
    memh->mem_type          = mem_type;
    memh->sys_dev           = info.sys_dev;
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

    *memh_p = memh;
    return UCS_OK;
}

static ucs_status_t
ucp_memh_rcache_get(ucs_rcache_t *rcache, void *address, size_t length,
                    size_t alignment, ucs_memory_type_t mem_type,
                    ucp_md_map_t reg_md_map, unsigned uct_flags,
                    const char *alloc_name, ucp_mem_h *memh_p)
{
    ucp_mem_rcache_reg_ctx_t reg_ctx = {
        .mem_type   = mem_type,
        .reg_md_map = reg_md_map,
        .uct_flags  = uct_flags,
        .alloc_name = alloc_name
    };
    ucs_rcache_region_t *rregion;
    ucs_status_t status;

    status = ucs_rcache_get(rcache, address, length, alignment,
                            PROT_READ | PROT_WRITE, &reg_ctx, &rregion);
    if (status != UCS_OK) {
        return status;
    }

    *memh_p = ucs_derived_of(rregion, ucp_mem_t);

    return UCS_OK;
}

static ucp_md_index_t ucp_mem_get_md_index(ucp_context_h context,
                                           const uct_md_h md,
                                           uct_alloc_method_t method)
{
    ucp_md_index_t md_index;

    if (method != UCT_ALLOC_METHOD_MD) {
        return UCP_NULL_RESOURCE;
    }

    for (md_index = 0; md_index < context->num_mds; md_index++) {
        if (md == context->tl_mds[md_index].md) {
            return md_index;
        }
    }

    return UCP_NULL_RESOURCE;
}

static ucs_status_t ucp_memh_create_from_mem(ucp_context_h context,
                                             const uct_allocated_memory_t *mem,
                                             ucp_mem_h *memh_p)
{
    ucs_status_t status;
    ucp_mem_h memh;

    status = ucp_memh_create(context, mem->address, mem->length, mem->mem_type,
                             mem->method, 0, &memh);
    if (status != UCS_OK) {
        return status;
    }

    memh->alloc_md_index = ucp_mem_get_md_index(context, mem->md, mem->method);
    if (memh->alloc_md_index != UCP_NULL_RESOURCE) {
        memh->uct[memh->alloc_md_index] = mem->memh;
        memh->md_map                   |= UCS_BIT(memh->alloc_md_index);
        ucs_trace("allocated address %p length %zu on md[%d]=%s %p",
                  mem->address, mem->length, memh->alloc_md_index,
                  context->tl_mds[memh->alloc_md_index].rsc.md_name,
                  memh->uct[memh->alloc_md_index]);
    }

    *memh_p = memh;
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE void
ucp_memh_init_from_parent(ucp_mem_h memh, ucp_md_map_t parent_md_map)
{
    ucp_md_index_t md_index;

    ucs_assertv(!(memh->md_map & parent_md_map),
                "memh %p: md_map 0x%lx parent_md_map 0x%lx", memh,
                memh->md_map, parent_md_map);

    memh->reg_id  = memh->parent->reg_id;
    memh->md_map |= parent_md_map;

    ucs_for_each_bit(md_index, parent_md_map) {
        ucs_assert(memh->uct[md_index] == NULL);
        memh->uct[md_index] = memh->parent->uct[md_index];
    }
}

static ucs_status_t ucp_memh_init_uct_reg(ucp_context_h context, ucp_mem_h memh,
                                          unsigned uct_flags,
                                          const char *alloc_name)
{
    ucs_memory_type_t mem_type = memh->mem_type;
    ucp_md_map_t reg_md_map    = context->reg_md_map[mem_type] & ~memh->md_map;
    ucp_md_map_t cache_md_map  = context->cache_md_map[mem_type] & reg_md_map;
    void *address              = ucp_memh_address(memh);
    size_t length              = ucp_memh_length(memh);
    ucs_status_t status;

    if (context->rcache == NULL) {
        status = ucp_memh_register(context, memh, reg_md_map, uct_flags,
                                   alloc_name);
        if (status != UCS_OK) {
            goto err;
        }

        memh->reg_id = context->next_memh_reg_id++;
        memh->parent = memh;
    } else {
        status = ucp_memh_get(context, address, length, mem_type, cache_md_map,
                              uct_flags, alloc_name, &memh->parent);
        if (status != UCS_OK) {
            goto err;
        }

        ucp_memh_init_from_parent(memh, cache_md_map);

        status = ucp_memh_register(context, memh, reg_md_map, uct_flags,
                                   alloc_name);
        if (status != UCS_OK) {
            goto err_put_rcache;
        }
    }

    ucs_assert(ucp_memh_is_user_memh(memh));
    return UCS_OK;

err_put_rcache:
    /* We assume the parent memh was retrieved from rcache */
    ucs_rcache_region_put_unsafe(context->rcache, &memh->parent->super);
err:
    return status;
}

static size_t ucp_memh_reg_align(ucp_context_h context, ucp_md_map_t reg_md_map)
{
    size_t reg_align = UCS_RCACHE_MIN_ALIGNMENT;
    const uct_md_attr_v2_t *md_attr;
    ucp_md_index_t md_index;

    ucs_for_each_bit(md_index, reg_md_map) {
        md_attr   = &context->tl_mds[md_index].attr;
        reg_align = ucs_max(md_attr->reg_alignment, reg_align);
    }

    return reg_align;
}

ucs_status_t ucp_memh_get_slow(ucp_context_h context, void *address,
                               size_t length, ucs_memory_type_t mem_type,
                               ucp_md_map_t reg_md_map, unsigned uct_flags,
                               const char *alloc_name, ucp_mem_h *memh_p)
{
    ucp_mem_h memh = NULL; /* To suppress compiler warning */
    void *reg_address;
    size_t reg_length;
    ucs_status_t status;
    ucs_memory_info_t mem_info;
    size_t reg_align;

    if (context->config.ext.reg_whole_alloc_bitmap & UCS_BIT(mem_type)) {
        ucp_memory_detect_internal(context, address, length, &mem_info);
        reg_address = mem_info.base_address;
        reg_length  = mem_info.alloc_length;
    } else {
        reg_address = address;
        reg_length  = length;
    }

    reg_align = ucp_memh_reg_align(context, reg_md_map);

    UCP_THREAD_CS_ENTER(&context->mt_lock);
    if (context->rcache == NULL) {
        status = ucp_memh_create(context, reg_address, reg_length, mem_type,
                                 UCT_ALLOC_METHOD_LAST, 0, &memh);
    } else {
        status = ucp_memh_rcache_get(context->rcache, reg_address, reg_length,
                                     reg_align, mem_type, reg_md_map, uct_flags,
                                     alloc_name, &memh);

        ucs_assert(memh->mem_type == mem_type);
        ucs_assert(ucs_padding((intptr_t)ucp_memh_address(memh), reg_align) == 0);
        ucs_assert(ucs_padding(ucp_memh_length(memh), reg_align) == 0);
    }

    if (status != UCS_OK) {
        goto out;
    }

    ucs_trace(
            "memh_get_slow: %s address %p/%p length %zu/%zu %s md_map %" PRIx64
            " flags 0x%x",
            alloc_name, address, ucp_memh_address(memh), length,
            ucp_memh_length(memh), ucs_memory_type_names[mem_type], reg_md_map,
            uct_flags);

    status = ucp_memh_register(context, memh, reg_md_map, uct_flags,
                               alloc_name);
    if (status != UCS_OK) {
        goto err_free_memh;
    }

    memh->reg_id = context->next_memh_reg_id++;
    *memh_p      = memh;

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

    status = ucp_memh_init_uct_reg(context, memh, uct_flags, alloc_name);
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
ucp_memh_import(ucp_context_h context, const void *export_mkey_buffer,
                ucp_mem_h *memh_p);

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
    const char *alloc_name = "user memory";
    uint8_t memh_flags     = 0;
    ucp_mem_h memh         = NULL;
    ucs_memory_type_t mem_type;
    ucp_memory_info_t mem_info;
    ucs_status_t status;
    unsigned uct_flags;
    unsigned flags;
    void *address;
    const void *exported_memh_buffer;
    size_t length;

    if (!(params->field_mask &
          (UCP_MEM_MAP_PARAM_FIELD_LENGTH |
           UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER))) {
        status = UCS_ERR_INVALID_PARAM;
        ucs_error("the length value or exported_memh_buffer for mapping memory"
                  " aren't set: %s", ucs_status_string(status));
        goto out;
    }

    address              = UCP_PARAM_VALUE(MEM_MAP, params, address, ADDRESS,
                                           NULL);
    length               = UCP_PARAM_VALUE(MEM_MAP, params, length, LENGTH, 0);
    flags                = UCP_PARAM_VALUE(MEM_MAP, params, flags, FLAGS, 0);
    exported_memh_buffer = UCP_PARAM_VALUE(MEM_MAP, params, exported_memh_buffer,
                                           EXPORTED_MEMH_BUFFER, NULL);
    mem_type             = UCP_PARAM_VALUE(MEM_MAP, params, memory_type,
                                           MEMORY_TYPE, UCS_MEMORY_TYPE_LAST);

    if ((flags & UCP_MEM_MAP_FIXED) &&
        ((uintptr_t)address % ucs_get_page_size())) {
        ucs_error("UCP_MEM_MAP_FIXED flag requires page aligned address");
        status = UCS_ERR_INVALID_PARAM;
        goto out;
    }

    if (mem_type == UCS_MEMORY_TYPE_LAST) {
        if (flags & UCP_MEM_MAP_ALLOCATE) {
            mem_type = UCS_MEMORY_TYPE_HOST;
        } else {
            ucp_memory_detect(context, address, length, &mem_info);
            mem_type = mem_info.type;
        }
    }

    if (exported_memh_buffer != NULL) {
        if (flags & UCP_MEM_MAP_ALLOCATE) {
            ucs_error("wrong combinations of parameters: exported memory handle"
                      " and memory allocation were requested altogether");
            status = UCS_ERR_INVALID_PARAM;
            goto out;
        }

        memh_flags |= UCP_MEMH_FLAG_IMPORTED;

        if (ucp_memh_buffer_is_dummy(exported_memh_buffer)) {
            /* Exported memory handle buffer packed for dummy memory handle */
            goto out_zero_mem;
        }
    } else if (length == 0) {
        goto out_zero_mem;
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

    uct_flags = ucp_mem_map_params2uct_flags(context, params);

    if (memh_flags & UCP_MEMH_FLAG_IMPORTED) {
        status = ucp_memh_import(context, exported_memh_buffer, &memh);
    } else if (flags & UCP_MEM_MAP_ALLOCATE) {
        status = ucp_memh_alloc(context, address, length, mem_type, uct_flags,
                                alloc_name, &memh);
    } else {
        status = ucp_memh_create(context, address, length, mem_type,
                                 UCT_ALLOC_METHOD_LAST, 0, &memh);
        if (status != UCS_OK) {
            goto out;
        }

        status = ucp_memh_init_uct_reg(context, memh, uct_flags, alloc_name);
        if (status != UCS_OK) {
            ucs_free(memh);
        }
    }

out:
    if (status == UCS_OK) {
        ucs_assert(ucp_memh_is_user_memh(memh));
        *memh_p = memh;
    }
    return status;

out_zero_mem:
    ucs_assert(ucp_memh_address(&ucp_mem_dummy_handle.memh) == NULL);
    ucs_assert(ucp_memh_length(&ucp_mem_dummy_handle.memh) == 0);

    ucs_debug("mapping zero length buffer, return dummy memh");
    *memh_p = &ucp_mem_dummy_handle.memh;
    return UCS_OK;
}

void ucp_memh_put_slow(ucp_context_h context, ucp_mem_h memh)
{
    ucs_assert(context->rcache == NULL);
    ucs_assert(memh->parent == NULL);
    ucp_memh_dereg(context, memh, memh->md_map);
    ucs_free(memh);
}

ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh)
{
    if (ucp_memh_is_zero_length(memh)) {
        return UCS_OK;
    }

    ucp_memh_cleanup(context, memh);
    return UCS_OK;
}

ucs_status_t ucp_mem_type_reg_buffers(ucp_worker_h worker, void *remote_addr,
                                      size_t length, ucs_memory_type_t mem_type,
                                      ucp_md_index_t md_index, ucp_mem_h *memh_p,
                                      uct_rkey_bundle_t *rkey_bundle)
{
    ucp_context_h context            = worker->context;
    const uct_md_attr_v2_t *md_attr  = &context->tl_mds[md_index].attr;
    ucp_mem_h memh                   = NULL; /* To suppress compiler warning */
    uct_md_mkey_pack_params_t params = { .field_mask = 0 };
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

    status = ucp_memh_get(context, remote_addr, length, mem_type,
                          UCS_BIT(md_index), UCT_MD_MEM_ACCESS_ALL, "mem_type",
                          &memh);
    if (status != UCS_OK) {
        goto out;
    }

    rkey_buffer = ucs_alloca(md_attr->rkey_packed_size);
    status      = uct_md_mkey_pack_v2(tl_md->md, memh->uct[md_index],
                                      remote_addr, length, &params,
                                      rkey_buffer);
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

    *memh_p = memh;
    return UCS_OK;

out_dereg_mem:
    ucp_memh_put(memh);
out:
    return status;
}

void ucp_mem_type_unreg_buffers(ucp_worker_h worker, ucp_md_index_t md_index,
                                ucp_mem_h memh, uct_rkey_bundle_t *rkey_bundle)
{
    ucp_context_h context = worker->context;
    ucp_rsc_index_t cmpt_index;

    if (rkey_bundle->rkey != UCT_INVALID_RKEY) {
        cmpt_index = context->tl_mds[md_index].cmpt_index;
        uct_rkey_release(context->tl_cmpts[cmpt_index].cmpt, rkey_bundle);
        ucp_memh_put(memh);
    }
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
    ucp_mem_desc_t *chunk_hdr;
    ucp_mem_h memh;
    ucs_status_t status;

    status = ucp_memh_alloc(worker->context, NULL,
                            *size_p + sizeof(*chunk_hdr), UCS_MEMORY_TYPE_HOST,
                            UCT_MD_MEM_ACCESS_RMA, ucs_mpool_name(mp), &memh);
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
    ucp_memh_cleanup(worker->context, chunk_hdr->memh);
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
    ucp_memh_cleanup(mpriv->worker->context, chunk_hdr->memh);
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

void ucp_mem_print_info(const char *mem_spec, ucp_context_h context,
                        FILE *stream)
{
    UCS_STRING_BUFFER_ONSTACK(strb, 128);
    size_t min_page_size, max_page_size;
    ucp_mem_map_params_t mem_params;
    char mem_types_buf[128];
    ssize_t mem_type_value;
    size_t mem_size_value;
    char memunits_str[32];
    ucs_status_t status;
    char *mem_size_str;
    char *mem_type_str;
    unsigned md_index;
    void *rkey_buffer;
    size_t rkey_size;
    ucp_mem_h memh;

    ucs_string_buffer_appendf(&strb, "%s", mem_spec);

    /* Parse memory size */
    mem_size_str = ucs_string_buffer_next_token(&strb, NULL, ",");
    status       = ucs_str_to_memunits(mem_size_str, &mem_size_value);
    if (status != UCS_OK) {
        printf("<Failed to convert a memunits string>\n");
        return;
    }

    /* Parse memory type */
    mem_type_str = ucs_string_buffer_next_token(&strb, mem_size_str, ",");
    if (mem_type_str != NULL) {
        mem_type_value = ucs_string_find_in_list(mem_type_str,
                                                 ucs_memory_type_names, 0);
        if ((mem_type_value < 0) ||
            !(UCS_BIT(mem_type_value) & context->mem_type_mask)) {
            printf("<Invalid memory type '%s', supported types: %s>\n",
                   mem_type_str,
                   ucs_flags_str(mem_types_buf, sizeof(mem_types_buf),
                                 context->mem_type_mask,
                                 ucs_memory_type_names));
            return;
        }
    } else {
        mem_type_value = UCS_MEMORY_TYPE_HOST;
    }

    mem_params.field_mask  = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                             UCP_MEM_MAP_PARAM_FIELD_LENGTH |
                             UCP_MEM_MAP_PARAM_FIELD_MEMORY_TYPE |
                             UCP_MEM_MAP_PARAM_FIELD_FLAGS;
    mem_params.address     = NULL;
    mem_params.length      = mem_size_value;
    mem_params.memory_type = mem_type_value;
    mem_params.flags       = UCP_MEM_MAP_ALLOCATE;

    status = ucp_mem_map(context, &mem_params, &memh);
    if (status != UCS_OK) {
        printf("<Failed to allocate memory of size %zd type %s>\n",
               mem_size_value, mem_type_str);
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

    status = ucp_rkey_pack(context, memh, &rkey_buffer, &rkey_size);
    if (status != UCS_OK) {
        printf("<Failed to pack rkey: %s>\n", ucs_status_string(status));
    } else {
        fprintf(stream, "#  rkey size: %zu\n", rkey_size);
        ucp_rkey_buffer_release(rkey_buffer);
    }

    status = ucp_mem_unmap(context, memh);
    if (status != UCS_OK) {
        printf("<Failed to unmap memory of size %zd>\n", mem_size_value);
    }
}

static ucs_status_t
ucp_mem_rcache_mem_reg_cb(void *ctx, ucs_rcache_t *rcache, void *arg,
                          ucs_rcache_region_t *rregion,
                          uint16_t rcache_mem_reg_flags)
{
    ucp_context_h context             = (ucp_context_h)ctx;
    ucp_mem_rcache_reg_ctx_t *reg_ctx = arg;
    ucp_mem_h memh                    = ucs_derived_of(rregion, ucp_mem_t);
    ucp_memory_info_t info;

    ucp_memory_detect(context, (void*)memh->super.super.start,
                      memh->super.super.end - memh->super.super.start, &info);
    memh->context        = context;
    memh->alloc_md_index = UCP_NULL_RESOURCE;
    memh->alloc_method   = UCT_ALLOC_METHOD_LAST;
    memh->mem_type       = reg_ctx->mem_type;
    memh->sys_dev        = info.sys_dev;

    if (rcache_mem_reg_flags & UCS_RCACHE_MEM_REG_HIDE_ERRORS) {
        /* Hide errors during registration but fail if any memory domain failed
           to register, to indicate that a part of the region may be invalid */
        return ucp_memh_register_internal(context, memh, reg_ctx->reg_md_map,
                                          reg_ctx->uct_flags |
                                                  UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                          reg_ctx->alloc_name,
                                          UCS_LOG_LEVEL_DEBUG, 0);
    }

    return ucp_memh_register(context, memh, reg_ctx->reg_md_map,
                             reg_ctx->uct_flags, reg_ctx->alloc_name);
}

static void ucp_mem_rcache_mem_dereg_cb(void *ctx, ucs_rcache_t *rcache,
                                        ucs_rcache_region_t *rregion)
{
    ucp_context_h context = ctx;
    ucp_mem_h memh        = ucs_derived_of(rregion, ucp_mem_t);

    ucp_memh_dereg(context, memh, memh->md_map);
}

static void ucp_mem_rcache_dump_region_cb(void *rcontext, ucs_rcache_t *rcache,
                                         ucs_rcache_region_t *rregion, char *buf,
                                         size_t max)
{
    UCS_STRING_BUFFER_FIXED(strb, buf, max);
    ucp_mem_h memh        = ucs_derived_of(rregion, ucp_mem_t);
    ucp_context_h context = rcontext;
    unsigned md_index;

    if (memh->md_map == 0) {
        ucs_string_buffer_appendf(&strb, "no mds");
        return;
    }

    ucs_for_each_bit(md_index, memh->md_map) {
        ucs_string_buffer_appendf(&strb, "md[%d]=%s", md_index,
                                  context->tl_mds[md_index].rsc.md_name);
        if (memh->alloc_md_index == md_index) {
            ucs_string_buffer_appendf(&strb, "(alloc)");
        }
        ucs_string_buffer_appendf(&strb, " ");
    }

    ucs_string_buffer_rtrim(&strb, NULL);
}

static ucs_rcache_ops_t ucp_mem_rcache_ops = {
    .mem_reg     = ucp_mem_rcache_mem_reg_cb,
    .mem_dereg   = ucp_mem_rcache_mem_dereg_cb,
    .dump_region = ucp_mem_rcache_dump_region_cb
};

static ucs_status_t
ucp_mem_rcache_create(ucp_context_h context, const char *name,
                      ucs_rcache_t **rcache_p, int events,
                      ucs_rcache_params_t *rcache_params)
{
    rcache_params->region_struct_size = ucp_memh_size(context);
    rcache_params->context            = context;
    rcache_params->ops                = &ucp_mem_rcache_ops;

    if (events) {
        rcache_params->flags         |= UCS_RCACHE_FLAG_SYNC_EVENTS;
        rcache_params->ucm_events     = UCM_EVENT_VM_UNMAPPED |
                                        UCM_EVENT_MEM_TYPE_FREE;
    }

    return ucs_rcache_create(rcache_params, name, ucs_stats_get_root(),
                             rcache_p);
}

ucs_status_t ucp_mem_rcache_init(ucp_context_h context,
                                 const ucs_rcache_config_t *rcache_config)
{
    ucs_status_t status;
    ucs_rcache_params_t rcache_params;

    ucs_rcache_set_params(&rcache_params, rcache_config);

    status = ucp_mem_rcache_create(context, "ucp_rcache", &context->rcache, 1,
                                   &rcache_params);
    if (status != UCS_OK) {
        goto err;
    }

    if (context->config.features & UCP_FEATURE_EXPORTED_MEMH) {
        context->imported_mem_hash = kh_init(ucp_context_imported_mem_hash);
        if (context->imported_mem_hash == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto err_rcache_destroy;
        }
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

    if (context->imported_mem_hash != NULL) {
        kh_foreach_value(context->imported_mem_hash, rcache, {
            ucs_rcache_destroy(rcache);
        })
        kh_destroy(ucp_context_imported_mem_hash, context->imported_mem_hash);
    }
}

ucs_status_t
ucp_mm_get_alloc_md_index(ucp_context_h context, ucp_md_index_t *md_idx)
{
    const ucs_memory_type_t alloc_mem_type = UCS_MEMORY_TYPE_HOST;
    ucs_status_t status;
    uct_allocated_memory_t mem;

    if (!context->alloc_md_index_initialized) {
        status = ucp_mem_do_alloc(context, NULL, 1,
                                  UCT_MD_MEM_ACCESS_RMA |
                                          UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                  alloc_mem_type, "get_alloc_md_id",
                                  &mem);
        if (status != UCS_OK) {
            return status;
        }

        context->alloc_md_index_initialized     = 1;
        context->alloc_md_index[alloc_mem_type] =
                ucp_mem_get_md_index(context, mem.md, mem.method);
        uct_mem_free(&mem);
    }

    *md_idx = context->alloc_md_index[alloc_mem_type];
    return UCS_OK;
}

static ucs_status_t
ucp_memh_import_attach(ucp_context_h context, ucp_mem_h memh,
                       ucp_unpacked_exported_tl_mkey_t *tl_mkeys,
                       unsigned num_tl_mkeys)
{
    ucp_md_index_t md_index;
    unsigned tl_mkey_index;
    const void *tl_mkey_buf;
    uct_md_mem_attach_params_t attach_params;
    uct_md_attr_v2_t *md_attr;
    ucs_status_t status;
    uct_mem_h uct_memh;

    for (tl_mkey_index = 0; tl_mkey_index < num_tl_mkeys; ++tl_mkey_index) {
        md_index    = tl_mkeys[tl_mkey_index].md_index;
        tl_mkey_buf = tl_mkeys[tl_mkey_index].tl_mkey_buf;
        md_attr     = &context->tl_mds[md_index].attr;
        ucs_assert_always(md_attr->flags & UCT_MD_FLAG_EXPORTED_MKEY);

        if (memh->uct[md_index] != NULL) {
            continue;
        }

        attach_params.field_mask = UCT_MD_MEM_ATTACH_FIELD_FLAGS;
        attach_params.flags      = UCT_MD_MEM_ATTACH_FLAG_HIDE_ERRORS;

        status = uct_md_mem_attach(context->tl_mds[md_index].md, tl_mkey_buf,
                                   &attach_params, &uct_memh);
        if (ucs_unlikely(status != UCS_OK)) {
            /* Don't print an error, because two MDs can have similar global
             * identifiers, but a memory key was exported on another MD */
            ucs_trace("failed to attach memory on '%s/%s': %s",
                      md_attr->component_name,
                      context->tl_mds[md_index].rsc.md_name,
                      ucs_status_string(status));
            continue;
        }

        memh->uct[md_index] = uct_memh;
        memh->md_map       |= UCS_BIT(md_index);

        ucs_trace("imported address %p length %zu on md[%d]=%s: uct_memh %p",
                  ucp_memh_address(memh), ucp_memh_length(memh), md_index,
                  context->tl_mds[md_index].rsc.md_name,
                  memh->uct[md_index]);
    }

    if (memh->md_map == 0) {
        ucs_error("no suitable UCT memory domains to perform importing on");
        return UCS_ERR_UNREACHABLE;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_memh_import_slow(ucp_context_h context, ucs_rcache_t *existing_rcache,
                     ucp_mem_h user_memh,
                     ucp_unpacked_exported_memh_t *unpacked)
{
    ucs_rcache_t *rcache;
    ucs_rcache_params_t rcache_params;
    ucs_status_t status;
    ucp_mem_h memh;
    char rcache_name[128];
    khiter_t iter;
    int ret;

    ucs_assert(user_memh != NULL);

    UCP_THREAD_CS_ENTER(&context->mt_lock);

    if (context->imported_mem_hash != NULL) {
        if (existing_rcache == NULL) {
            ucs_snprintf_safe(rcache_name, sizeof(rcache_name),
                              "ucp_import_rcache[0x%" PRIx64 "]",
                              unpacked->remote_uuid);

            ucs_rcache_set_default_params(&rcache_params);

            status = ucp_mem_rcache_create(context, rcache_name, &rcache, 0,
                                           &rcache_params);
            if (status != UCS_OK) {
                goto out;
            }

            iter = kh_put(ucp_context_imported_mem_hash,
                          context->imported_mem_hash, unpacked->remote_uuid,
                          &ret);
            ucs_assertv((ret != UCS_KH_PUT_FAILED) &&
                        (ret != UCS_KH_PUT_KEY_PRESENT), "ret %d", ret);

            kh_value(context->imported_mem_hash, iter) = rcache;
        } else {
            rcache = existing_rcache;
        }

        status = ucp_memh_rcache_get(rcache, unpacked->address,
                                     unpacked->length, UCS_RCACHE_MIN_ALIGNMENT,
                                     unpacked->mem_type, 0, 0, "", &memh);
        if (status != UCS_OK) {
            goto err_rcache_destroy;
        }

        memh->flags       |= UCP_MEMH_FLAG_IMPORTED;
        user_memh->parent  = memh;
    } else {
        memh         = user_memh;
        memh->parent = memh;
        rcache       = NULL;
    }

    memh->reg_id      = unpacked->reg_id;
    memh->remote_uuid = unpacked->remote_uuid;
    status            = ucp_memh_import_attach(context, memh,
                                               unpacked->tl_mkeys,
                                               unpacked->num_tl_mkeys);
    if (status != UCS_OK) {
        goto err_memh_free;
    }

out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;

err_memh_free:
    if (rcache != NULL) {
        ucs_rcache_region_put_unsafe(rcache, &memh->super);
    }
err_rcache_destroy:
    if ((rcache != NULL) && (rcache != existing_rcache)) {
        /* Rcache was allocated here - remove it from hash and destroy */
        iter = kh_get(ucp_context_imported_mem_hash,
                      context->imported_mem_hash, unpacked->remote_uuid);
        ucs_assert(iter != kh_end(context->imported_mem_hash));
        kh_del(ucp_context_imported_mem_hash, context->imported_mem_hash,
               iter);
        ucs_rcache_destroy(rcache);
    }
    goto out;
}

static ucs_status_t
ucp_memh_import(ucp_context_h context, const void *export_mkey_buffer,
                ucp_mem_h *memh_p)
{
    ucs_rcache_t *rcache = NULL;
    ucp_mem_h memh, rcache_memh;
    ucs_status_t status;
    ucp_unpacked_exported_memh_t unpacked_memh;
    ucs_rcache_region_t *rregion;
    khiter_t iter;

    status = ucp_memh_exported_unpack(context, export_mkey_buffer,
                                      &unpacked_memh);
    if (status != UCS_OK) {
        goto out;
    }

    status = ucp_memh_create(context, unpacked_memh.address,
                             unpacked_memh.length, unpacked_memh.mem_type,
                             UCT_ALLOC_METHOD_LAST, 0, &memh);
    if (status != UCS_OK) {
        goto out;
    }

    if (ucs_likely(context->imported_mem_hash != NULL)) {
        /* Try to find rcache of imported memory buffers for the specific peer
         * with UUID packed in the exported_mkey_buffer */
        UCP_THREAD_CS_ENTER(&context->mt_lock);

        iter = kh_get(ucp_context_imported_mem_hash,
                      context->imported_mem_hash, unpacked_memh.remote_uuid);
        if (iter != kh_end(context->imported_mem_hash)) {
            /* Found rcache for the specific peer */
            rcache = kh_value(context->imported_mem_hash, iter);
            ucs_assert(rcache != NULL);

            rregion = ucs_rcache_lookup_unsafe(rcache, unpacked_memh.address,
                                               unpacked_memh.length, 1,
                                               PROT_READ | PROT_WRITE);
            if (rregion != NULL) {
                rcache_memh = ucs_derived_of(rregion, ucp_mem_t);
                if (ucs_likely(rcache_memh->reg_id == unpacked_memh.reg_id)) {
                    ucp_memh_rcache_print(rcache_memh, unpacked_memh.address,
                                          unpacked_memh.length);
                    memh->parent = rcache_memh;
                    status       = UCS_OK;
                    UCP_THREAD_CS_EXIT(&context->mt_lock);
                    goto out_memh_update;
                }

                ucs_debug("found memh %p (reg_id %" PRIu64 ") in rcache %p,"
                          " but reg_id is not matched with %" PRIu64,
                          rcache_memh, rcache_memh->reg_id, rcache,
                          unpacked_memh.reg_id);

                /* If registration IDs are not matched, but a region still
                 * exists in the RCACHE of imported regions, it means that
                 * an exported memory handle has already been destroyed for a
                 * given address, but an imported memory handle hasn't been
                 * removed from the RCACHE yet. So, it had refcount == 1 and
                 * now it should be 2. */
                ucs_assertv(rregion->refcount == 2, "%u", rregion->refcount);
                ucs_rcache_region_invalidate(rcache, rregion,
                                             ucs_empty_function, NULL);
                ucs_rcache_region_put_unsafe(rcache, rregion);
            }
        }

        UCP_THREAD_CS_EXIT(&context->mt_lock);
    }

    status = ucp_memh_import_slow(context, rcache, memh, &unpacked_memh);
    if (status != UCS_OK) {
        ucs_free(memh);
        goto out;
    }

out_memh_update:
    if (memh->parent != memh) {
        ucp_memh_init_from_parent(memh, memh->parent->md_map);
    }

    *memh_p = memh;
out:
    return status;
}
