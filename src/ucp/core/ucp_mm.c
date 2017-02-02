/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_mm.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>

#include <string.h>
#include <inttypes.h>


static ucp_mem_t ucp_mem_dummy_handle = {
    .address      = NULL,
    .length       = 0,
    .alloc_method = UCT_ALLOC_METHOD_LAST,
    .alloc_md     = NULL,
    .md_map       = 0
};

/**
 * Unregister memory from all memory domains.
 * Save in *alloc_md_memh_p the memory handle of the allocating MD, if such exists.
 */
static ucs_status_t ucp_memh_dereg_mds(ucp_context_h context, ucp_mem_h memh,
                                       uct_mem_h* alloc_md_memh_p)
{
    unsigned md_index, uct_index;
    ucs_status_t status;

    uct_index        = 0;
    *alloc_md_memh_p = UCT_INVALID_MEM_HANDLE;

    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        if (!(memh->md_map & UCS_BIT(md_index))) {
            /* MD not present in the array */
            continue;
        }

        if (memh->alloc_md == context->tl_mds[md_index].md) {
            /* If we used a md to register the memory, remember the memh - for
             * releasing the memory later. We cannot release the memory at this
             * point because we have to unregister it from other MDs first.
             */
            ucs_assert(memh->alloc_method == UCT_ALLOC_METHOD_MD);
            *alloc_md_memh_p = memh->uct[uct_index];
        } else {
            status = uct_md_mem_dereg(context->tl_mds[md_index].md,
                                      memh->uct[uct_index]);
            if (status != UCS_OK) {
                ucs_error("Failed to dereg address %p with md %s", memh->address,
                         context->tl_mds[md_index].rsc.md_name);
                return status;
            }
        }

        ++uct_index;
    }

    return UCS_OK;
}

/**
 * Register the memory on all MDs, except maybe for alloc_md.
 * In case alloc_md != NULL, alloc_md_memh will hold the memory key obtained from
 * allocation. It will be put in the array of keys in the proper index.
 */
static ucs_status_t ucp_memh_reg_mds(ucp_context_h context, ucp_mem_h memh,
                                     unsigned uct_flags, uct_mem_h alloc_md_memh)
{
    uct_mem_h dummy_md_memh;
    unsigned uct_memh_count;
    ucs_status_t status;
    unsigned md_index;

    memh->md_map   = 0;
    uct_memh_count = 0;

    /* Register on all transports (except the one we used to allocate) */
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        if (context->tl_mds[md_index].md == memh->alloc_md) {
            /* Add the memory handle we got from allocation */
            ucs_assert(memh->alloc_method == UCT_ALLOC_METHOD_MD);
            memh->md_map |= UCS_BIT(md_index);
            memh->uct[uct_memh_count++] = alloc_md_memh;
        } else if (context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_REG) {
            /* If the MD supports registration, register on it as well */
            status = uct_md_mem_reg(context->tl_mds[md_index].md, memh->address,
                                    memh->length, uct_flags,
                                    &memh->uct[uct_memh_count]);
            if (status != UCS_OK) {
                ucp_memh_dereg_mds(context, memh, &dummy_md_memh);
                return status;
            }

            memh->md_map |= UCS_BIT(md_index);
            ++uct_memh_count;
        }
    }
    return UCS_OK;
}

/**
 * @return Whether MD number 'md_index' is selected by the configuration as part
 *         of allocation method number 'config_method_index'.
 */
static int ucp_is_md_selected_by_config(ucp_context_h context,
                                        unsigned config_method_index,
                                        unsigned md_index)
{
    const char *config_mdc_name = context->config.alloc_methods[config_method_index].mdc_name;
    const char *mdc_name        = context->tl_mds[md_index].attr.component_name;

    return !strncmp(config_mdc_name, "*",      UCT_MD_COMPONENT_NAME_MAX) ||
           !strncmp(config_mdc_name, mdc_name, UCT_MD_COMPONENT_NAME_MAX);
}

static ucs_status_t ucp_mem_alloc(ucp_context_h context, size_t length,
                                  unsigned uct_flags, const char *name, ucp_mem_h memh)
{
    uct_allocated_memory_t mem;
    uct_alloc_method_t method;
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
        if (method == UCT_ALLOC_METHOD_MD && !(uct_flags & UCT_MD_MEM_FLAG_FIXED)) {
            for (md_index = 0; md_index < context->num_mds; ++md_index) {
                if (ucp_is_md_selected_by_config(context, method_index, md_index)) {
                    mds[num_mds++] = context->tl_mds[md_index].md;
                }
            }
        } else if ((method == UCT_ALLOC_METHOD_MMAP) &&
                   (uct_flags & UCT_MD_MEM_FLAG_FIXED)) {
            mem.address = memh->address;
        }

        status = uct_mem_alloc(length, uct_flags, &method, 1, mds, num_mds, name, &mem);
        if (status == UCS_OK) {
            goto allocated;
        }
    }

    status = UCS_ERR_NO_MEMORY;
    goto out;

allocated:
    ucs_debug("allocated memory at %p with method %s, now registering it",
             mem.address, uct_alloc_method_names[mem.method]);
    memh->address      = mem.address;
    memh->length       = mem.length;
    memh->alloc_method = mem.method;
    memh->alloc_md     = mem.md;
    status = ucp_memh_reg_mds(context, memh, uct_flags, mem.memh);
    if (status != UCS_OK) {
        uct_mem_free(&mem);
    }
out:
    ucs_free(mds);
    return status;
}


static inline unsigned uct_flags(ucp_mem_map_params_t *params)
{
    unsigned flags = 0;

    if (params && (params->field_mask & UCP_MEM_MAP_PARAM_FIELD_FLAGS)) {
        if (params->flags & UCP_MEM_MAP_NONBLOCK) {
            flags |= UCT_MD_MEM_FLAG_NONBLOCK;
        }

        if (params->flags & UCP_MEM_MAP_FIXED) {
            flags |= UCT_MD_MEM_FLAG_FIXED;
        }
    }

    return flags;
}

ucs_status_t ucp_mem_map(ucp_context_h context, ucp_mem_map_params_t *params,
                         ucp_mem_h *memh_p)
{
    ucs_status_t status;
    unsigned flags = uct_flags(params);
    ucp_mem_h memh;

    /* always acquire context lock */
    UCP_THREAD_CS_ENTER(&context->mt_lock);

    if (!(params->field_mask & UCP_MEM_MAP_PARAM_FIELD_LENGTH)) {
        status = UCS_ERR_INVALID_PARAM;
        ucs_error("The length value for mapping memory isn't set: %s",
                  ucs_status_string(status));
        goto out;
    }

    if (params->length == 0) {
        ucs_debug("mapping zero length buffer, return dummy memh");
        *memh_p = &ucp_mem_dummy_handle;
        status  = UCS_OK;
        goto out;
    }

    /* Allocate the memory handle */
    ucs_assert(context->num_mds > 0);
    memh = ucs_malloc(sizeof(*memh) + context->num_mds * sizeof(memh->uct[0]),
                      "ucp_memh");
    if (memh == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    if (!(params->field_mask & UCP_MEM_MAP_PARAM_FIELD_ADDRESS)) {
        params->address = NULL;

        /* Address must be defined */
        if (flags & UCT_MD_MEM_FLAG_FIXED) {
            status = UCS_ERR_INVALID_PARAM;
            goto out;
        }
    }

    memh->address      = params->address;
    memh->length       = params->length;

    if (params->address == NULL) {
        status = ucp_mem_alloc(context, params->length, flags, "user allocation", memh);
        if (status != UCS_OK) {
            goto err_free_memh;
        }

        params->address = memh->address;
    } else {
        ucs_debug("registering user memory at %p length %zu", params->address, params->length);
        memh->alloc_method = UCT_ALLOC_METHOD_LAST;
        memh->alloc_md     = NULL;
        status = ucp_memh_reg_mds(context, memh, flags, UCT_INVALID_MEM_HANDLE);
        if (status != UCS_OK) {
            goto err_free_memh;
        }
    }

    ucs_debug("%s buffer %p length %zu memh %p md_map 0x%x",
              (memh->alloc_method == UCT_ALLOC_METHOD_LAST) ? "mapped" : "allocated",
              memh->address, memh->length, memh, memh->md_map);
    *memh_p = memh;
    status  = UCS_OK;
    goto out;

err_free_memh:
    ucs_free(memh);
out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh)
{
    uct_allocated_memory_t mem;
    uct_mem_h alloc_md_memh;
    ucs_status_t status;

    /* always acquire context lock */
    UCP_THREAD_CS_ENTER(&context->mt_lock);

    ucs_debug("unmapping buffer %p memh %p", memh->address, memh);
    if (memh == &ucp_mem_dummy_handle) {
        status = UCS_OK;
        goto out;
    }

    /* Unregister from all memory domains */
    status = ucp_memh_dereg_mds(context, memh, &alloc_md_memh);
    if (status != UCS_OK) {
        goto out;
    }

    /* If the memory was also allocated, release it */
    if (memh->alloc_method != UCT_ALLOC_METHOD_LAST) {
        mem.address = memh->address;
        mem.length  = memh->length;
        mem.method  = memh->alloc_method;
        mem.md      = memh->alloc_md;  /* May be NULL if method is not MD */
        mem.memh    = alloc_md_memh;   /* May be INVALID if method is not MD */

        status = uct_mem_free(&mem);
        if (status != UCS_OK) {
            goto out;
        }
    }

    ucs_free(memh);
    status = UCS_OK;
out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

static ucs_status_t ucp_advice2uct(unsigned ucp_advice, unsigned *uct_advice) 
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
    unsigned uct_advice;

    if (!ucs_test_all_flags(params->field_mask,
                            UCP_MEM_ADVISE_PARAM_FIELD_ADDRESS|
                            UCP_MEM_ADVISE_PARAM_FIELD_LENGTH|
                            UCP_MEM_ADVISE_PARAM_FIELD_ADVICE)) {
        return UCS_ERR_INVALID_PARAM;
    }

    if ((params->address < memh->address) ||
        (params->address + params->length > memh->address + memh->length)) {
        return UCS_ERR_INVALID_PARAM;
    }

    status = ucp_advice2uct(params->advice, &uct_advice);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("advice buffer %p length %llu memh %p flags %x",
               params->address, (unsigned long long)params->length, memh,
               params->advice);

    if (memh == &ucp_mem_dummy_handle) {
        return UCS_OK;
    }

    UCP_THREAD_CS_ENTER(&context->mt_lock);

    status = UCS_OK;
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        if (!(context->tl_mds[md_index].attr.cap.flags & UCT_MD_FLAG_ADVISE)) {
            continue;
        }
        tmp_status = uct_md_mem_advise(context->tl_mds[md_index].md, memh->uct[md_index],
                                       params->address, params->length, uct_advice);
        if (tmp_status != UCS_OK) {
            status = tmp_status;
        }
    }

    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}
