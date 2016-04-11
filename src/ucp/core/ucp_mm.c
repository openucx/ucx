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


static ucp_mem_t dummy_mem = {
    .address      = NULL,
    .length       = 0,
    .alloc_method = UCT_ALLOC_METHOD_LAST,
    .alloc_pd     = NULL,
    .pd_map       = 0
};

/**
 * Unregister memory from all protection domains.
 * Save in *alloc_pd_memh_p the memory handle of the allocating PD, if such exists.
 */
static ucs_status_t ucp_memh_dereg_pds(ucp_context_h context, ucp_mem_h memh,
                                       uct_mem_h* alloc_pd_memh_p)
{
    unsigned pd_index, uct_index;
    ucs_status_t status;

    uct_index        = 0;
    *alloc_pd_memh_p = UCT_INVALID_MEM_HANDLE;

    for (pd_index = 0; pd_index < context->num_pds; ++pd_index) {
        if (!(memh->pd_map & UCS_BIT(pd_index))) {
            /* PD not present in the array */
            continue;
        }

        if (memh->alloc_pd == context->pds[pd_index]) {
            /* If we used a pd to register the memory, remember the memh - for
             * releasing the memory later. We cannot release the memory at this
             * point because we have to unregister it from other PDs first.
             */
            ucs_assert(memh->alloc_method == UCT_ALLOC_METHOD_PD);
            *alloc_pd_memh_p = memh->uct[uct_index];
        } else {
            status = uct_pd_mem_dereg(context->pds[pd_index],
                                      memh->uct[uct_index]);
            if (status != UCS_OK) {
                ucs_error("Failed to dereg address %p with pd %s", memh->address,
                         context->pd_rscs[pd_index].pd_name);
                return status;
            }
        }

        ++uct_index;
    }

    return UCS_OK;
}

/**
 * Register the memory on all PDs, except maybe for alloc_pd.
 * In case alloc_pd != NULL, alloc_pd_memh will hold the memory key obtained from
 * allocation. It will be put in the array of keys in the proper index.
 */
static ucs_status_t ucp_memh_reg_pds(ucp_context_h context, ucp_mem_h memh,
                                     uct_mem_h alloc_pd_memh)
{
    uct_mem_h dummy_pd_memh;
    unsigned uct_memh_count;
    ucs_status_t status;
    unsigned pd_index;

    memh->pd_map   = 0;
    uct_memh_count = 0;

    /* Register on all transports (except the one we used to allocate) */
    for (pd_index = 0; pd_index < context->num_pds; ++pd_index) {
        if (context->pds[pd_index] == memh->alloc_pd) {
            /* Add the memory handle we got from allocation */
            ucs_assert(memh->alloc_method == UCT_ALLOC_METHOD_PD);
            memh->pd_map |= UCS_BIT(pd_index);
            memh->uct[uct_memh_count++] = alloc_pd_memh;
        } else if (context->pd_attrs[pd_index].cap.flags & UCT_PD_FLAG_REG) {
            /* If the PD supports registration, register on it as well */
            status = uct_pd_mem_reg(context->pds[pd_index], memh->address,
                                    memh->length, &memh->uct[uct_memh_count]);
            if (status != UCS_OK) {
                ucp_memh_dereg_pds(context, memh, &dummy_pd_memh);
                return status;
            }

            memh->pd_map |= UCS_BIT(pd_index);
            ++uct_memh_count;
        }
    }
    return UCS_OK;
}

/**
 * @return Whether PD number 'pd_index' is selected by the configuration as part
 *         of allocation method number 'config_method_index'.
 */
static int ucp_is_pd_selected_by_config(ucp_context_h context,
                                        unsigned config_method_index,
                                        unsigned pd_index)
{
    const char *config_pdc_name = context->config.alloc_methods[config_method_index].pdc_name;
    const char *pdc_name        = context->pd_attrs[pd_index].component_name;

    return !strncmp(config_pdc_name, "*",      UCT_PD_COMPONENT_NAME_MAX) ||
           !strncmp(config_pdc_name, pdc_name, UCT_PD_COMPONENT_NAME_MAX);
}

static ucs_status_t ucp_mem_alloc(ucp_context_h context, size_t length,
                                  const char *name, ucp_mem_h memh)
{
    uct_allocated_memory_t mem;
    uct_alloc_method_t method;
    unsigned method_index, pd_index, num_pds;
    ucs_status_t status;
    uct_pd_h *pds;

    pds = ucs_calloc(context->num_pds, sizeof(*pds), "temp pds");
    if (pds == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (method_index = 0; method_index < context->config.num_alloc_methods;
                    ++method_index)
    {
        method = context->config.alloc_methods[method_index].method;

        /* If we are trying PD method, gather all PDs which match the component
         * name specified in the configuration.
         */
        num_pds = 0;
        if (method == UCT_ALLOC_METHOD_PD) {
            for (pd_index = 0; pd_index < context->num_pds; ++pd_index) {
                if (ucp_is_pd_selected_by_config(context, method_index, pd_index)) {
                    pds[num_pds++] = context->pds[pd_index];
                }
            }
        }

        status = uct_mem_alloc(length, &method, 1, pds, num_pds, name, &mem);
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
    memh->alloc_pd     = mem.pd;
    status = ucp_memh_reg_pds(context, memh, mem.memh);
    if (status != UCS_OK) {
        uct_mem_free(&mem);
    }
out:
    ucs_free(pds);
    return status;
}


ucs_status_t ucp_mem_map(ucp_context_h context, void **address_p, size_t length,
                         unsigned flags, ucp_mem_h *memh_p)
{
    ucs_status_t status;
    ucp_mem_h memh;

    if (length == 0) {
        if (flags & UCP_MEM_FLAG_ZERO_REG) {
            ucs_debug("mapping zero length buffer, return dummy memh");
            *memh_p = &dummy_mem;
            return UCS_OK;
        }
        return UCS_ERR_INVALID_PARAM;
    }

    /* Allocate the memory handle */
    ucs_assert(context->num_pds > 0);
    memh = ucs_malloc(sizeof(*memh) + context->num_pds * sizeof(memh->uct[0]),
                      "ucp_memh");
    if (memh == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    if (*address_p == NULL) {
        status = ucp_mem_alloc(context, length, "user allocation", memh);
        if (status != UCS_OK) {
            goto err_free_memh;
        }

        *address_p = memh->address;
    } else {
        ucs_debug("registering user memory at %p length %zu", *address_p, length);
        memh->address      = *address_p;
        memh->length       = length;
        memh->alloc_method = UCT_ALLOC_METHOD_LAST;
        memh->alloc_pd     = NULL;
        status = ucp_memh_reg_pds(context, memh, UCT_INVALID_MEM_HANDLE);
        if (status != UCS_OK) {
            goto err_free_memh;
        }
    }

    ucs_debug("%s buffer %p length %zu memh %p pd_map 0x%"PRIx64,
              (memh->alloc_method == UCT_ALLOC_METHOD_LAST) ? "mapped" : "allocated",
              memh->address, memh->length, memh, memh->pd_map);
    *memh_p = memh;
    return UCS_OK;

err_free_memh:
    ucs_free(memh);
err:
    return status;
}

ucs_status_t ucp_mem_unmap(ucp_context_h context, ucp_mem_h memh)
{
    uct_allocated_memory_t mem;
    uct_mem_h alloc_pd_memh;
    ucs_status_t status;

    ucs_debug("unmapping buffer %p memh %p", memh->address, memh);
    if (memh == &dummy_mem) {
        return UCS_OK;
    }

    /* Unregister from all protection domains */
    status = ucp_memh_dereg_pds(context, memh, &alloc_pd_memh);
    if (status != UCS_OK) {
        return status;
    }

    /* If the memory was also allocated, release it */
    if (memh->alloc_method != UCT_ALLOC_METHOD_LAST) {
        mem.address = memh->address;
        mem.length  = memh->length;
        mem.method  = memh->alloc_method;
        mem.pd      = memh->alloc_pd;  /* May be NULL if method is not PD */
        mem.memh    = alloc_pd_memh;   /* May be INVALID if method is not PD */

        status = uct_mem_free(&mem);
        if (status != UCS_OK) {
            return status;
        }
    }

    ucs_free(memh);
    return UCS_OK;
}
