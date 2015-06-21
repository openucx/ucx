/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "ucp_int.h"



#define UCP_RMA_RKEY_LOOKUP(_ep, _rkey) \
    ({ \
        if (ENABLE_PARAMS_CHECK && \
            !((_rkey)->pd_map & UCS_BIT((_ep)->uct.dst_pd_index))) \
        { \
            ucs_error("Remote key does not support current transport " \
                       "(remote pd index: %d rkey map: 0x%"PRIx64")", \
                       (_ep)->uct.dst_pd_index, (_rkey)->pd_map); \
            return UCS_ERR_UNREACHABLE; \
        } \
        \
        ucp_lookup_uct_rkey(_ep, _rkey); \
    })

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

ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p)
{
    unsigned pd_index, uct_memh_index;
    void *rkey_buffer, *p;
    size_t size, pd_size;

    ucs_trace("packing rkeys for buffer %p memh %p pd_map 0x%"PRIx64,
              memh->address, memh, memh->pd_map);

    size = sizeof(uint64_t);
    for (pd_index = 0; pd_index < context->num_pds; ++pd_index) {
        size += sizeof(uint8_t);
        pd_size = context->pd_attrs[pd_index].rkey_packed_size;
        ucs_assert_always(pd_size < UINT8_MAX);
        size += pd_size;
    }

    rkey_buffer = ucs_malloc(size, "ucp_rkey_buffer");
    if (rkey_buffer == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    p = rkey_buffer;

    /* Write the PD map */
    *(uint64_t*)p = memh->pd_map;
    p += sizeof(uint64_t);

    /* Write both size and rkey_buffer for each UCT rkey */
    uct_memh_index = 0;
    for (pd_index = 0; pd_index < context->num_pds; ++pd_index) {
        if (!(memh->pd_map & UCS_BIT(pd_index))) {
            continue;
        }

        pd_size = context->pd_attrs[pd_index].rkey_packed_size;
        *((uint8_t*)p++) = pd_size;
        uct_pd_mkey_pack(context->pds[pd_index], memh->uct[uct_memh_index], p);
        ++uct_memh_index;
        p += pd_size;
    }

    *rkey_buffer_p = rkey_buffer;
    *size_p        = size;
    return UCS_OK;
}

void ucp_rkey_buffer_release(void *rkey_buffer)
{
    ucs_free(rkey_buffer);
}

ucs_status_t ucp_ep_rkey_unpack(ucp_ep_h ep, void *rkey_buffer, ucp_rkey_h *rkey_p)
{
    unsigned remote_pd_index, remote_pd_gap;
    unsigned rkey_index;
    unsigned pd_count;
    ucs_status_t status;
    ucp_rkey_h rkey;
    uint8_t pd_size;
    uint64_t pd_map;
    void *p;

    /* Count the number of remote PDs in the rkey buffer */
    p = rkey_buffer;

    /* Read remote PD map */
    pd_map   = *(uint64_t*)p;
    pd_count = ucs_count_one_bits(pd_map);
    p       += sizeof(uint64_t);

    /* Allocate rkey handle which holds UCT rkeys for all remote PDs.
     * We keep all of them to handle a future transport switch.
     */
    rkey = ucs_malloc(sizeof(*rkey) + (sizeof(rkey->uct[0]) * pd_count), "ucp_rkey");
    if (rkey == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rkey->pd_map    = 0;
    remote_pd_index = 0; /* Index of remote PD */
    rkey_index      = 0; /* Index of the rkey in the array */

    /* Unpack rkey of each UCT PD */
    ucs_trace("unpacking rkey with pd_map 0x%"PRIx64, pd_map);
    while (pd_map > 0) {
        pd_size = *((uint8_t*)p++);

        /* Use bit operations to iterate through the indices of the remote PDs
         * as provided in the pd_map. pd_map always holds a bitmap of PD indices
         * that remain to be used. Every time we find the "gap" until the next
         * valid PD index using ffs operation. If some rkeys cannot be unpacked,
         * we remove them from the local map.
         */
        remote_pd_gap    = ucs_ffs64(pd_map); /* Find the offset for next PD index */
        remote_pd_index += remote_pd_gap;      /* Calculate next index of remote PD*/
        pd_map >>= remote_pd_gap;                   /* Remove the gap from the map */
        ucs_assert(pd_map & 1);

        /* Unpack only reachable rkeys */
        if (ep->uct.reachable_pds & UCS_BIT(remote_pd_index)) {

            ucs_assert(rkey_index < pd_count);
            status = uct_rkey_unpack(p, &rkey->uct[rkey_index]);
            if (status != UCS_OK) {
                ucs_error("Failed to unpack remote key from remote pd[%d]: %s",
                          remote_pd_index, ucs_status_string(status));
                goto err_destroy;
            }

            ucs_trace("rkey[%d] for remote pd %d is 0x%lx", rkey_index,
                      remote_pd_index, rkey->uct[rkey_index].rkey);
            rkey->pd_map |= UCS_BIT(remote_pd_index);
            ++rkey_index;
        }

        ++remote_pd_index;
        pd_map >>= 1;
        p += pd_size;
    }

    *rkey_p = rkey;
    return UCS_OK;

err_destroy:
    ucp_rkey_destroy(rkey);
err:
    return status;
}

void ucp_rkey_destroy(ucp_rkey_h rkey)
{
    unsigned num_rkeys = ucs_count_one_bits(rkey->pd_map);
    unsigned i;

    for (i = 0; i < num_rkeys; ++i) {
        uct_rkey_release(&rkey->uct[i]);
    }
    ucs_free(rkey);
}

static inline uct_rkey_t ucp_lookup_uct_rkey(ucp_ep_h ep, ucp_rkey_h rkey)
{
    unsigned rkey_index;

    /*
     * Calculate the rkey index inside the compact array. This is actually the
     * number of PDs in the map with index less-than ours. So mask pd_map to get
     * only the less-than indices, and then count them using popcount operation.
     * TODO save the mask in ep->uct, to avoid the shift operation.
     */
    rkey_index = ucs_count_one_bits(rkey->pd_map & UCS_MASK(ep->uct.dst_pd_index));
    return rkey->uct[rkey_index].rkey;
}

ucs_status_t ucp_rma_put(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;

    uct_rkey = UCP_RMA_RKEY_LOOKUP(ep, rkey);

    /* Loop until all message has been sent.
     * We re-check the configuration on every iteration, because it can be
     * changed by transport switch.
     */
    for (;;) {
        if (length <= ep->config.max_short_put) {
            status = uct_ep_put_short(ep->uct.ep, buffer, length, remote_addr,
                                      uct_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                break;
            }
        } else {
            if (length <= ep->worker->context->config.bcopy_thresh) {
                frag_length = ucs_min(length, ep->config.max_short_put);
                status = uct_ep_put_short(ep->uct.ep, buffer, frag_length, remote_addr,
                                          uct_rkey);
            } else {
                frag_length = ucs_min(length, ep->config.max_bcopy_put);
                status = uct_ep_put_bcopy(ep->uct.ep, (uct_pack_callback_t)memcpy,
                                          (void*)buffer, frag_length, remote_addr,
                                          uct_rkey);
            }
            if (ucs_likely(status == UCS_OK)) {
                length      -= frag_length;
                if (length == 0) {
                    break;
                }

                buffer      += frag_length;
                remote_addr += frag_length;
            } else if (status != UCS_ERR_NO_RESOURCE) {
                break;
            }
        }
        ucp_worker_progress(ep->worker);
    }

    return status;
}

ucs_status_t ucp_rma_get(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_rma_fence(ucp_worker_h worker)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_rma_flush(ucp_worker_h worker)
{
    unsigned rsc_index;

    /* TODO flush in parallel */
    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index] == NULL) {
            continue;
        }

        while (uct_iface_flush(worker->ifaces[rsc_index]) != UCS_OK) {
            ucp_worker_progress(worker);
        }
    }

    return UCS_OK;
}
