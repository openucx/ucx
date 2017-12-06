/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_mm.h"
#include "ucp_request.h"
#include "ucp_ep.inl"

#include <ucs/datastruct/mpool.inl>
#include <inttypes.h>


static ucp_rkey_t ucp_mem_dummy_rkey = {
                                        // TODO cache?
    .md_map = 0
};

static ucp_md_map_t ucp_mem_dummy_buffer = 0;

ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p)
{
    unsigned md_index, uct_memh_index;
    void *rkey_buffer, *p;
    size_t size, md_size;
    ucs_status_t status;
    char UCS_V_UNUSED buf[128];

    /* always acquire context lock */
    UCP_THREAD_CS_ENTER(&context->mt_lock);

    ucs_trace("packing rkeys for buffer %p memh %p md_map 0x%lx",
              memh->address, memh, memh->md_map);

    if (memh->length == 0) {
        /* dummy memh, return dummy key */
        *rkey_buffer_p = &ucp_mem_dummy_buffer;
        *size_p        = sizeof(ucp_mem_dummy_buffer);
        status         = UCS_OK;
        goto out;
    }

    size = sizeof(ucp_md_map_t);
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        if (!(memh->md_map & UCS_BIT(md_index))) {
            continue;
        }

        size += sizeof(uint8_t);
        md_size = context->tl_mds[md_index].attr.rkey_packed_size;
        ucs_assert_always(md_size < UINT8_MAX);
        size += md_size;
    }

    rkey_buffer = ucs_malloc(size, "ucp_rkey_buffer");
    if (rkey_buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    p = rkey_buffer;

    /* Write the MD map */
    *(ucp_md_map_t*)p = memh->md_map;
    p += sizeof(ucp_md_map_t);

    /* Write both size and rkey_buffer for each UCT rkey */
    uct_memh_index = 0;
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        if (!(memh->md_map & UCS_BIT(md_index))) {
            continue;
        }

        md_size = context->tl_mds[md_index].attr.rkey_packed_size;
        *((uint8_t*)p++) = md_size;
        uct_md_mkey_pack(context->tl_mds[md_index].md, memh->uct[uct_memh_index], p);

        ucs_trace("rkey[%d]=%s for md[%d]=%s", uct_memh_index,
                  ucs_log_dump_hex(p, md_size, buf, sizeof(buf)), md_index,
                  context->tl_mds[md_index].rsc.md_name);

        ++uct_memh_index;
        p += md_size;
    }

    if (uct_memh_index == 0) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_destroy;
    }

    *rkey_buffer_p = rkey_buffer;
    *size_p        = size;
    status         = UCS_OK;
    goto out;

err_destroy:
    ucs_free(rkey_buffer);
out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

void ucp_rkey_buffer_release(void *rkey_buffer)
{
    if (rkey_buffer == &ucp_mem_dummy_buffer) {
        /* Dummy key, just return */
        return;
    }
    ucs_free(rkey_buffer);
}

ucs_status_t ucp_ep_rkey_unpack(ucp_ep_h ep, void *rkey_buffer, ucp_rkey_h *rkey_p)
{
    ucp_context_t *context = ep->worker->context;
    unsigned remote_md_index, remote_md_gap;
    ucp_md_map_t md_map, remote_md_map;
    unsigned rkey_index;
    unsigned md_count;
    ucs_status_t status;
    ucp_rkey_h rkey;
    uint8_t md_size;
    void *p;

    /* Count the number of remote MDs in the rkey buffer */
    p = rkey_buffer;

    /* Read remote MD map */
    remote_md_map = *(ucp_md_map_t*)p;
    ucs_trace("unpacking rkey with md_map 0x%lx", remote_md_map);

    /* MD map for the unpacked rkey */
    md_map = remote_md_map & ucp_ep_config(ep)->key.reachable_md_map;
    if (md_map == 0) {
        /* Dummy key return ok */
        *rkey_p = &ucp_mem_dummy_rkey;
        return UCS_OK;
    }

    md_count = ucs_count_one_bits(md_map);
    p       += sizeof(ucp_md_map_t);

    /* Allocate rkey handle which holds UCT rkeys for all remote MDs. Small key
     * allocations are done from a memory pool.
     * We keep all of them to handle a future transport switch.
     */
    if (md_count <= UCP_RKEY_MPOOL_MAX_MD) {
        UCP_THREAD_CS_ENTER_CONDITIONAL(&context->mt_lock);
        rkey = ucs_mpool_get(&context->rkey_mp);
        UCP_THREAD_CS_EXIT_CONDITIONAL(&context->mt_lock);
    } else {
        rkey = ucs_malloc(sizeof(*rkey) + (sizeof(rkey->uct[0]) * md_count),
                          "ucp_rkey");
    }
    if (rkey == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rkey->md_map = md_map;

    /* Unpack rkey of each UCT MD */
    remote_md_index = 0; /* Index of remote MD */
    rkey_index      = 0; /* Index of the rkey in the array */
    while (remote_md_map > 0) {
        md_size = *((uint8_t*)p++);

        /* Use bit operations to iterate through the indices of the remote MDs
         * as provided in the md_map. md_map always holds a bitmap of MD indices
         * that remain to be used. Every time we find the "gap" until the next
         * valid MD index using ffs operation. If some rkeys cannot be unpacked,
         * we remove them from the local map.
         */
        remote_md_gap    = ucs_ffs64(remote_md_map); /* Find the offset for next MD index */
        remote_md_index += remote_md_gap;            /* Calculate next index of remote MD*/
        remote_md_map  >>= remote_md_gap;            /* Remove the gap from the map */
        ucs_assert(remote_md_map & 1);
        ucs_assert_always(remote_md_index <= UCP_MD_INDEX_BITS);

        /* Unpack only reachable rkeys */
        if (UCS_BIT(remote_md_index) & rkey->md_map) {
            ucs_assert(rkey_index < md_count);

            status = uct_rkey_unpack(p, &rkey->uct[rkey_index]);
            if (status != UCS_OK) {
                ucs_error("Failed to unpack remote key from remote md[%d]: %s",
                          remote_md_index, ucs_status_string(status));
                goto err_destroy;
            }

            ucs_trace("rkey[%d] for remote md %d is 0x%lx", rkey_index,
                      remote_md_index, rkey->uct[rkey_index].rkey);
            rkey->md_map |= UCS_BIT(remote_md_index);
            ++rkey_index;
        }

        ++remote_md_index;
        remote_md_map >>= 1;
        p += md_size;
    }

    ucs_assert(rkey_index == md_count);

    ucp_rkey_resolve_inner(rkey, ep);
    *rkey_p = rkey;
    return UCS_OK;

err_destroy:
    ucp_rkey_destroy(rkey);
err:
    return status;
}

ucs_status_t ucp_rkey_ptr(ucp_rkey_h rkey, uint64_t raddr, void **addr_p)
{
    unsigned num_rkeys;
    unsigned i;
    ucs_status_t status;

    if (rkey == &ucp_mem_dummy_rkey) {
        return UCS_ERR_UNREACHABLE;
    }

    num_rkeys = ucs_count_one_bits(rkey->md_map);

    for (i = 0; i < num_rkeys; ++i) {
        status = uct_rkey_ptr(&rkey->uct[i], raddr, addr_p);
        if ((status == UCS_OK) ||
            (status == UCS_ERR_INVALID_ADDR)) {
            return status;
        }
    }

    return UCS_ERR_UNREACHABLE;
}

void ucp_rkey_destroy(ucp_rkey_h rkey)
{
    ucp_context_h UCS_V_UNUSED context;
    unsigned num_rkeys;
    unsigned i;

    if (rkey == &ucp_mem_dummy_rkey) {
        return;
    }

    num_rkeys = ucs_count_one_bits(rkey->md_map);

    for (i = 0; i < num_rkeys; ++i) {
        uct_rkey_release(&rkey->uct[i]);
    }

    if (ucs_count_one_bits(rkey->md_map) <= UCP_RKEY_MPOOL_MAX_MD) {
        context = ucs_container_of(ucs_mpool_obj_owner(rkey), ucp_context_t,
                                   rkey_mp);
        UCP_THREAD_CS_ENTER_CONDITIONAL(&context->mt_lock);
        ucs_mpool_put(rkey);
        UCP_THREAD_CS_EXIT_CONDITIONAL(&context->mt_lock);
    } else {
        ucs_free(rkey);
    }
}

static ucp_lane_index_t ucp_config_find_rma_lane(const ucp_ep_config_t *config,
                                                 const ucp_lane_index_t *lanes,
                                                 ucp_md_map_t rkey_md_map,
                                                 ucp_md_index_t *rkey_index_p)
{
    ucp_md_index_t dst_md_index;
    ucp_lane_index_t lane;
    ucp_md_map_t dst_md_mask;
    int prio;

    for (prio = 0; prio < UCP_MAX_LANES; ++prio) {
        lane = lanes[prio];
        if (lane == UCP_NULL_LANE) {
            return UCP_NULL_LANE; /* No more lanes */
        }

        dst_md_index = config->key.lanes[lane].dst_md_index;
        dst_md_mask  = UCS_BIT(dst_md_index);
        if (rkey_md_map & dst_md_mask) {
            /* Return first matching lane */
            *rkey_index_p = ucs_count_one_bits(rkey_md_map & (dst_md_mask - 1));
            return lane;
        }
    }

    return UCP_NULL_LANE;
}

void ucp_rkey_resolve_inner(ucp_rkey_h rkey, ucp_ep_h ep)
{
    ucp_ep_config_t *config = ucp_ep_config(ep);
    ucp_md_index_t rkey_index;

    rkey->cache.rma_lane = ucp_config_find_rma_lane(config, config->key.rma_lanes,
                                                    rkey->md_map, &rkey_index);
    if (rkey->cache.rma_lane != UCP_NULL_LANE) {
        rkey->cache.rma_rkey      = rkey->uct[rkey_index].rkey;
        rkey->cache.max_put_short = config->rma[rkey->cache.rma_lane].max_put_short;
    }

    rkey->cache.amo_lane = ucp_config_find_rma_lane(config, config->key.amo_lanes,
                                                    rkey->md_map, &rkey_index);
    if (rkey->cache.amo_lane != UCP_NULL_LANE) {
        rkey->cache.amo_rkey      = rkey->uct[rkey_index].rkey;
    }

    rkey->cache.ep_cfg_index  = ep->cfg_index;
    ucs_trace("rkey %p ep %p @ cfg[%d] rma_lane %d amo_lane %d", rkey, ep,
              ep->cfg_index, rkey->cache.rma_lane, rkey->cache.amo_lane);
}

