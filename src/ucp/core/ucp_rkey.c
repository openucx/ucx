/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_mm.h"
#include "ucp_request.h"
#include "ucp_ep.inl"

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

    ucs_trace("packing rkeys for buffer %p memh %p md_map 0x%x",
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
    unsigned remote_md_index, remote_md_gap;
    unsigned rkey_index;
    unsigned md_count;
    ucs_status_t status;
    ucp_rkey_h rkey;
    uint8_t md_size;
    ucp_md_map_t md_map;
    void *p;

    /* Count the number of remote MDs in the rkey buffer */
    p = rkey_buffer;

    /* Read remote MD map */
    md_map   = *(ucp_md_map_t*)p;

    ucs_trace("unpacking rkey with md_map 0x%x", md_map);

    if (md_map == 0) {
        /* Dummy key return ok */
        *rkey_p = &ucp_mem_dummy_rkey;
        return UCS_OK;
    }

    md_count = ucs_count_one_bits(md_map);
    p       += sizeof(ucp_md_map_t);

    /* Allocate rkey handle which holds UCT rkeys for all remote MDs.
     * We keep all of them to handle a future transport switch.
     */
    rkey = ucs_malloc(sizeof(*rkey) + (sizeof(rkey->uct[0]) * md_count), "ucp_rkey");
    if (rkey == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    rkey->md_map    = 0;
    remote_md_index = 0; /* Index of remote MD */
    rkey_index      = 0; /* Index of the rkey in the array */

    /* Unpack rkey of each UCT MD */
    while (md_map > 0) {
        md_size = *((uint8_t*)p++);

        /* Use bit operations to iterate through the indices of the remote MDs
         * as provided in the md_map. md_map always holds a bitmap of MD indices
         * that remain to be used. Every time we find the "gap" until the next
         * valid MD index using ffs operation. If some rkeys cannot be unpacked,
         * we remove them from the local map.
         */
        remote_md_gap    = ucs_ffs64(md_map); /* Find the offset for next MD index */
        remote_md_index += remote_md_gap;      /* Calculate next index of remote MD*/
        md_map >>= remote_md_gap;                   /* Remove the gap from the map */
        ucs_assert(md_map & 1);
        ucs_assert_always(remote_md_index <= UCP_MD_INDEX_BITS);

        /* Unpack only reachable rkeys */
        if (UCS_BIT(remote_md_index) & ucp_ep_config(ep)->key.reachable_md_map) {
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
        md_map >>= 1;
        p += md_size;
    }

    if (rkey->md_map == 0) {
        ucs_debug("The unpacked rkey from the destination is unreachable");
        status = UCS_ERR_UNREACHABLE;
        goto err_destroy;
    }

    ucp_rkey_resolve_inner(rkey, ep);
    *rkey_p = rkey;
    return UCS_OK;

err_destroy:
    ucp_rkey_destroy(rkey);
err:
    return status;
}

void ucp_rkey_destroy(ucp_rkey_h rkey)
{
    unsigned num_rkeys;
    unsigned i;

    if (rkey == &ucp_mem_dummy_rkey) {
        return;
    }

    num_rkeys = ucs_count_one_bits(rkey->md_map);

    for (i = 0; i < num_rkeys; ++i) {
        uct_rkey_release(&rkey->uct[i]);
    }
    ucs_free(rkey);
}

static inline ucp_md_lane_map_t ucp_ep_md_map_expand(ucp_md_map_t md_map)
{
    /* "Broadcast" md_map to all bit groups in ucp_md_lane_map_t, so that
     * if would look like this: <md_map><md_map>....<md_map>.
     * The expanded value can be used to select which lanes support a given md_map.
     * TODO use SSE and support larger md_lane_map with shuffle / broadcast
     */
    UCS_STATIC_ASSERT(UCP_MD_LANE_MAP_BITS == 64);
    UCS_STATIC_ASSERT(UCP_MD_INDEX_BITS    == 8);
    return md_map * 0x0101010101010101ul;
}

/*
 * Calculate lane and rkey index based of the lane_map in ep configuration: the
 * lane_map holds the md_index which each lane supports, so we do 'and' between
 * that, and the md_map of the rkey (we duplicate the md_map in rkey to fill the
 * mask for each possible lane). The first set bit in the 'and' product represents
 * the first matching lane and its md_index.
 */
#define UCP_EP_RESOLVE_RKEY(_ep, _rkey, _name, _config, _lane, _uct_rkey) \
    { \
        ucp_md_lane_map_t ep_lane_map, rkey_md_map; \
        ucp_rsc_index_t dst_md_index, rkey_index; \
        uint8_t bit_index; \
        \
        _config     = ucp_ep_config(_ep); \
        ep_lane_map = (_config)->key._name##_lane_map; \
        rkey_md_map = ucp_ep_md_map_expand((_rkey)->md_map); \
        \
        if (!(ep_lane_map & rkey_md_map)) { \
            _lane     = UCP_NULL_LANE; \
            _uct_rkey = UCT_INVALID_RKEY; \
        } else { \
            /* Find the first lane which supports one of the remote md's in the rkey*/ \
            bit_index    = ucs_ffs64(ep_lane_map & rkey_md_map); \
            _lane        = bit_index / UCP_MD_INDEX_BITS; \
            dst_md_index = bit_index % UCP_MD_INDEX_BITS; \
            rkey_index   = ucs_count_one_bits(rkey_md_map & UCS_MASK(dst_md_index)); \
            _uct_rkey    = (_rkey)->uct[rkey_index].rkey; \
        } \
    }

void ucp_rkey_resolve_inner(ucp_rkey_h rkey, ucp_ep_h ep)
{
    ucp_lane_index_t amo_index;
    ucp_ep_config_t *config;

    UCP_EP_RESOLVE_RKEY(ep, rkey, rma, config, rkey->c.rma_lane, rkey->c.rma_rkey);
    if (rkey->c.rma_lane != UCP_NULL_LANE) {
        rkey->c.max_put_short = config->rma[rkey->c.rma_lane].max_put_short;
    }

    UCP_EP_RESOLVE_RKEY(ep, rkey, amo, config, amo_index, rkey->c.amo_rkey);
    if (amo_index != UCP_NULL_LANE) {
        rkey->c.amo_lane      = config->key.amo_lanes[amo_index];
    } else {
        rkey->c.amo_lane      = UCP_NULL_LANE;
    }

    rkey->c.ep_cfg_index  = ep->cfg_index;
    ucs_trace("rkey %p ep %p @ cfg[%d] rma_lane %d amo_lane %d", rkey, ep,
              ep->cfg_index, rkey->c.rma_lane, rkey->c.amo_lane);
}

