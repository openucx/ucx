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

    ucs_trace("packing rkeys for buffer %p memh %p md_map 0x%x",
              memh->address, memh, memh->md_map);

    if (memh->length == 0) {
        /* dummy memh, return dummy key */
        *rkey_buffer_p = &ucp_mem_dummy_buffer;
        *size_p        = sizeof(ucp_mem_dummy_buffer);
        return UCS_OK;
    }

    size = sizeof(ucp_md_map_t);
    for (md_index = 0; md_index < context->num_mds; ++md_index) {
        size += sizeof(uint8_t);
        md_size = context->md_attrs[md_index].rkey_packed_size;
        ucs_assert_always(md_size < UINT8_MAX);
        size += md_size;
    }

    rkey_buffer = ucs_malloc(size, "ucp_rkey_buffer");
    if (rkey_buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
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

        md_size = context->md_attrs[md_index].rkey_packed_size;
        *((uint8_t*)p++) = md_size;
        uct_md_mkey_pack(context->mds[md_index], memh->uct[uct_memh_index], p);

        ucs_trace("rkey[%d]=%s for md[%d]=%s", uct_memh_index,
                  ucs_log_dump_hex(p, md_size, buf, sizeof(buf)), md_index,
                  context->md_rscs[md_index].md_name);

        ++uct_memh_index;
        p += md_size;
    }

    if (uct_memh_index == 0) {
        status = UCS_ERR_UNSUPPORTED;
        goto err_destroy;
    }

    *rkey_buffer_p = rkey_buffer;
    *size_p        = size;
    return UCS_OK;

err_destroy:
    ucs_free(rkey_buffer);
err:
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
