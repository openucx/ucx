/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_rkey.h"
#include "ucp_request.h"
#include "ucp_ep.inl"

#include <ucp/rma/rma.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/profile/profile.h>
#include <ucs/sys/string.h>
#include <inttypes.h>


static struct {
    ucp_md_map_t md_map;
    uint8_t      mem_type;
} UCS_S_PACKED ucp_mem_dummy_buffer = {0, UCS_MEMORY_TYPE_HOST};


size_t ucp_rkey_packed_size(ucp_context_h context, ucp_md_map_t md_map)
{
    size_t size, md_size;
    unsigned md_index;

    size = sizeof(ucp_md_map_t);
    size += sizeof(uint8_t);
    ucs_for_each_bit (md_index, md_map) {
        md_size = context->tl_mds[md_index].attr.rkey_packed_size;
        ucs_assert_always(md_size <= UINT8_MAX);
        size += sizeof(uint8_t) + md_size;
    }
    return size;
}

void ucp_rkey_packed_copy(ucp_context_h context, ucp_md_map_t md_map,
                          ucs_memory_type_t mem_type, void *rkey_buffer,
                          const void* uct_rkeys[])
{
    uint8_t *p = rkey_buffer;
    unsigned md_index;
    size_t md_size;

    *(ucp_md_map_t*)p = md_map;
    p += sizeof(ucp_md_map_t);

    *(p++) = mem_type;

    ucs_for_each_bit(md_index, md_map) {
        md_size = context->tl_mds[md_index].attr.rkey_packed_size;
        ucs_assert_always(md_size <= UINT8_MAX);
        *(p++) = md_size;
        memcpy(p, *uct_rkeys, md_size);
        p += md_size;
        ++uct_rkeys;
    }
}

ssize_t ucp_rkey_pack_uct(ucp_context_h context, ucp_md_map_t md_map,
                          const uct_mem_h *memh, ucs_memory_type_t mem_type,
                          void *rkey_buffer)
{
    uint8_t *p          = rkey_buffer;
    ucs_status_t status = UCS_OK;
    unsigned md_index, uct_memh_index;
    size_t md_size;
    char UCS_V_UNUSED buf[128];

    /* Check that md_map is valid */
    ucs_assert(ucs_test_all_flags(UCS_MASK(context->num_mds), md_map));

    /* Write the MD map */
    *(ucp_md_map_t*)p = md_map;
    p += sizeof(ucp_md_map_t);

    /* Write memory type */
    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_LAST <= 255);
    *(p++) = mem_type;

    /* Write both size and rkey_buffer for each UCT rkey */
    uct_memh_index = 0;
    ucs_for_each_bit (md_index, md_map) {
        md_size = context->tl_mds[md_index].attr.rkey_packed_size;
        *(p++) = md_size;
        status = uct_md_mkey_pack(context->tl_mds[md_index].md,
                                  memh[uct_memh_index], p);
        if (status != UCS_OK) {
            return status;
        }

        ucs_trace("rkey[%d]=%s for md[%d]=%s", uct_memh_index,
                  ucs_str_dump_hex(p, md_size, buf, sizeof(buf), SIZE_MAX),
                  md_index, context->tl_mds[md_index].rsc.md_name);

        ++uct_memh_index;
        p += md_size;
    }

    return UCS_PTR_BYTE_DIFF(rkey_buffer, p);
}

ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p)
{
    void *rkey_buffer, *p;
    ucs_status_t status;
    ssize_t packed_size;
    size_t size;

    /* always acquire context lock */
    UCP_THREAD_CS_ENTER(&context->mt_lock);

    ucs_trace("packing rkeys for buffer %p memh %p md_map 0x%"PRIx64,
              memh->address, memh, memh->md_map);

    if (memh->length == 0) {
        /* dummy memh, return dummy key */
        *rkey_buffer_p = &ucp_mem_dummy_buffer;
        *size_p        = sizeof(ucp_mem_dummy_buffer);
        status         = UCS_OK;
        goto out;
    }

    size = ucp_rkey_packed_size(context, memh->md_map);
    rkey_buffer = ucs_malloc(size, "ucp_rkey_buffer");
    if (rkey_buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    p = rkey_buffer;

    packed_size = ucp_rkey_pack_uct(context, memh->md_map, memh->uct,
                                    memh->mem_type, p);
    if (packed_size < 0) {
        status = (ucs_status_t)packed_size;
        goto err_destroy;
    }

    ucs_assert(packed_size == size);

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

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_rkey_unpack, (ep, rkey_buffer, rkey_p),
                 ucp_ep_h ep, const void *rkey_buffer,
                 ucp_rkey_h *rkey_p)
{
    ucp_worker_h  worker = ep->worker;
    const ucp_ep_config_t *ep_config;
    ucp_rkey_config_key_t rkey_config_key;
    unsigned remote_md_index;
    ucp_md_map_t md_map, remote_md_map;
    ucp_rsc_index_t cmpt_index;
    ucp_tl_rkey_t *tl_rkey;
    unsigned rkey_index;
    unsigned md_count;
    ucs_status_t status;
    ucp_rkey_h rkey;
    ucs_memory_type_t mem_type;
    uint8_t md_size;
    const uint8_t *p;
    uint8_t flags;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);

    ep_config = ucp_ep_config(ep);

    /* Count the number of remote MDs in the rkey buffer */
    p = rkey_buffer;

    /* Read remote MD map */
    remote_md_map = *(ucp_md_map_t*)p;
    ucs_trace("ep %p: unpacking rkey with md_map 0x%"PRIx64, ep, remote_md_map);

    /* MD map for the unpacked rkey */
    md_map   = remote_md_map & ucp_ep_config(ep)->key.reachable_md_map;
    md_count = ucs_popcount(md_map);
    p       += sizeof(ucp_md_map_t);

    /* Allocate rkey handle which holds UCT rkeys for all remote MDs. Small key
     * allocations are done from a memory pool.
     * We keep all of them to handle a future transport switch.
     */
    flags = 0;
    if (md_count <= UCP_RKEY_MPOOL_MAX_MD) {
        rkey  = ucs_mpool_get_inline(&worker->rkey_mp);
        flags = UCP_RKEY_DESC_FLAG_POOL;
    } else {
        rkey = ucs_malloc(sizeof(*rkey) + (sizeof(rkey->tl_rkey[0]) * md_count),
                          "ucp_rkey");
    }
    if (rkey == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }

    /* Read memory type */
    mem_type = (ucs_memory_type_t)*(p++);

    rkey->md_map   = md_map;
    rkey->mem_type = mem_type;
    rkey->flags    = flags;
#if ENABLE_PARAMS_CHECK
    rkey->ep       = ep;
#endif

    /* Unpack rkey of each UCT MD */
    rkey_index = 0; /* Index of the rkey in the array */
    /* Go over remote MD indices */
    ucs_for_each_bit (remote_md_index, remote_md_map) {
        md_size = *(p++);

        /* Use bit operations to iterate through the indices of the remote MDs
         * as provided in the md_map. md_map always holds a bitmap of MD indices
         * that remain to be used. Every time we find the next valid MD index.
         * If some rkeys cannot be unpacked, we remove them from the local map.
         */
        ucs_assert(UCS_BIT(remote_md_index) & remote_md_map);
        ucs_assert_always(remote_md_index <= UCP_MD_INDEX_BITS);

        /* Unpack only reachable rkeys */
        if (UCS_BIT(remote_md_index) & rkey->md_map) {
            ucs_assert(rkey_index < md_count);

            tl_rkey       = &rkey->tl_rkey[rkey_index];
            cmpt_index    = ucp_ep_config_get_dst_md_cmpt(&ep_config->key,
                                                          remote_md_index);
            tl_rkey->cmpt = worker->context->tl_cmpts[cmpt_index].cmpt;

            status = uct_rkey_unpack(tl_rkey->cmpt, p, &tl_rkey->rkey);
            if (status == UCS_OK) {
                ucs_trace("rkey[%d] for remote md %d is 0x%lx", rkey_index,
                          remote_md_index, tl_rkey->rkey.rkey);
                ++rkey_index;
            } else if (status == UCS_ERR_UNREACHABLE) {
                rkey->md_map &= ~UCS_BIT(remote_md_index);
                ucs_trace("rkey[%d] for remote md %d is 0x%lx not reachable",
                          rkey_index, remote_md_index, tl_rkey->rkey.rkey);
                /* FIXME this can make malloc allocated key be released to mpool */
            } else {
                ucs_error("failed to unpack remote key from remote md[%d]: %s",
                          remote_md_index, ucs_status_string(status));
                goto err_destroy;
            }
        }

        p += md_size;
    }

    /* Silence clang checker - assert that if some rkeys are unpacked, then
     * rkey->md_map is nozero.
     */
    ucs_assert((rkey_index > 0) || (rkey->md_map == 0));

    if (worker->context->config.ext.proto_enable) {
        rkey_config_key.ep_cfg_index = ep->cfg_index;
        rkey_config_key.md_map       = rkey->md_map;
        rkey_config_key.mem_type     = rkey->mem_type;
        rkey_config_key.sys_dev      = 0;

        status = ucp_worker_get_rkey_config(worker, &rkey_config_key,
                                            &rkey->cfg_index);
        if (status != UCS_OK) {
            goto err_destroy;
        }

        /* Avoid calling ucp_ep_resolve_remote_id() from rkey_unpack, and let
         * the APIs which are not yet using new protocols resolve the remote key
         * on-demand.
         */
        rkey->cache.ep_cfg_index = UCP_WORKER_CFG_INDEX_NULL;
    } else {
        ucp_rkey_resolve_inner(rkey, ep);
    }

    *rkey_p = rkey;
    status  = UCS_OK;

out_unlock:
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    return status;

err_destroy:
    ucp_rkey_destroy(rkey);
    goto out_unlock;
}

void ucp_rkey_dump_packed(const void *rkey_buffer, char *buffer, size_t max)
{
    char *p                 = buffer;
    char *endp              = buffer + max;
    const uint8_t *rkey_buf = rkey_buffer;
    ucp_md_map_t md_map;
    unsigned md_index;
    uint8_t md_size;
    int first;

    snprintf(p, endp - p, "{");
    p += strlen(p);

    md_map    = *(ucp_md_map_t*)(rkey_buf);
    rkey_buf += sizeof(ucp_md_map_t) + sizeof(uint8_t);

    first = 1;
    ucs_for_each_bit(md_index, md_map) {
         md_size   = *rkey_buf;
         rkey_buf += sizeof(uint8_t);

         if (!first) {
             snprintf(p, endp - p, ",");
             p += strlen(p);
         }
         first = 0;

         snprintf(p, endp - p, "%d:", md_index);
         p += strlen(p);

         ucs_str_dump_hex(rkey_buf, md_size, p, endp - p, SIZE_MAX);
         p += strlen(p);

         rkey_buf += md_size;
    }

    snprintf(p, endp - p, "}");
}

ucs_status_t ucp_rkey_ptr(ucp_rkey_h rkey, uint64_t raddr, void **addr_p)
{
    unsigned remote_md_index, rkey_index;
    ucs_status_t status;

    rkey_index = 0;
    ucs_for_each_bit(remote_md_index, rkey->md_map) {
        status = uct_rkey_ptr(rkey->tl_rkey[rkey_index].cmpt,
                              &rkey->tl_rkey[rkey_index].rkey, raddr, addr_p);
        if ((status == UCS_OK) ||
            (status == UCS_ERR_INVALID_ADDR)) {
            return status;
        }

        ++rkey_index;
    }

    return UCS_ERR_UNREACHABLE;
}

void ucp_rkey_destroy(ucp_rkey_h rkey)
{
    unsigned remote_md_index, rkey_index;
    ucp_worker_h UCS_V_UNUSED worker;

    rkey_index = 0;
    ucs_for_each_bit(remote_md_index, rkey->md_map) {
        uct_rkey_release(rkey->tl_rkey[rkey_index].cmpt,
                         &rkey->tl_rkey[rkey_index].rkey);
        ++rkey_index;
    }

    if (rkey->flags & UCP_RKEY_DESC_FLAG_POOL) {
        worker = ucs_container_of(ucs_mpool_obj_owner(rkey), ucp_worker_t,
                                  rkey_mp);
        UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(worker);
        ucs_mpool_put_inline(rkey);
        UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(worker);
    } else {
        ucs_free(rkey);
    }
}

ucp_lane_index_t ucp_rkey_find_rma_lane(ucp_context_h context,
                                        const ucp_ep_config_t *config,
                                        ucs_memory_type_t mem_type,
                                        const ucp_lane_index_t *lanes,
                                        ucp_rkey_h rkey,
                                        ucp_lane_map_t ignore,
                                        uct_rkey_t *uct_rkey_p)
{
    ucp_md_index_t dst_md_index;
    ucp_lane_index_t lane;
    ucp_md_index_t md_index;
    uct_md_attr_t *md_attr;
    uint64_t mem_types;
    uint8_t rkey_index;
    int prio;

    for (prio = 0; prio < UCP_MAX_LANES; ++prio) {
        lane = lanes[prio];
        if (lane == UCP_NULL_LANE) {
            return UCP_NULL_LANE; /* No more lanes */
        } else if (ignore & UCS_BIT(lane)) {
            continue; /* lane is in ignore mask, do not process it */
        }

        md_index = config->md_index[lane];
        md_attr  = &context->tl_mds[md_index].attr;

        if ((md_index != UCP_NULL_RESOURCE) &&
            (!(md_attr->cap.flags & UCT_MD_FLAG_NEED_RKEY)))
        {
            /* Lane does not need rkey, can use the lane with invalid rkey  */
            if (!rkey || ((md_attr->cap.access_mem_types & UCS_BIT(mem_type)) &&
                          (mem_type == rkey->mem_type))) {
                *uct_rkey_p = UCT_INVALID_RKEY;
                return lane;
            }
        }

        mem_types = md_attr->cap.reg_mem_types | md_attr->cap.alloc_mem_types;
        if ((md_index != UCP_NULL_RESOURCE) && !(mem_types & UCS_BIT(mem_type))) {
            continue;
        }

        dst_md_index = config->key.lanes[lane].dst_md_index;
        if (rkey->md_map & UCS_BIT(dst_md_index)) {
            /* Return first matching lane */
            rkey_index  = ucs_bitmap2idx(rkey->md_map, dst_md_index);
            *uct_rkey_p = rkey->tl_rkey[rkey_index].rkey.rkey;
            return lane;
        }
    }

    return UCP_NULL_LANE;
}

void ucp_rkey_resolve_inner(ucp_rkey_h rkey, ucp_ep_h ep)
{
    ucp_context_h context   = ep->worker->context;
    ucp_ep_config_t *config = ucp_ep_config(ep);
    int rma_sw              = 0;
    int amo_sw              = 0;
    ucs_status_t status;
    uct_rkey_t uct_rkey;

    rkey->cache.rma_lane = ucp_rkey_find_rma_lane(context, config,
                                                  UCS_MEMORY_TYPE_HOST,
                                                  config->key.rma_lanes, rkey,
                                                  0, &uct_rkey);
    if (rkey->cache.rma_lane == UCP_NULL_LANE) {
        rkey->cache.rma_proto     = &ucp_rma_sw_proto;
        rkey->cache.rma_rkey      = UCT_INVALID_RKEY;
        rkey->cache.max_put_short = 0;
        rma_sw                    = !!(context->config.features & UCP_FEATURE_RMA);
    } else {
        rkey->cache.rma_proto     = &ucp_rma_basic_proto;
        rkey->cache.rma_rkey      = uct_rkey;
        rkey->cache.rma_proto     = &ucp_rma_basic_proto;
        rkey->cache.max_put_short = config->rma[rkey->cache.rma_lane].max_put_short;
    }

    rkey->cache.amo_lane = ucp_rkey_find_rma_lane(context, config,
                                                  UCS_MEMORY_TYPE_HOST,
                                                  config->key.amo_lanes, rkey,
                                                  0, &uct_rkey);
    if (rkey->cache.amo_lane == UCP_NULL_LANE) {
        rkey->cache.amo_proto     = &ucp_amo_sw_proto;
        rkey->cache.amo_rkey      = UCT_INVALID_RKEY;
        amo_sw                    = !!(context->config.features &
                                       (UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64));
    } else {
        rkey->cache.amo_proto     = &ucp_amo_basic_proto;
        rkey->cache.amo_rkey      = uct_rkey;
    }

    /* If we use sw rma/amo need to resolve destination endpoint in order to
     * receive responses and completion messages
     */
    if ((amo_sw || rma_sw) && (config->key.am_lane != UCP_NULL_LANE)) {
        status = ucp_ep_resolve_remote_id(ep, config->key.am_lane);
        if (status != UCS_OK) {
            ucs_debug("ep %p: failed to resolve destination ep, "
                      "sw rma cannot be used", ep);
        } else {
            /* if we can resolve destination ep, save the active message lane
             * as the rma/amo lane in the rkey cache
             */
            if (amo_sw) {
                rkey->cache.amo_lane = config->key.am_lane;
            }
            if (rma_sw) {
                rkey->cache.rma_lane = config->key.am_lane;
            }
        }
    }

    rkey->cache.ep_cfg_index  = ep->cfg_index;

    ucs_trace("rkey %p ep %p @ cfg[%d] %s: lane[%d] rkey 0x%"PRIxPTR
              " %s: lane[%d] rkey 0x%"PRIxPTR,
              rkey, ep, ep->cfg_index,
              rkey->cache.rma_proto->name, rkey->cache.rma_lane, rkey->cache.rma_rkey,
              rkey->cache.amo_proto->name, rkey->cache.amo_lane, rkey->cache.amo_rkey);
}
