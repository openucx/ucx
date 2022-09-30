/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "ucp_rkey.inl"
#include "ucp_request.h"
#include "ucp_ep.inl"

#include <ucp/core/ucp_mm.inl>
#include <ucp/rma/rma.h>
#include <ucp/proto/proto_debug.h>
#include <ucs/datastruct/mpool.inl>
#include <ucs/profile/profile.h>
#include <ucs/type/float8.h>
#include <ucs/type/serialize.h>
#include <ucs/sys/string.h>
#include <ucs/sys/topo/base/topo.h>
#include <inttypes.h>


typedef struct {
    uint8_t   sys_dev;
    ucs_fp8_t latency;
    ucs_fp8_t bandwidth;
} UCS_S_PACKED ucp_rkey_packed_distance_t;

static struct {
    ucp_md_map_t md_map;
    uint8_t      mem_type;
} UCS_S_PACKED ucp_memh_rkey_dummy_buffer = {0, UCS_MEMORY_TYPE_HOST};


ucp_memh_dummy_buffer_t ucp_memh_dummy_buffer = { 
    sizeof(ucp_memh_dummy_buffer)
};

const ucp_amo_proto_t *ucp_amo_proto_list[] = {
    [UCP_RKEY_BASIC_PROTO] = &ucp_amo_basic_proto,
    [UCP_RKEY_SW_PROTO]    = &ucp_amo_sw_proto
};

const ucp_rma_proto_t *ucp_rma_proto_list[] = {
    [UCP_RKEY_BASIC_PROTO] = &ucp_rma_basic_proto,
    [UCP_RKEY_SW_PROTO]    = &ucp_rma_sw_proto
};


/* Storage for a default value returned from  */
uint64_t memh_serialize_default_val = 0;


size_t ucp_rkey_packed_size(ucp_context_h context, ucp_md_map_t md_map,
                            ucs_sys_device_t sys_dev,
                            ucp_sys_dev_map_t sys_dev_map)
{
    size_t size, tl_rkey_size;
    unsigned md_index;

    size  = sizeof(ucp_md_map_t); /* Memory domains map */
    size += sizeof(uint8_t); /* Memory type */

    ucs_for_each_bit(md_index, md_map) {
        tl_rkey_size = context->tl_mds[md_index].attr.rkey_packed_size;
        ucs_assert_always(tl_rkey_size <= UINT8_MAX);
        size += sizeof(uint8_t) + tl_rkey_size;
    }

    if (sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        /* System device id */
        size += sizeof(uint8_t);

        /* Distance of each device */
        size += ucs_popcount(sys_dev_map) * sizeof(ucp_rkey_packed_distance_t);
    }

    return size;
}

void ucp_rkey_packed_copy(ucp_context_h context, ucp_md_map_t md_map,
                          ucs_memory_type_t mem_type, void *buffer,
                          const void *uct_rkeys[])
{
    void *p = buffer;
    size_t tl_rkey_size;
    unsigned md_index;

    *ucs_serialize_next(&p, ucp_md_map_t) = md_map;
    *ucs_serialize_next(&p, uint8_t)      = mem_type;

    ucs_for_each_bit(md_index, md_map) {
        tl_rkey_size = context->tl_mds[md_index].attr.rkey_packed_size;
        ucs_assert_always(tl_rkey_size <= UINT8_MAX);
        *ucs_serialize_next(&p, uint8_t) = tl_rkey_size;
        memcpy(ucs_serialize_next_raw(&p, void, tl_rkey_size), *(uct_rkeys++),
               tl_rkey_size);
    }
}

static void ucp_rkey_pack_distance(ucs_sys_device_t sys_dev,
                                   const ucs_sys_dev_distance_t *distance,
                                   ucp_rkey_packed_distance_t *packed_distance)
{
    double latency_nsec = distance->latency * UCS_NSEC_PER_SEC;

    packed_distance->sys_dev   = sys_dev;
    packed_distance->latency   = UCS_FP8_PACK(LATENCY, latency_nsec);
    packed_distance->bandwidth = UCS_FP8_PACK(BANDWIDTH, distance->bandwidth);
}

static void
ucp_rkey_unpack_distance(const ucp_rkey_packed_distance_t *packed_distance,
                         ucs_sys_device_t *sys_dev_p,
                         ucs_sys_dev_distance_t *distance)
{
    double latency_nsec = UCS_FP8_UNPACK(LATENCY, packed_distance->latency);

    *sys_dev_p          = packed_distance->sys_dev;
    distance->latency   = latency_nsec / UCS_NSEC_PER_SEC;
    distance->bandwidth = UCS_FP8_UNPACK(BANDWIDTH, packed_distance->bandwidth);
}

static ssize_t
ucp_rkey_pack_common(ucp_context_h context, ucp_md_map_t md_map,
                     const uct_mem_h *memh, const ucp_memory_info_t *mem_info,
                     ucp_sys_dev_map_t sys_dev_map,
                     const ucs_sys_dev_distance_t *sys_distance, void *buffer,
                     int sparse_memh, unsigned uct_flags)
{
    void *p = buffer;
    uct_md_mkey_pack_params_t params;
    unsigned md_index, uct_memh_index;
    char UCS_V_UNUSED buf[128];
    ucs_sys_device_t sys_dev;
    size_t tl_rkey_size;
    ucs_status_t status;
    void *tl_rkey_buf;
    ssize_t result;

    /* Check that md_map is valid */
    ucs_assert(ucs_test_all_flags(UCS_MASK(context->num_mds), md_map));

    ucs_trace("packing rkey type %s md_map 0x%" PRIx64 "dev_map 0x%" PRIx64,
              ucs_memory_type_names[mem_info->type], md_map, sys_dev_map);
    ucs_log_indent(1);

    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_LAST <= 255);
    *ucs_serialize_next(&p, ucp_md_map_t) = md_map;
    *ucs_serialize_next(&p, uint8_t)      = mem_info->type;

    params.field_mask = UCT_MD_MKEY_PACK_FIELD_FLAGS;
    /* Write both size and rkey_buffer for each UCT rkey */
    uct_memh_index = 0;
    ucs_for_each_bit (md_index, md_map) {
        tl_rkey_size = context->tl_mds[md_index].attr.rkey_packed_size;
        *ucs_serialize_next(&p, uint8_t) = tl_rkey_size;

        tl_rkey_buf  = ucs_serialize_next_raw(&p, void, tl_rkey_size);
        params.flags = context->tl_mds[md_index].pack_flags_mask & uct_flags;

        status = uct_md_mkey_pack_v2(context->tl_mds[md_index].md,
                                     memh[sparse_memh ? md_index :
                                     uct_memh_index], &params, tl_rkey_buf);
        if (status != UCS_OK) {
            result = status;
            goto out;
        }

        ucs_trace("rkey[%d]=%s for md[%d]=%s", uct_memh_index,
                  ucs_str_dump_hex(p, tl_rkey_size, buf, sizeof(buf), SIZE_MAX),
                  md_index, context->tl_mds[md_index].rsc.md_name);
        ++uct_memh_index;
    }

    if (ucs_likely(mem_info->sys_dev == UCS_SYS_DEVICE_ID_UNKNOWN)) {
        goto out_packed_size;
    }

    /* Pack system device id */
    *ucs_serialize_next(&p, uint8_t) = mem_info->sys_dev;

    /* Pack distance from sys_dev to each device in distance_dev_map */
    ucs_for_each_bit(sys_dev, sys_dev_map) {
        ucp_rkey_pack_distance(sys_dev, sys_distance++,
                               ucs_serialize_next(&p,
                                                  ucp_rkey_packed_distance_t));
    }

out_packed_size:
    result = UCS_PTR_BYTE_DIFF(buffer, p);
out:
    ucs_log_indent(-1);
    return result;
}

UCS_PROFILE_FUNC(ssize_t, ucp_rkey_pack_uct,
                 (context, md_map, memh, mem_info, sys_dev_map, uct_flags,
                  sys_distance, buffer),
                 ucp_context_h context, ucp_md_map_t md_map,
                 const uct_mem_h *memh, const ucp_memory_info_t *mem_info,
                 ucp_sys_dev_map_t sys_dev_map, unsigned uct_flags,
                 const ucs_sys_dev_distance_t *sys_distance, void *buffer)
{
    return ucp_rkey_pack_common(context, md_map, memh, mem_info,
                                sys_dev_map, sys_distance, buffer, 0, uct_flags);
}

UCS_PROFILE_FUNC(ssize_t, ucp_rkey_pack_memh,
                 (context, md_map, memh, mem_info, sys_dev_map, sys_distance,
                  buffer),
                 ucp_context_h context, ucp_md_map_t md_map,
                 const ucp_mem_h memh, const ucp_memory_info_t *mem_info,
                 ucp_sys_dev_map_t sys_dev_map,
                 const ucs_sys_dev_distance_t *sys_distance, void *buffer)
{
    return ucp_rkey_pack_common(context, md_map, memh->uct, mem_info,
                                sys_dev_map, sys_distance, buffer, 1, 0);
}

static size_t ucp_memh_common_packed_size()
{
    size_t size;

    /* Size of mkey information which comes prior TL mkey data of all MDs */
    size = sizeof(uint16_t);

    /* Flags */
    size += sizeof(uint64_t);

    /* Memory domains map */
    size += sizeof(ucp_md_map_t);

    /* Memory type */
    size += sizeof(uint8_t);

    return size;
}

static void* ucp_memh_common_pack(const ucp_mem_h memh, void *buffer,
                                  uint64_t flags, uint16_t **memh_info_size_p)
{
    ucp_context_h UCS_V_UNUSED context = memh->context;
    void *p                            = buffer;

    /* Check that md_map is valid */
    ucs_assert(ucs_test_all_flags(UCS_MASK(context->num_mds), memh->md_map));

    /* Size of mkey information which comes prior TL mkey data of all MDs */
    *memh_info_size_p                 = p;
    *ucs_serialize_next(&p, uint16_t) = ucp_memh_common_packed_size();

    /* Flags */
    *ucs_serialize_next(&p, uint64_t) = flags;

    /* Memory domains map */
    *ucs_serialize_next(&p, ucp_md_map_t) =
            context->export_md_map[memh->mem_type];

    /* Memory type */
    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_LAST <= 255);
    *ucs_serialize_next(&p, uint8_t) = memh->mem_type;

    return p;
}

size_t ucp_memh_global_id_packed_size(uct_md_attr_v2_t *md_attr)
{
    size_t size = UCT_MD_GLOBAL_ID_MAX;

    while ((size != 0) && (md_attr->global_id[size - 1] == '\0')) {
        --size;
    }

    ucs_assert(size < UCT_MD_GLOBAL_ID_MAX);
    return size;
}

static size_t
ucp_memh_exported_packed_size(ucp_context_h context, ucp_md_map_t md_map)
{
    size_t size = ucp_memh_common_packed_size();
    uct_md_attr_v2_t *md_attr;
    ucp_md_index_t md_index;

    /* Address */
    size += sizeof(uint64_t);

    /* Length */
    size += sizeof(uint64_t);

    /* UCP uuid */
    size += sizeof(uint64_t);

    /* Registration ID */
    size += sizeof(uint64_t);

    ucs_for_each_bit(md_index, md_map) {
        md_attr = &context->tl_mds[md_index].attr;

        /* Size of packed TL mkey data */
        size += sizeof(uint16_t);

        /* TL mkey size */
        size += sizeof(uint8_t);

        /* TL mkey */
        size += md_attr->exported_mkey_packed_size;

        /* Size of component name */
        size += sizeof(uint8_t);

        /* Component name */
        size += strlen(md_attr->component_name);

        /* Size of global MD identifier */
        size += sizeof(uint8_t);

        /* Global MD identifier */
        size += ucp_memh_global_id_packed_size(md_attr);
    }

    return size;
}

static ssize_t
ucp_memh_exported_pack(const ucp_mem_h memh, void *buffer)
{
    ucp_context_h context            = memh->context;
    uint64_t address                 = (uint64_t)ucp_memh_address(memh);
    uint64_t length                  = ucp_memh_length(memh);
    uct_mem_h *uct_memhs             = memh->uct;
    void *p                          = buffer;
    ucp_tl_md_t *tl_mds              = context->tl_mds;
    uct_md_mkey_pack_params_t params = {
        .field_mask = UCT_MD_MKEY_PACK_FIELD_FLAGS,
        .flags      = UCT_MD_MKEY_PACK_FLAG_EXPORT
    };
    uct_md_attr_v2_t *md_attr;
    void *md_data_p, *common_memh_info_p;
    uint16_t *memh_info_size_p;
    char UCS_V_UNUSED buf[128];
    ucs_status_t status;
    ssize_t result;
    ucp_md_index_t md_index;
    unsigned uct_memh_index;
    size_t tl_mkey_size, component_name_size, global_id_size;
    void *tl_mkey_buf, *component_name_buf, *global_id_buf;
    size_t tl_mkey_data_size;

    ucs_log_indent(1);

    p                  = ucp_memh_common_pack(memh, p,
                                              UCP_MEMH_BUFFER_FLAG_EXPORTED,
                                              &memh_info_size_p);
    common_memh_info_p = p;

    /* Address */
    *ucs_serialize_next(&p, uint64_t) = address;

    /* Length */
    *ucs_serialize_next(&p, uint64_t) = length;

    /* UCP uuid */
    *ucs_serialize_next(&p, uint64_t) = context->uuid;

    /* Registration ID */
    *ucs_serialize_next(&p, uint64_t) = memh->reg_id;

    /* Store the size of an exported memh information */
    *memh_info_size_p += UCS_PTR_BYTE_DIFF(common_memh_info_p, p);

    uct_memh_index = 0;
    ucs_for_each_bit(md_index, context->export_md_map[memh->mem_type]) {
        md_attr = &tl_mds[md_index].attr;

        /* Save pointer to the begging of the TL mkey data to fill later by the
         * resulted size of TL mkey data */
        md_data_p                         = p;
        *ucs_serialize_next(&p, uint16_t) = 0;

        /* TL mkey size */
        tl_mkey_size = md_attr->exported_mkey_packed_size;
        ucs_assertv((tl_mkey_size <= UINT8_MAX) && (tl_mkey_size != 0),
                    "tl_mkey_size %zu", tl_mkey_size);
        *ucs_serialize_next(&p, uint8_t) = tl_mkey_size;

        /* TL mkey */
        tl_mkey_buf = ucs_serialize_next_raw(&p, void, tl_mkey_size);
        status      = uct_md_mkey_pack_v2(tl_mds[md_index].md,
                                          uct_memhs[md_index], &params,
                                          tl_mkey_buf);
        if (status != UCS_OK) {
            result = status;
            goto out;
        }

        /* Size of component name */
        component_name_size              = strlen(md_attr->component_name);
        *ucs_serialize_next(&p, uint8_t) = component_name_size;

        /* Component name */
        component_name_buf = ucs_serialize_next_raw(&p, void,
                                                    component_name_size);
        memcpy(component_name_buf, md_attr->component_name,
               component_name_size);

        /* Size of global MD identifier */
        global_id_size                   =
                ucp_memh_global_id_packed_size(md_attr);
        *ucs_serialize_next(&p, uint8_t) = global_id_size;

        /* Global MD identifier */
        global_id_buf = ucs_serialize_next_raw(&p, void, global_id_size);
        memcpy(global_id_buf, md_attr->global_id, global_id_size);

        /* Size of TL mkey data */
        tl_mkey_data_size = UCS_PTR_BYTE_DIFF(md_data_p, p);
        ucs_assertv((tl_mkey_data_size <= UINT16_MAX) &&
                    (tl_mkey_data_size != 0),
                    "tl_mkey_data_size %zu", tl_mkey_data_size);
        *(uint16_t*)md_data_p = tl_mkey_data_size;

        ucs_trace("exported mkey[%d]=%s for md[%d]=%s", uct_memh_index,
                  ucs_str_dump_hex(p, tl_mkey_size, buf, sizeof(buf),
                                   SIZE_MAX),
                  md_index, tl_mds[md_index].rsc.md_name);
        ++uct_memh_index;
    }

    result = UCS_PTR_BYTE_DIFF(buffer, p);

out:
    ucs_log_indent(-1);
    return result;
}

static size_t
ucp_memh_packed_size(ucp_mem_h memh, uint64_t flags, int rkey_compat)
{
    ucp_context_h context = memh->context;

    if (flags & UCP_MEMH_PACK_FLAG_EXPORT) {
        ucs_assert(!rkey_compat);
        return ucp_memh_exported_packed_size(
                context, context->export_md_map[memh->mem_type]);
    }

    if (rkey_compat) {
        return ucp_rkey_packed_size(context, memh->md_map,
                                    UCS_SYS_DEVICE_ID_UNKNOWN, 0);
    }

    ucs_fatal("packing rkey using ucp_memh_pack() is unsupported");
}

static ssize_t ucp_memh_do_pack(ucp_mem_h memh, uint64_t flags,
                                int rkey_compat, void *memh_buffer)
{
    ucp_memory_info_t mem_info;

    if (flags & UCP_MEMH_PACK_FLAG_EXPORT) {
        return ucp_memh_exported_pack(memh, memh_buffer);
    }

    if (rkey_compat) {
        mem_info.type    = memh->mem_type;
        mem_info.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
        return ucp_rkey_pack_memh(memh->context, memh->md_map, memh, &mem_info,
                                  0, NULL, memh_buffer);
    }

    ucs_fatal("packing rkey using ucp_memh_pack() is unsupported");
}

static ucs_status_t
ucp_memh_pack_internal(ucp_mem_h memh, const ucp_memh_pack_params_t *params,
                       int rkey_compat, void **buffer_p, size_t *buffer_size_p)
{
    ucp_context_h context = memh->context;
    ucs_status_t status;
    ssize_t packed_size;
    void *memh_buffer;
    size_t size;
    uint64_t flags;

    ucs_trace("packing memh %p for buffer %p md_map 0x%" PRIx64
              " export_md_map 0x%" PRIx64, memh, ucp_memh_address(memh),
              memh->md_map, context->export_md_map[memh->mem_type]);

    if (ucp_memh_is_zero_length(memh)) {
        /* Dummy memh, return dummy key */
        if (rkey_compat) {
            *buffer_p      = &ucp_memh_rkey_dummy_buffer;
            *buffer_size_p = sizeof(ucp_memh_rkey_dummy_buffer);
        } else {
            *buffer_p      = &ucp_memh_dummy_buffer;
            *buffer_size_p = sizeof(ucp_memh_dummy_buffer);
        }
        return UCS_OK;
    }

    UCP_THREAD_CS_ENTER(&context->mt_lock);

    flags = UCP_PARAM_VALUE(MEMH_PACK, params, flags, FLAGS, 0);
    size  = ucp_memh_packed_size(memh, flags, rkey_compat);

    if ((flags & UCP_MEMH_PACK_FLAG_EXPORT) &&
        (context->export_md_map[memh->mem_type] == 0)) {
        status = UCS_ERR_UNSUPPORTED;
        goto out;
    }

    memh_buffer = ucs_malloc(size, "ucp_memh_buffer");
    if (memh_buffer == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    packed_size = ucp_memh_do_pack(memh, flags, rkey_compat, memh_buffer);
    if (packed_size < 0) {
        status = (ucs_status_t)packed_size;
        goto err_destroy;
    }

    ucs_assertv(packed_size == size, "packed_size=%zd size=%zu", packed_size,
                size);

    *buffer_p      = memh_buffer;
    *buffer_size_p = size;
    status         = UCS_OK;
    goto out;

err_destroy:
    ucs_free(memh_buffer);
out:
    UCP_THREAD_CS_EXIT(&context->mt_lock);
    return status;
}

ucs_status_t
ucp_memh_pack(ucp_mem_h memh, const ucp_memh_pack_params_t *params,
              void **buffer_p, size_t *buffer_size_p)
{
    return ucp_memh_pack_internal(memh, params, 0, buffer_p, buffer_size_p);
}

void ucp_memh_buffer_release(void *buffer,
                             const ucp_memh_buffer_release_params_t *params)
{
    if ((buffer == &ucp_memh_dummy_buffer) ||
        (buffer == &ucp_memh_rkey_dummy_buffer)) {
        /* Dummy key, just return */
        return;
    }
    ucs_free(buffer);
}

ucs_status_t ucp_rkey_pack(ucp_context_h context, ucp_mem_h memh,
                           void **rkey_buffer_p, size_t *size_p)
{
    ucp_memh_pack_params_t params = {0};
    return ucp_memh_pack_internal(memh, &params, 1, rkey_buffer_p, size_p);
}

void ucp_rkey_buffer_release(void *rkey_buffer)
{
    ucp_memh_buffer_release_params_t params = {0};
    ucp_memh_buffer_release(rkey_buffer, &params);
}

static void UCS_F_NOINLINE
ucp_rkey_unpack_lanes_distance(const ucp_ep_config_key_t *ep_config_key,
                               ucs_sys_dev_distance_t *lanes_distance,
                               const void *buffer, const void *buffer_end)
{
    const void *p                 = buffer;
    ucp_sys_dev_map_t sys_dev_map = 0;
    ucs_sys_dev_distance_t distance, distance_by_dev[UCS_SYS_DEVICE_ID_MAX];
    ucs_sys_device_t sys_dev;
    ucp_lane_index_t lane;
    char buf[128];

    /* Unpack lane distances and update distance_by_dev lookup */
    while (p < buffer_end) {
        ucp_rkey_unpack_distance(
                ucs_serialize_next(&p, const ucp_rkey_packed_distance_t),
                &sys_dev, &distance);
        distance_by_dev[sys_dev] = distance;
        sys_dev_map             |= UCS_BIT(sys_dev);
    }

    /* Initialize lane distances according to distance_by_dev */
    for (lane = 0; lane < ep_config_key->num_lanes; ++lane) {
        sys_dev              = ep_config_key->lanes[lane].dst_sys_dev;
        lanes_distance[lane] = (sys_dev_map & UCS_BIT(sys_dev)) ?
                                       distance_by_dev[sys_dev] :
                                       ucs_topo_default_distance;
        ucs_trace("lane[%d] dev %d distance %s", lane, sys_dev,
                  ucs_topo_distance_str(&lanes_distance[lane], buf,
                                        sizeof(buf)));
    }
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rkey_proto_resolve,
                 (rkey, ep, buffer, buffer_end), ucp_rkey_h rkey, ucp_ep_h ep,
                 const void *buffer, const void *buffer_end)
{
    ucp_worker_h worker = ep->worker;
    const void *p       = buffer;
    ucs_sys_dev_distance_t *lanes_distance;
    ucp_rkey_config_key_t rkey_config_key;
    khiter_t khiter;

    /* Avoid calling ucp_ep_resolve_remote_id() from rkey_unpack, and let
     * the APIs which are not yet using new protocols resolve the remote key
     * on-demand.
     */
    rkey->cache.ep_cfg_index = UCP_WORKER_CFG_INDEX_NULL;

    /* Look up remote key's configration */
    rkey_config_key.ep_cfg_index = ep->cfg_index;
    rkey_config_key.md_map       = rkey->md_map;
    rkey_config_key.mem_type     = rkey->mem_type;

    if (buffer < buffer_end) {
        rkey_config_key.sys_dev = *ucs_serialize_next(&p, const uint8_t);
    } else {
        rkey_config_key.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;
    }

    khiter = kh_get(ucp_worker_rkey_config, &worker->rkey_config_hash,
                    rkey_config_key);
    if (ucs_likely(khiter != kh_end(&worker->rkey_config_hash))) {
        /* Found existing configuration in hash */
        rkey->cfg_index = kh_val(&worker->rkey_config_hash, khiter);
        return UCS_OK;
    }

    lanes_distance = ucs_alloca(sizeof(*lanes_distance) * UCP_MAX_LANES);
    ucp_rkey_unpack_lanes_distance(&ucp_ep_config(ep)->key, lanes_distance, p,
                                   buffer_end);
    return ucp_worker_add_rkey_config(worker, &rkey_config_key, lanes_distance,
                                      &rkey->cfg_index);
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_ep_rkey_unpack_internal,
                 (ep, buffer, length, unpack_md_map, rkey_p), ucp_ep_h ep,
                 const void *buffer, size_t length, ucp_md_map_t unpack_md_map,
                 ucp_rkey_h *rkey_p)
{
    ucp_worker_h worker              = ep->worker;
    const ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    const void *p                    = buffer;
    ucp_md_map_t md_map, remote_md_map;
    ucp_rsc_index_t cmpt_index;
    unsigned remote_md_index;
    const void *tl_rkey_buf;
    ucp_tl_rkey_t *tl_rkey;
    size_t tl_rkey_size;
    unsigned rkey_index;
    ucs_status_t status;
    ucp_rkey_h rkey;
    uint8_t flags;
    int md_count;

    UCS_STATIC_ASSERT(ucs_offsetof(ucp_rkey_t, mem_type) ==
                      ucs_offsetof(ucp_rkey_t, cache.mem_type));
    UCS_STATIC_ASSERT(ucs_same_type(ucs_field_type(ucp_rkey_t, mem_type),
                                    ucs_field_type(ucp_rkey_t, cache.mem_type)));

    ucs_trace("ep %p: unpacking rkey buffer %p length %zu", ep, buffer, length);
    ucs_log_indent(1);

    /* MD map for the unpacked rkey */
    remote_md_map = *ucs_serialize_next(&p, const ucp_md_map_t);
    md_map        = remote_md_map & unpack_md_map;
    md_count      = ucs_popcount(md_map);

    /* Allocate rkey handle which holds UCT rkeys for all remote MDs. Small key
     * allocations are done from a memory pool.
     * We keep all of them to handle a future transport switch.
     */
    if (md_count <= worker->context->config.ext.rkey_mpool_max_md) {
        rkey  = ucs_mpool_get_inline(&worker->rkey_mp);
        flags = UCP_RKEY_DESC_FLAG_POOL;
    } else {
        rkey  = ucs_malloc(sizeof(*rkey) + (sizeof(rkey->tl_rkey[0]) * md_count),
                           "ucp_rkey");
        flags = 0;
    }
    if (rkey == NULL) {
        ucs_error("failed to allocate remote key");
        status = UCS_ERR_NO_MEMORY;
        goto out;
    }

    rkey->md_map   = md_map;
    rkey->mem_type = *ucs_serialize_next(&p, const uint8_t);
    rkey->flags    = flags;
#if ENABLE_PARAMS_CHECK
    rkey->ep       = ep;
#endif

    /* Go over remote MD indices and unpack rkey of each UCT MD */
    rkey_index = 0; /* Index of the rkey in the array */
    ucs_for_each_bit(remote_md_index, remote_md_map) {
        tl_rkey_size = *ucs_serialize_next(&p, const uint8_t);
        tl_rkey_buf  = ucs_serialize_next_raw(&p, const void, tl_rkey_size);

        /* Use bit operations to iterate through the indices of the remote MDs
         * as provided in the md_map. md_map always holds a bitmap of MD indices
         * that remain to be used. Every time we find the next valid MD index.
         * If some rkeys cannot be unpacked, we remove them from the local map.
         */
        ucs_assert(UCS_BIT(remote_md_index) & remote_md_map);
        ucs_assert_always(remote_md_index <= UCP_MD_INDEX_BITS);

        /* Unpack only reachable rkeys */
        if (!(UCS_BIT(remote_md_index) & rkey->md_map)) {
            continue;
        }

        ucs_assert(rkey_index < md_count);
        tl_rkey       = &rkey->tl_rkey[rkey_index];
        cmpt_index    = ucp_ep_config_get_dst_md_cmpt(&ep_config->key,
                                                      remote_md_index);
        tl_rkey->cmpt = worker->context->tl_cmpts[cmpt_index].cmpt;

        status = uct_rkey_unpack(tl_rkey->cmpt, tl_rkey_buf, &tl_rkey->rkey);
        if (status == UCS_OK) {
            ucs_trace("rkey[%d] for remote md %d is 0x%lx", rkey_index,
                      remote_md_index, tl_rkey->rkey.rkey);
            ++rkey_index;
        } else if (status == UCS_ERR_UNREACHABLE) {
            rkey->md_map &= ~UCS_BIT(remote_md_index);
            ucs_trace("rkey[%d] for remote md %d is 0x%lx not reachable",
                      rkey_index, remote_md_index, tl_rkey->rkey.rkey);
        } else {
            ucs_error("failed to unpack remote key from remote md[%d]: %s",
                      remote_md_index, ucs_status_string(status));
            goto err_destroy;
        }
    }

    if (worker->context->config.ext.proto_enable) {
        status = ucp_rkey_proto_resolve(rkey, ep, p,
                                        UCS_PTR_BYTE_OFFSET(buffer, length));
        if (status != UCS_OK) {
            goto err_destroy;
        }
    } else {
        ucp_rkey_resolve_inner(rkey, ep);
    }

    ucs_trace("ep %p: unpacked rkey %p md_map 0x%" PRIx64 " type %s", ep, rkey,
              rkey->md_map, ucs_memory_type_names[rkey->mem_type]);
    *rkey_p = rkey;
    status  = UCS_OK;
    goto out;

err_destroy:
    ucp_rkey_destroy(rkey);
out:
    ucs_log_indent(-1);
    return status;
}

ucs_status_t ucp_ep_rkey_unpack(ucp_ep_h ep, const void *rkey_buffer,
                                ucp_rkey_h *rkey_p)
{
    ucs_status_t status;

    UCP_WORKER_THREAD_CS_ENTER_CONDITIONAL(ep->worker);
    status = ucp_ep_rkey_unpack_reachable(ep, rkey_buffer, 0, rkey_p);
    UCP_WORKER_THREAD_CS_EXIT_CONDITIONAL(ep->worker);

    return status;
}

void ucp_rkey_dump_packed(const void *buffer, size_t length,
                          ucs_string_buffer_t *strb)
{
    const void *p          = buffer;
    const void *buffer_end = UCS_PTR_BYTE_OFFSET(buffer, length);
    ucs_sys_dev_distance_t distance;
    ucs_memory_type_t mem_type;
    ucs_sys_device_t sys_dev;
    const void *tl_tkey;
    ucp_md_map_t md_map;
    unsigned md_index;
    uint8_t tl_rkey_size;
    char buf[128];

    md_map   = *ucs_serialize_next(&p, const ucp_md_map_t);
    mem_type = *ucs_serialize_next(&p, const uint8_t);

    ucs_string_buffer_appendf(strb, "{%s", ucs_memory_type_names[mem_type]);

    ucs_for_each_bit(md_index, md_map) {
        tl_rkey_size = *ucs_serialize_next(&p, const uint8_t);
        tl_tkey      = ucs_serialize_next_raw(&p, const void, tl_rkey_size);
        ucs_string_buffer_appendf(strb, ",%u:", md_index);
        ucs_string_buffer_append_hex(strb, tl_tkey, tl_rkey_size, SIZE_MAX);
    }

    if (p < buffer_end) {
        sys_dev = *ucs_serialize_next(&p, const uint8_t);
        ucs_string_buffer_appendf(strb, ",sys:%u", sys_dev);
    }

    while (p < buffer_end) {
        ucp_rkey_unpack_distance(
                ucs_serialize_next(&p, const ucp_rkey_packed_distance_t),
                &sys_dev, &distance);
        ucs_string_buffer_appendf(strb, ",dev:%u:%s", sys_dev,
                                  ucs_topo_distance_str(&distance, buf,
                                                        sizeof(buf)));
    }

    ucs_string_buffer_appendf(strb, "}");
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
    uct_md_attr_v2_t *md_attr;
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
            (!(md_attr->flags & UCT_MD_FLAG_NEED_RKEY))) {
            /* Lane does not need rkey, can use the lane with invalid rkey  */
            if (!rkey || ((md_attr->access_mem_types & UCS_BIT(mem_type)) &&
                          (mem_type == rkey->mem_type))) {
                *uct_rkey_p = UCT_INVALID_RKEY;
                return lane;
            }
        }

        mem_types = md_attr->reg_mem_types | md_attr->alloc_mem_types;
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

    UCS_STATIC_ASSERT(ucs_offsetof(ucp_rkey_t, flags) ==
                      ucs_offsetof(ucp_rkey_t, cache.flags));
    UCS_STATIC_ASSERT(ucs_same_type(ucs_field_type(ucp_rkey_t, flags),
                                    ucs_field_type(ucp_rkey_t, cache.flags)));

    rkey->cache.rma_lane = ucp_rkey_find_rma_lane(context, config,
                                                  UCS_MEMORY_TYPE_HOST,
                                                  config->key.rma_lanes, rkey,
                                                  0, &uct_rkey);
    if (rkey->cache.rma_lane == UCP_NULL_LANE) {
        rkey->cache.rma_proto_index = UCP_RKEY_SW_PROTO;
        rkey->cache.rma_rkey        = UCT_INVALID_RKEY;
        rkey->cache.max_put_short   = 0;
        rma_sw                      = !!(context->config.features & UCP_FEATURE_RMA);
    } else {
        rkey->cache.rma_proto_index = UCP_RKEY_BASIC_PROTO;
        rkey->cache.rma_rkey        = uct_rkey;
        UCS_STATIC_ASSERT(ucs_same_type(ucs_field_type(ucp_rkey_t,
                                        cache.max_put_short), int8_t));
        rkey->cache.max_put_short   =
            ucs_min(config->rma[rkey->cache.rma_lane].max_put_short, INT8_MAX);
    }

    rkey->cache.amo_lane = ucp_rkey_find_rma_lane(context, config,
                                                  UCS_MEMORY_TYPE_HOST,
                                                  config->key.amo_lanes, rkey,
                                                  0, &uct_rkey);
    if (rkey->cache.amo_lane == UCP_NULL_LANE) {
        rkey->cache.amo_proto_index = UCP_RKEY_SW_PROTO;
        rkey->cache.amo_rkey        = UCT_INVALID_RKEY;
        amo_sw                      = !!(context->config.features &
                                         (UCP_FEATURE_AMO32 | UCP_FEATURE_AMO64));
    } else {
        rkey->cache.amo_proto_index = UCP_RKEY_BASIC_PROTO;
        rkey->cache.amo_rkey        = uct_rkey;
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
              UCP_RKEY_RMA_PROTO(rkey->cache.rma_proto_index)->name,
              rkey->cache.rma_lane, rkey->cache.rma_rkey,
              UCP_RKEY_AMO_PROTO(rkey->cache.amo_proto_index)->name,
              rkey->cache.amo_lane, rkey->cache.amo_rkey);
}

void ucp_rkey_config_dump_brief(const ucp_rkey_config_key_t *rkey_config_key,
                                ucs_string_buffer_t *strb)
{
    ucs_string_buffer_appendf(strb, "%s",
                              ucs_memory_type_names[rkey_config_key->mem_type]);
    if (rkey_config_key->sys_dev != UCS_SYS_DEVICE_ID_UNKNOWN) {
        ucs_string_buffer_appendf(strb, "/dev[%d]", rkey_config_key->sys_dev);
    }
}

void ucp_rkey_proto_select_dump(ucp_worker_h worker,
                                ucp_worker_cfg_index_t rkey_cfg_index,
                                ucs_string_buffer_t *strb)
{
    const ucp_rkey_config_t *rkey_config = &worker->rkey_config[rkey_cfg_index];

    ucp_proto_select_dump_short(&rkey_config->put_short, "put_short", strb);
    ucp_proto_select_info(worker, rkey_config->key.ep_cfg_index, rkey_cfg_index,
                          &rkey_config->proto_select, strb);
}
