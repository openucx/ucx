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


static struct {
    uint8_t  size_part1;
    uint16_t size_part2;
} UCS_S_PACKED ucp_memh_dummy_buffer = { 0, 0 };


const ucp_amo_proto_t *ucp_amo_proto_list[] = {
    [UCP_RKEY_BASIC_PROTO] = &ucp_amo_basic_proto,
    [UCP_RKEY_SW_PROTO]    = &ucp_amo_sw_proto
};

const ucp_rma_proto_t *ucp_rma_proto_list[] = {
    [UCP_RKEY_BASIC_PROTO] = &ucp_rma_basic_proto,
    [UCP_RKEY_SW_PROTO]    = &ucp_rma_sw_proto
};


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
        ucs_assertv_always(tl_rkey_size <= UINT8_MAX, "md %s: tl_rkey_size=%zu",
                           context->tl_mds[md_index].rsc.md_name, tl_rkey_size);
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
        ucs_assertv_always(tl_rkey_size <= UINT8_MAX, "md %s: tl_rkey_size=%zu",
                           context->tl_mds[md_index].rsc.md_name, tl_rkey_size);
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

UCS_PROFILE_FUNC(ssize_t, ucp_rkey_pack_memh,
                 (context, md_map, memh, address, length, mem_info, sys_dev_map,
                  sys_distance, uct_flags, buffer),
                 ucp_context_h context, ucp_md_map_t md_map,
                 const ucp_mem_h memh, void *address, size_t length,
                 const ucp_memory_info_t *mem_info,
                 ucp_sys_dev_map_t sys_dev_map,
                 const ucs_sys_dev_distance_t *sys_distance, unsigned uct_flags,
                 void *buffer)
{
    void *p = buffer;
    uct_md_mkey_pack_params_t params;
    unsigned md_index;
    char UCS_V_UNUSED buf[128];
    ucs_sys_device_t sys_dev;
    size_t tl_rkey_size;
    ucs_status_t status;
    void *tl_rkey_buf;
    ssize_t result;

    /* Check that md_map is valid */
    ucs_assert(ucs_test_all_flags(UCS_MASK(context->num_mds), md_map));

    ucs_trace("packing rkey type %s md_map 0x%" PRIx64 " dev_map 0x%" PRIx64,
              ucs_memory_type_names[mem_info->type], md_map, sys_dev_map);
    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE, 1);

    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_LAST <= 255);
    *ucs_serialize_next(&p, ucp_md_map_t) = md_map;
    *ucs_serialize_next(&p, uint8_t)      = mem_info->type;

    params.field_mask = UCT_MD_MKEY_PACK_FIELD_FLAGS;
    /* Write both size and rkey_buffer for each UCT rkey */
    ucs_for_each_bit (md_index, md_map) {
        tl_rkey_size = context->tl_mds[md_index].attr.rkey_packed_size;
        *ucs_serialize_next(&p, uint8_t) = tl_rkey_size;

        tl_rkey_buf  = ucs_serialize_next_raw(&p, void, tl_rkey_size);
        params.flags = context->tl_mds[md_index].pack_flags_mask & uct_flags;

        status = uct_md_mkey_pack_v2(context->tl_mds[md_index].md,
                                     memh->uct[md_index], address, length,
                                     &params, tl_rkey_buf);
        if (status != UCS_OK) {
            result = status;
            goto out;
        }

        ucs_trace("rkey %s for md[%d]=%s",
                  ucs_str_dump_hex(p, tl_rkey_size, buf, sizeof(buf), SIZE_MAX),
                  md_index, context->tl_mds[md_index].rsc.md_name);
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
    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE, -1);
    return result;
}


static UCS_F_ALWAYS_INLINE ucp_md_map_t ucp_memh_export_md_map(ucp_mem_h memh)
{
    return memh->context->export_md_map & memh->md_map;
}

static size_t ucp_memh_extended_info_size(size_t size)
{
    if ((size - sizeof(uint8_t)) > UINT8_MAX) {
        /* This field is packed after the 8-bit size field in case of a total
         * info size is greater than UINT8_MAX. */
        return sizeof(uint16_t);
    }

    return 0;
}

static size_t ucp_memh_common_packed_size(size_t specific_info_size)
{
    size_t size = specific_info_size +
            sizeof(uint8_t) /* Size of mkey information */ +
            sizeof(uint16_t) /* Flags */ +
            sizeof(uint64_t) /* Memory domains map */ +
            sizeof(uint8_t) /* Memory type */;

    return size + ucp_memh_extended_info_size(size);
}

static size_t ucp_memh_exported_info_packed_size()
{
    size_t size =
            sizeof(uint64_t) /* Address */ +
            sizeof(uint64_t) /* Length */ +
            sizeof(uint64_t) /* UCP uuid */ +
            sizeof(uint64_t) /* Registration ID */;

    /* Common data (size, md_map, mem_type and etc) */
    return ucp_memh_common_packed_size(size);
}

static void ucp_memh_info_size_pack(void **p, size_t info_size)
{
    size_t packed_info_size;

    ucs_assert(info_size <= UINT16_MAX);

    packed_info_size = info_size - sizeof(uint8_t);
    if (packed_info_size > UINT8_MAX) {
        packed_info_size -= sizeof(uint16_t);

        *ucs_serialize_next(p, uint8_t)  = 0;
        *ucs_serialize_next(p, uint16_t) = packed_info_size;
    } else {
        *ucs_serialize_next(p, uint8_t) = packed_info_size;
    }
}

static uint16_t ucp_memh_info_size_unpack(const void **p)
{
    uint16_t info_size = *ucs_serialize_next(p, uint8_t);

    if (info_size == 0) {
        info_size = *ucs_serialize_next(p, uint16_t) + sizeof(uint16_t);
    }

    return info_size + sizeof(uint8_t);
}

static void ucp_memh_common_pack(const ucp_mem_h memh, void **p,
                                 uint64_t flags, size_t memh_info_size)
{
    ucp_context_h UCS_V_UNUSED context = memh->context;

    /* Check that md_map is valid */
    ucs_assertv(ucs_test_all_flags(UCS_MASK(context->num_mds), memh->md_map),
                "mask 0x%lx memh %p md_map 0x%lx", UCS_MASK(context->num_mds),
                memh, memh->md_map);
    UCS_STATIC_ASSERT(UCS_MEMORY_TYPE_LAST <= 255);

    ucp_memh_info_size_pack(p, memh_info_size);

    *ucs_serialize_next(p, uint16_t) = flags;
    *ucs_serialize_next(p, uint64_t) = ucp_memh_export_md_map(memh);
    *ucs_serialize_next(p, uint8_t)  = memh->mem_type;
}

static size_t ucp_memh_global_id_packed_size(const uct_md_attr_v2_t *md_attr)
{
    size_t size = UCT_MD_GLOBAL_ID_MAX;

    while ((size != 0) && (md_attr->global_id[size - 1] == '\0')) {
        --size;
    }

    ucs_assertv(size < UINT8_MAX, "size %zu", size);
    return size;
}

static size_t
ucp_memh_tl_mkey_common_packed_size(size_t specific_info_size)
{
    size_t size = specific_info_size;

    /* Size of packed TL mkey data */
    size += sizeof(uint8_t);

    /* TL mkey size */
    size += sizeof(uint8_t);

    return size + ucp_memh_extended_info_size(size);
}

static size_t
ucp_memh_exported_tl_mkey_packed_size(ucp_context_h context,
                                      ucp_md_index_t md_index)
{
    uct_md_attr_v2_t *md_attr = &context->tl_mds[md_index].attr;
    size_t size;

    /* Size of exported TL mkey packed size */
    size = md_attr->exported_mkey_packed_size;

    /* Size of global MD identifier */
    size += sizeof(uint8_t);

    /* Global MD identifier */
    size += ucp_memh_global_id_packed_size(md_attr);

    return ucp_memh_tl_mkey_common_packed_size(size);
}

static size_t
ucp_memh_exported_packed_size(ucp_context_h context, ucp_md_map_t md_map)
{
    size_t size = ucp_memh_exported_info_packed_size();
    ucp_md_index_t md_index;

    ucs_for_each_bit(md_index, md_map) {
        size += ucp_memh_exported_tl_mkey_packed_size(context, md_index);
    }

    return size;
}

/**
 *  memh_info_size_part1    :  8
 * [memh_info_size_part2    : 16  - packed only if memh_info_size > 255,
 *                                  memh_info_size_part1 is set to 0]
 *  flags                   : 16
 *  md_map                  : 64
 *  mem_type                :  8
 *
 * exported_memh :
 * [
 *  address          : 64
 *  length           : 64
 *  ucp_context_uuid : 64
 *  registration_id  : 64
 * ]
 *
 * ucs_for_each_bit(md_map) {
 *     tl_mkey_data_size_part1 :  8
 *    [tl_mkey_data_size_part2 : 16 - packed only if tl_mkey_data_size > 255,
 *                                    tl_mkey_data_size_part1 is set to 0]
 *     tl_mkey_size            :  8
 *     tl_mkey_packed          : <size>
 *
 *     exported_memh :
 *     [
 *      global_md_id_size   :  8
 *      global_md_id        : <size>
 *     ]
 * }
 */
static ssize_t
ucp_memh_exported_pack(const ucp_mem_h memh, void *buffer)
{
    ucp_context_h context            = memh->context;
    void* address                    = ucp_memh_address(memh);
    uint64_t length                  = ucp_memh_length(memh);
    void *p                          = buffer;
    ucp_tl_md_t *tl_mds              = context->tl_mds;
    uct_md_mkey_pack_params_t params = {
        .field_mask = UCT_MD_MKEY_PACK_FIELD_FLAGS,
        .flags      = UCT_MD_MKEY_PACK_FLAG_EXPORT
    };
    size_t memh_info_size            = ucp_memh_exported_info_packed_size();
    ucp_md_map_t export_md_map       = ucp_memh_export_md_map(memh);
    size_t UCS_V_UNUSED memh_info_packed_size;
    const uct_md_attr_v2_t *md_attr;
    char UCS_V_UNUSED buf[128];
    ucs_status_t status;
    ssize_t result;
    ucp_md_index_t md_index;
    size_t tl_mkey_size, global_id_size;
    size_t tl_mkey_data_size;

    ucs_log_indent(1);

    ucp_memh_common_pack(memh, &p, UCP_MEMH_BUFFER_FLAG_EXPORTED,
                         memh_info_size);

    *ucs_serialize_next(&p, uint64_t) = (uint64_t)address;
    *ucs_serialize_next(&p, uint64_t) = length;
    *ucs_serialize_next(&p, uint64_t) = context->uuid;
    *ucs_serialize_next(&p, uint64_t) = memh->reg_id;

    memh_info_packed_size = UCS_PTR_BYTE_DIFF(buffer, p);
    ucs_assertv(memh_info_size == memh_info_packed_size,
                "memh_info_size %zu memh_info_packed_size %zu",
                memh_info_size, memh_info_packed_size);

    ucs_for_each_bit(md_index, export_md_map) {
        md_attr           = &tl_mds[md_index].attr;
        global_id_size    = ucp_memh_global_id_packed_size(md_attr);
        tl_mkey_data_size = ucp_memh_exported_tl_mkey_packed_size(context,
                                                                  md_index);
        ucp_memh_info_size_pack(&p, tl_mkey_data_size);

        tl_mkey_size = md_attr->exported_mkey_packed_size;
        ucs_assertv((tl_mkey_size <= UINT8_MAX) && (tl_mkey_size != 0),
                    "tl_mkey_size %zu", tl_mkey_size);
        *ucs_serialize_next(&p, uint8_t) = tl_mkey_size;

        status = uct_md_mkey_pack_v2(
                tl_mds[md_index].md, memh->uct[md_index], address, length,
                &params, ucs_serialize_next_raw(&p, void, tl_mkey_size));
        if (status != UCS_OK) {
            result = status;
            goto out;
        }

        *ucs_serialize_next(&p, uint8_t) = global_id_size;
        memcpy(ucs_serialize_next_raw(&p, void, global_id_size),
               md_attr->global_id, global_id_size);

        ucs_trace("exported mkey[%d]=%s for md[%d]=%s",
                  ucs_bitmap2idx(export_md_map, md_index),
                  ucs_str_dump_hex(p, tl_mkey_size, buf, sizeof(buf),
                                   SIZE_MAX),
                  md_index, tl_mds[md_index].rsc.md_name);
    }

    result = UCS_PTR_BYTE_DIFF(buffer, p);

out:
    ucs_log_indent(-1);
    return result;
}

static ucp_md_map_t ucp_rkey_find_global_id_md_map(ucp_context_h context,
                                                   const void *md_global_id,
                                                   size_t global_id_size)
{
    ucp_md_map_t md_map = 0;
    ucp_md_index_t md_index;
    uct_md_attr_v2_t *md_attr;

    for (md_index = 0; md_index < context->num_mds; md_index++) {
        md_attr = &context->tl_mds[md_index].attr;
        if ((ucp_memh_global_id_packed_size(md_attr) == global_id_size) &&
            (memcmp(md_attr->global_id, md_global_id, global_id_size) == 0)) {
            md_map |= UCS_BIT(md_index);
        }
    }

    return md_map;
}

static void
ucp_memh_exported_tl_mkey_data_unpack(ucp_context_h context, 
                                      const void **start_p,
                                      const void **tl_mkey_buf_p,
                                      ucp_md_map_t *md_map_p)
{
    const void *p = *start_p;
    const void *next_tl_md_p;
    size_t tl_mkey_data_size;
    size_t tl_mkey_size, global_id_size;
    const void *tl_mkey_buf;
    ucp_md_map_t md_map;

    ucs_assert(p != NULL);

    tl_mkey_data_size = ucp_memh_info_size_unpack(&p);
    ucs_assert(tl_mkey_data_size != 0);

    tl_mkey_size = *ucs_serialize_next(&p, uint8_t);
    ucs_assert(tl_mkey_size != 0);

    tl_mkey_buf = ucs_serialize_next_raw(&p, void, tl_mkey_size);

    global_id_size = *ucs_serialize_next(&p, uint8_t);
    ucs_assert(global_id_size != 0);

    /* Get local MD indices which corresponds to the remote ones */
    md_map = ucp_rkey_find_global_id_md_map(
            context, ucs_serialize_next_raw(&p, void, global_id_size),
            global_id_size);

    next_tl_md_p = UCS_PTR_BYTE_OFFSET(*start_p, tl_mkey_data_size);
    ucs_assertv(p <= next_tl_md_p, "p=%p, next_tl_md_p=%p", p, next_tl_md_p);

    *start_p       = next_tl_md_p;
    *tl_mkey_buf_p = tl_mkey_buf;
    *md_map_p      = md_map;
}

ucs_status_t
ucp_memh_exported_unpack(ucp_context_h context, const void *export_mkey_buffer,
                         ucp_unpacked_exported_memh_t *unpacked)
{
    const void *p = export_mkey_buffer;
    uint16_t memh_info_size;
    uint16_t UCS_V_UNUSED mem_info_parsed_size;
    ucp_md_index_t remote_md_index;
    ucp_unpacked_exported_tl_mkey_t *tl_mkey;

    ucs_assert(p != NULL);

    /* Common memory handle information */
    memh_info_size = ucp_memh_info_size_unpack(&p);
    ucs_assert(memh_info_size != 0);

    unpacked->flags         = *ucs_serialize_next(&p, uint16_t);
    unpacked->remote_md_map = *ucs_serialize_next(&p, uint64_t);
    unpacked->mem_type      = *ucs_serialize_next(&p, uint8_t);

    if (ucs_unlikely(!(unpacked->flags & UCP_MEMH_BUFFER_FLAG_EXPORTED))) {
        ucs_error("passed memory handle buffer which does not contain exported"
                  " memory handle: flags 0x%" PRIx16, unpacked->flags);
        return UCS_ERR_INVALID_PARAM;
    }

    /* Exported memory handle stuff */
    unpacked->address     = (void*)*ucs_serialize_next(&p, uint64_t);
    unpacked->length      = *ucs_serialize_next(&p, uint64_t);
    unpacked->remote_uuid = *ucs_serialize_next(&p, uint64_t);
    unpacked->reg_id      = *ucs_serialize_next(&p, uint64_t);

    ucs_assert(unpacked->length != 0);

    mem_info_parsed_size = UCS_PTR_BYTE_DIFF(export_mkey_buffer, p);
    ucs_assertv(mem_info_parsed_size <= memh_info_size,
                "mem_info: parsed_size %" PRIu16 " memh_info_size %" PRIu16,
                mem_info_parsed_size, memh_info_size);
    p = UCS_PTR_BYTE_OFFSET(export_mkey_buffer, memh_info_size);

    unpacked->num_tl_mkeys = 0;
    ucs_for_each_bit(remote_md_index, unpacked->remote_md_map) {
        ucs_assertv(unpacked->num_tl_mkeys < UCP_MAX_MDS, "num_tl_mkeys=%u"
                    " UCP_MAX_MDS=%u", unpacked->num_tl_mkeys, UCP_MAX_MDS);
        tl_mkey = &unpacked->tl_mkeys[unpacked->num_tl_mkeys++];
        ucp_memh_exported_tl_mkey_data_unpack(context, &p, &tl_mkey->tl_mkey_buf,
                                              &tl_mkey->local_md_map);
    }

    if (unpacked->num_tl_mkeys == 0) {
        ucs_diag("couldn't find local MDs which correspond to remote MDs");
        return UCS_ERR_UNREACHABLE;
    }

    return UCS_OK;
}

static size_t
ucp_memh_packed_size(ucp_mem_h memh, uint64_t flags, int rkey_compat)
{
    ucp_context_h context = memh->context;

    if (flags & UCP_MEMH_PACK_FLAG_EXPORT) {
        ucs_assert(!rkey_compat);
        return ucp_memh_exported_packed_size(context,
                                             ucp_memh_export_md_map(memh));
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
        return ucp_rkey_pack_memh(memh->context, memh->md_map, memh,
                                  ucp_memh_address(memh), ucp_memh_length(memh),
                                  &mem_info, 0, NULL, 0, memh_buffer);
    }

    ucs_fatal("packing rkey using ucp_memh_pack() is unsupported");
}

int ucp_memh_buffer_is_dummy(const void *exported_memh_buffer)
{
    return memcmp(exported_memh_buffer, &ucp_memh_dummy_buffer,
                  sizeof(ucp_memh_dummy_buffer)) == 0;
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

    flags = UCP_PARAM_VALUE(MEMH_PACK, params, flags, FLAGS, 0);

    ucs_trace("packing %smemh %p for buffer %p md_map 0x%" PRIx64
              " export_md_map 0x%" PRIx64,
              (flags & UCP_MEMH_PACK_FLAG_EXPORT) ? "exported " : "", memh,
              ucp_memh_address(memh), memh->md_map, context->export_md_map);

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

    size = ucp_memh_packed_size(memh, flags, rkey_compat);

    if ((flags & UCP_MEMH_PACK_FLAG_EXPORT) &&
        (ucp_memh_export_md_map(memh) == 0)) {
        ucs_diag("packing memory handle as exported was requested, but"
                 " no UCT MDs which support exported memory keys");
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
                 (rkey, ep, buffer, buffer_end, unreachable_md_map),
                 ucp_rkey_h rkey, ucp_ep_h ep, const void *buffer,
                 const void *buffer_end, ucp_md_map_t unreachable_md_map)
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

    /* Look up remote key's configuration */
    rkey_config_key.ep_cfg_index       = ep->cfg_index;
    rkey_config_key.md_map             = rkey->md_map;
    rkey_config_key.mem_type           = rkey->mem_type;
    rkey_config_key.unreachable_md_map = unreachable_md_map;

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
                 (ep, buffer, length, unpack_md_map, skip_md_map, rkey_p),
                 ucp_ep_h ep, const void *buffer, size_t length,
                 ucp_md_map_t unpack_md_map, ucp_md_map_t skip_md_map,
                 ucp_rkey_h *rkey_p)
{
    ucp_worker_h worker              = ep->worker;
    const ucp_ep_config_t *ep_config = ucp_ep_config(ep);
    const void *p                    = buffer;
    ucp_md_map_t md_map, remote_md_map, unreachable_md_map;
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
    remote_md_map      = *ucs_serialize_next(&p, const ucp_md_map_t);
    md_map             = remote_md_map & unpack_md_map;
    md_count           = ucs_popcount(md_map);
    unreachable_md_map = 0;

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
        ucs_assert_always(remote_md_index <= UCP_MAX_MDS);

        /* Unpack only reachable rkeys */
        if (!(UCS_BIT(remote_md_index) & rkey->md_map)) {
            continue;
        }

        ucs_assert(rkey_index < md_count);
        tl_rkey = &rkey->tl_rkey[rkey_index];

        if (UCS_BIT(remote_md_index) & skip_md_map) {
            tl_rkey->rkey.rkey   = UCT_INVALID_RKEY;
            tl_rkey->rkey.handle = NULL;
            tl_rkey->cmpt        = NULL;
            ucs_trace("rkey[%d] for remote md %d is not unpacked",
                      rkey_index, remote_md_index);
            ++rkey_index;
            continue;
        }

        cmpt_index    = ucp_ep_config_get_dst_md_cmpt(&ep_config->key,
                                                      remote_md_index);
        tl_rkey->cmpt = worker->context->tl_cmpts[cmpt_index].cmpt;

        status = uct_rkey_unpack(tl_rkey->cmpt, tl_rkey_buf, &tl_rkey->rkey);
        if (status == UCS_OK) {
            ucs_trace("rkey[%d] for remote md %d is 0x%lx", rkey_index,
                      remote_md_index, tl_rkey->rkey.rkey);
            ++rkey_index;
        } else if (status == UCS_ERR_UNREACHABLE) {
            rkey->md_map       &= ~UCS_BIT(remote_md_index);
            unreachable_md_map |= UCS_BIT(remote_md_index);
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
                                        UCS_PTR_BYTE_OFFSET(buffer, length),
                                        unreachable_md_map);
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
        if (rkey->tl_rkey[rkey_index].rkey.rkey != UCT_INVALID_RKEY) {
            uct_rkey_release(rkey->tl_rkey[rkey_index].cmpt,
                             &rkey->tl_rkey[rkey_index].rkey);
        }
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

        /* Should not use md_attr->reg_mem_types with protov2 */
        ucs_assert(!context->config.ext.proto_enable);

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
                          &rkey_config->proto_select, 0, strb);
}

ucs_status_t
ucp_rkey_compare(ucp_worker_h worker, ucp_rkey_h rkey1, ucp_rkey_h rkey2,
                 const ucp_rkey_compare_params_t *params, int *result)
{
    ucs_status_t status;
    uct_rkey_compare_params_t uct_params;
    uct_component_h cmpt;
    ucp_md_index_t remote_md_index;
    unsigned rkey_index;
    uct_rkey_t uct_rkey1, uct_rkey2;
    int diff;

    if ((params->field_mask != 0) || (result == NULL)) {
        ucs_error("invalid field_mask 0x%" PRIu64 " or null result passed",
                  params->field_mask);
        return UCS_ERR_INVALID_PARAM;
    }

    /* Matching config indices means that the possibly unrelated remote MDs all
     * resolve to the same local components.
     */
    diff = worker->context->config.ext.proto_enable ?
                   (int)rkey1->cfg_index - (int)rkey2->cfg_index :
                   (int)rkey1->cache.ep_cfg_index -
                           (int)rkey2->cache.ep_cfg_index;
    if (diff != 0) {
        *result = (diff > 0) ? 1 : -1;
        return UCS_OK;
    }

    if (rkey1->md_map != rkey2->md_map) {
        *result = (rkey1->md_map > rkey2->md_map) ? 1 : -1;
        return UCS_OK;
    }

    *result    = 0;
    rkey_index = 0;
    status     = UCS_OK;
    ucs_for_each_bit(remote_md_index, rkey1->md_map) {
        cmpt      = rkey1->tl_rkey[rkey_index].cmpt;
        uct_rkey1 = rkey1->tl_rkey[rkey_index].rkey.rkey;
        uct_rkey2 = rkey2->tl_rkey[rkey_index].rkey.rkey;

        ucs_assert(cmpt == rkey2->tl_rkey[rkey_index].cmpt);

        uct_params.field_mask = 0;
        status = uct_rkey_compare(cmpt, uct_rkey1, uct_rkey2, &uct_params,
                                  result);
        if ((status != UCS_OK) || (*result != 0)) {
            break;
        }

        rkey_index++;
    }

    return status;
}
