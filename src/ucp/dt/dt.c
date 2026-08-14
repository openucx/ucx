/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2026. ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2024. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "dt.h"
#include "dt_iov.h"
#include "dt_contig.h"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_mm.inl>
#include <ucs/profile/profile.h>


const char * ucp_datatype_class_names[] = {
    [UCP_DATATYPE_CONTIG]   = "contiguous",
    [UCP_DATATYPE_STRIDED]  = "strided",
    [UCP_DATATYPE_IOV]      = "iov",
    [UCP_DATATYPE_SGL]      = "sgl",
    [UCP_DATATYPE_GENERIC]  = "generic"
};


ucs_status_t ucp_dt_mem_info_check_elem(ucp_context_h context,
                                        const void *buffer, size_t length,
                                        const ucp_memory_info_t *ref,
                                        const char *dt_name, size_t index,
                                        size_t count)
{
    ucp_memory_info_t memory_info;

    ucp_memory_detect(context, buffer, length, &memory_info);
    return ucp_dt_mem_info_verify(dt_name, index, &memory_info, ref, count);
}

ucs_status_t ucp_dt_mem_info_verify(const char *dt_name, size_t index,
                                    const ucp_memory_info_t *cur,
                                    const ucp_memory_info_t *ref,
                                    size_t count)
{
    if (ucp_memory_info_equal(cur, ref)) {
        return UCS_OK;
    }

    ucs_error("inconsistent %s mem_info: [%zu]=%s-%s flags=0x%x "
              "[0]=%s-%s flags=0x%x count=%zu",
              dt_name, index, ucs_memory_type_names[cur->type],
              ucs_topo_sys_device_get_name(cur->sys_dev), cur->flags,
              ucs_memory_type_names[ref->type],
              ucs_topo_sys_device_get_name(ref->sys_dev), ref->flags,
              count);
    return UCS_ERR_INVALID_PARAM;
}


/*
 * Select an RMA lane whose MD accepts the buffer mem_flags, starting the search
 * from the given lane index. Returns UCP_NULL_LANE if no lane is compatible.
 */
static ucp_lane_index_t
ucp_mem_type_ep_lane_by_flags(ucp_ep_h ep, uint8_t mem_flags, unsigned first)
{
    ucp_context_h context          = ep->worker->context;
    const ucp_ep_config_key_t *key = &ucp_ep_config(ep)->key;
    const uct_md_attr_v2_t *md_attr;
    ucp_lane_index_t lane;
    unsigned i;

    for (i = first; key->rma_lanes[i] != UCP_NULL_LANE; ++i) {
        lane    = key->rma_lanes[i];
        md_attr = &context->tl_mds[ucp_ep_md_index(ep, lane)].attr;
        if (ucs_test_all_flags(mem_flags, md_attr->required_mem_flags)) {
            return lane;
        }
    }

    return UCP_NULL_LANE;
}

/* Select the mem-type endpoint lane which can access the buffer and register */
static ucs_status_t
ucp_mem_type_lane_reg(ucp_worker_h worker, ucp_ep_h ep, void *address,
                      size_t length, ucs_memory_type_t mem_type,
                      ucp_lane_index_t *lane_p,
                      ucp_mtype_pack_context_t *pack_context)
{
    ucp_context_h context            = worker->context;
    const ucp_ep_config_key_t *key   = &ucp_ep_config(ep)->key;
    ucp_lane_index_t lane            = key->rma_lanes[0];
    ucp_md_index_t md_index          = ucp_ep_md_index(ep, lane);
    const uct_md_attr_v2_t *md_attr  = &context->tl_mds[md_index].attr;
    ucs_memory_info_t cache_info;
    ucp_memory_info_t mem_info;
    ucs_status_t status;

    /* The preferred MD accepts any memory, no need to know the buffer flags */
    if (md_attr->required_mem_flags == 0) {
        goto out_reg;
    }

    /* The memory flags may already be known, for example set by a memory hook
     * on allocation, use them to pick a compatible lane */
    status = ucs_memtype_cache_lookup(address, length, &cache_info);
    if ((status == UCS_OK) && ucp_memory_info_is_complete(&cache_info)) {
        lane = ucp_mem_type_ep_lane_by_flags(ep, cache_info.mem_flags, 0);
        goto out_reg;
    }

    /* Unknown memory: attempt the preferred MD instead of querying attributes,
     * a memory domain such as gdr_copy fails on non-registrable memory */
    status = ucp_mem_type_reg_buffers(worker, address, length, mem_type,
                                      md_index, UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                      pack_context);
    if (status == UCS_OK) {
        *lane_p = lane;
        return UCS_OK;
    }

    /* Preferred MD rejected the buffer: detect attributes for a fallback lane
     * (also warms the memtype cache). Coverity fnptr model false positive. */
    /* coverity[use_after_free] */
    ucp_memory_detect(context, address, length, &mem_info);
    lane = ucp_mem_type_ep_lane_by_flags(ep, mem_info.flags, 1);

out_reg:
    if (lane == UCP_NULL_LANE) {
        ucs_error("no mem type rma lane can register %s buffer %p length %zu",
                  ucs_memory_type_names[mem_type], address, length);
        return UCS_ERR_UNSUPPORTED;
    }

    md_index = ucp_ep_md_index(ep, lane);
    *lane_p  = lane;
    return ucp_mem_type_reg_buffers(worker, address, length, mem_type, md_index,
                                    0, pack_context);
}

UCS_PROFILE_FUNC_VOID(ucp_mem_type_unpack,
                      (worker, buffer, recv_data, recv_length, mem_type),
                      ucp_worker_h worker, void *buffer, const void *recv_data,
                      size_t recv_length, ucs_memory_type_t mem_type)
{
    ucp_ep_h ep = worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane;
    ucs_status_t status;
    ucp_mtype_pack_context_t pack_context;

    if (recv_length == 0) {
        return;
    }

    status = ucp_mem_type_lane_reg(worker, ep, buffer, recv_length, mem_type,
                                   &lane, &pack_context);
    if (status != UCS_OK) {
        ucs_fatal("failed to register buffer with mem type domain %s",
                  ucs_memory_type_names[mem_type]);
    }

    status = uct_ep_put_short(ucp_ep_get_lane(ep, lane), recv_data, recv_length,
                              (uint64_t)buffer, pack_context.rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_fatal("mem type unpack failed to uct_ep_put_short() %s",
                  ucs_status_string(status));
    }

    ucp_mem_type_unreg_buffers(worker, &pack_context);
}

UCS_PROFILE_FUNC_VOID(ucp_mem_type_pack,
                      (worker, dest, src, length, mem_type),
                      ucp_worker_h worker, void *dest, const void *src,
                      size_t length, ucs_memory_type_t mem_type)
{
    ucp_ep_h ep = worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane;
    ucs_status_t status;
    ucp_mtype_pack_context_t pack_context;

    if (length == 0) {
        return;
    }

    status = ucp_mem_type_lane_reg(worker, ep, (void*)src, length, mem_type,
                                   &lane, &pack_context);
    if (status != UCS_OK) {
        ucs_fatal("failed to register buffer with mem type domain %s",
                  ucs_memory_type_names[mem_type]);
    }

    status = uct_ep_get_short(ucp_ep_get_lane(ep, lane), dest, length,
                              (uint64_t)src, pack_context.rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_fatal("mem type pack failed to uct_ep_get_short() %s",
                  ucs_status_string(status));
    }

    ucp_mem_type_unreg_buffers(worker, &pack_context);
}

size_t ucp_dt_pack(ucp_worker_h worker, ucp_datatype_t datatype,
                   ucs_memory_type_t mem_type, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length)
{
    size_t result_len = 0;
    ucp_dt_generic_t *dt;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        ucp_dt_contig_pack(worker, dest,
                           UCS_PTR_BYTE_OFFSET(src, state->offset),
                           length, mem_type, length);
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, worker, dest, src, length,
                              &state->dt.iov.iov_offset,
                              &state->dt.iov.iovcnt_offset, mem_type, length);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt         = ucp_dt_to_generic(datatype);
        result_len = UCS_PROFILE_NAMED_CALL("dt_pack", dt->ops.pack,
                                            state->dt.generic.state,
                                            state->offset, dest, length);
        break;

    default:
        ucs_error("Invalid data type");
    }

    state->offset += result_len;
    return result_len;
}

ucs_status_t ucp_dt_query(ucp_datatype_t datatype, ucp_datatype_attr_t *attr)
{
    ucp_dt_generic_t *dt_gen;
    void *state_gen;
    size_t count;

    /* Currently, the only datatype attribute to query is the packed size. */
    if (!(attr->field_mask & UCP_DATATYPE_ATTR_FIELD_PACKED_SIZE)) {
        return UCS_OK;
    }

    count = UCP_ATTR_VALUE(DATATYPE, attr, count, COUNT, 1);

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        attr->packed_size = ucp_contig_dt_elem_size(datatype) * count;
        return UCS_OK;
    case UCP_DATATYPE_IOV:
        if (!(attr->field_mask & UCP_DATATYPE_ATTR_FIELD_BUFFER) ||
            (attr->buffer == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }

        attr->packed_size = ucp_dt_iov_length(attr->buffer, count);
        return UCS_OK;
    case UCP_DATATYPE_GENERIC:
        if (!(attr->field_mask & UCP_DATATYPE_ATTR_FIELD_BUFFER) ||
            (attr->buffer == NULL)) {
            return UCS_ERR_INVALID_PARAM;
        }

        dt_gen = ucp_dt_to_generic(datatype);
        if (dt_gen == NULL) {
            return UCS_ERR_INVALID_PARAM;
        }

        state_gen = dt_gen->ops.start_pack(dt_gen->context, attr->buffer,
                                           count);
        attr->packed_size = dt_gen->ops.packed_size(state_gen);
        dt_gen->ops.finish(state_gen);
        return UCS_OK;
    default:
        return UCS_ERR_INVALID_PARAM;
    }
}
