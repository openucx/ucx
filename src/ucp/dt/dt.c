/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2017. ALL RIGHTS RESERVED.
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
    [UCP_DATATYPE_GENERIC]  = "generic"
};


UCS_PROFILE_FUNC_VOID(ucp_mem_type_unpack,
                      (worker, buffer, recv_data, recv_length, mem_type),
                      ucp_worker_h worker, void *buffer, const void *recv_data,
                      size_t recv_length, ucs_memory_type_t mem_type)
{
    ucp_ep_h ep = worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane;
    unsigned md_index;
    ucp_mem_h memh;
    ucs_status_t status;
    uct_rkey_bundle_t rkey_bundle;

    if (recv_length == 0) {
        return;
    }

    memh     = ucs_alloca(ucp_memh_size(worker->context));
    lane     = ucp_ep_config(ep)->key.rma_lanes[0];
    md_index = ucp_ep_md_index(ep, lane);

    status = ucp_mem_type_reg_buffers(worker, buffer, recv_length, mem_type,
                                      md_index, memh, &rkey_bundle);
    if (status != UCS_OK) {
        ucs_fatal("failed to register buffer with mem type domain %s",
                  ucs_memory_type_names[mem_type]);
    }

    status = uct_ep_put_short(ucp_ep_get_lane(ep, lane), recv_data, recv_length,
                              (uint64_t)buffer, rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_fatal("mem type unpack failed to uct_ep_put_short() %s",
                  ucs_status_string(status));
    }

    ucp_mem_type_unreg_buffers(worker, md_index, memh, &rkey_bundle);
}

UCS_PROFILE_FUNC_VOID(ucp_mem_type_pack,
                      (worker, dest, src, length, mem_type),
                      ucp_worker_h worker, void *dest, const void *src,
                      size_t length, ucs_memory_type_t mem_type)
{
    ucp_ep_h ep = worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane;
    ucp_md_index_t md_index;
    ucs_status_t status;
    ucp_mem_h memh;
    uct_rkey_bundle_t rkey_bundle;

    if (length == 0) {
        return;
    }

    memh     = ucs_alloca(ucp_memh_size(worker->context));
    lane     = ucp_ep_config(ep)->key.rma_lanes[0];
    md_index = ucp_ep_md_index(ep, lane);

    status = ucp_mem_type_reg_buffers(worker, (void *)src, length, mem_type,
                                      md_index, memh, &rkey_bundle);
    if (status != UCS_OK) {
        ucs_fatal("failed to register buffer with mem type domain %s",
                  ucs_memory_type_names[mem_type]);
    }

    status = uct_ep_get_short(ucp_ep_get_lane(ep, lane), dest, length,
                              (uint64_t)src, rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_fatal("mem type pack failed to uct_ep_get_short() %s",
                  ucs_status_string(status));
    }

    ucp_mem_type_unreg_buffers(worker, md_index, memh, &rkey_bundle);
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
