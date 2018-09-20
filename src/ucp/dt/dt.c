/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "dt.h"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.h>
#include <ucp/core/ucp_mm.h>
#include <ucs/profile/profile.h>


UCS_PROFILE_FUNC(ucs_status_t, ucp_mem_type_unpack,
                 (worker, buffer, recv_data, recv_length, mem_type),
                 ucp_worker_h worker, void *buffer, const void *recv_data,
                 size_t recv_length, uct_memory_type_t mem_type)
{
    ucp_ep_h ep         = worker->mem_type_ep[mem_type];
    ucp_md_map_t md_map = 0;
    ucp_lane_index_t lane;
    unsigned md_index;
    uct_mem_h memh[1];
    ucs_status_t status;
    uct_rkey_bundle_t rkey_bundle;

    if (recv_length == 0) {
        return UCS_OK;
    }

    lane     = ucp_ep_config(ep)->key.rma_lanes[0];
    md_index = ucp_ep_md_index(ep, lane);

    status = ucp_mem_type_reg_buffers(worker, buffer, recv_length,
                                      mem_type, md_index, memh, &md_map,
                                      &rkey_bundle);
    if (status != UCS_OK) {
        ucs_error("failed to register buffer with mem type domian");
        return status;
    }

    status = uct_ep_put_short(ep->uct_eps[lane], recv_data, recv_length,
                              (uint64_t)buffer, rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_error("uct_ep_put_short() failed %s", ucs_status_string(status));
    }

    ucp_mem_type_unreg_buffers(worker, mem_type, memh,
                               &md_map, &rkey_bundle);
    return status;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_mem_type_pack,
                 (worker, dest, src, length, mem_type),
                 ucp_worker_h worker, void *dest, const void *src, size_t length,
                 uct_memory_type_t mem_type)
{
    ucp_ep_h ep         = worker->mem_type_ep[mem_type];
    ucp_md_map_t md_map = 0;
    ucp_lane_index_t lane;
    unsigned md_index;
    ucs_status_t status;
    uct_mem_h memh[1];
    uct_rkey_bundle_t rkey_bundle;

    if (length == 0) {
        return UCS_OK;
    }

    lane     = ucp_ep_config(ep)->key.rma_lanes[0];
    md_index = ucp_ep_md_index(ep, lane);

    status = ucp_mem_type_reg_buffers(worker, (void *)src, length, mem_type, md_index,
                                      memh, &md_map, &rkey_bundle);
    if (status != UCS_OK) {
        ucs_error("failed to register buffer with mem type domian");
        return status;
    }

    status = uct_ep_get_short(ep->uct_eps[lane], dest, length,
                              (uint64_t)src, rkey_bundle.rkey);
    if (status != UCS_OK) {
        ucs_error("uct_ep_put_short() failed %s", ucs_status_string(status));
    }

    ucp_mem_type_unreg_buffers(worker, mem_type, memh,
                               &md_map, &rkey_bundle);
    return status;
}

size_t ucp_dt_pack(ucp_worker_h worker, ucp_datatype_t datatype,
                   uct_memory_type_t mem_type, void *dest, const void *src,
                   ucp_dt_state_t *state, size_t length)
{
    size_t result_len = 0;
    ucp_dt_generic_t *dt;

    if (!length) {
        return length;
    }

    switch (datatype & UCP_DATATYPE_CLASS_MASK) {
    case UCP_DATATYPE_CONTIG:
        if ((ucs_likely(UCP_MEM_IS_HOST(mem_type))) ||
            (ucs_likely(UCP_MEM_IS_CUDA_MANAGED(mem_type)))) {
            UCS_PROFILE_CALL(memcpy, dest, src + state->offset, length);
        } else {
            ucp_mem_type_pack(worker, dest, src + state->offset, length, mem_type);
        }
        result_len = length;
        break;

    case UCP_DATATYPE_IOV:
        UCS_PROFILE_CALL_VOID(ucp_dt_iov_gather, dest, src, length,
                              &state->dt.iov.iov_offset,
                              &state->dt.iov.iovcnt_offset);
        result_len = length;
        break;

    case UCP_DATATYPE_GENERIC:
        dt = ucp_dt_generic(datatype);
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
