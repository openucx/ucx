/*
 * Copyright (C) 2021, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCP_RNDV_MTYPE_INL_
#define UCP_RNDV_MTYPE_INL_

#include "proto_rndv.inl"
#include "rndv.h"

#include <ucp/core/ucp_worker.h>


static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_mtype_init(const ucp_proto_init_params_t *init_params,
                          ucp_md_map_t *mdesc_md_map_p, size_t *frag_size_p)
{
    ucp_worker_h worker        = init_params->worker;
    ucs_memory_type_t mem_type = init_params->select_param->mem_type;
    ucs_status_t status;

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (worker->mem_type_ep[mem_type] == NULL)) {
        return UCS_ERR_UNSUPPORTED;
    }

    if ((init_params->select_param->op_id != UCP_OP_ID_RNDV_SEND) &&
        (init_params->select_param->op_id != UCP_OP_ID_RNDV_RECV)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_mm_get_alloc_md_map(worker->context, mdesc_md_map_p);
    if (status != UCS_OK) {
        return status;
    }

    *frag_size_p = worker->context->config.ext.rndv_frag_size[UCS_MEMORY_TYPE_HOST];
    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_mtype_request_init(ucp_request_t *req)
{
    ucp_worker_h worker = req->send.ep->worker;

    req->send.rndv.mdesc = ucp_rndv_mpool_get(worker, UCS_MEMORY_TYPE_HOST,
                                              UCS_SYS_DEVICE_ID_UNKNOWN);
    if (req->send.rndv.mdesc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_proto_rndv_mtype_get_memh(ucp_mem_desc_t *mdesc, ucp_rsc_index_t memh_index)
{
    if (memh_index == UCP_NULL_RESOURCE) {
        return UCT_MEM_HANDLE_NULL;
    }

    ucs_assertv(UCS_BIT(memh_index) & mdesc->memh->md_map,
                "memh_index=%d md_map=0x%" PRIx64, memh_index,
                mdesc->memh->md_map);
    return mdesc->memh->uct[memh_index];
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_iov_init(ucp_request_t *req, size_t length, size_t offset,
                              ucp_rsc_index_t memh_index, uct_iov_t *iov)
{
    ucp_mem_desc_t *mdesc = req->send.rndv.mdesc;

    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);

    iov->length = length;
    iov->buffer = UCS_PTR_BYTE_OFFSET(mdesc->ptr, offset);
    iov->memh   = ucp_proto_rndv_mtype_get_memh(mdesc, memh_index);
    iov->stride = 0;
    iov->count  = 1;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_next_iov(ucp_request_t *req,
                              const ucp_proto_rndv_bulk_priv_t *rpriv,
                              const ucp_proto_multi_lane_priv_t *lpriv,
                              ucp_datatype_iter_t *next_iter, uct_iov_t *iov)
{
    size_t max_payload = ucp_proto_rndv_bulk_max_payload(req, rpriv, lpriv);
    size_t length      = ucp_datatype_iter_next(&req->send.state.dt_iter,
                                                max_payload, next_iter);

    ucp_proto_rndv_mtype_iov_init(req, length, req->send.state.dt_iter.offset,
                                  lpriv->super.md_index, iov);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_mtype_copy(
        ucp_request_t *req, uct_ep_put_zcopy_func_t copy_func,
        uct_completion_callback_t comp_func, const char *mode)
{
    ucp_ep_h ep                = req->send.ep;
    ucp_worker_h worker        = ep->worker;
    ucs_memory_type_t mem_type = req->send.state.dt_iter.mem_info.type;
    ucp_ep_h mtype_ep          = worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane      = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    ucp_md_index_t md_index    = ucp_ep_md_index(mtype_ep, lane);
    ucp_mem_desc_t *mdesc      = req->send.rndv.mdesc;
    ucs_status_t status;
    uct_iov_t iov;

    ucs_assert(lane != UCP_NULL_LANE);
    ucs_assert(mdesc != NULL);

    ucp_trace_req(req, "mdesc %p copy-%s %p %s using memtype-ep %p lane[%d]",
                  mdesc, mode, req->send.state.dt_iter.type.contig.buffer,
                  ucs_memory_type_names[mem_type], mtype_ep, lane);

    ucp_proto_completion_init(&req->send.state.uct_comp, comp_func);

    /* Set up IOV pointing to the mdesc */
    ucp_proto_rndv_mtype_iov_init(req, req->send.state.dt_iter.length, 0,
                                  md_index, &iov);

    /* Copy from mdesc to user buffer */
    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);
    status = copy_func(mtype_ep->uct_eps[lane], &iov, 1,
                       (uintptr_t)req->send.state.dt_iter.type.contig.buffer,
                       UCT_INVALID_RKEY, &req->send.state.uct_comp);
    ucp_trace_req(req, "mdesc %p copy returned %s", mdesc,
                  ucs_status_string(status));
    ucs_assert(status != UCS_ERR_NO_RESOURCE);

    if (status != UCS_INPROGRESS) {
        ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
    }

    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_query_desc(const ucp_proto_query_params_t *params,
                                ucp_proto_query_attr_t *attr,
                                const char *xfer_desc)
{
    UCS_STRING_BUFFER_FIXED(strb, attr->desc, sizeof(attr->desc));
    ucp_context_h context      = params->worker->context;
    ucs_memory_type_t mem_type = params->select_param->mem_type;
    ucp_ep_h mtype_ep          = params->worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane      = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    ucp_rsc_index_t rsc_index  = ucp_ep_get_rsc_index(mtype_ep, lane);
    const char *tl_name        = context->tl_rscs[rsc_index].tl_rsc.tl_name;

    if (params->select_param->op_id == UCP_OP_ID_RNDV_SEND) {
        ucs_string_buffer_appendf(&strb, "%s, ", tl_name);
    }

    ucs_string_buffer_appendf(&strb, "%s", xfer_desc);

    if (params->select_param->op_id == UCP_OP_ID_RNDV_RECV) {
        ucs_string_buffer_appendf(&strb, ", %s", tl_name);
    }
}

#endif
