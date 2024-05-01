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
    ucp_worker_h worker             = init_params->worker;
    ucp_context_h context           = worker->context;
    ucs_memory_type_t mem_type      = init_params->select_param->mem_type;
    ucs_memory_type_t frag_mem_type = context->config.ext.rndv_frag_mem_type;

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (worker->mem_type_ep[mem_type] == NULL) ||
        !ucp_proto_init_check_op(init_params, UCP_PROTO_RNDV_OP_ID_MASK)) {
        return UCS_ERR_UNSUPPORTED;
    }

    *mdesc_md_map_p = context->reg_md_map[frag_mem_type];
    *frag_size_p    = context->config.ext.rndv_frag_size[frag_mem_type];

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_mtype_request_init(ucp_request_t *req)
{
    ucp_worker_h worker             = req->send.ep->worker;
    ucs_memory_type_t frag_mem_type =
                                 worker->context->config.ext.rndv_frag_mem_type;

    req->send.rndv.mdesc = ucp_rndv_mpool_get(worker, frag_mem_type,
                                              UCS_SYS_DEVICE_ID_UNKNOWN);
    if (req->send.rndv.mdesc == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_proto_rndv_mtype_get_memh(ucp_request_t *req, ucp_md_index_t md_index)
{
    ucp_mem_desc_t *mdesc = req->send.rndv.mdesc;

    if (md_index == UCP_NULL_RESOURCE) {
        return UCT_MEM_HANDLE_NULL;
    }

    ucs_assertv(UCS_BIT(md_index) & mdesc->memh->md_map,
                "md_index=%d md_map=0x%" PRIx64, md_index,
                mdesc->memh->md_map);
    return mdesc->memh->uct[md_index];
}

static UCS_F_ALWAYS_INLINE uct_mem_h
ucp_proto_rndv_mtype_get_req_memh(ucp_request_t *req)
{
    ucs_memory_type_t mem_type = req->send.state.dt_iter.mem_info.type;
    ucp_ep_h mtype_ep          = req->send.ep->worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane      = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    ucp_md_index_t md_index    = ucp_ep_md_index(mtype_ep, lane);

    return ucp_proto_rndv_mtype_get_memh(req, md_index);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_mtype_iov_init(ucp_request_t *req, void *buffer,
                              size_t length, size_t offset, uct_mem_h memh,
                              uct_iov_t *iov)
{
    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);

    iov->length = length;
    iov->buffer = UCS_PTR_BYTE_OFFSET(buffer, offset);
    iov->memh   = memh;
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
    uct_mem_h memh     = ucp_proto_rndv_mtype_get_memh(req,
                                                       lpriv->super.md_index);

    ucp_proto_rndv_mtype_iov_init(req, req->send.rndv.mdesc->ptr, length,
                                  req->send.state.dt_iter.offset, memh, iov);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_mtype_copy(
        ucp_request_t *req, void *buffer, uct_mem_h memh,
        uct_ep_put_zcopy_func_t copy_func, uct_completion_callback_t comp_func,
        const char *mode)
{
    ucp_ep_h ep                = req->send.ep;
    ucp_worker_h worker        = ep->worker;
    ucs_memory_type_t mem_type = req->send.state.dt_iter.mem_info.type;
    ucp_ep_h mtype_ep          = worker->mem_type_ep[mem_type];
    ucp_lane_index_t lane      = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];
    ucs_status_t status;
    uct_iov_t iov;

    ucp_trace_req(req, "buffer %p copy-%s %p %s using memtype-ep %p lane[%d]",
                  buffer, mode, req->send.state.dt_iter.type.contig.buffer,
                  ucs_memory_type_names[mem_type], mtype_ep, lane);

    ucp_proto_completion_init(&req->send.state.uct_comp, comp_func);

    /* Set up IOV pointing to the mdesc */
    ucp_proto_rndv_mtype_iov_init(req, buffer, req->send.state.dt_iter.length,
                                  0, memh, &iov);

    /* Copy from mdesc to user buffer */
    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);
    status = copy_func(ucp_ep_get_lane(mtype_ep, lane), &iov, 1,
                       (uintptr_t)req->send.state.dt_iter.type.contig.buffer,
                       UCT_INVALID_RKEY, &req->send.state.uct_comp);
    ucp_trace_req(req, "buffer %p copy returned %s", buffer,
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

    if (ucp_proto_select_op_id(params->select_param) == UCP_OP_ID_RNDV_SEND) {
        ucs_string_buffer_appendf(&strb, "%s, ", tl_name);
    }

    ucs_string_buffer_appendf(&strb, "%s", xfer_desc);

    if (ucp_proto_select_op_id(params->select_param) == UCP_OP_ID_RNDV_RECV) {
        ucs_string_buffer_appendf(&strb, ", %s", tl_name);
    }
}

#endif
