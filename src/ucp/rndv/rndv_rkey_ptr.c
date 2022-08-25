/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"
#include "rndv_mtype.inl"

#include <ucp/proto/proto_debug.h>


#define UCP_PROTO_RNDV_RKEY_PTR_DESC "write to attached"


enum {
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_FETCH = UCP_PROTO_STAGE_START,
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATS,
    UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATP
};


typedef struct {
    ucp_proto_rndv_ack_priv_t ack;
    ucp_proto_single_priv_t   spriv;
} ucp_proto_rndv_rkey_ptr_priv_t;


static double ucp_proto_rndv_rkey_ptr_overhead()
{
    switch (ucs_arch_get_cpu_vendor()) {
    case UCS_CPU_VENDOR_FUJITSU_ARM:
        return 500e-9;
    default:
        return 0;
    }
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_init(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_rndv_rkey_ptr_priv_t *rpriv = init_params->priv;
    ucp_context_t *context                = init_params->worker->context;
    uint64_t rndv_modes                   = UCS_BIT(UCP_RNDV_MODE_RKEY_PTR);
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 0,
        .super.overhead      = ucp_proto_rndv_rkey_ptr_overhead(),
        .super.latency       = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_LAST,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG,
        .lane_type           = UCP_LANE_TYPE_RKEY_PTR,
        .tl_cap_flags        = 0,
    };
    ucp_proto_caps_t rkey_ptr_caps;
    ucs_status_t status;

    if (!ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_RECV, 0)) {
        return UCS_ERR_UNSUPPORTED;
    }

    params.super.super.caps = &rkey_ptr_caps;
    status = ucp_proto_single_init_priv(&params, &rpriv->spriv);
    if (status != UCS_OK) {
        return status;
    }

    *init_params->priv_size = sizeof(*rpriv);
    status = ucp_proto_rndv_ack_init(init_params, UCP_PROTO_RNDV_ATS_NAME,
                                     &rkey_ptr_caps, UCS_LINEAR_FUNC_ZERO,
                                     &rpriv->ack);
    ucp_proto_select_caps_cleanup(&rkey_ptr_caps);

    return status;
}

static void
ucp_proto_rndv_rkey_ptr_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr)
{
    UCS_STRING_BUFFER_FIXED(config_strb, attr->config, sizeof(attr->config));
    const ucp_proto_rndv_rkey_ptr_priv_t *rpriv = params->priv;

    ucp_proto_default_query(params, attr);
    ucp_proto_common_lane_priv_str(params, &rpriv->spriv.super, 1, 0,
                                   &config_strb);
}

static unsigned ucp_proto_rndv_progress_rkey_ptr(void *arg)
{
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucp_request_t *req  = ucs_queue_head_elem_non_empty(&worker->rkey_ptr_reqs,
                                                        ucp_request_t,
                                                        send.rndv.rkey_ptr.queue_elem);
    size_t max_seg_size = worker->context->config.ext.rkey_ptr_seg_size;
    size_t length       = req->send.state.dt_iter.length;
    size_t offset       = req->send.state.completed_size;
    size_t seg_size     = ucs_min(max_seg_size, length - offset);
    ucs_status_t status;
    const void *src;

    src = UCS_PTR_BYTE_OFFSET(req->send.rndv.rkey_ptr_addr, offset);

    ucp_trace_req(req, "rkey_ptr unpack %zd from %p at offset %zd/%zd",
                  seg_size, src, offset, length);

    status = ucp_datatype_iter_unpack(&req->send.state.dt_iter, worker,
                                      seg_size, offset, src);
    if (ucs_unlikely(status != UCS_OK)) {
        ucp_proto_request_abort(req, status);
        return 0;
    }

    if (!ucp_proto_common_frag_complete(req, seg_size, "rkey_ptr")) {
        return 1;
    }

    ucs_queue_pull_non_empty(&worker->rkey_ptr_reqs);
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, UCP_DT_MASK_ALL);

    ucp_proto_rndv_recv_complete_with_ats(req,
                                          UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATS);
    if (ucs_queue_is_empty(&worker->rkey_ptr_reqs)) {
        uct_worker_progress_unregister_safe(worker->uct,
                                            &worker->rkey_ptr_cb_id);
    }

    return 1;
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_fetch_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req                          = ucs_container_of(uct_req,
                                                                   ucp_request_t,
                                                                   send.uct);
    const ucp_proto_rndv_rkey_ptr_priv_t *rpriv = req->send.proto_config->priv;
    ucp_worker_h worker                         = req->send.ep->worker;
    unsigned rkey_index                         = rpriv->spriv.super.rkey_index;
    ucp_rkey_h rkey                             = req->send.rndv.rkey;
    ucs_status_t status;

    ucs_assert(rkey_index != UCP_NULL_RESOURCE);
    status = uct_rkey_ptr(rkey->tl_rkey[rkey_index].cmpt,
                          &rkey->tl_rkey[rkey_index].rkey,
                          req->send.rndv.remote_address,
                          &req->send.rndv.rkey_ptr_addr);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    req->send.state.completed_size = 0;
    UCP_WORKER_STAT_RNDV(worker, RKEY_PTR, 1);
    ucs_queue_push(&worker->rkey_ptr_reqs, &req->send.rndv.rkey_ptr.queue_elem);
    uct_worker_progress_register_safe(worker->uct,
            ucp_proto_rndv_progress_rkey_ptr, worker,
            UCS_CALLBACKQ_FLAG_FAST, &worker->rkey_ptr_cb_id);

    return UCS_OK;
}

ucp_proto_t ucp_rndv_rkey_ptr_proto = {
    .name     = "rndv/rkey_ptr",
    .desc     = "copy from mapped remote memory",
    .flags    = 0,
    .init     = ucp_proto_rndv_rkey_ptr_init,
    .query    = ucp_proto_rndv_rkey_ptr_query,
    .progress = {
         [UCP_PROTO_RNDV_RKEY_PTR_STAGE_FETCH] = ucp_proto_rndv_rkey_ptr_fetch_progress,
         [UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATS]   = ucp_proto_rndv_ats_progress
    },
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_fatal_not_implemented_void,
    .reset    = (ucp_request_reset_func_t)ucs_empty_function_fatal_not_implemented_void
};

static ucs_status_t ucp_proto_rndv_rkey_ptr_mtype_init_params(
        const ucp_proto_init_params_t *init_params,
        ucp_md_map_t initial_reg_md_map, size_t max_length)
{
    ucp_context_t *context               = init_params->worker->context;
    uint64_t rndv_modes                  = UCS_BIT(UCP_RNDV_MODE_PUT_PIPELINE);
    ucp_proto_rndv_put_priv_t *rpriv     = init_params->priv;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.overhead      = ucp_proto_rndv_rkey_ptr_overhead(),
        .super.latency       = 0,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = max_length,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.put.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_LAST,
        .super.memtype_op    = UCT_EP_OP_GET_ZCOPY,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RKEY_PTR,
        .max_lanes           = context->config.ext.max_rndv_lanes,
        .initial_reg_md_map  = initial_reg_md_map,
        .first.tl_cap_flags  = 0,
        .first.lane_type     = UCP_LANE_TYPE_RKEY_PTR,
        .middle.tl_cap_flags = 0,
        .middle.lane_type    = UCP_LANE_TYPE_RKEY_PTR
    };
    size_t bulk_priv_size;
    ucs_status_t status;

    status = ucp_proto_rndv_bulk_init(&params, &rpriv->bulk,
                                      UCP_PROTO_RNDV_RKEY_PTR_DESC,
                                      UCP_PROTO_RNDV_ATP_NAME, &bulk_priv_size);
    if (status != UCS_OK) {
        return status;
    }

    rpriv->atp_map       = rpriv->bulk.mpriv.lane_map;
    rpriv->atp_num_lanes = ucs_popcount(rpriv->atp_map);

    return UCS_OK;
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_mtype_init(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context = init_params->worker->context;
    ucs_status_t status;
    ucp_md_map_t mdesc_md_map;
    size_t frag_size;

    ucp_md_index_t md_index;
    const uct_md_attr_t *md_attr;
    int found_md_map = 0;

    if (!context->config.ext.rndv_shm_ppln_enable) {
        return UCS_ERR_UNSUPPORTED;
    }

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_SEND, 1) ||
        init_params->rkey_config_key == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_rndv_mtype_init(init_params, &mdesc_md_map, &frag_size);
    if (status != UCS_OK) {
        return status;
    }

    ucs_for_each_bit(md_index, init_params->rkey_config_key->md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if ((md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR) &&
            /* Do not use xpmem, because cuda_copy registration will fail and
             * performance will not be optimal. */
            !(md_attr->cap.flags & UCT_MD_FLAG_REG) &&
            (md_attr->cap.access_mem_types &
             UCS_BIT(init_params->rkey_config_key->mem_type))) {
            found_md_map = UCS_BIT(md_index);
            break;
        }
    }
    if (found_md_map == 0) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_rkey_ptr_mtype_init_params(init_params, mdesc_md_map,
                                                     frag_size);
}


static void ucp_proto_rndv_rkey_ptr_mtype_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "ucp_proto_rndv_rkey_ptr_mtype_completion");
    ucp_proto_rndv_rkey_destroy(req);
    ucp_proto_request_zcopy_complete(req, req->send.state.uct_comp.status);
}

static void
ucp_proto_rndv_rkey_ptr_mtype_copy_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "ucp_proto_rndv_rkey_ptr_mtype_copy_completion");

    ucp_proto_completion_init(&req->send.state.uct_comp,
                              ucp_proto_rndv_rkey_ptr_mtype_completion);
    ucp_proto_request_set_stage(req, UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATP);
    ucp_request_send(req);
}


static ucs_status_t
ucp_proto_rndv_rkey_ptr_mtype_copy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req                = ucs_container_of(uct_req, ucp_request_t,
                                                         send.uct);
    ucp_context_h context             = req->send.ep->worker->context;
    ucp_md_index_t rkey_ptr_md_index  = UCP_NULL_RESOURCE;
    uint64_t remote_address           = req->send.rndv.remote_address;
    ucs_memory_type_t local_mem_type  = req->send.state.dt_iter.mem_info.type;
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;
    ucs_memory_type_t remote_mem_type      = req->send.rndv.rkey->mem_type;
    ucp_md_map_t md_map                    = req->send.rndv.rkey->md_map;
    void *rkey_buffer                      = req->send.rndv.rkey_buffer;
    ucp_lane_index_t mem_type_rma_lane;
    ucp_ep_peer_mem_data_t *ppln_data;
    const uct_md_attr_t *md_attr;
    ucp_md_index_t md_index;
    ucp_ep_h mem_type_ep;
    unsigned rkey_index;
    void *local_ptr;
    uct_iov_t iov;
    ucs_status_t status;

    req->send.rndv.rkey_buffer = NULL;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED));

    mem_type_ep = req->send.ep->worker->mem_type_ep[local_mem_type];
    if (mem_type_ep == NULL) {
        return UCS_OK;
    }
    mem_type_rma_lane = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];

    ucs_for_each_bit(md_index, md_map) {
        md_attr = &context->tl_mds[md_index].attr;
        if ((md_attr->cap.flags & UCT_MD_FLAG_RKEY_PTR) &&
            /* Do not use xpmem, because cuda_copy registration will fail and
             * performance will not be optimal. */
            !(md_attr->cap.flags & UCT_MD_FLAG_REG) &&
            (md_attr->cap.access_mem_types & UCS_BIT(remote_mem_type))) {
            rkey_ptr_md_index = md_index;
            break;
        }
    }

    if (rkey_ptr_md_index == UCP_NULL_RESOURCE) {
        ucp_proto_request_abort(req, UCP_NULL_RESOURCE);
        return UCS_OK;
    }

    ppln_data = ucp_ep_peer_mem_get(context, req->send.ep, remote_address,
                                    req->send.state.dt_iter.length, rkey_buffer,
                                    rkey_ptr_md_index);
    if (ppln_data->rkey == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    rkey_index = ucs_bitmap2idx(ppln_data->rkey->md_map, rkey_ptr_md_index);
    status     = uct_rkey_ptr(ppln_data->rkey->tl_rkey[rkey_index].cmpt,
                              &ppln_data->rkey->tl_rkey[rkey_index].rkey,
                              req->send.rndv.remote_address, &local_ptr);
    if (status != UCS_OK) {
        ppln_data->size = 0; /* Make sure hash element is updated next time */
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    if (ppln_data->uct_memh == NULL) {
        /* Register remote memory segment with memtype ep MD. Without
        * registration fetching data from GPU to CPU will be performance
        * inefficient. */
        md_map              = 0;
        ppln_data->md_index = ucp_ep_md_index(mem_type_ep, mem_type_rma_lane);

        status = ucp_mem_rereg_mds(context, UCS_BIT(ppln_data->md_index),
                                   local_ptr, ppln_data->size,
                                   UCT_MD_MEM_ACCESS_RMA |
                                           UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                   NULL, remote_mem_type, NULL,
                                   &ppln_data->uct_memh, &md_map);

        if (status != UCS_OK) {
            ppln_data->md_index = UCP_NULL_RESOURCE;
        } else {
            ucs_assertv(md_map == UCS_BIT(ppln_data->md_index),
                        "mdmap=0x%lx, md_index=%u", md_map,
                        ppln_data->md_index);
        }
    }

    ucp_proto_completion_init(&req->send.state.uct_comp,
                              ucp_proto_rndv_rkey_ptr_mtype_copy_completion);

    req->send.rndv.put.atp_map = rpriv->atp_map;

    /* Set up IOV pointing to attached remote buffer */
    ucs_assert(req->send.state.dt_iter.dt_class == UCP_DATATYPE_CONTIG);

    iov.length = req->send.state.dt_iter.length;
    iov.buffer = local_ptr;
    iov.memh   = ppln_data->uct_memh;
    iov.stride = 0;
    iov.count  = 1;

    status = uct_ep_get_zcopy(
            ucp_ep_get_lane(mem_type_ep, mem_type_rma_lane), &iov, 1,
            (uintptr_t)req->send.state.dt_iter.type.contig.buffer,
            UCT_INVALID_RKEY, &req->send.state.uct_comp);
    ucp_trace_req(req, "local_ptr %p copy returned %s", local_ptr,
                  ucs_status_string(status));
    ucs_assert(status != UCS_ERR_NO_RESOURCE);
    if (status != UCS_INPROGRESS) {
        ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
    }

    req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    return UCS_OK;
}

static ucs_status_t
ucp_proto_rndv_rkey_ptr_mtype_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;
    ucp_lane_map_t atp_map                 = rpriv->atp_map;

    return ucp_proto_multi_lane_map_progress(
            req, &atp_map, ucp_proto_rndv_put_common_atp_send);
}

ucp_proto_t ucp_rndv_rkey_ptr_mtype_proto = {
    .name     = "rndv/rkey_ptr/mtype",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_proto_rndv_rkey_ptr_mtype_init,
    .query    = ucp_proto_rndv_rkey_ptr_query,
    .progress = {
        [UCP_PROTO_RNDV_RKEY_PTR_STAGE_FETCH] = ucp_proto_rndv_rkey_ptr_mtype_copy_progress,
        [UCP_PROTO_RNDV_RKEY_PTR_STAGE_ATP]   = ucp_proto_rndv_rkey_ptr_mtype_atp_progress,
    },
    .abort    = (ucp_request_abort_func_t)ucs_empty_function_do_assert_void
};
