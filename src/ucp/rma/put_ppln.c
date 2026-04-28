/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "rma.h"
#include "rma.inl"

#include <ucp/core/ucp_mm.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_common.h>
#include <ucp/proto/proto_select.inl>


/*
 * ppln fragment pool — lazy-alloc per (mem_type, sys_dev), separate from rndv
 */

static ucs_mpool_ops_t ucp_ppln_frag_mpool_ops = {
    .chunk_alloc   = ucp_ppln_frag_mpool_malloc,
    .chunk_release = ucp_ppln_frag_mpool_free,
    .obj_init      = ucp_ppln_frag_mpool_obj_init,
    .obj_cleanup   = (ucs_mpool_obj_cleanup_func_t)ucs_empty_function
};

ucp_mem_desc_t *
ucp_ppln_mpool_get(ucp_worker_h worker, ucs_memory_type_t mem_type,
                   ucs_sys_device_t sys_dev)
{
    ucp_rndv_mpool_priv_t *mpriv;
    ucp_worker_mpool_key_t key;
    ucs_status_t status;
    unsigned num_frags;
    ucs_mpool_t *mpool;
    khiter_t khiter;
    int khret;
    ucs_mpool_params_t mp_params;

    key.sys_dev  = sys_dev;
    key.mem_type = mem_type;

    khiter = kh_get(ucp_worker_mpool_hash, &worker->ppln_mpool_hash, key);
    if (ucs_likely(khiter != kh_end(&worker->ppln_mpool_hash))) {
        mpool = &kh_val(&worker->ppln_mpool_hash, khiter);
        goto out_mp_get;
    }

    khiter = kh_put(ucp_worker_mpool_hash, &worker->ppln_mpool_hash, key,
                    &khret);
    if (khret == UCS_KH_PUT_FAILED) {
        return NULL;
    }

    ucs_assert_always(khret != UCS_KH_PUT_KEY_PRESENT);

    mpool     = &kh_value(&worker->ppln_mpool_hash, khiter);
    num_frags = worker->context->config.ext.ppln_num_frags[key.mem_type];

    ucs_mpool_params_reset(&mp_params);
    mp_params.priv_size       = sizeof(ucp_rndv_mpool_priv_t);
    mp_params.elem_size       = sizeof(ucp_mem_desc_t);
    mp_params.alignment       = 1;
    mp_params.elems_per_chunk = num_frags;
    mp_params.ops             = &ucp_ppln_frag_mpool_ops;
    mp_params.name            = "ucp_ppln_frags";
    status = ucs_mpool_init(&mp_params, mpool);
    if (status != UCS_OK) {
        return NULL;
    }

    mpriv            = ucs_mpool_priv(mpool);
    mpriv->worker    = worker;
    mpriv->mem_type  = key.mem_type;
    mpriv->sys_dev   = sys_dev;

out_mp_get:
    return ucp_worker_mpool_get(mpool);
}


/*
 * put/mtype — copy-in (GPU → bounce) + multi-lane RDMA send (bounce → remote)
 */

enum {
    UCP_PROTO_PUT_MTYPE_STAGE_COPY = UCP_PROTO_STAGE_START,
    UCP_PROTO_PUT_MTYPE_STAGE_SEND,
    UCP_PROTO_PUT_MTYPE_STAGE_FENCE,
    UCP_PROTO_PUT_MTYPE_STAGE_AM
};

static void
ucp_proto_put_mtype_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                    = init_params->worker->context;
    const ucp_proto_select_param_t *sel_param = init_params->select_param;
    ucs_memory_type_t frag_mem_type;
    ucp_md_map_t mdesc_md_map;
    ucp_memory_info_t frag_mem_info;
    ucp_md_index_t UCS_V_UNUSED dummy_md_id;
    ucs_status_t status;
    size_t frag_size;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.overhead      = 0,
        .super.latency       = 0,
        .super.cfg_thresh    = 0,
        .super.cfg_priority  = 40,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.put.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_GET_ZCOPY,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY   |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .max_lanes           = context->config.ext.max_rma_lanes,
        .min_chunk           = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_ZCOPY |
                               UCT_IFACE_FLAG_AM_BCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_ZCOPY |
                               UCT_IFACE_FLAG_AM_BCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.opt_zcopy_align),
    };

    if (sel_param->dt_class != UCP_DATATYPE_CONTIG) {
        return;
    }

    /* Only inter-node */
    if (!ucp_ep_config_is_inter_node(init_params->ep_config_key)) {
        return;
    }

    /* Only GPU-to-GPU */
    if (!UCP_MEM_IS_CUDA(sel_param->mem_type) ||
        ((init_params->rkey_config_key != NULL) &&
         !UCP_MEM_IS_CUDA(init_params->rkey_config_key->mem_type))) {
        return;
    }

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return;
    }

    /* Only probe as a pipeline fragment protocol */
    if (!(ucp_proto_select_op_flags(sel_param) &
          UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG)) {
        return;
    }

    if (context->config.ext.ppln_frag_mem_types == 0) {
        return;
    }

    /* Find a usable fragment memory type (prefer CUDA, fall back to host) */
    frag_mem_type = UCP_MEM_IS_CUDA(sel_param->mem_type) ?
                    UCS_MEMORY_TYPE_CUDA : UCS_MEMORY_TYPE_HOST;
    if (!UCS_BIT_GET(context->config.ext.ppln_frag_mem_types,
                     frag_mem_type)) {
        frag_mem_type = UCS_MEMORY_TYPE_HOST;
        if (!UCS_BIT_GET(context->config.ext.ppln_frag_mem_types,
                         frag_mem_type)) {
            return;
        }
    }

    /* Check mem_type ep exists for staging copy */
    if (init_params->worker->mem_type_ep[sel_param->mem_type] == NULL &&
        init_params->worker->mem_type_ep[frag_mem_type] == NULL) {
        return;
    }

    mdesc_md_map = context->reg_md_map[frag_mem_type];
    frag_size    = context->config.ext.ppln_frag_size[frag_mem_type];

    status = ucp_mm_get_alloc_md_index(context, frag_mem_type,
                                       sel_param->sys_dev, &dummy_md_id,
                                       &frag_mem_info);
    if (status != UCS_OK) {
        return;
    }

    params.super.max_length     = frag_size;
    params.super.reg_mem_info   = frag_mem_info;
    params.initial_reg_md_map   = mdesc_md_map;

    ucp_proto_multi_probe(&params);
}

static ucs_status_t
ucp_proto_put_mtype_copy_progress(uct_pending_req_t *self)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t
ucp_proto_put_mtype_send_progress(uct_pending_req_t *self)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t
ucp_proto_put_mtype_fence_progress(uct_pending_req_t *self)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

static ucs_status_t
ucp_proto_put_mtype_am_progress(uct_pending_req_t *self)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucp_proto_t ucp_put_mtype_proto = {
    .name     = "put/mtype",
    .desc     = UCP_PROTO_PPLN_DESC " " UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .probe    = ucp_proto_put_mtype_probe,
    .query    = ucp_proto_multi_query,
    .progress = {
        [UCP_PROTO_PUT_MTYPE_STAGE_COPY]  = ucp_proto_put_mtype_copy_progress,
        [UCP_PROTO_PUT_MTYPE_STAGE_SEND]  = ucp_proto_put_mtype_send_progress,
        [UCP_PROTO_PUT_MTYPE_STAGE_FENCE] = ucp_proto_put_mtype_fence_progress,
        [UCP_PROTO_PUT_MTYPE_STAGE_AM]    = ucp_proto_put_mtype_am_progress,
    },
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};


/*
 * put/ppln — orchestrator: RTS/RTS_RESP handshake, bounce buffer lifecycle,
 *            ATP signaling, completion tracking
 */

static void
ucp_proto_put_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context                     = init_params->worker->context;
    const ucp_proto_select_param_t *sel_param  = init_params->select_param;
    const ucp_rkey_config_key_t *rkey_cfg_key  = init_params->rkey_config_key;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_multi,
        .super.cfg_thresh    = 0,
        .super.cfg_priority  = 60,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.put.max_iov),
        .super.hdr_size      = 0,
        .super.send_op       = UCT_EP_OP_PUT_ZCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY   |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY   |
                               UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                         sel_param),
        .max_lanes           = context->config.ext.max_rma_lanes,
        .min_chunk           = context->config.ext.min_rma_chunk_size,
        .initial_reg_md_map  = 0,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_ZCOPY  |
                               UCT_IFACE_FLAG_AM_BCOPY    |
                               UCT_IFACE_FLAG_AM_SHORT,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_ZCOPY  |
                               UCT_IFACE_FLAG_AM_BCOPY    |
                               UCT_IFACE_FLAG_AM_SHORT,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .opt_align_offs      = UCP_PROTO_COMMON_OFFSET_INVALID,
    };

    /* Only inter-node */
    if (!ucp_ep_config_is_inter_node(init_params->ep_config_key)) {
        return;
    }

    /* Only GPU-to-GPU */
    if (!UCP_MEM_IS_CUDA(sel_param->mem_type) ||
        ((rkey_cfg_key != NULL) &&
         !UCP_MEM_IS_CUDA(rkey_cfg_key->mem_type))) {
        return;
    }

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return;
    }

    ucp_proto_multi_probe(&params);
}

static ucs_status_t
ucp_proto_put_ppln_progress(uct_pending_req_t *self)
{
    return UCS_ERR_NOT_IMPLEMENTED;
}

ucp_proto_t ucp_put_ppln_proto = {
    .name     = "put/ppln",
    .desc     = UCP_PROTO_PPLN_DESC,
    .flags    = 0,
    .probe    = ucp_proto_put_ppln_probe,
    .query    = ucp_proto_multi_query,
    .progress = {ucp_proto_put_ppln_progress},
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};
