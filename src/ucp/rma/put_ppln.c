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
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_perf.h>
#include <ucp/proto/proto_select.h>
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

    if ((sel_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT)) ||
        !(ucp_proto_select_op_flags(sel_param) &
          UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG) ||
        !ucp_ep_config_is_inter_node(init_params->ep_config_key) ||
        !UCP_MEM_IS_CUDA(sel_param->mem_type) ||
        ((init_params->rkey_config_key != NULL) &&
         !UCP_MEM_IS_CUDA(init_params->rkey_config_key->mem_type)) ||
        (context->config.ext.ppln_frag_mem_types == 0)) {
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
 * put/ppln — orchestrator: bounce buffer lifecycle, pipeline fragment dispatch,
 *            ATP signaling, completion tracking
 */

/* Private data for put/ppln protocol */
typedef struct {
    size_t             frag_size;
    ucp_proto_config_t frag_proto_cfg;
    size_t             frag_proto_min_length;
} ucp_proto_put_ppln_priv_t;

static ucs_status_t
ucp_proto_put_ppln_add_overhead(ucp_proto_perf_t *ppln_perf, size_t frag_size)
{
    static const double frag_overhead = 30e-9;
    ucp_proto_perf_factors_t factors  = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    char frag_str[64];
    ucp_proto_perf_node_t *node;

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));
    factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU] =
            ucs_linear_func_make(frag_overhead, frag_overhead / frag_size);
    node = ucp_proto_perf_node_new_data("fragment overhead", "frag size: %s",
                                        frag_str);
    return ucp_proto_perf_add_funcs(ppln_perf, frag_size + 1, SIZE_MAX, factors,
                                    node, NULL);
}

static void
ucp_proto_put_ppln_set_frag_proto_config(
        const ucp_proto_init_params_t *init_params,
        const ucp_proto_init_elem_t *proto,
        const ucp_proto_select_param_t *select_param, const void *priv,
        ucp_proto_config_t *proto_config)
{
    proto_config->proto          = ucp_protocols[proto->proto_id];
    proto_config->priv           = priv;
    proto_config->ep_cfg_index   = init_params->ep_cfg_index;
    proto_config->rkey_cfg_index = init_params->rkey_cfg_index;
    proto_config->select_param   = *select_param;
    proto_config->init_elem      = proto;
    proto_config->selections     = 0;
    ucp_request_progress_wrapper_init(init_params->worker, proto_config);
}

static void
ucp_proto_put_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_worker_h worker                       = init_params->worker;
    const ucp_proto_select_param_t *sel_param = init_params->select_param;
    const ucp_proto_perf_segment_t *frag_seg, *first_seg;
    const ucp_proto_select_elem_t *select_elem;
    UCS_STRING_BUFFER_ONSTACK(seg_strb, 128);
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t frag_sel_param;
    ucp_proto_put_ppln_priv_t rpriv;
    ucp_proto_select_t *proto_select;
    ucp_proto_perf_t *ppln_perf;
    ucp_proto_init_elem_t *proto;
    char frag_size_str[32];
    void *frag_proto_priv;
    ucs_status_t status;
    uint8_t proto_flags;

    if ((sel_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_PUT)) ||
        (ucp_proto_select_op_flags(sel_param) &
         UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG) ||
        !ucp_ep_config_is_inter_node(init_params->ep_config_key) ||
        (init_params->ep_config_key->am_lane == UCP_NULL_LANE) ||
        !UCP_MEM_IS_CUDA(sel_param->mem_type) ||
        ((init_params->rkey_config_key != NULL) &&
         !UCP_MEM_IS_CUDA(init_params->rkey_config_key->mem_type))) {
        return;
    }

    /* Look up the fragment protocol (put/mtype) */
    frag_sel_param             = *sel_param;
    frag_sel_param.op_id_flags = ucp_proto_select_op_id(sel_param) |
                                 UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG;
    frag_sel_param.op_attr     = ucp_proto_select_op_attr_pack(
            UCP_OP_ATTR_FLAG_MULTI_SEND, UCP_PROTO_SELECT_OP_ATTR_MASK);

    proto_select = ucp_proto_select_get(worker, init_params->ep_cfg_index,
                                        init_params->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return;
    }

    select_elem = ucp_proto_select_lookup_slow(worker, proto_select, 1,
                                               init_params->ep_cfg_index,
                                               init_params->rkey_cfg_index,
                                               &frag_sel_param);
    if (select_elem == NULL) {
        return;
    }

    /* Add each fragment proto variant as a separate ppln variant */
    ucs_array_for_each(proto, &select_elem->proto_init.protocols) {
        proto_flags = ucp_proto_id_field(proto->proto_id, flags);
        if (proto_flags & UCP_PROTO_FLAG_INVALID) {
            continue;
        }

        ucs_assert(!ucp_proto_perf_is_empty(proto->perf));

        status = ucp_proto_perf_create("pipeline", &ppln_perf);
        if (status != UCS_OK) {
            continue;
        }

        frag_seg = ucp_proto_perf_add_ppln(proto->perf, ppln_perf, SIZE_MAX);
        if (frag_seg == NULL) {
            goto out_destroy_ppln_perf;
        }

        rpriv.frag_size = ucp_proto_perf_segment_end(frag_seg);
        first_seg       = ucp_proto_perf_find_segment_lb(proto->perf, 0);
        rpriv.frag_proto_min_length = ucp_proto_perf_segment_start(first_seg);
        ucs_assertv(rpriv.frag_size >= rpriv.frag_proto_min_length,
                    "rpriv.frag_size=%zu rpriv.frag_proto_min_length=%zu",
                    rpriv.frag_size, rpriv.frag_proto_min_length);

        frag_proto_priv = &ucs_array_elem(&select_elem->proto_init.priv_buf,
                                          proto->priv_offset);
        ucp_proto_put_ppln_set_frag_proto_config(init_params, proto,
                                                 &frag_sel_param,
                                                 frag_proto_priv,
                                                 &rpriv.frag_proto_cfg);

        ucp_proto_perf_segment_str(frag_seg, &seg_strb);
        ucs_trace("put_ppln frag=%s proto=%s segment=%s",
                  ucs_memunits_to_str(rpriv.frag_size, frag_size_str,
                                      sizeof(frag_size_str)),
                  ucp_proto_id_field(proto->proto_id, name),
                  ucs_string_buffer_cstr(&seg_strb));

        status = ucp_proto_put_ppln_add_overhead(ppln_perf, rpriv.frag_size);
        if (status != UCS_OK) {
            goto out_destroy_ppln_perf;
        }

        ucp_proto_select_add_proto(init_params, proto->cfg_thresh,
                                   proto->cfg_priority, ppln_perf, &rpriv,
                                   sizeof(rpriv));

    out_destroy_ppln_perf:
        ucp_proto_perf_destroy(ppln_perf);
    }
}

static void
ucp_proto_put_ppln_query(const ucp_proto_query_params_t *params,
                         ucp_proto_query_attr_t *attr)
{
    const ucp_proto_put_ppln_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t frag_attr;

    if (params->msg_length <= rpriv->frag_size) {
        ucp_proto_config_query(params->worker, &rpriv->frag_proto_cfg,
                               params->msg_length, attr);
        attr->max_msg_length = rpriv->frag_size;
    } else {
        ucp_proto_config_query(params->worker, &rpriv->frag_proto_cfg,
                               rpriv->frag_size, &frag_attr);

        attr->max_msg_length = SIZE_MAX;
        attr->is_estimation  = 0;
        attr->lane_map       = frag_attr.lane_map;
        ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "pipeline %s",
                          frag_attr.desc);
        ucs_strncpy_safe(attr->config, frag_attr.config, sizeof(attr->config));
    }
}

enum {
    UCP_PROTO_PUT_PPLN_STAGE_SEND = UCP_PROTO_STAGE_START,
};

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
    .query    = ucp_proto_put_ppln_query,
    .progress = {
        [UCP_PROTO_PUT_PPLN_STAGE_SEND] = ucp_proto_put_ppln_progress,
    },
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};
