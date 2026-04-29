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
#include <ucp/proto/proto_am.h>
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

static void
ucp_proto_put_mtype_copy_complete(ucp_request_t *req)
{
    ucp_trace_req(req, "put/mtype: copy-in complete local_mdesc=%p",
                  req->send.frag_ppln.local_mdesc);
    ucp_proto_request_set_stage(req, UCP_PROTO_PUT_MTYPE_STAGE_SEND);
}

static void
ucp_proto_put_mtype_copy_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_proto_put_mtype_copy_complete(req);
    ucp_request_send(req);
}

static ucs_status_t
ucp_proto_put_mtype_copy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req     = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_mem_desc_t *mdesc  = req->send.frag_ppln.local_mdesc;
    ucs_memory_type_t frag_mem_type = mdesc->memh->mem_type;
    ucp_worker_h worker    = req->send.ep->worker;
    ucp_ep_h mtype_ep;
    ucp_lane_index_t lane;
    ucs_status_t status;
    uct_iov_t iov;

    /* Find mem_type EP for the copy */
    mtype_ep = worker->mem_type_ep[req->send.state.dt_iter.mem_info.type];
    ucs_assertv(mtype_ep != NULL, "no mem_type_ep for %s",
                ucs_memory_type_names[req->send.state.dt_iter.mem_info.type]);

    lane = ucp_ep_config(mtype_ep)->key.rma_bw_lanes[0];

    ucp_trace_req(req, "put/mtype: copy-in src=%p dst=%p size=%zu "
                  "mem_type=%s-%s",
                  req->send.state.dt_iter.type.contig.buffer, mdesc->ptr,
                  req->send.state.dt_iter.length,
                  ucs_memory_type_names[req->send.state.dt_iter.mem_info.type],
                  ucs_memory_type_names[frag_mem_type]);

    ucp_proto_completion_init(&req->send.state.uct_comp,
                              ucp_proto_put_mtype_copy_completion);

    iov.buffer = mdesc->ptr;
    iov.length = req->send.state.dt_iter.length;
    iov.memh   = mdesc->memh->uct[ucp_ep_md_index(mtype_ep, lane)];
    iov.stride = 0;
    iov.count  = 1;

    status = uct_ep_get_zcopy(ucp_ep_get_lane(mtype_ep, lane), &iov, 1,
                              (uintptr_t)req->send.state.dt_iter.type.contig.buffer,
                              UCT_INVALID_RKEY, &req->send.state.uct_comp);
    if (status == UCS_OK) {
        ucp_proto_put_mtype_copy_complete(req);
        return UCS_INPROGRESS;
    } else if (status != UCS_INPROGRESS) {
        ucp_proto_request_abort(req, status);
    }

    return UCS_OK;
}

/* Called when all PUT ZCOPY submissions have completed, release bounce buffer */
static void
ucp_proto_put_mtype_send_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucs_mpool_put_inline(req->send.frag_ppln.local_mdesc);
    ucp_request_send(req);
}

/* All data posted on lanes, transition to fence stage, decrement uct completion */
static ucs_status_t
ucp_proto_put_mtype_data_sent(ucp_request_t *req)
{
    ucp_proto_request_set_stage(req, UCP_PROTO_PUT_MTYPE_STAGE_FENCE);
    return ucp_request_invoke_uct_completion_success(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_mtype_send_func(ucp_request_t *req,
                               const ucp_proto_multi_lane_priv_t *lpriv,
                               ucp_datatype_iter_t *next_iter,
                               ucp_lane_index_t *lane_shift)
{
    ucp_mem_desc_t *mdesc  = req->send.frag_ppln.local_mdesc;
    uint64_t remote_addr   = req->send.frag_ppln.remote_addr +
                             req->send.state.dt_iter.offset;
    uct_rkey_t tl_rkey     = ucp_rkey_get_tl_rkey(req->send.frag_ppln.remote_rkey,
                                                  lpriv->super.rkey_index);
    size_t max_payload     = ucp_proto_multi_max_payload(req, lpriv, 0);
    size_t length;
    uct_iov_t iov;

    length     = ucp_datatype_iter_next(&req->send.state.dt_iter, max_payload,
                                        next_iter);
    iov.buffer = UCS_PTR_BYTE_OFFSET(mdesc->ptr, req->send.state.dt_iter.offset);
    iov.length = length;
    iov.memh   = mdesc->memh->uct[lpriv->super.md_index];
    iov.stride = 0;
    iov.count  = 1;

    return uct_ep_put_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            &iov, 1, remote_addr, tl_rkey,
                            &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_put_mtype_send_init(ucp_request_t *req)
{
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;

    req->send.multi_lane_idx = req->send.frag_ppln.frag_id % mpriv->num_lanes;
    ucp_proto_multi_set_send_lane(req);
    return UCS_OK;
}

static ucs_status_t
ucp_proto_put_mtype_send_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_multi_priv_t *mpriv = req->send.proto_config->priv;

    /* Remote rkey not yet available: park until RTS_RESP arrives */
    if (req->send.frag_ppln.remote_rkey == UCP_RKEY_INVALID) {
        return UCS_OK;
    }

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_put_mtype_send_completion);
        ucp_proto_put_mtype_send_init(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_multi_progress(req, mpriv,
                                    ucp_proto_put_mtype_send_func,
                                    ucp_proto_put_mtype_data_sent,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
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

/* Sub-IDs for UCP_AM_ID_RMA_PPLN */
enum {
    UCP_RMA_PPLN_AM_RTS,
    UCP_RMA_PPLN_AM_RTS_RESP,
    UCP_RMA_PPLN_AM_ATP,
};

/* Common header for all RMA ppln active messages */
typedef struct {
    ucp_request_hdr_t super;
    uint8_t           sub_id;
} UCS_S_PACKED ucp_rma_ppln_hdr_t;

/* RTS header: sender → receiver, requesting remote bounce buffers */
typedef struct {
    ucp_rma_ppln_hdr_t super;
    int              count;     /* Number of fragments requested */
    ucp_md_map_t     md_map;   /* MD map for fragment registration */
    ucs_sys_device_t sys_dev;  /* System device for fragment allocation */
    size_t           frag_size; /* Fragment size */
} UCS_S_PACKED ucp_put_ppln_rts_hdr_t;

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
    UCP_PROTO_PUT_PPLN_STAGE_RTS = UCP_PROTO_STAGE_START,
    UCP_PROTO_PUT_PPLN_STAGE_SEND,
};

static ucp_md_map_t
ucp_proto_put_ppln_remote_md_map(const ucp_request_t *req,
                                 const ucp_proto_multi_priv_t *mpriv)
{
    ucp_worker_h worker = req->send.ep->worker;
    const ucp_ep_config_key_t *ep_config_key;
    ucp_md_map_t remote_md_map = 0;
    ucp_lane_index_t i, lane;

    ep_config_key = &ucs_array_elem(&worker->ep_config,
                                    req->send.proto_config->ep_cfg_index).key;

    for (i = 0; i < mpriv->num_lanes; i++) {
        lane           = mpriv->lanes[i].super.lane;
        remote_md_map |= UCS_BIT(ep_config_key->lanes[lane].dst_md_index);
    }

    return remote_md_map;
}

static size_t
ucp_proto_put_ppln_rts_pack(void *dest, void *arg)
{
    ucp_put_ppln_rts_hdr_t *rts = dest;
    ucp_request_t *req          = arg;
    const ucp_proto_put_ppln_priv_t *rpriv;
    const ucp_proto_multi_priv_t *mpriv;
    size_t length;

    rpriv  = req->send.proto_config->priv;
    mpriv  = rpriv->frag_proto_cfg.priv;
    length = req->send.state.dt_iter.length;

    rts->super.super.req_id = ucp_send_request_get_id(req);
    rts->super.super.ep_id  = ucp_send_request_get_ep_remote_id(req);
    rts->super.sub_id       = UCP_RMA_PPLN_AM_RTS;
    rts->frag_size          = rpriv->frag_size;
    rts->count              = ucs_div_round_up(length, rpriv->frag_size);
    rts->md_map             = ucp_proto_put_ppln_remote_md_map(req, mpriv);
    rts->sys_dev            = ucp_rkey_config(req->send.ep->worker,
                                  req->send.proto_config->rkey_cfg_index)->key.sys_dev;

    return sizeof(*rts);
}

static ucs_status_t
ucp_proto_put_ppln_rts_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    status = ucp_do_am_single(self, UCP_AM_ID_RMA_PPLN,
                              ucp_proto_put_ppln_rts_pack,
                              sizeof(ucp_put_ppln_rts_hdr_t));
    if (status != UCS_OK) {
        return status;
    }

    ucp_proto_request_set_stage(req, UCP_PROTO_PUT_PPLN_STAGE_SEND);
    return UCS_INPROGRESS;
}

static ucs_status_t
ucp_proto_put_ppln_send_progress(uct_pending_req_t *self)
{
    ucp_request_t *req  = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_worker_h worker = req->send.ep->worker;
    const ucp_proto_put_ppln_priv_t *rpriv;
    ucp_datatype_iter_t next_iter;
    ucp_request_t *freq;
    unsigned num_freqs;
    unsigned frag_idx;
    size_t overlap;

    ucs_assert(req->send.state.dt_iter.length > 0);

    req->send.state.completed_size = 0;
    rpriv     = req->send.proto_config->priv;
    num_freqs = ucs_div_round_up(req->send.state.dt_iter.length,
                                 rpriv->frag_size);

    req->send.ppln.freqs = ucs_malloc(num_freqs * sizeof(*req->send.ppln.freqs),
                                      "put_ppln_freqs");
    if (req->send.ppln.freqs == NULL) {
        ucp_proto_request_abort(req, UCS_ERR_NO_MEMORY);
        return UCS_OK;
    }
    req->send.ppln.num_freqs = num_freqs;

    frag_idx = 0;
    while (!ucp_datatype_iter_is_end(&req->send.state.dt_iter)) {
        freq = ucp_request_get(worker);
        if (freq == NULL) {
            ucp_proto_request_abort(req, UCS_ERR_NO_MEMORY);
            return UCS_OK;
        }

        ucp_proto_request_send_init(freq, req->send.ep, 0);
        ucp_request_set_super(freq, req);

        freq->send.ppln.freqs          = NULL;
        freq->send.ppln.num_freqs      = 0;
        freq->send.frag_ppln.remote_addr  = 0;
        freq->send.frag_ppln.remote_rkey  = UCP_RKEY_INVALID;
        freq->send.frag_ppln.remote_mdesc = NULL;

        /* Allocate local bounce buffer for this fragment */
        freq->send.frag_ppln.local_mdesc = ucp_ppln_mpool_get(
                worker,
                req->send.state.dt_iter.mem_info.type,
                req->send.state.dt_iter.mem_info.sys_dev);
        if (freq->send.frag_ppln.local_mdesc == NULL) {
            ucp_request_put(freq);
            ucp_proto_request_abort(req, UCS_ERR_NO_MEMORY);
            return UCS_OK;
        }

        overlap = ucp_datatype_iter_next_slice_overlap(
                        &req->send.state.dt_iter, rpriv->frag_size,
                        rpriv->frag_proto_min_length,
                        &freq->send.state.dt_iter, &next_iter);
        ucs_assertv(overlap == 0, "overlap=%zu", overlap);
        ucs_assert(freq->send.state.dt_iter.length > 0);

        ucp_proto_request_set_proto(freq, &rpriv->frag_proto_cfg,
                                    freq->send.state.dt_iter.length);

        req->send.ppln.freqs[frag_idx] = freq;
        freq->send.frag_ppln.frag_id = frag_idx++;

        ucp_trace_req(req, "put_ppln_frag frag_id=%d freq=%p offset=%zu "
                      "size=%zu local_mdesc=%p",
                      freq->send.frag_ppln.frag_id, freq,
                      req->send.state.dt_iter.offset,
                      freq->send.state.dt_iter.length,
                      freq->send.frag_ppln.local_mdesc);
        UCS_PROFILE_CALL_VOID_ALWAYS(ucp_request_send, freq);

        ucp_datatype_iter_copy_position(&req->send.state.dt_iter, &next_iter,
                                        UCS_BIT(UCP_DATATYPE_CONTIG));
    }

    ucs_assertv(frag_idx == num_freqs, "frag_idx=%u num_freqs=%u",
                frag_idx, num_freqs);
    return UCS_OK;
}

ucp_proto_t ucp_put_ppln_proto = {
    .name     = "put/ppln",
    .desc     = UCP_PROTO_PPLN_DESC,
    .flags    = 0,
    .probe    = ucp_proto_put_ppln_probe,
    .query    = ucp_proto_put_ppln_query,
    .progress = {
        [UCP_PROTO_PUT_PPLN_STAGE_RTS]  = ucp_proto_put_ppln_rts_progress,
        [UCP_PROTO_PUT_PPLN_STAGE_SEND] = ucp_proto_put_ppln_send_progress,
    },
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};


/*
 * AM handler for UCP_AM_ID_RMA_PPLN — dispatches by sub_id
 */

static ucs_status_t
ucp_rma_ppln_handler(ucp_worker_h worker, void *data, size_t length,
                     unsigned flags)
{
    const ucp_rma_ppln_hdr_t *hdr = data;

    switch (hdr->sub_id) {
    case UCP_RMA_PPLN_AM_RTS:
        return UCS_OK;
    case UCP_RMA_PPLN_AM_RTS_RESP:
        return UCS_OK;
    case UCP_RMA_PPLN_AM_ATP:
        return UCS_OK;
    default:
        ucs_error("rma_ppln: unknown sub_id=%u", hdr->sub_id);
        return UCS_ERR_INVALID_PARAM;
    }
}

static void
ucp_rma_ppln_dump(ucp_worker_h worker, uct_am_trace_type_t type, uint8_t id,
                  const void *data, size_t length, char *buffer, size_t max)
{
    const ucp_rma_ppln_hdr_t *hdr = data;
    UCS_STRING_BUFFER_FIXED(strb, buffer, max);

    switch (hdr->sub_id) {
    case UCP_RMA_PPLN_AM_RTS:
    {
        const ucp_put_ppln_rts_hdr_t *rts = data;
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_RTS ep_id=0x%" PRIx64 " req_id=0x%" PRIx64
                " count=%d frag_size=%zu sys_dev=%u md_map=0x%" PRIx64,
                rts->super.super.ep_id, rts->super.super.req_id,
                rts->count, rts->frag_size, rts->sys_dev,
                (uint64_t)rts->md_map);
        break;
    }
    case UCP_RMA_PPLN_AM_RTS_RESP:
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_RTS_RESP ep_id=0x%" PRIx64 " req_id=0x%" PRIx64,
                hdr->super.ep_id, hdr->super.req_id);
        break;
    case UCP_RMA_PPLN_AM_ATP:
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_ATP ep_id=0x%" PRIx64 " req_id=0x%" PRIx64,
                hdr->super.ep_id, hdr->super.req_id);
        break;
    default:
        ucs_string_buffer_appendf(&strb, "RMA_PPLN unknown sub_id=%u",
                                  hdr->sub_id);
        break;
    }
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_RMA, UCP_AM_ID_RMA_PPLN,
                         ucp_rma_ppln_handler, ucp_rma_ppln_dump, 0);
