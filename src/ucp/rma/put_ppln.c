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
#include <ucp/core/ucp_worker.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_single.inl>
#include <ucp/proto/proto_am.h>
#include <ucp/proto/proto_common.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_perf.h>
#include <ucp/proto/proto_select.h>
#include <ucp/proto/proto_select.inl>


/*
 * ppln fragment pool — lazy-alloc per (mem_type, sys_dev)
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


static UCS_F_ALWAYS_INLINE ucs_memory_type_t
ucp_rma_ppln_frag_mem_type(ucp_context_t *context)
{
    return context->config.ext.ppln_frag_mem_type;
}

static UCS_F_ALWAYS_INLINE ucs_sys_device_t
ucp_rma_ppln_frag_sys_dev(ucp_context_t *context, ucs_sys_device_t sys_dev)
{
    return (ucp_rma_ppln_frag_mem_type(context) == UCS_MEMORY_TYPE_HOST)
            ? UCS_SYS_DEVICE_ID_UNKNOWN : sys_dev;
}


/* Sub-IDs for UCP_AM_ID_RMA_PPLN */
enum {
    UCP_RMA_PPLN_AM_RTS,
    UCP_RMA_PPLN_AM_RTR,
    UCP_RMA_PPLN_AM_ATP,
    UCP_RMA_PPLN_AM_ATS,
};

/* Common header for all RMA ppln active messages */
typedef struct {
    ucp_request_hdr_t super;
    uint8_t           sub_id;
} UCS_S_PACKED ucp_rma_ppln_hdr_t;

/* RTS header: sender to receiver, requesting remote bounce buffers */
typedef struct {
    ucp_rma_ppln_hdr_t super;
    int              frag_count; /* Number of fragments requested */
    ucp_md_map_t     md_map;   /* MD map for fragment registration */
    ucs_sys_device_t sys_dev;  /* System device for fragment allocation */
    size_t           frag_size; /* Fragment size */
    size_t           total_length;  /* Total transfer length */
    uint64_t         remote_addr;   /* Final destination address on receiver */
} UCS_S_PACKED ucp_put_ppln_rts_hdr_t;

/* ATP header: sender to receiver, one per lane, signals partial data arrival */
typedef struct {
    ucp_rma_ppln_hdr_t super;
    ucp_lane_index_t frag_id;       /* Fragment index */
    ucp_lane_index_t lane_id;       /* ATP lane index */
    int              total;         /* Total ATPs for this fragment */
    ucp_mem_desc_t   *remote_mdesc; /* Remote bounce buffer (source for copy-out) */
    size_t           length;        /* Bytes written by this ATP */
} UCS_S_PACKED ucp_put_ppln_atp_hdr_t;

/* RTR entry: one per fragment, variable length (packed rkey follows) */
typedef struct {
    ucp_mem_desc_t *mdesc;          /* Remote bounce buffer descriptor */
    uint64_t       addr;            /* Remote bounce buffer address */
    uint8_t        packed_rkey_len; /* Length of packed rkey that follows */
    uint8_t        packed_rkey[];   /* Packed remote key */
} UCS_S_PACKED ucp_put_ppln_rtr_entry_t;

/* RTR header: receiver to sender, providing remote bounce buffer info */
typedef struct {
    ucp_rma_ppln_hdr_t super;
    ucs_ptr_map_key_t sender_req_id; /* Sender's request ID for lookup */
    int              frag_count;     /* Number of fragments */
} UCS_S_PACKED ucp_put_ppln_rtr_hdr_t;

/* ATS header: receiver to sender, signals all copy-outs are done */
typedef struct {
    ucp_rma_ppln_hdr_t super;
    ucs_ptr_map_key_t sender_req_id; /* Sender's request ID for completion */
} UCS_S_PACKED ucp_put_ppln_ats_hdr_t;

/* Receiver-side per-fragment tracking */
typedef struct {
    ucp_mem_desc_t   *mdesc;        /* Local bounce buffer for this fragment */
    int              atp_count;     /* ATPs received for this fragment */
    int              atp_total;     /* Total ATPs expected for this fragment */
    ucs_queue_elem_t queue;         /* Copy-out ready queue element */
    uct_completion_t comp;          /* Per-frag copy-out completion */
    ucp_request_t    *req;          /* Back-pointer to receiver request */
} ucp_put_ppln_recv_frag_t;

/* Private data for put/ppln and get/ppln protocols */
typedef struct {
    size_t             frag_size;
    ucp_proto_config_t frag_proto_cfg;
    size_t             frag_proto_min_length;
} ucp_proto_ppln_priv_t;


/*
 * put/mtype — copy-in (GPU to bounce) + multi-lane RDMA send (bounce to remote)
 */

enum {
    UCP_PROTO_PUT_MTYPE_STAGE_COPY = UCP_PROTO_STAGE_START,
    UCP_PROTO_PUT_MTYPE_STAGE_SEND,
    UCP_PROTO_PUT_MTYPE_STAGE_FENCED_ATP,
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
        .super.cfg_priority  = 0, /* TODO: Loose against all put/offload* */
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
        !ucp_proto_init_check_op(init_params,
                                 UCS_BIT(UCP_OP_ID_PUT_MTYPE)) ||
        ucp_ep_config_is_self(init_params->ep_config_key) ||
        !UCP_MEM_IS_CUDA(sel_param->mem_type) ||
        ((init_params->rkey_config_key != NULL) &&
         !UCP_MEM_IS_CUDA(init_params->rkey_config_key->mem_type))) {
        return;
    }

    frag_mem_type = ucp_rma_ppln_frag_mem_type(context);
    if (frag_mem_type == UCS_MEMORY_TYPE_LAST) {
        return;
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

/* Called when all ATPs have been sent on all lanes */
static void
ucp_proto_put_mtype_atp_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req   = ucs_container_of(uct_comp, ucp_request_t,
                                            send.state.uct_comp);
    ucp_request_t *super = ucp_request_get_super(req);

    ucp_trace_req(req, "put/mtype: frag_id=%u atp done",
                  req->send.frag_ppln.frag_id);
    super->send.ppln.freqs[req->send.frag_ppln.frag_id] = NULL;
    ucp_rkey_destroy(req->send.frag_ppln.remote_rkey);
    ucp_request_put(req);
}

/* Called when all PUT ZCOPY submissions have completed, release bounce buffer */
static void
ucp_proto_put_mtype_send_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucs_mpool_put_inline(req->send.frag_ppln.local_mdesc);
    /* Reset for lane_map iteration in fenced ATP stage */
    req->send.multi_lane_idx = 0;
    ucp_proto_completion_init(&req->send.state.uct_comp,
                              ucp_proto_put_mtype_atp_completion);
    ucp_request_send(req);
}

/* All data posted on lanes, transition to fence stage, decrement uct completion */
static ucs_status_t
ucp_proto_put_mtype_data_sent(ucp_request_t *req)
{
    ucp_proto_request_set_stage(req, UCP_PROTO_PUT_MTYPE_STAGE_FENCED_ATP);
    return ucp_request_invoke_uct_completion_success(req);
}

static UCS_F_ALWAYS_INLINE ucp_md_index_t
ucp_put_ppln_get_rkey_index(ucp_request_t *req, ucp_rkey_h rkey,
                            ucp_lane_index_t lane)
{
    ucp_ep_config_t *ep_config  = ucp_ep_config(req->send.ep);
    ucp_md_index_t md_index     = ep_config->md_index[lane];
    ucp_md_index_t dst_md_index = ep_config->key.lanes[lane].dst_md_index;

    ucs_assertv_always((UCS_BIT(dst_md_index) & rkey->md_map) &&
                       (md_index != UCP_NULL_RESOURCE),
                       "dst_md_index=%u rkey->md_map=0x%lx md_index=%u",
                       dst_md_index, rkey->md_map, md_index);

    return ucs_bitmap2idx(rkey->md_map, dst_md_index);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_put_mtype_send_func(ucp_request_t *req,
                               const ucp_proto_multi_lane_priv_t *lpriv,
                               ucp_datatype_iter_t *next_iter,
                               ucp_lane_index_t *lane_shift)
{
    ucp_rkey_h rkey        = req->send.frag_ppln.remote_rkey;
    ucp_md_index_t rkey_index = ucp_put_ppln_get_rkey_index(req, rkey,
                                                            lpriv->super.lane);
    ucp_mem_desc_t *mdesc  = req->send.frag_ppln.local_mdesc;
    uint64_t remote_addr   = req->send.frag_ppln.remote_addr +
                             req->send.state.dt_iter.offset;
    uct_rkey_t tl_rkey     = ucp_rkey_get_tl_rkey(rkey, rkey_index);
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

    req->send.frag_ppln.send_lane_map |= UCS_BIT(lpriv->super.lane);

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

    /* Remote rkey not yet available: park until RTR arrives */
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

typedef struct {
    ucp_request_t    *req;
    ucp_lane_index_t lane;
} ucp_proto_put_mtype_atp_pack_ctx_t;

static size_t
ucp_proto_put_mtype_atp_pack(void *dest, void *arg)
{
    ucp_proto_put_mtype_atp_pack_ctx_t *ctx = arg;
    ucp_request_t *req                      = ctx->req;
    ucp_put_ppln_atp_hdr_t *atp             = dest;

    atp->super.super.req_id = ucp_request_get_super(req)->send.ppln.remote_req_id;
    atp->super.super.ep_id  = ucp_send_request_get_ep_remote_id(req);
    atp->super.sub_id       = UCP_RMA_PPLN_AM_ATP;
    atp->frag_id            = req->send.frag_ppln.frag_id;
    atp->lane_id            = ctx->lane;
    atp->total        = ucs_popcount(req->send.frag_ppln.send_lane_map);
    atp->remote_mdesc       = req->send.frag_ppln.remote_mdesc;
    atp->length             = req->send.state.dt_iter.length;

    return sizeof(*atp);
}

static ucs_status_t
ucp_proto_put_mtype_fenced_atp_send(ucp_request_t *req,
                                    ucp_lane_index_t lane)
{
    ucp_proto_put_mtype_atp_pack_ctx_t ctx;
    ucs_status_t status;

    status = uct_ep_fence(ucp_ep_get_lane(req->send.ep, lane), 0);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    ctx.req  = req;
    ctx.lane = lane;

    return ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RMA_PPLN, lane,
                                          ucp_proto_put_mtype_atp_pack, &ctx,
                                          sizeof(ucp_put_ppln_atp_hdr_t), 0);
}

static ucs_status_t
ucp_proto_put_mtype_fenced_atp_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_multi_lane_map_progress(
            req, &req->send.multi_lane_idx,
            req->send.frag_ppln.send_lane_map,
            ucp_proto_put_mtype_fenced_atp_send);
}

ucp_proto_t ucp_put_mtype_proto = {
    .name     = "put/mtype",
    .desc     = UCP_PROTO_PPLN_DESC " " UCP_PROTO_COPY_IN_DESC,
    .flags    = 0,
    .probe    = ucp_proto_put_mtype_probe,
    .query    = ucp_proto_multi_query,
    .progress = {
        [UCP_PROTO_PUT_MTYPE_STAGE_COPY]       = ucp_proto_put_mtype_copy_progress,
        [UCP_PROTO_PUT_MTYPE_STAGE_SEND]       = ucp_proto_put_mtype_send_progress,
        [UCP_PROTO_PUT_MTYPE_STAGE_FENCED_ATP] = ucp_proto_put_mtype_fenced_atp_progress,
    },
    /* TODO: custom abort to destroy remote_rkey and release local_mdesc */
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};


/*
 * put/ppln — orchestrator: bounce buffer lifecycle, pipeline fragment dispatch,
 *            ATP signaling, completion tracking
 */

static ucs_status_t
ucp_proto_ppln_add_overhead(ucp_proto_perf_t *ppln_perf, size_t frag_size)
{
    static const double rts_rtr_overhead = 1000e-9; /* RTS/RTR AM round-trip */
    static const double ats_overhead     = 500e-9; /* ATS one-way (~RTT/2) */
    static const double frag_overhead    = 2000e-9;  /* Per-fragment mgmt */
    ucp_proto_perf_factors_t factors     = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    char frag_str[64];
    ucp_proto_perf_node_t *node;

    ucs_memunits_to_str(frag_size, frag_str, sizeof(frag_str));

    /* Fixed latency: RTS/RTR round-trip + ATS — not pipelined away */
    factors[UCP_PROTO_PERF_FACTOR_LATENCY] =
            ucs_linear_func_make(rts_rtr_overhead + ats_overhead, 0);
    /* Per-fragment management overhead */
    factors[UCP_PROTO_PERF_FACTOR_LOCAL_CPU] =
            ucs_linear_func_make(frag_overhead, frag_overhead);
    node = ucp_proto_perf_node_new_data("ppln overhead",
                                        "rts_rtr + ats + frag %s", frag_str);
    return ucp_proto_perf_add_funcs(ppln_perf, 0, SIZE_MAX, factors,
                                    node, NULL);
}

static void
ucp_proto_ppln_set_frag_proto_config(
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

static int
ucp_proto_ppln_check_common(const ucp_proto_init_params_t *init_params,
                            uint64_t op_id_mask)
{
    const ucp_proto_select_param_t *sel_param = init_params->select_param;

    return (sel_param->dt_class == UCP_DATATYPE_CONTIG) &&
           ucp_proto_init_check_op(init_params, op_id_mask) &&
           !(ucp_proto_select_op_flags(sel_param) &
             UCP_PROTO_SELECT_OP_FLAG_PPLN_FRAG) &&
           !ucp_ep_config_is_self(init_params->ep_config_key) &&
           (init_params->ep_config_key->am_lane != UCP_NULL_LANE) &&
           UCP_MEM_IS_CUDA(sel_param->mem_type) &&
           ((init_params->rkey_config_key == NULL) ||
            UCP_MEM_IS_CUDA(init_params->rkey_config_key->mem_type));
}

static void
ucp_proto_ppln_probe_perf(const ucp_proto_init_params_t *init_params,
                          ucp_operation_id_t frag_op_id)
{
    ucp_worker_h worker                       = init_params->worker;
    const ucp_proto_select_param_t *sel_param = init_params->select_param;
    const ucp_proto_perf_segment_t *frag_seg, *first_seg;
    const ucp_proto_select_elem_t *select_elem;
    UCS_STRING_BUFFER_ONSTACK(seg_strb, 128);
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_param_t frag_sel_param;
    ucp_proto_ppln_priv_t rpriv;
    ucp_proto_select_t *proto_select;
    ucp_proto_perf_t *ppln_perf;
    ucp_proto_init_elem_t *proto;
    char frag_size_str[32];
    void *frag_proto_priv;
    ucs_status_t status;
    uint8_t proto_flags;

    frag_sel_param             = *sel_param;
    frag_sel_param.op_id_flags = frag_op_id;
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
        ucp_proto_ppln_set_frag_proto_config(init_params, proto,
                                             &frag_sel_param,
                                             frag_proto_priv,
                                             &rpriv.frag_proto_cfg);

        ucp_proto_perf_segment_str(frag_seg, &seg_strb);
        ucs_trace("ppln frag=%s proto=%s segment=%s",
                  ucs_memunits_to_str(rpriv.frag_size, frag_size_str,
                                      sizeof(frag_size_str)),
                  ucp_proto_id_field(proto->proto_id, name),
                  ucs_string_buffer_cstr(&seg_strb));

        status = ucp_proto_ppln_add_overhead(ppln_perf, rpriv.frag_size);
        if (status != UCS_OK) {
            goto out_destroy_ppln_perf;
        }

        ucp_proto_select_add_proto(init_params, proto->cfg_thresh,
                                   proto->cfg_priority, ppln_perf, &rpriv,
                                   sizeof(rpriv));
        continue;

    out_destroy_ppln_perf:
        ucp_proto_perf_destroy(ppln_perf);
    }
}

static void
ucp_proto_put_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    if (!ucp_proto_ppln_check_common(init_params, UCS_BIT(UCP_OP_ID_PUT))) {
        return;
    }

    ucp_proto_ppln_probe_perf(init_params, UCP_OP_ID_PUT_MTYPE);
}

static void
ucp_proto_ppln_query(const ucp_proto_query_params_t *params,
                         ucp_proto_query_attr_t *attr)
{
    const ucp_proto_ppln_priv_t *rpriv = params->priv;
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
        ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "multi %s",
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
    const ucp_proto_ppln_priv_t *rpriv;
    const ucp_proto_multi_priv_t *mpriv;
    size_t length;

    rpriv  = req->send.proto_config->priv;
    mpriv  = rpriv->frag_proto_cfg.priv;
    length = req->send.state.dt_iter.length;

    rts->super.super.req_id = ucp_send_request_get_id(req);
    rts->super.super.ep_id  = ucp_send_request_get_ep_remote_id(req);
    rts->super.sub_id       = UCP_RMA_PPLN_AM_RTS;
    rts->frag_size          = rpriv->frag_size;
    rts->frag_count              = ucs_div_round_up(length, rpriv->frag_size);
    rts->total_length       = length;
    rts->remote_addr        = req->send.rma.remote_addr;
    rts->md_map             = ucp_proto_put_ppln_remote_md_map(req, mpriv);
    rts->sys_dev            = ucs_array_elem(&req->send.ep->worker->rkey_config,
                                  req->send.proto_config->rkey_cfg_index).key.sys_dev;

    return sizeof(*rts);
}

static ucs_status_t
ucp_proto_put_ppln_rts_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_send_request_id_alloc(req);
        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

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
    const ucp_proto_ppln_priv_t *rpriv;
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
        freq->send.frag_ppln.remote_addr   = 0;
        freq->send.frag_ppln.remote_rkey   = UCP_RKEY_INVALID;
        freq->send.frag_ppln.remote_mdesc  = NULL;
        freq->send.frag_ppln.send_lane_map = 0;

        /* Allocate local bounce buffer for this fragment */
        freq->send.frag_ppln.local_mdesc = ucp_ppln_mpool_get(
                worker,
                ucp_rma_ppln_frag_mem_type(worker->context),
                ucp_rma_ppln_frag_sys_dev(worker->context,
                                          req->send.state.dt_iter.mem_info.sys_dev));
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
    .query    = ucp_proto_ppln_query,
    .progress = {
        [UCP_PROTO_PUT_PPLN_STAGE_RTS]  = ucp_proto_put_ppln_rts_progress,
        [UCP_PROTO_PUT_PPLN_STAGE_SEND] = ucp_proto_put_ppln_send_progress,
    },
    /* TODO: custom abort to free freqs array and abort fragment requests */
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};


/*
 * get/ppln — receiver side: bounce buffer lifecycle, RTR, copy-out, completion
 *            Also used for user-initiated GET with pipelining.
 *            Performance is symmetric to put/ppln (triggers a remote PUT).
 */

enum {
    UCP_PROTO_GET_PPLN_STAGE_RTR = UCP_PROTO_STAGE_START,
    UCP_PROTO_GET_PPLN_STAGE_COPY_OUT,
    UCP_PROTO_GET_PPLN_STAGE_ATS,
};

static void
ucp_proto_get_ppln_probe(const ucp_proto_init_params_t *init_params)
{
    if (!ucp_proto_ppln_check_common(init_params,
                                     UCS_BIT(UCP_OP_ID_GET) |
                                     UCS_BIT(UCP_OP_ID_GET_PPLN))) {
        return;
    }

    ucp_proto_ppln_probe_perf(init_params, UCP_OP_ID_PUT_MTYPE);
}

static size_t
ucp_rma_ppln_ats_pack(void *dest, void *arg)
{
    ucp_request_t *req        = arg;
    ucp_put_ppln_ats_hdr_t *ats = dest;

    ats->super.super.req_id = ucp_send_request_get_id(req);
    ats->super.super.ep_id  = ucp_ep_remote_id(req->send.ep);
    ats->super.sub_id       = UCP_RMA_PPLN_AM_ATS;
    ats->sender_req_id      = req->send.recv_ppln.sender_req_id;

    return sizeof(*ats);
}

static int
ucp_rma_ppln_frag_copy_out_done(ucp_request_t *req,
                                ucp_put_ppln_recv_frag_t *frag)
{
    int frag_id = (int)(frag -
                        (ucp_put_ppln_recv_frag_t *)req->send.recv_ppln.frags);

    ucs_mpool_put_inline(frag->mdesc);
    frag->mdesc = NULL;

    ucs_trace_req("rma_ppln: frag copy-out done req=%p frag_id=%d", req,
                  frag_id);

    return (--req->send.state.uct_comp.count == 0);
}

static void
ucp_rma_ppln_copy_out_set_ats(ucp_request_t *req)
{
    ucs_trace_req("rma_ppln: all copy-outs done req=%p", req);
    ucp_proto_request_set_stage(req, UCP_PROTO_GET_PPLN_STAGE_ATS);
}

static void
ucp_rma_ppln_frag_copy_out_complete(uct_completion_t *self)
{
    ucp_put_ppln_recv_frag_t *frag =
            ucs_container_of(self, ucp_put_ppln_recv_frag_t, comp);
    ucp_request_t *req = frag->req;

    if (ucp_rma_ppln_frag_copy_out_done(req, frag)) {
        ucp_rma_ppln_copy_out_set_ats(req);
        ucp_request_send(req);
    }
}

static ucs_status_t
ucp_rma_ppln_recv_alloc_bufs(ucp_worker_h worker, ucp_request_t *req)
{
    ucs_memory_type_t mem_type      = ucp_rma_ppln_frag_mem_type(worker->context);
    ucp_put_ppln_recv_frag_t *frags = req->send.recv_ppln.frags;
    int count                       = req->send.recv_ppln.frag_count;
    ucp_mem_desc_t *mdesc;
    int i;

    if (mem_type == UCS_MEMORY_TYPE_LAST) {
        ucs_error("rma_ppln: no ppln_frag_mem_type configured");
        return UCS_ERR_UNSUPPORTED;
    }

    for (i = 0; i < count; i++) {
        mdesc = ucp_ppln_mpool_get(worker, mem_type,
                                   ucp_rma_ppln_frag_sys_dev(
                                           worker->context,
                                           req->send.recv_ppln.sys_dev));
        if (mdesc == NULL) {
            ucs_error("rma_ppln: bounce buffer alloc failed frag=%d", i);
            while (i-- > 0) {
                ucs_mpool_put_inline(frags[i].mdesc);
            }
            return UCS_ERR_NO_MEMORY;
        }

        frags[i].mdesc        = mdesc;
        frags[i].atp_count    = 0;
        frags[i].atp_total    = 0;
        ucp_proto_completion_init(&frags[i].comp,
                                  ucp_rma_ppln_frag_copy_out_complete);
        frags[i].req          = req;

        ucs_trace_req("rma_ppln: alloc frag=%d mdesc=%p addr=%p", i, mdesc,
                      mdesc->ptr);
    }

    req->send.state.uct_comp.func   = NULL;
    req->send.state.uct_comp.count  = count;
    req->send.state.uct_comp.status = UCS_OK;

    return UCS_OK;
}

static ssize_t
ucp_rma_ppln_rtr_serialize(ucp_request_t *req, void *buf, size_t buf_size)
{
    ucp_worker_h worker                     = req->send.ep->worker;
    const ucp_put_ppln_recv_frag_t *frags   = req->send.recv_ppln.frags;
    int count                               = req->send.recv_ppln.frag_count;
    ucp_put_ppln_rtr_hdr_t *rtr = buf;
    ucp_put_ppln_rtr_entry_t *entry;
    ucp_memory_info_t mem_info;
    ssize_t packed_rkey_size;
    ucp_mem_desc_t *mdesc;
    void *p;
    int i;

    rtr->super.super.req_id = ucp_send_request_get_id(req);
    rtr->super.super.ep_id  = ucp_ep_remote_id(req->send.ep);
    rtr->super.sub_id       = UCP_RMA_PPLN_AM_RTR;
    rtr->sender_req_id      = req->send.recv_ppln.sender_req_id;
    rtr->frag_count              = count;

    p = rtr + 1;

    for (i = 0; i < count; i++) {
        mdesc = frags[i].mdesc;

        entry = p;
        entry->mdesc = mdesc;
        entry->addr  = (uint64_t)mdesc->ptr;

        mem_info.type    = mdesc->memh->mem_type;
        mem_info.sys_dev = UCS_SYS_DEVICE_ID_UNKNOWN;

        packed_rkey_size = ucp_rkey_pack_memh(
                worker->context, req->send.recv_ppln.md_map,
                mdesc->memh, mdesc->ptr,
                req->send.recv_ppln.frag_size, &mem_info,
                0, NULL, 0, 0, entry->packed_rkey);
        if (packed_rkey_size < 0) {
            ucs_error("rma_ppln: rkey pack failed frag=%d", i);
            return packed_rkey_size;
        }

        ucs_assertv(packed_rkey_size <= UINT8_MAX,
                    "packed_rkey_size=%zd", packed_rkey_size);
        entry->packed_rkey_len = (uint8_t)packed_rkey_size;
        p = UCS_PTR_BYTE_OFFSET(p, sizeof(*entry) + packed_rkey_size);

        ucs_trace_req("rma_ppln: RTR frag=%d/%d mdesc=%p addr=0x%" PRIx64
                      " mem_type=%s rkey_len=%u md_map=0x%" PRIx64,
                      i, count, mdesc, entry->addr,
                      ucs_memory_type_names[mdesc->memh->mem_type],
                      entry->packed_rkey_len,
                      (uint64_t)req->send.recv_ppln.md_map);
    }

    /* TODO: handle splitting RTR across multiple bcopy messages */
    ucs_assertv((size_t)UCS_PTR_BYTE_DIFF(buf, p) <= buf_size,
                "RTR packed size %zu exceeds max_bcopy %zu",
                (size_t)UCS_PTR_BYTE_DIFF(buf, p), buf_size);

    return (ssize_t)UCS_PTR_BYTE_DIFF(buf, p);
}

static size_t
ucp_rma_ppln_rtr_pack(void *dest, void *arg)
{
    ucp_request_t *req         = arg;
    ucp_lane_index_t am_lane   = ucp_ep_config(req->send.ep)->key.am_lane;
    size_t max_bcopy           = ucp_ep_get_iface_attr(req->send.ep,
                                                       am_lane)->cap.am.max_bcopy;
    ssize_t packed;

    packed = ucp_rma_ppln_rtr_serialize(req, dest, max_bcopy);
    ucs_assertv(packed > 0, "RTR serialize failed packed=%zd", packed);
    return (size_t)packed;
}

static ucs_status_t
ucp_proto_get_ppln_rtr_complete(ucp_request_t *req)
{
    /* Park; ATP handler will re-enqueue for copy-out */
    ucp_proto_request_set_stage(req, UCP_PROTO_GET_PPLN_STAGE_COPY_OUT);
    return UCS_OK;
}

static ucs_status_t
ucp_proto_get_ppln_rtr_progress(uct_pending_req_t *self)
{
    ucp_request_t *req              = ucs_container_of(self, ucp_request_t,
                                                       send.uct);
    ucp_worker_h worker             = req->send.ep->worker;
    ucp_put_ppln_recv_frag_t *frags;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        ucp_send_request_id_alloc(req);

        frags = ucs_malloc(req->send.recv_ppln.frag_count * sizeof(*frags),
                           "ppln_recv_frags");
        if (frags == NULL) {
            ucp_proto_request_abort(req, UCS_ERR_NO_MEMORY);
            return UCS_OK;
        }

        req->send.recv_ppln.frags = frags;
        ucs_queue_head_init(&req->send.recv_ppln.copy_out_queue);

        status = ucp_rma_ppln_recv_alloc_bufs(worker, req);
        if (status != UCS_OK) {
            ucs_free(frags);
            req->send.recv_ppln.frags = NULL;
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_RMA_PPLN,
            ucp_ep_config(req->send.ep)->key.am_lane,
            ucp_rma_ppln_rtr_pack, req, SIZE_MAX,
            ucp_proto_get_ppln_rtr_complete, 0);
}

static ucs_status_t
ucp_proto_get_ppln_copy_out_progress(uct_pending_req_t *self)
{
    ucp_request_t *req              = ucs_container_of(self, ucp_request_t,
                                                       send.uct);
    ucp_worker_h worker             = req->send.ep->worker;
    ucp_put_ppln_recv_frag_t *frags = req->send.recv_ppln.frags;
    size_t frag_size                 = req->send.recv_ppln.frag_size;
    size_t total_length              = req->send.recv_ppln.total_length;
    ucp_put_ppln_recv_frag_t *frag;
    ucp_ep_h mem_type_ep;
    ucp_lane_index_t lane;
    uct_iov_t iov;
    ucs_status_t status;
    uint64_t dest_addr;
    size_t offset;
    size_t length;
    int frag_id;

    while (!ucs_queue_is_empty(&req->send.recv_ppln.copy_out_queue)) {
        frag    = ucs_queue_head_elem_non_empty(
                &req->send.recv_ppln.copy_out_queue,
                ucp_put_ppln_recv_frag_t, queue);
        frag_id = (int)(frag - frags);

        offset    = (size_t)frag_id * frag_size;
        dest_addr = req->send.recv_ppln.remote_addr + offset;
        length    = ucs_min(frag_size, total_length - offset);

        ucs_assertv((frag_id < req->send.recv_ppln.frag_count - 1)
                            ? (length == frag_size)
                            : (length <= frag_size),
                    "frag_id=%d length=%zu frag_size=%zu frag_count=%d",
                    frag_id, length, frag_size,
                    req->send.recv_ppln.frag_count);

        mem_type_ep = worker->mem_type_ep[req->send.proto_config->select_param.mem_type];
        ucs_assert(mem_type_ep != NULL);
        lane        = ucp_ep_config(mem_type_ep)->key.rma_bw_lanes[0];
        ucs_assert(lane != UCP_NULL_LANE);

        iov.buffer = frag->mdesc->ptr;
        iov.length = length;
        iov.memh   = frag->mdesc->memh->uct[ucp_ep_md_index(mem_type_ep, lane)];
        iov.stride = 0;
        iov.count  = 1;

        status = uct_ep_put_zcopy(ucp_ep_get_lane(mem_type_ep, lane),
                                  &iov, 1, dest_addr, UCT_INVALID_RKEY,
                                  &frag->comp);
        if (status == UCS_ERR_NO_RESOURCE) {
            return UCS_ERR_NO_RESOURCE;
        }

        ucs_queue_pull_non_empty(&req->send.recv_ppln.copy_out_queue);

        ucs_trace_req("rma_ppln: copy-out req=%p frag_id=%d src=%p"
                      " dst=0x%" PRIx64 " length=%zu",
                      req, frag_id, frag->mdesc->ptr, dest_addr, length);

        if (status == UCS_OK) {
            ucp_rma_ppln_frag_copy_out_done(req, frag);
        } else if (ucs_unlikely(status != UCS_INPROGRESS)) {
            ucs_error("rma_ppln: copy-out failed req=%p frag_id=%d status=%s",
                      req, frag_id, ucs_status_string(status));
        }
    }

    if (req->send.state.uct_comp.count == 0) {
        ucp_rma_ppln_copy_out_set_ats(req);
        return UCS_INPROGRESS;
    }

    return UCS_OK;
}

static ucs_status_t
ucp_proto_get_ppln_ats_complete(ucp_request_t *req)
{
    ucs_trace_req("rma_ppln: ATS sent req=%p, cleaning up receiver", req);
    ucs_assert(ucs_queue_is_empty(&req->send.recv_ppln.copy_out_queue));
    ucp_send_request_id_release(req);
    ucs_free(req->send.recv_ppln.frags);
    ucp_request_put(req);
    return UCS_OK;
}

static ucs_status_t
ucp_proto_get_ppln_ats_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    return ucp_proto_am_bcopy_single_progress(
            req, UCP_AM_ID_RMA_PPLN,
            ucp_ep_config(req->send.ep)->key.am_lane,
            ucp_rma_ppln_ats_pack, req, SIZE_MAX,
            ucp_proto_get_ppln_ats_complete, 0);
}

static void
ucp_proto_get_ppln_query(const ucp_proto_query_params_t *params,
                         ucp_proto_query_attr_t *attr)
{
    const ucp_proto_ppln_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t frag_attr;

    if (params->msg_length <= rpriv->frag_size) {
        ucp_proto_config_query(params->worker, &rpriv->frag_proto_cfg,
                               params->msg_length, &frag_attr);
        attr->max_msg_length = rpriv->frag_size;
        attr->is_estimation  = 0;
        attr->lane_map       = frag_attr.lane_map;
        ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "%s %s",
                          UCP_PROTO_PPLN_DESC, UCP_PROTO_COPY_OUT_DESC);
        ucs_strncpy_safe(attr->config, frag_attr.config, sizeof(attr->config));
    } else {
        ucp_proto_config_query(params->worker, &rpriv->frag_proto_cfg,
                               rpriv->frag_size, &frag_attr);
        attr->max_msg_length = SIZE_MAX;
        attr->is_estimation  = 0;
        attr->lane_map       = frag_attr.lane_map;
        ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "multi %s %s",
                          UCP_PROTO_PPLN_DESC, UCP_PROTO_COPY_OUT_DESC);
        ucs_strncpy_safe(attr->config, frag_attr.config, sizeof(attr->config));
    }
}

ucp_proto_t ucp_get_ppln_proto = {
    .name     = "get/ppln",
    .desc     = UCP_PROTO_PPLN_DESC " " UCP_PROTO_COPY_OUT_DESC,
    .flags    = 0,
    .probe    = ucp_proto_get_ppln_probe,
    .query    = ucp_proto_get_ppln_query,
    .progress = {
        [UCP_PROTO_GET_PPLN_STAGE_RTR]      = ucp_proto_get_ppln_rtr_progress,
        [UCP_PROTO_GET_PPLN_STAGE_COPY_OUT] = ucp_proto_get_ppln_copy_out_progress,
        [UCP_PROTO_GET_PPLN_STAGE_ATS]      = ucp_proto_get_ppln_ats_progress,
    },
    /* TODO: custom abort to free frags, return bounce buffers, release req_id */
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_offload_zcopy_reset
};

static ucs_status_t
ucp_rma_ppln_atp_handler(ucp_worker_h worker,
                          const ucp_put_ppln_atp_hdr_t *atp)
{
    ucp_request_t *req;
    ucp_put_ppln_recv_frag_t *frag;
    int was_empty;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, atp->super.super.req_id, 0,
                               return UCS_OK, "rma_ppln ATP");

    ucs_assertv(atp->frag_id < req->send.recv_ppln.frag_count,
                "frag_id=%u frag_count=%d", atp->frag_id,
                req->send.recv_ppln.frag_count);

    frag = &((ucp_put_ppln_recv_frag_t *)req->send.recv_ppln.frags)[atp->frag_id];
    frag->atp_total = atp->total;
    frag->atp_count++;

    ucs_trace_req("rma_ppln: ATP received req=%p frag_id=%u lane_id=%u"
                  " atp=%d/%d length=%zu remote_mdesc=%p",
                  req, atp->frag_id, atp->lane_id,
                  frag->atp_count, frag->atp_total, atp->length,
                  atp->remote_mdesc);

    if (frag->atp_count == frag->atp_total) {
        was_empty = ucs_queue_is_empty(&req->send.recv_ppln.copy_out_queue);
        ucs_queue_push(&req->send.recv_ppln.copy_out_queue, &frag->queue);
        if (was_empty) {
            ucp_request_send(req);
        }
    }

    return UCS_OK;
}

static ucs_status_t
ucp_rma_ppln_rts_handler(ucp_worker_h worker,
                         const ucp_put_ppln_rts_hdr_t *rts)
{
    ucp_proto_select_param_t sel_param;
    ucp_memory_info_t mem_info;
    ucp_request_t *req;
    ucs_status_t status;
    ucp_ep_h ep;

    UCP_WORKER_GET_EP_BY_ID(&ep, worker, rts->super.super.ep_id,
                            return UCS_OK, "rma_ppln RTS");

    ucs_trace_req("rma_ppln: RTS received ep=%p sender_req_id=0x%" PRIx64
                  " count=%d frag_size=%zu total_length=%zu"
                  " remote_addr=0x%" PRIx64 " md_map=0x%" PRIx64
                  " sys_dev=%u",
                  ep, rts->super.super.req_id, rts->frag_count, rts->frag_size,
                  rts->total_length, rts->remote_addr,
                  (uint64_t)rts->md_map, rts->sys_dev);

    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_error("rma_ppln: failed to allocate receiver request");
        return UCS_ERR_NO_MEMORY;
    }

    ucp_proto_request_send_init(req, ep, 0);

    req->send.recv_ppln.frag_count         = rts->frag_count;
    req->send.recv_ppln.sender_req_id = rts->super.super.req_id;
    req->send.recv_ppln.md_map        = rts->md_map;
    req->send.recv_ppln.sys_dev       = rts->sys_dev;
    req->send.recv_ppln.frag_size     = rts->frag_size;
    req->send.recv_ppln.remote_addr   = rts->remote_addr;
    req->send.recv_ppln.total_length  = rts->total_length;

    mem_info.type    = UCS_MEMORY_TYPE_CUDA;
    mem_info.sys_dev = worker->context->alloc_md[UCS_MEMORY_TYPE_CUDA].sys_dev;

    ucp_proto_select_param_init(&sel_param, UCP_OP_ID_GET_PPLN, 0, 0,
                                UCP_DATATYPE_CONTIG, &mem_info, 1);

    status = ucp_proto_request_lookup_proto(
            worker, ep, req, &ucp_ep_config(ep)->proto_select,
            UCP_WORKER_CFG_INDEX_NULL, &sel_param,
            rts->total_length);
    if (status != UCS_OK) {
        ucp_request_put(req);
        return status;
    }

    ucp_request_send(req);
    return UCS_OK;
}

static ucs_status_t
ucp_rma_ppln_rtr_unpack_frags(ucp_request_t *req,
                              const ucp_put_ppln_rtr_hdr_t *rtr,
                              size_t rtr_length)
{
    ucp_ep_h ep = req->send.ep;
    const ucp_put_ppln_rtr_entry_t *entry;
    ucp_request_t *freq;
    ucs_status_t status;
    const void *p;
    int i;

    ucs_assertv(rtr->frag_count == (int)req->send.ppln.num_freqs,
                "RTR count=%d num_freqs=%u", rtr->frag_count,
                req->send.ppln.num_freqs);

    p = rtr + 1;

    for (i = 0; i < rtr->frag_count; i++) {
        entry = p;
        freq  = req->send.ppln.freqs[i];

        ucs_assert(freq->send.frag_ppln.remote_rkey == UCP_RKEY_INVALID);

        status = ucp_ep_rkey_unpack_reachable(ep, entry->packed_rkey,
                                              entry->packed_rkey_len,
                                              &freq->send.frag_ppln.remote_rkey);
        if (status != UCS_OK) {
            ucs_error("rma_ppln: rkey unpack failed frag=%d status=%s",
                      i, ucs_status_string(status));
            return status;
        }

        freq->send.frag_ppln.remote_addr  = entry->addr;
        freq->send.frag_ppln.remote_mdesc = entry->mdesc;

        ucs_trace_req("rma_ppln: RTR frag=%d freq=%p remote_addr=0x%" PRIx64
                      " remote_mdesc=%p rkey_len=%u",
                      i, freq, entry->addr, entry->mdesc,
                      entry->packed_rkey_len);

        p = UCS_PTR_BYTE_OFFSET(p, sizeof(*entry) + entry->packed_rkey_len);
    }

    ucs_assertv(p == UCS_PTR_BYTE_OFFSET(rtr, rtr_length),
                "RTR parsed %zu bytes, expected %zu",
                (size_t)UCS_PTR_BYTE_DIFF(rtr, p), rtr_length);

    return UCS_OK;
}

static ucs_status_t
ucp_rma_ppln_rtr_handler(ucp_worker_h worker,
                         const ucp_put_ppln_rtr_hdr_t *rtr,
                         size_t rtr_length)
{
    ucp_request_t *req;
    ucp_request_t *freq;
    ucs_status_t status;
    unsigned i;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, rtr->sender_req_id, 0,
                               return UCS_OK, "rma_ppln RTR");

    ucs_trace_req("rma_ppln: RTR received req=%p sender_req_id=0x%" PRIx64
                  " recv_req_id=0x%" PRIx64 " count=%d",
                  req, rtr->sender_req_id, rtr->super.super.req_id,
                  rtr->frag_count);

    req->send.ppln.remote_req_id = rtr->super.super.req_id;

    status = ucp_rma_ppln_rtr_unpack_frags(req, rtr, rtr_length);
    if (status != UCS_OK) {
        return status;
    }

    for (i = 0; i < req->send.ppln.num_freqs; i++) {
        freq = req->send.ppln.freqs[i];
        if (freq->send.proto_stage == UCP_PROTO_PUT_MTYPE_STAGE_SEND) {
            ucp_request_send(freq);
        }
    }

    return UCS_OK;
}

static ucs_status_t
ucp_rma_ppln_ats_handler(ucp_worker_h worker,
                         const ucp_put_ppln_ats_hdr_t *ats)
{
    ucp_request_t *req;

    UCP_SEND_REQUEST_GET_BY_ID(&req, worker, ats->sender_req_id, 0,
                               return UCS_OK, "rma_ppln ATS");

    ucs_trace_req("rma_ppln: ATS received req=%p sender_req_id=0x%" PRIx64,
                  req, ats->sender_req_id);

    ucp_send_request_id_release(req);
    ucs_free(req->send.ppln.freqs);
    req->send.ppln.freqs = NULL;
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, UCP_DT_MASK_ALL);
    ucp_request_complete_send(req, UCS_OK);

    return UCS_OK;
}

static ucs_status_t
ucp_rma_ppln_handler(void *arg, void *data, size_t length,
                     unsigned flags)
{
    ucp_worker_h worker           = arg;
    const ucp_rma_ppln_hdr_t *hdr = data;

    switch (hdr->sub_id) {
    case UCP_RMA_PPLN_AM_RTS:
        return ucp_rma_ppln_rts_handler(worker,
                                        (const ucp_put_ppln_rts_hdr_t *)data);
    case UCP_RMA_PPLN_AM_RTR:
        return ucp_rma_ppln_rtr_handler(worker,
                                        (const ucp_put_ppln_rtr_hdr_t *)data,
                                        length);
    case UCP_RMA_PPLN_AM_ATP:
        return ucp_rma_ppln_atp_handler(worker,
                                        (const ucp_put_ppln_atp_hdr_t *)data);
    case UCP_RMA_PPLN_AM_ATS:
        return ucp_rma_ppln_ats_handler(worker,
                                        (const ucp_put_ppln_ats_hdr_t *)data);
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
    const ucp_put_ppln_rts_hdr_t *rts;
    const ucp_put_ppln_rtr_hdr_t *rtr;
    const ucp_put_ppln_atp_hdr_t *atp;
    const ucp_put_ppln_ats_hdr_t *ats;
    UCS_STRING_BUFFER_FIXED(strb, buffer, max);

    switch (hdr->sub_id) {
    case UCP_RMA_PPLN_AM_RTS:
        rts = data;
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_RTS ep_id=0x%" PRIx64 " req_id=0x%" PRIx64
                " frag_count=%d frag_size=%zu total_length=%zu"
                " remote_addr=0x%" PRIx64
                " sys_dev=%u md_map=0x%" PRIx64,
                rts->super.super.ep_id, rts->super.super.req_id,
                rts->frag_count, rts->frag_size, rts->total_length,
                rts->remote_addr, rts->sys_dev, (uint64_t)rts->md_map);
        break;
    case UCP_RMA_PPLN_AM_RTR:
        rtr = data;
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_RTR ep_id=0x%" PRIx64 " req_id=0x%" PRIx64
                " sender_req_id=0x%" PRIx64 " frag_count=%d",
                rtr->super.super.ep_id, rtr->super.super.req_id,
                rtr->sender_req_id, rtr->frag_count);
        break;
    case UCP_RMA_PPLN_AM_ATP:
        atp = data;
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_ATP ep_id=0x%" PRIx64 " req_id=0x%" PRIx64
                " frag_id=%u lane_id=%u total_count=%d"
                " remote_mdesc=%p length=%zu",
                atp->super.super.ep_id, atp->super.super.req_id,
                atp->frag_id, atp->lane_id, atp->total,
                atp->remote_mdesc, atp->length);
        break;
    case UCP_RMA_PPLN_AM_ATS:
        ats = data;
        ucs_string_buffer_appendf(&strb,
                "RMA_PPLN_ATS ep_id=0x%" PRIx64 " req_id=0x%" PRIx64
                " sender_req_id=0x%" PRIx64,
                ats->super.super.ep_id, ats->super.super.req_id,
                ats->sender_req_id);
        break;
    default:
        ucs_string_buffer_appendf(&strb, "RMA_PPLN unknown sub_id=%u",
                                  hdr->sub_id);
        break;
    }
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_RMA, UCP_AM_ID_RMA_PPLN,
                         ucp_rma_ppln_handler, ucp_rma_ppln_dump, 0);
