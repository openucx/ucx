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

#include <ucp/core/ucp_request.inl>
#include <ucp/dt/datatype_iter.inl>
#include <ucp/proto/proto.h>
#include <ucp/proto/proto_debug.h>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.inl>
#include <ucp/rndv/proto_rndv.inl>


#define UCP_PROTO_RMA_RNDV_PUT_RTS_NAME "RMA PUT RTS"
#define UCP_PROTO_RMA_RNDV_GET_REQ_NAME "RMA GET REQ"
#define UCP_PROTO_RMA_RNDV_DESC         "rendezvous"
#define UCP_PROTO_RMA_RNDV_PUSH_DESC    "rendezvous push"


enum {
    UCP_RMA_RNDV_AM_PUT_RTS,
    UCP_RMA_RNDV_AM_GET_REQ,
    UCP_RMA_RNDV_AM_GET_RTS
};


typedef struct {
    ucp_rndv_rts_hdr_t super;
    uint64_t           address;
    ucs_sys_device_t   sys_dev;
    ucs_memory_type_t  mem_type;
} UCS_S_PACKED ucp_rma_rndv_put_rts_hdr_t;


typedef struct {
    uint64_t          hdr;
    ucp_request_hdr_t req;
    uint64_t          address;
    size_t            size;
    ucs_sys_device_t  sys_dev;
    ucs_memory_type_t mem_type;
} UCS_S_PACKED ucp_rma_rndv_get_req_hdr_t;


typedef struct {
    ucp_rndv_rts_hdr_t super;
    ucs_ptr_map_key_t  get_req_id;
} UCS_S_PACKED ucp_rma_rndv_get_rts_hdr_t;


static void
ucp_rma_rndv_dt_iter_init(ucp_datatype_iter_t *dt_iter, uint64_t address,
                          size_t length, ucs_memory_type_t mem_type,
                          ucs_sys_device_t sys_dev)
{
    dt_iter->dt_class           = UCP_DATATYPE_CONTIG;
    dt_iter->mem_info.type      = mem_type;
    dt_iter->mem_info.sys_dev   = sys_dev;
    dt_iter->length             = length;
    dt_iter->offset             = 0;
    dt_iter->type.contig.buffer = (void*)(uintptr_t)address;
    dt_iter->type.contig.memh   = NULL;
}

static int
ucp_proto_rma_rndv_probe_check(const ucp_proto_init_params_t *init_params,
                               ucp_operation_id_t op_id)
{
    const ucp_proto_select_param_t *sel_param = init_params->select_param;

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(op_id)) ||
        ucp_proto_rndv_init_params_is_ppln_frag(init_params) ||
        (sel_param->dt_class != UCP_DATATYPE_CONTIG) ||
        (init_params->rkey_config_key == NULL)) {
        return 0;
    }

    return !UCP_MEM_IS_HOST(sel_param->mem_type) ||
           !UCP_MEM_IS_HOST(init_params->rkey_config_key->mem_type);
}


static size_t ucp_proto_put_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *req              = arg;
    ucp_rma_rndv_put_rts_hdr_t *rts = dest;
    ucp_rkey_config_t *rkey_config;

    rkey_config = ucp_rkey_config(req->send.ep->worker, req->send.rma.rkey);

    rts->super.hdr    = UCP_RMA_RNDV_AM_PUT_RTS;
    rts->super.opcode = UCP_RNDV_RTS_TAG_OK;
    rts->address      = req->send.rma.remote_addr;
    rts->sys_dev      = rkey_config->key.sys_dev;
    rts->mem_type     = req->send.rma.rkey->mem_type;

    return ucp_proto_rndv_rts_pack(req, &rts->super, sizeof(*rts));
}

static ucs_status_t ucp_proto_put_rndv_init(ucp_request_t *req)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    int was_initialized;
    ucs_status_t status;

    was_initialized = req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    status          = ucp_proto_rndv_rts_request_init(req);
    if ((status != UCS_OK) || was_initialized) {
        return status;
    }

    /* Nested RNDV data protocols are not RMA protocols, so the wrapper handles
     * RMA fence ordering before exposing the operation to the peer. */
    return ucp_ep_rma_handle_fence(req->send.ep, req, UCS_BIT(rpriv->lane));
}

static ucs_status_t ucp_proto_put_rndv_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    size_t max_rts_size;
    ucs_status_t status;
    ucp_ep_h ep;

    status = ucp_proto_put_rndv_init(req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    ep           = req->send.ep;
    rpriv        = req->send.proto_config->priv;
    max_rts_size = sizeof(ucp_rma_rndv_put_rts_hdr_t) +
                   rpriv->packed_rkey_size;

    ucp_worker_flush_ops_count_add(ep->worker, +1);
    status = ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RMA_RNDV,
                                            rpriv->lane,
                                            ucp_proto_put_rndv_rts_pack, req,
                                            max_rts_size, 0);
    if (status == UCS_ERR_NO_RESOURCE) {
        ucp_worker_flush_ops_count_add(ep->worker, -1);
        req->send.lane = rpriv->lane;
        return status;
    } else if (status != UCS_OK) {
        ucp_worker_flush_ops_count_add(ep->worker, -1);
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    ucp_ep_rma_remote_request_sent(ep);
    return UCS_OK;
}

static void
ucp_proto_put_rndv_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rts,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 5,
        .super.min_length    = 1,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_rma_rndv_put_rts_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        /* For performance modeling, this control protocol is followed on the
         * peer by a regular RNDV receive flow over the final RMA address. */
        .remote_op_id        = UCP_OP_ID_RNDV_RECV,
        .lane                = ucp_proto_rndv_find_ctrl_lane(init_params),
        .unpack_perf         = NULL,
        .perf_bias           = 0,
        .ctrl_msg_name       = UCP_PROTO_RMA_RNDV_PUT_RTS_NAME,
        .md_map              = 0
    };
    ucp_proto_rndv_ctrl_priv_t rpriv = {0};

    if (!ucp_proto_rma_rndv_probe_check(init_params, UCP_OP_ID_PUT)) {
        return;
    }

    ucp_proto_rndv_ctrl_probe(&params, &rpriv, sizeof(rpriv));
}

static void
ucp_rma_rndv_send_ats_err(ucp_ep_h ep, ucs_ptr_map_key_t remote_req_id,
                          ucs_status_t status)
{
    ucp_request_t *req;

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        ucs_error("failed to allocate RMA RNDV error ATS");
        return;
    }

    ucp_proto_request_send_init(req, ep, 0);
    ucp_rndv_req_send_ack(req, 0, remote_req_id, status, UCP_AM_ID_RNDV_ATS,
                          "send_ats_err");
}

static void ucp_proto_rma_rndv_query(const ucp_proto_query_params_t *params,
                                     ucp_proto_query_attr_t *attr,
                                     const char *desc)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = params->priv;
    ucp_proto_query_attr_t remote_attr;

    ucp_proto_config_query(params->worker, &rpriv->remote_proto_config,
                           params->msg_length, &remote_attr);

    attr->is_estimation  = 1;
    attr->max_msg_length = remote_attr.max_msg_length;
    attr->lane_map       = UCS_BIT(rpriv->lane) | remote_attr.lane_map;

    ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "%s using %s", desc,
                      remote_attr.desc);
    ucs_snprintf_safe(attr->config, sizeof(attr->config), "%s",
                      remote_attr.config);
}

static void
ucp_proto_rma_rndv_default_query(const ucp_proto_query_params_t *params,
                                 ucp_proto_query_attr_t *attr)
{
    ucp_proto_rma_rndv_query(params, attr, UCP_PROTO_RMA_RNDV_DESC);
}

static void
ucp_proto_get_rndv_push_query(const ucp_proto_query_params_t *params,
                              ucp_proto_query_attr_t *attr)
{
    ucp_proto_rma_rndv_query(params, attr,
                             UCP_PROTO_RMA_RNDV_PUSH_DESC);
}

static size_t ucp_proto_get_rndv_req_pack(void *dest, void *arg)
{
    ucp_request_t *req              = arg;
    ucp_rma_rndv_get_req_hdr_t *hdr = dest;
    ucp_rkey_config_t *rkey_config;

    rkey_config = ucp_rkey_config(req->send.ep->worker, req->send.rma.rkey);

    hdr->hdr        = UCP_RMA_RNDV_AM_GET_REQ;
    hdr->req.ep_id  = ucp_send_request_get_ep_remote_id(req);
    hdr->req.req_id = ucp_send_request_get_id(req);
    hdr->address    = req->send.rma.remote_addr;
    hdr->size       = req->send.state.dt_iter.length;
    hdr->sys_dev    = rkey_config->key.sys_dev;
    hdr->mem_type   = req->send.rma.rkey->mem_type;

    return sizeof(*hdr);
}

static ucs_status_t ucp_proto_get_rndv_init(ucp_request_t *req)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        return UCS_OK;
    }

    status = ucp_ep_resolve_remote_id(req->send.ep, rpriv->lane);
    if (status != UCS_OK) {
        return status;
    }

    req->send.buffer = req->send.state.dt_iter.type.contig.buffer;
    req->send.length = req->send.state.dt_iter.length;
    ucp_send_request_id_alloc(req);
    req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;

    /* The nested RNDV receive starts only after GET_RTS; this wrapper still
     * has to respect RMA fence ordering before the target can expose data. */
    return ucp_ep_rma_handle_fence(req->send.ep, req, UCS_BIT(rpriv->lane));
}

static ucs_status_t ucp_proto_get_rndv_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    ucs_status_t status;

    status = ucp_proto_get_rndv_init(req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    rpriv  = req->send.proto_config->priv;
    status = ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RMA_RNDV,
                                            rpriv->lane,
                                            ucp_proto_get_rndv_req_pack, req,
                                            sizeof(ucp_rma_rndv_get_req_hdr_t),
                                            0);
    if (status == UCS_ERR_NO_RESOURCE) {
        req->send.lane = rpriv->lane;
        return status;
    } else if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    return UCS_OK;
}

static void
ucp_proto_get_rndv_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = context->config.ext.proto_overhead_rndv_rts,
        .super.cfg_thresh    = context->config.ext.zcopy_thresh,
        .super.cfg_priority  = 5,
        .super.min_length    = 1,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 1,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t, cap.am.max_bcopy),
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = sizeof(ucp_rma_rndv_get_req_hdr_t),
        .super.send_op       = UCT_EP_OP_AM_BCOPY,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = ucp_proto_common_select_param_mem_info(
                                                     init_params->select_param),
        /* The peer turns GET_REQ into a synthetic RNDV sender. */
        .remote_op_id        = UCP_OP_ID_RNDV_SEND,
        .lane                = ucp_proto_rndv_find_ctrl_lane(init_params),
        .unpack_perf         = NULL,
        .perf_bias           = 0,
        .ctrl_msg_name       = UCP_PROTO_RMA_RNDV_GET_REQ_NAME,
        .md_map              = 0
    };
    ucp_proto_rndv_ctrl_priv_t rpriv = {0};

    if (!ucp_proto_rma_rndv_probe_check(init_params, UCP_OP_ID_GET)) {
        return;
    }

    ucp_proto_rndv_ctrl_probe(&params, &rpriv, sizeof(rpriv));
}

static void
ucp_proto_get_rndv_push_add_variant(
        const ucp_proto_init_params_t *init_params,
        const ucp_proto_select_param_t *select_param,
        ucp_worker_cfg_index_t rkey_cfg_index, ucp_lane_index_t lane,
        ucp_proto_init_elem_t *proto, const void *proto_priv)
{
    ucp_context_h context = init_params->worker->context;
    const ucp_proto_perf_t *perf_elems[1];
    ucp_proto_rndv_ctrl_priv_t rpriv = {0};
    ucp_proto_init_params_t variant_params;
    UCS_STRING_BUFFER_ONSTACK(perf_name, 128);
    ucp_proto_perf_t *perf;
    size_t cfg_thresh;
    ucs_status_t status;

    perf_elems[0] = proto->perf;
    ucs_string_buffer_appendf(&perf_name, "%s" UCP_PROTO_PERF_NODE_NEW_LINE
                              "%s", UCP_PROTO_RNDV_RTR_REQ_NAME,
                              ucp_proto_perf_name(proto->perf));
    status = ucp_proto_perf_aggregate(ucs_string_buffer_cstr(&perf_name),
                                      perf_elems, 1, &perf);
    if (status != UCS_OK) {
        return;
    } else if (ucp_proto_perf_is_empty(perf)) {
        ucp_proto_perf_destroy(perf);
        return;
    }

    rpriv.lane = lane;

    variant_params                = *init_params;
    variant_params.rkey_cfg_index = rkey_cfg_index;
    ucp_proto_rndv_set_variant_config(&variant_params, proto, select_param,
                                      proto_priv, &rpriv.remote_proto_config);

    cfg_thresh = context->config.ext.zcopy_thresh;
    if (proto->cfg_thresh != UCS_MEMUNITS_AUTO) {
        cfg_thresh = proto->cfg_thresh;
    }

    ucp_proto_select_add_proto(init_params, cfg_thresh, 6, perf, &rpriv,
                               sizeof(rpriv));
}

static void
ucp_proto_get_rndv_push_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_worker_h worker                     = init_params->worker;
    const ucp_proto_select_init_protocols_t *proto_init;
    ucp_proto_select_param_t rndv_sel_param;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_elem_t *select_elem;
    ucp_proto_select_t *proto_select;
    ucp_proto_init_elem_t *proto;
    ucp_memory_info_t mem_info;
    ucp_lane_index_t lane;
    const void *priv;

    if (!ucp_proto_rma_rndv_probe_check(init_params, UCP_OP_ID_GET)) {
        return;
    }

    lane = ucp_proto_rndv_find_ctrl_lane(init_params);
    if (lane == UCP_NULL_LANE) {
        return;
    }

    mem_info = ucp_proto_common_select_param_mem_info(init_params->select_param);
    ucp_proto_select_param_init(&rndv_sel_param, UCP_OP_ID_RNDV_RECV, 0,
                                UCP_PROTO_SELECT_OP_FLAG_RNDV_PUSH,
                                UCP_DATATYPE_CONTIG, &mem_info, 1);

    proto_select = ucp_proto_select_get(worker, init_params->ep_cfg_index,
                                        init_params->rkey_cfg_index,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return;
    }

    select_elem = ucp_proto_select_lookup_slow(worker, proto_select, 1,
                                               init_params->ep_cfg_index,
                                               rkey_cfg_index,
                                               &rndv_sel_param);
    if (select_elem == NULL) {
        return;
    }

    proto_init = &select_elem->proto_init;
    ucs_array_for_each(proto, &proto_init->protocols) {
        if (ucp_proto_id_field(proto->proto_id, flags) &
            UCP_PROTO_FLAG_INVALID) {
            continue;
        }

        priv = &ucs_array_elem(&proto_init->priv_buf, proto->priv_offset);
        ucp_proto_get_rndv_push_add_variant(init_params, &rndv_sel_param,
                                            rkey_cfg_index, lane, proto, priv);
    }
}

static void ucp_proto_get_rndv_abort(ucp_request_t *req, ucs_status_t status)
{
    if (req->id != UCS_PTR_MAP_KEY_INVALID) {
        ucp_send_request_id_release(req);
    }

    if (req->send.state.dt_iter.dt_class != UCP_DATATYPE_CLASS_MASK) {
        ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                                  UCP_DT_MASK_ALL);
    }

    ucp_request_complete_send(req, status);
}

static ucs_status_t ucp_proto_get_rndv_reset(ucp_request_t *req)
{
    if (req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        if (req->id != UCS_PTR_MAP_KEY_INVALID) {
            ucp_send_request_id_release(req);
        }

        if (req->send.state.dt_iter.dt_class != UCP_DATATYPE_CLASS_MASK) {
            ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                                      UCP_DT_MASK_ALL);
        }
    }

    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    return UCS_OK;
}

static void
ucp_rma_rndv_get_send_complete(void *request,
                               ucs_status_t UCS_V_UNUSED status,
                               void *UCS_V_UNUSED user_data)
{
    ucp_request_t *req = (ucp_request_t*)request - 1;

    ucp_request_put(req);
}

static void
ucp_rma_rndv_get_send_abort(ucp_request_t *req, ucs_status_t status)
{
    if (req->send.rndv.remote_req_id != UCS_PTR_MAP_KEY_INVALID) {
        ucp_rma_rndv_send_ats_err(req->send.ep, req->send.rndv.remote_req_id,
                                  status);
        req->send.rndv.remote_req_id = UCS_PTR_MAP_KEY_INVALID;
    }

    if (req->id != UCS_PTR_MAP_KEY_INVALID) {
        ucp_send_request_id_release(req);
    }

    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, UCP_DT_MASK_ALL);
    ucp_request_complete_send(req, status);
}

static size_t ucp_rma_rndv_get_rts_pack(void *dest, void *arg)
{
    ucp_request_t *req              = arg;
    ucp_rma_rndv_get_rts_hdr_t *rts = dest;
    ucp_datatype_iter_t *dt_iter    = &req->send.state.dt_iter;
    void *rkey_buffer               = UCS_PTR_BYTE_OFFSET(rts, sizeof(*rts));
    size_t rkey_size                = 0;

    rts->super.hdr         = UCP_RMA_RNDV_AM_GET_RTS;
    rts->super.opcode      = UCP_RNDV_RTS_TAG_OK;
    rts->super.sreq.req_id = ucp_send_request_get_id(req);
    rts->super.sreq.ep_id  = ucp_send_request_get_ep_remote_id(req);
    rts->super.size        = dt_iter->length;
    rts->super.address     = 0;
    rts->get_req_id        = req->send.rndv.remote_req_id;

    if ((dt_iter->length > 0) && (req->send.rndv.md_map != 0)) {
        rkey_size = UCS_PROFILE_CALL(ucp_proto_request_pack_rkey, req,
                                     req->send.rndv.md_map, 0, NULL,
                                     rkey_buffer);
        if (rkey_size > 0) {
            rts->super.address = (uintptr_t)dt_iter->type.contig.buffer;
        }
    }

    return sizeof(*rts) + rkey_size;
}

static ucs_status_t ucp_rma_rndv_get_rts_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_lane_index_t lane;
    ucs_status_t status;

    lane           = ucp_ep_get_am_lane(req->send.ep);
    req->send.lane = lane;
    status         = ucp_proto_am_bcopy_single_send(
            req, UCP_AM_ID_RMA_RNDV, lane, ucp_rma_rndv_get_rts_pack, req,
            sizeof(ucp_rma_rndv_get_rts_hdr_t) +
            ucp_ep_config(req->send.ep)->rndv.rkey_size, 0);
    if (status == UCS_ERR_NO_RESOURCE) {
        return status;
    } else if (status != UCS_OK) {
        ucp_rma_rndv_get_send_abort(req, status);
    }

    return UCS_OK;
}

static ucs_status_t
ucp_rma_rndv_get_sreq_init(ucp_ep_h ep, ucp_request_t *req,
                           const ucp_rma_rndv_get_req_hdr_t *get_req)
{
    ucp_proto_select_param_t sel_param;
    ucp_md_map_t md_map;
    ucs_status_t status;

    ucp_proto_request_send_init(req, ep,
                                UCP_REQUEST_FLAG_RNDV_SEND_INTERNAL);
    ucp_request_set_callback(req, send.cb, ucp_rma_rndv_get_send_complete);
    req->send.buffer                  = (void*)(uintptr_t)get_req->address;
    req->send.length                  = get_req->size;
    req->send.mem_type                = get_req->mem_type;
    req->send.rndv.remote_req_id      = get_req->req.req_id;
    req->send.rndv.rkey               = NULL;
    req->send.rndv.remote_address     = get_req->address;
    req->send.rndv.md_map             = 0;
    ucp_rma_rndv_dt_iter_init(&req->send.state.dt_iter, get_req->address,
                              get_req->size, get_req->mem_type,
                              get_req->sys_dev);

    status = ucp_ep_resolve_remote_id(ep, ucp_ep_get_am_lane(ep));
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    md_map = ucp_ep_config(ep)->key.rma_bw_md_map;
    if ((get_req->size > 0) && (md_map != 0)) {
        status = ucp_datatype_iter_mem_reg(ep->worker->context,
                                           &req->send.state.dt_iter, md_map,
                                           UCT_MD_MEM_ACCESS_RMA |
                                           UCT_MD_MEM_FLAG_HIDE_ERRORS,
                                           UCP_DT_MASK_ALL);
        if (status != UCS_OK) {
            goto err_cleanup;
        }

        req->send.rndv.md_map =
                req->send.state.dt_iter.type.contig.memh->md_map & md_map;
    }

    ucp_proto_select_param_init(&sel_param, UCP_OP_ID_RNDV_SEND, 0, 0,
                                UCP_DATATYPE_CONTIG,
                                &req->send.state.dt_iter.mem_info, 1);
    status = UCS_PROFILE_CALL(ucp_proto_request_lookup_proto, ep->worker, ep,
                              req, &ucp_ep_config(ep)->proto_select,
                              UCP_WORKER_CFG_INDEX_NULL, &sel_param,
                              get_req->size);
    if (status != UCS_OK) {
        goto err_cleanup;
    }

    ucp_send_request_id_alloc(req);
    req->flags        |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    req->send.uct.func = ucp_rma_rndv_get_rts_progress;
    return UCS_OK;

err_cleanup:
    ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 1, UCP_DT_MASK_ALL);
    return status;
}

static void ucp_rma_rndv_get_recv_complete(ucp_request_t *recv_req)
{
    ucp_request_t *get_req = ucp_request_get_super(recv_req);

    ucp_request_complete_send(get_req, recv_req->status);
    ucp_request_put(recv_req);
}

static ucs_status_t ucp_proto_get_rndv_push_init(ucp_request_t *get_req,
                                                 ucp_request_t **rndv_req_p)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = get_req->send.proto_config->priv;
    ucp_worker_h worker                     = get_req->send.ep->worker;
    ucp_rkey_config_t *rkey_config;
    ucp_request_t *recv_req;
    ucp_request_t *rndv_req;
    uint8_t UCS_V_UNUSED sg_count;
    ucp_memory_info_t mem_info;
    ucs_status_t status;
    uint64_t address;
    size_t length;

    status = ucp_ep_resolve_remote_id(get_req->send.ep, rpriv->lane);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_ep_rma_handle_fence(get_req->send.ep, get_req,
                                     UCS_BIT(rpriv->lane));
    if (status != UCS_OK) {
        return status;
    }

    address              = get_req->send.rma.remote_addr;
    rkey_config          = ucp_rkey_config(worker, get_req->send.rma.rkey);
    mem_info.type        = get_req->send.rma.rkey->mem_type;
    mem_info.sys_dev     = rkey_config->key.sys_dev;
    length               = get_req->send.state.dt_iter.length;
    get_req->send.buffer =
            get_req->send.state.dt_iter.type.contig.buffer;
    get_req->send.length = length;

    recv_req = ucp_request_get(worker);
    if (recv_req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    rndv_req = ucp_request_get(worker);
    if (rndv_req == NULL) {
        ucp_request_put(recv_req);
        return UCS_ERR_NO_MEMORY;
    }

    get_req->flags                 |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    recv_req->flags                 = UCP_REQUEST_FLAG_RNDV_RECV_INTERNAL;
    recv_req->recv.worker           = worker;
    recv_req->recv.op_attr          = 0;
    recv_req->recv.remote_req_id    = UCS_PTR_MAP_KEY_INVALID;
    recv_req->recv.rndv.complete_cb = ucp_rma_rndv_get_recv_complete;
    recv_req->status                = UCS_OK;
    ucp_request_set_super(recv_req, get_req);

    UCS_PROFILE_CALL_VOID(ucp_datatype_iter_move, &recv_req->recv.dt_iter,
                          &get_req->send.state.dt_iter, length, &sg_count);

    ucp_proto_request_send_init(rndv_req, get_req->send.ep,
                                UCP_REQUEST_FLAG_RNDV_RTR_REQ);
    ucp_request_set_super(rndv_req, recv_req);
    rndv_req->send.rndv.remote_req_id      = UCS_PTR_MAP_KEY_INVALID;
    rndv_req->send.rndv.remote_address     = address;
    rndv_req->send.rndv.remote_mem_info    = mem_info;
    rndv_req->send.rndv.rkey               = NULL;
    rndv_req->send.rndv.offset             = 0;

    UCS_PROFILE_CALL_VOID(ucp_datatype_iter_move,
                          &rndv_req->send.state.dt_iter,
                          &recv_req->recv.dt_iter, length, &sg_count);
    ucp_proto_request_set_proto(rndv_req, &rpriv->remote_proto_config, length);

    *rndv_req_p = rndv_req;
    return UCS_OK;
}

static ucs_status_t ucp_proto_get_rndv_push_progress(uct_pending_req_t *self)
{
    ucp_request_t *get_req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_request_t *rndv_req;
    ucs_status_t status;

    if (get_req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED) {
        return UCS_OK;
    }

    status = ucp_proto_get_rndv_push_init(get_req, &rndv_req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(get_req, status);
        return UCS_OK;
    }

    ucp_request_send(rndv_req);
    return UCS_OK;
}

static ucs_status_t
ucp_rma_rndv_handle_get_req(ucp_worker_h worker, void *data, size_t length)
{
    const ucp_rma_rndv_get_req_hdr_t *get_req = data;
    ucp_request_t *req;
    ucs_status_t status;
    ucp_ep_h ep;

    if (length < sizeof(*get_req)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    UCP_WORKER_GET_EP_BY_ID(&ep, worker, get_req->req.ep_id, return UCS_OK,
                            "RMA RNDV GET request");

    req = ucp_request_get(worker);
    if (req == NULL) {
        ucs_error("failed to allocate RMA RNDV GET send request");
        ucp_rma_rndv_send_ats_err(ep, get_req->req.req_id, UCS_ERR_NO_MEMORY);
        return UCS_OK;
    }

    status = ucp_rma_rndv_get_sreq_init(ep, req, get_req);
    if (status != UCS_OK) {
        ucp_rma_rndv_send_ats_err(ep, get_req->req.req_id, status);
        ucp_request_put(req);
        return UCS_OK;
    }

    ucp_request_send(req);
    return UCS_OK;
}

static ucs_status_t
ucp_rma_rndv_handle_get_rts(ucp_worker_h worker, void *data, size_t length)
{
    const ucp_rma_rndv_get_rts_hdr_t *rts = data;
    ucp_request_t *get_req, *recv_req;
    uint8_t UCS_V_UNUSED sg_count;
    const void *rkey_buffer;
    ucp_ep_h ep;

    if (length < sizeof(*rts)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    UCP_SEND_REQUEST_GET_BY_ID(&get_req, worker, rts->get_req_id, 0,
                               return UCS_OK, "RMA RNDV GET_RTS %p", rts);

    recv_req = ucp_request_get(worker);
    if (recv_req == NULL) {
        ucs_error("failed to allocate RMA RNDV GET receive request");
        UCP_WORKER_GET_EP_BY_ID(&ep, worker, rts->super.sreq.ep_id,
                                return UCS_OK, "RMA RNDV GET_RTS error");
        ucp_rma_rndv_send_ats_err(ep, rts->super.sreq.req_id,
                                  UCS_ERR_NO_MEMORY);
        ucp_proto_get_rndv_abort(get_req, UCS_ERR_NO_MEMORY);
        return UCS_OK;
    }

    ucp_send_request_id_release(get_req);
    recv_req->flags              = UCP_REQUEST_FLAG_RNDV_RECV_INTERNAL;
    recv_req->recv.worker        = worker;
    recv_req->recv.op_attr       = 0;
    recv_req->recv.remote_req_id = rts->super.sreq.req_id;
    recv_req->recv.rndv.ep_id    = rts->super.sreq.ep_id;
    recv_req->recv.rndv.complete_cb = ucp_rma_rndv_get_recv_complete;
    ucp_request_set_super(recv_req, get_req);

    UCS_PROFILE_CALL_VOID(ucp_datatype_iter_move, &recv_req->recv.dt_iter,
                          &get_req->send.state.dt_iter,
                          get_req->send.state.dt_iter.length, &sg_count);

    rkey_buffer = UCS_PTR_BYTE_OFFSET(rts, sizeof(*rts));
    ucp_proto_rndv_receive_start(worker, recv_req, &rts->super, rkey_buffer,
                                 length - sizeof(*rts));
    return UCS_OK;
}

static void ucp_rma_rndv_put_recv_complete(ucp_request_t *recv_req)
{
    ucp_worker_h worker = recv_req->recv.worker;
    ucp_ep_h ep;

    UCP_WORKER_GET_EP_BY_ID(&ep, worker, recv_req->recv.rndv.ep_id, {
        ucp_request_put(recv_req);
        return;
    }, "RMA RNDV PUT completion");

    ucp_rma_sw_send_cmpl(ep);
    if (recv_req->status != UCS_OK) {
        ucp_rma_rndv_send_ats_err(ep, recv_req->recv.remote_req_id,
                                  recv_req->status);
    }

    ucp_request_put(recv_req);
}

static ucs_status_t
ucp_rma_rndv_handle_put_rts(ucp_worker_h worker, void *data, size_t length)
{
    const ucp_rma_rndv_put_rts_hdr_t *rts = data;
    const void *rkey_buffer;
    ucp_request_t *recv_req;
    ucp_ep_h ep;

    if (length < sizeof(*rts)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    recv_req = ucp_request_get(worker);
    if (recv_req == NULL) {
        ucs_error("failed to allocate RMA RNDV PUT receive request");
        UCP_WORKER_GET_EP_BY_ID(&ep, worker, rts->super.sreq.ep_id,
                                return UCS_OK, "RMA RNDV PUT error");
        ucp_rma_sw_send_cmpl(ep);
        ucp_rma_rndv_send_ats_err(ep, rts->super.sreq.req_id,
                                  UCS_ERR_NO_MEMORY);
        return UCS_OK;
    }

    recv_req->flags              = UCP_REQUEST_FLAG_RNDV_RECV_INTERNAL;
    recv_req->recv.worker        = worker;
    recv_req->recv.op_attr       = 0;
    recv_req->recv.remote_req_id = rts->super.sreq.req_id;
    ucp_rma_rndv_dt_iter_init(&recv_req->recv.dt_iter, rts->address,
                              rts->super.size, rts->mem_type, rts->sys_dev);
    recv_req->recv.rndv.ep_id                 = rts->super.sreq.ep_id;
    recv_req->recv.rndv.complete_cb           = ucp_rma_rndv_put_recv_complete;

    rkey_buffer = UCS_PTR_BYTE_OFFSET(rts, sizeof(*rts));
    ucp_proto_rndv_receive_start(worker, recv_req, &rts->super, rkey_buffer,
                                 length - sizeof(*rts));
    return UCS_OK;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_rma_rndv_handler,
                 (arg, data, length, am_flags), void *arg, void *data,
                 size_t length, unsigned am_flags)
{
    const uint64_t *hdr = data;
    ucp_worker_h worker = arg;

    if (length < sizeof(*hdr)) {
        return UCS_ERR_MESSAGE_TRUNCATED;
    }

    switch (*hdr) {
    case UCP_RMA_RNDV_AM_PUT_RTS:
        return ucp_rma_rndv_handle_put_rts(worker, data, length);
    case UCP_RMA_RNDV_AM_GET_REQ:
        return ucp_rma_rndv_handle_get_req(worker, data, length);
    case UCP_RMA_RNDV_AM_GET_RTS:
        return ucp_rma_rndv_handle_get_rts(worker, data, length);
    default:
        ucs_debug("unexpected RMA RNDV AM sub-id %" PRIu64, *hdr);
        return UCS_ERR_UNSUPPORTED;
    }
}

static void
ucp_rma_rndv_dump_packet(ucp_worker_h worker, uct_am_trace_type_t type,
                         uint8_t id, const void *data, size_t length,
                         char *buffer, size_t max)
{
    const ucp_rma_rndv_put_rts_hdr_t *put_rts = data;
    const ucp_rma_rndv_get_req_hdr_t *get_req = data;
    const ucp_rma_rndv_get_rts_hdr_t *get_rts = data;
    const uint64_t *hdr                       = data;

    if (length < sizeof(*hdr)) {
        return;
    }

    switch (*hdr) {
    case UCP_RMA_RNDV_AM_PUT_RTS:
        if (length < sizeof(*put_rts)) {
            return;
        }

        snprintf(buffer, max, "RMA_PUT_RTS [src 0x%" PRIx64
                 " dst 0x%" PRIx64 " len %zu req_id 0x%" PRIx64
                 " ep_id 0x%" PRIx64 " %s]", put_rts->super.address,
                 put_rts->address, put_rts->super.size,
                 put_rts->super.sreq.req_id, put_rts->super.sreq.ep_id,
                 ucs_memory_type_names[put_rts->mem_type]);
        break;
    case UCP_RMA_RNDV_AM_GET_REQ:
        if (length < sizeof(*get_req)) {
            return;
        }

        snprintf(buffer, max, "RMA_GET_REQ [src 0x%" PRIx64
                 " len %zu req_id 0x%" PRIx64 " ep_id 0x%" PRIx64
                 " %s]", get_req->address, get_req->size,
                 get_req->req.req_id, get_req->req.ep_id,
                 ucs_memory_type_names[get_req->mem_type]);
        break;
    case UCP_RMA_RNDV_AM_GET_RTS:
        if (length < sizeof(*get_rts)) {
            return;
        }

        snprintf(buffer, max, "RMA_GET_RTS [src 0x%" PRIx64
                 " len %zu sreq_id 0x%" PRIx64 " ep_id 0x%" PRIx64
                 " get_req_id 0x%" PRIx64 "]", get_rts->super.address,
                 get_rts->super.size, get_rts->super.sreq.req_id,
                 get_rts->super.sreq.ep_id, get_rts->get_req_id);
        break;
    default:
        snprintf(buffer, max, "RMA_RNDV [sub-id %" PRIu64 "]", *hdr);
        break;
    }
}

UCP_DEFINE_AM_WITH_PROXY(UCP_FEATURE_RMA, UCP_AM_ID_RMA_RNDV,
                         ucp_rma_rndv_handler, ucp_rma_rndv_dump_packet, 0);

ucp_proto_t ucp_put_rndv_proto = {
    .name     = "put/rndv",
    .desc     = UCP_PROTO_RMA_RNDV_DESC,
    .flags    = 0,
    .probe    = ucp_proto_put_rndv_probe,
    .query    = ucp_proto_rma_rndv_default_query,
    .progress = {ucp_proto_put_rndv_progress},
    .abort    = ucp_proto_rndv_rts_abort,
    .reset    = ucp_proto_rndv_rts_reset
};

ucp_proto_t ucp_get_rndv_proto = {
    .name     = "get/rndv",
    .desc     = UCP_PROTO_RMA_RNDV_DESC,
    .flags    = 0,
    .probe    = ucp_proto_get_rndv_probe,
    .query    = ucp_proto_rma_rndv_default_query,
    .progress = {ucp_proto_get_rndv_progress},
    .abort    = ucp_proto_get_rndv_abort,
    .reset    = ucp_proto_get_rndv_reset
};

ucp_proto_t ucp_get_rndv_push_proto = {
    .name     = "get/rndv/push",
    .desc     = UCP_PROTO_RMA_RNDV_PUSH_DESC,
    .flags    = 0,
    .probe    = ucp_proto_get_rndv_push_probe,
    .query    = ucp_proto_get_rndv_push_query,
    .progress = {ucp_proto_get_rndv_push_progress},
    .abort    = ucp_proto_get_rndv_abort,
    .reset    = ucp_proto_get_rndv_reset
};
