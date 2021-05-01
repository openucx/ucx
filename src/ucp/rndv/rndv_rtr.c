/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "proto_rndv.inl"

#include <ucp/proto/proto_single.inl>


static ucs_status_t
ucp_proto_rndv_rtr_common_init(const ucp_proto_init_params_t *init_params,
                               uint64_t rndv_modes, ucs_memory_type_t mem_type,
                               ucs_sys_device_t sys_dev)
{
    ucp_context_h context                    = init_params->worker->context;
    ucp_proto_rndv_ctrl_init_params_t params = {
        .super.super        = *init_params,
        .super.latency      = 0,
        .super.overhead     = 40e-9,
        .super.cfg_thresh   = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority = 0,
        .super.flags        = UCP_PROTO_COMMON_INIT_FLAG_RESPONSE,
        .remote_op_id       = UCP_OP_ID_RNDV_SEND,
        .perf_bias          = 0.0,
        .mem_info.type      = mem_type,
        .mem_info.sys_dev   = sys_dev,
        .min_length         = 1
    };

    return ucp_proto_rndv_ctrl_init(&params);
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rtr_common_request_init(ucp_request_t *req)
{
    ucp_request_t *recv_req = ucp_request_get_super(req);

    recv_req->status         = UCS_OK;
    recv_req->recv.remaining = req->send.state.dt_iter.length;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rdnv_rtr_common_comp_init(ucp_ep_h ep, uct_completion_t *comp,
                                    ucs_ptr_map_key_t *ptr_id_p,
                                    uct_completion_callback_t comp_func)
{
    /* RTR sends the id of its &req->comp field, and not of the request, to
       support fragmented protocol with multiple RTRs per request */
    ucp_proto_completion_init(comp, comp_func);
    ucp_ep_ptr_id_alloc(ep, comp, ptr_id_p);
}

static ucs_status_t
ucp_proto_rndv_rtr_common_send(ucp_request_t *req, uct_pack_callback_t pack_cb)
{
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    size_t max_rtr_size = sizeof(ucp_rndv_rtr_hdr_t) + rpriv->packed_rkey_size;

    return ucp_proto_am_bcopy_single_progress(req, UCP_AM_ID_RNDV_RTR,
                                              rpriv->lane, pack_cb, req,
                                              max_rtr_size, NULL);
}

static void ucp_proto_rndv_rtr_common_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);
    ucp_proto_rndv_rtr_common_complete(req, req->send.state.uct_comp.status);
}

static size_t ucp_proto_rndv_rtr_pack(void *dest, void *arg)
{
    ucp_rndv_rtr_hdr_t *rtr = dest;
    ucp_request_t *req      = arg;
    const UCS_V_UNUSED ucp_proto_rndv_ctrl_priv_t *rpriv;

    rtr->sreq_id = req->send.rndv.remote_req_id;
    rtr->rreq_id = req->send.rndv.rtr.rreq_id;
    rtr->size    = req->send.state.dt_iter.length;
    rtr->offset  = 0;
    rtr->address = (uintptr_t)req->send.state.dt_iter.type.contig.buffer;

    rpriv = req->send.proto_config->priv;
    ucs_assert(rtr->size > 0);
    ucs_assert(rpriv->md_map == req->send.state.dt_iter.type.contig.reg.md_map);
    return sizeof(*rtr) + ucp_proto_request_pack_rkey(req, rtr + 1);
}

static ucs_status_t ucp_proto_rndv_rtr_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv = req->send.proto_config->priv;
    ucs_status_t status;

    if (!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED)) {
        status = ucp_datatype_iter_mem_reg(req->send.ep->worker->context,
                                           &req->send.state.dt_iter,
                                           rpriv->md_map,
                                           UCT_MD_MEM_ACCESS_REMOTE_PUT);
        if (status != UCS_OK) {
            ucp_proto_request_abort(req, status);
            return UCS_OK;
        }

        ucp_proto_rtr_common_request_init(req);
        ucp_proto_rdnv_rtr_common_comp_init(
                req->send.ep, &req->send.state.uct_comp,
                &req->send.rndv.rtr.rreq_id,
                ucp_proto_rndv_rtr_common_completion);

        req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    }

    return ucp_proto_rndv_rtr_common_send(req, ucp_proto_rndv_rtr_pack);
}

static ucs_status_t
ucp_proto_rndv_rtr_init(const ucp_proto_init_params_t *init_params)
{
    static const uint64_t rndv_modes = UCS_BIT(UCP_RNDV_MODE_PUT_ZCOPY) |
                                       UCS_BIT(UCP_RNDV_MODE_AM);

    if (init_params->select_param->op_id != UCP_OP_ID_RNDV_RECV) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_rtr_common_init(init_params, rndv_modes,
                                          init_params->select_param->mem_type,
                                          init_params->select_param->sys_dev);
}

static ucp_proto_t ucp_rndv_rtr_proto = {
    .name       = "rndv/rtr",
    .flags      = 0,
    .init       = ucp_proto_rndv_rtr_init,
    .config_str = ucp_proto_rndv_ctrl_config_str,
    .progress   = ucp_proto_rndv_rtr_progress
};
UCP_PROTO_REGISTER(&ucp_rndv_rtr_proto);
