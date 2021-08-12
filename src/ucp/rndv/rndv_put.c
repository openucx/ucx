/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_rndv.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_single.inl>


enum {
    UCP_PROTO_RNDV_PUT_STAGE_SEND = UCP_PROTO_STAGE_START,

    /* Flush all lanes to ensure remote delivery */
    UCP_PROTO_RNDV_PUT_STAGE_FLUSH,

    /* Send ATP without fence (could be done after a flush) */
    UCP_PROTO_RNDV_PUT_STAGE_ATP,

    /* Send ATP with fence (could be done if using send lanes for ATP) */
    UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP,
};

typedef struct ucp_proto_rndv_put_priv {
    uct_completion_callback_t  put_comp_cb;
    uct_completion_callback_t  atp_comp_cb;
    uint8_t                    stage_after_put;
    ucp_lane_map_t             flush_map;
    ucp_lane_map_t             atp_map;
    ucp_lane_index_t           atp_num_lanes;
    ucp_proto_rndv_bulk_priv_t bulk;
} ucp_proto_rndv_put_priv_t;


static void
ucp_proto_rndv_put_common_completion_send_atp(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    ucp_trace_req(req, "rndv_put_common_completion_send_atp");
    ucp_proto_completion_init(&req->send.state.uct_comp, rpriv->atp_comp_cb);
    ucp_proto_request_set_stage(req, UCP_PROTO_RNDV_PUT_STAGE_ATP);
    ucp_request_send(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_flush_send(ucp_request_t *req, ucp_lane_index_t lane)
{
    ucp_ep_h ep = req->send.ep;

    ucp_trace_req(req, "flush lane[%d] " UCT_TL_RESOURCE_DESC_FMT, lane,
                  UCT_TL_RESOURCE_DESC_ARG(ucp_ep_get_tl_rsc(ep, lane)));
    return uct_ep_flush(ep->uct_eps[lane], 0, &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_rndv_put_common_flush_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_multi_lane_map_progress(
            req, &req->send.rndv.put.flush_map,
            ucp_proto_rndv_put_common_flush_send);
}

static size_t ucp_proto_rndv_put_common_pack_atp(void *dest, void *arg)
{
    ucp_request_t *req                     = arg;
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_rndv_send_pack_atp(req, dest, rpriv->atp_num_lanes);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_atp_send(ucp_request_t *req, ucp_lane_index_t lane)
{
    const ucp_proto_rndv_put_priv_t UCS_V_UNUSED *rpriv =
            req->send.proto_config->priv;

    ucp_trace_req(req, "send ATP lane %d count %d", lane, rpriv->atp_num_lanes);
    return ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RNDV_ATP, lane,
                                          ucp_proto_rndv_put_common_pack_atp,
                                          req, sizeof(ucp_rndv_atp_hdr_t));
}

static ucs_status_t
ucp_proto_rndv_put_common_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_multi_lane_map_progress(req, &req->send.rndv.put.atp_map,
                                             ucp_proto_rndv_put_common_atp_send);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_fenced_atp_send(ucp_request_t *req,
                                          ucp_lane_index_t lane)
{
    ucs_status_t status;

    status = uct_ep_fence(req->send.ep->uct_eps[lane], 0);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    return ucp_proto_rndv_put_common_atp_send(req, lane);
}

static ucs_status_t
ucp_proto_rndv_put_common_fenced_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);

    return ucp_proto_multi_lane_map_progress(
            req, &req->send.rndv.put.atp_map,
            ucp_proto_rndv_put_common_fenced_atp_send);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_data_sent(ucp_request_t *req)
{
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    ucp_trace_req(req, "rndv_put_common_data_sent");
    ucp_proto_request_set_stage(req, rpriv->stage_after_put);
    return UCS_INPROGRESS;
}

static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_put_common_request_init(ucp_request_t *req)
{
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    req->send.rndv.put.atp_map   = rpriv->atp_map;
    req->send.rndv.put.flush_map = rpriv->flush_map;
    ucp_proto_rndv_bulk_request_init(req, &rpriv->bulk);
}

static void ucp_proto_rndv_put_zcopy_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "put_zcopy_completion");
    ucp_proto_rndv_rkey_destroy(req);
    ucp_proto_request_zcopy_complete(req, req->send.state.uct_comp.status);
}

static ucs_status_t
ucp_proto_rndv_put_zcopy_init(const ucp_proto_init_params_t *init_params)
{
    const size_t atp_size                = sizeof(ucp_rndv_atp_hdr_t);
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_rndv_put_priv_t *rpriv     = init_params->priv;
    uint64_t rndv_modes                  = UCS_BIT(UCP_RNDV_MODE_PUT_ZCOPY);
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.overhead      = 0,
        .super.latency       = 0,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 0,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_zcopy),
        .super.hdr_size      = 0,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS,
        .max_lanes           = context->config.ext.max_rndv_lanes,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
    };
    const uct_iface_attr_t *iface_attr;
    ucp_lane_index_t lane_idx, lane;
    size_t bulk_priv_size;
    ucs_status_t status;
    int use_fence;

    if ((init_params->select_param->op_id != UCP_OP_ID_RNDV_SEND) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    status = ucp_proto_rndv_bulk_init(&params, &rpriv->bulk, &bulk_priv_size);
    if (status != UCS_OK) {
        return status;
    }

    *init_params->priv_size = ucs_offsetof(ucp_proto_rndv_put_priv_t, bulk) +
                              bulk_priv_size;

    /* Check if all potential lanes support sending ATP */
    use_fence = !context->config.ext.rndv_put_force_flush;
    lane_idx  = 0;
    while (use_fence && (lane_idx < rpriv->bulk.mpriv.num_lanes)) {
        lane       = rpriv->bulk.mpriv.lanes[lane_idx++].super.lane;
        iface_attr = ucp_proto_common_get_iface_attr(init_params, lane);
        use_fence  = use_fence &&
                     (((iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) &&
                       (iface_attr->cap.am.max_short >= atp_size)) ||
                      ((iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) &&
                       (iface_attr->cap.am.max_bcopy >= atp_size)));
    }

    /* All lanes can send ATP - invalidate am_lane, to use mpriv->lanes.
     * Otherwise, would need to flush all lanes and send ATP on
     * rpriv->super.lane when the flush is completed
     */
    if (use_fence) {
        /* Send fence followed by ATP on all lanes */
        rpriv->bulk.super.lane = UCP_NULL_LANE;
        rpriv->put_comp_cb     = ucp_proto_rndv_put_zcopy_completion;
        rpriv->atp_comp_cb     = NULL;
        rpriv->stage_after_put = UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP;
        rpriv->flush_map       = 0;
        rpriv->atp_map         = rpriv->bulk.mpriv.lane_map;
    } else {
        /* Flush all lanes and send single ATP on control message lane */
        rpriv->put_comp_cb     = ucp_proto_rndv_put_common_completion_send_atp;
        rpriv->atp_comp_cb     = ucp_proto_rndv_put_zcopy_completion;
        rpriv->atp_map         = UCS_BIT(rpriv->bulk.super.lane);
        rpriv->stage_after_put = UCP_PROTO_RNDV_PUT_STAGE_FLUSH;
        rpriv->flush_map       = rpriv->bulk.mpriv.lane_map;
        ucs_assert(rpriv->flush_map != 0);
    }

    ucs_assert(rpriv->atp_map != 0);
    rpriv->atp_num_lanes = ucs_popcount(rpriv->atp_map);

    return UCS_OK;
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_put_zcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter)
{
    ucp_rkey_h rkey    = req->send.rndv.rkey;
    uct_rkey_t tl_rkey = rkey->tl_rkey[lpriv->super.rkey_index].rkey.rkey;
    size_t max_payload;
    uct_iov_t iov;

    max_payload = ucp_proto_multi_max_payload(req, lpriv, 0);
    ucp_datatype_iter_next_iov(&req->send.state.dt_iter,
                               lpriv->super.memh_index, max_payload, next_iter,
                               &iov);
    return uct_ep_put_zcopy(req->send.ep->uct_eps[lpriv->super.lane], &iov, 1,
                            req->send.rndv.remote_address +
                                    req->send.state.dt_iter.offset,
                            tl_rkey, &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_rndv_put_zcopy_send_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_multi_zcopy_progress(
            req, &rpriv->bulk.mpriv, ucp_proto_rndv_put_common_request_init,
            UCT_MD_MEM_ACCESS_LOCAL_READ, ucp_proto_rndv_put_zcopy_send_func,
            ucp_proto_rndv_put_common_data_sent, rpriv->put_comp_cb);
}

static void ucp_proto_rndv_put_config_str(size_t min_length, size_t max_length,
                                          const void *priv,
                                          ucs_string_buffer_t *strb)
{
    const ucp_proto_rndv_put_priv_t *rpriv = priv;

    ucp_proto_rndv_bulk_config_str(min_length, max_length, &rpriv->bulk, strb);
    if (rpriv->flush_map != 0) {
        ucs_string_buffer_appendf(strb, " flush:");
        ucs_string_buffer_append_flags(strb, rpriv->flush_map, NULL);
    }
    if (rpriv->atp_map != 0) {
        ucs_string_buffer_appendf(strb, " atp:");
        ucs_string_buffer_append_flags(strb, rpriv->atp_map, NULL);
    }
}

static ucp_proto_t ucp_rndv_put_zcopy_proto = {
    .name        = "rndv/put/zcopy",
    .flags       = 0,
    .init        = ucp_proto_rndv_put_zcopy_init,
    .config_str  = ucp_proto_rndv_put_config_str,
    .progress    = {
        [UCP_PROTO_RNDV_PUT_STAGE_SEND]       = ucp_proto_rndv_put_zcopy_send_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_FLUSH]      = ucp_proto_rndv_put_common_flush_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_ATP]        = ucp_proto_rndv_put_common_atp_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP] = ucp_proto_rndv_put_common_fenced_atp_progress,
    },
};
UCP_PROTO_REGISTER(&ucp_rndv_put_zcopy_proto);
