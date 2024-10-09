/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2021. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_rndv.inl"
#include "rndv_mtype.inl"

#include <ucp/core/ucp_request.inl>
#include <ucp/proto/proto_am.inl>
#include <ucp/proto/proto_multi.inl>
#include <ucp/proto/proto_single.inl>


#define UCP_PROTO_RNDV_PUT_DESC "write to remote"

typedef struct ucp_proto_rndv_put_priv {
    uct_completion_callback_t  put_comp_cb;
    uct_completion_callback_t  atp_comp_cb;
    uint8_t                    stage_after_put;
    ucp_lane_map_t             flush_map;
    ucp_lane_map_t             atp_map;
    ucp_lane_index_t           atp_num_lanes;
    uint8_t                    stat_counter;
    ucp_proto_rndv_bulk_priv_t bulk;
} ucp_proto_rndv_put_priv_t;


/* Context for packing ATP message */
typedef struct {
    ucp_request_t *req;     /* rndv/put request */
    size_t        ack_size; /* Size to send in ATP message */
} ucp_proto_rndv_put_atp_pack_ctx_t;


static UCS_F_ALWAYS_INLINE void
ucp_proto_rndv_put_common_complete(ucp_request_t *req)
{
    const ucp_proto_rndv_put_priv_t UCS_V_UNUSED *rpriv =
                                                   req->send.proto_config->priv;
    ucp_trace_req(req, "rndv_put_common_complete");
    UCS_STATS_UPDATE_COUNTER(req->send.ep->worker->stats, rpriv->stat_counter,
                             +1);
    ucp_proto_rndv_rkey_destroy(req);
    ucp_proto_request_zcopy_complete(req, req->send.state.uct_comp.status);
}

static void ucp_proto_rndv_put_zcopy_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);
    ucp_proto_rndv_put_common_complete(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_send(ucp_request_t *req,
                               const ucp_proto_multi_lane_priv_t *lpriv,
                               const uct_iov_t *iov, uct_completion_t *comp)
{
    uct_rkey_t tl_rkey      = ucp_rkey_get_tl_rkey(req->send.rndv.rkey,
                                                   lpriv->super.rkey_index);
    uint64_t remote_address = req->send.rndv.remote_address +
                              req->send.state.dt_iter.offset;

    return uct_ep_put_zcopy(ucp_ep_get_lane(req->send.ep, lpriv->super.lane),
                            iov, 1, remote_address, tl_rkey, comp);
}

static void
ucp_proto_rndv_put_common_flush_completion_send_atp(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    ucp_trace_req(req, "rndv_put_common_completion_send_atp status %s",
                  ucs_status_string(uct_comp->status));

    if (ucs_unlikely(uct_comp->status != UCS_OK)) {
        ucp_proto_rndv_put_common_complete(req);
        return;
    }

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
    return uct_ep_flush(ucp_ep_get_lane(ep, lane), 0,
                        &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_rndv_put_common_flush_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv;

    rpriv = req->send.proto_config->priv;
    return ucp_proto_multi_lane_map_progress(
            req, &req->send.rndv.put.flush_lane, rpriv->flush_map,
            ucp_proto_rndv_put_common_flush_send);
}

static size_t ucp_proto_rndv_put_common_pack_atp(void *dest, void *arg)
{
    ucp_proto_rndv_put_atp_pack_ctx_t *pack_ctx = arg;

    return ucp_proto_rndv_pack_ack(pack_ctx->req, dest, pack_ctx->ack_size);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_atp_send(ucp_request_t *req, ucp_lane_index_t lane)
{
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;
    ucp_proto_rndv_put_atp_pack_ctx_t pack_ctx;
    ucs_status_t status;

    /* Make sure the sum of ack_size field in all ATP messages we send will not
       exceed request length, since each ATP message has to acknowledge at least
       one byte. */
    ucs_assertv(req->send.rndv.put.atp_count <= req->send.state.dt_iter.length,
                "atp_count=%u length=%zu", req->send.rndv.put.atp_count,
                req->send.state.dt_iter.length);
    if (req->send.rndv.put.atp_count == req->send.state.dt_iter.length) {
        return UCS_OK;
    }

    /* Ensure ATP is sent on the same lane as the data to prevent ATP from
     * arriving before the data. If data transmission starts from a non-zero
     * lane, ATP may never be sent on the data lane. */
    if (ucs_unlikely((req->send.state.dt_iter.length < rpriv->atp_num_lanes) &&
                     (lane < req->send.multi_lane_idx))) {
        return UCS_OK;
    }

    pack_ctx.req = req;

    /* When we need to send multiple ATP messages: each will acknowledge 1 byte,
       except the last ATP which will acknowledge the remaining payload size.
       This is simpler than keeping track of how much was sent on each lane */
    if (req->send.rndv.put.atp_count == (rpriv->atp_num_lanes - 1)) {
        pack_ctx.ack_size = req->send.state.dt_iter.length -
                            req->send.rndv.put.atp_count;
    } else {
        pack_ctx.ack_size = 1;
    }

    status = ucp_proto_am_bcopy_single_send(req, UCP_AM_ID_RNDV_ATP, lane,
                                            ucp_proto_rndv_put_common_pack_atp,
                                            &pack_ctx,
                                            sizeof(ucp_rndv_ack_hdr_t), 0);
    if (status != UCS_OK) {
        return status;
    }

    ++req->send.rndv.put.atp_count;
    return UCS_OK;
}

static ucs_status_t
ucp_proto_rndv_put_common_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv;

    rpriv = req->send.proto_config->priv;
    return ucp_proto_multi_lane_map_progress(req, &req->send.rndv.put.atp_lane,
                                             rpriv->atp_map,
                                             ucp_proto_rndv_put_common_atp_send);
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_proto_rndv_put_common_fenced_atp_send(ucp_request_t *req,
                                          ucp_lane_index_t lane)
{
    ucs_status_t status;

    status = uct_ep_fence(ucp_ep_get_lane(req->send.ep, lane), 0);
    if (ucs_unlikely(status != UCS_OK)) {
        return status;
    }

    return ucp_proto_rndv_put_common_atp_send(req, lane);
}

static ucs_status_t
ucp_proto_rndv_put_common_fenced_atp_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv;

    rpriv = req->send.proto_config->priv;
    return ucp_proto_multi_lane_map_progress(
            req, &req->send.rndv.put.atp_lane, rpriv->atp_map,
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

    req->send.rndv.put.flush_lane = 0;
    req->send.rndv.put.atp_lane   = 0;
    req->send.rndv.put.atp_count  = 0;
    ucp_proto_rndv_bulk_request_init(req, &rpriv->bulk);
}

static void
ucp_proto_rndv_put_common_probe(const ucp_proto_init_params_t *init_params,
                                uint64_t rndv_modes, size_t max_length,
                                uct_ep_operation_t memtype_op, unsigned flags,
                                ucp_md_map_t initial_reg_md_map,
                                uct_completion_callback_t comp_cb,
                                int support_ppln, uint8_t stat_counter,
                                const ucp_memory_info_t *reg_mem_info)
{
    const size_t atp_size                = sizeof(ucp_rndv_ack_hdr_t);
    ucp_context_t *context               = init_params->worker->context;
    ucp_proto_multi_init_params_t params = {
        .super.super         = *init_params,
        .super.overhead      = 0,
        .super.latency       = 0,
        .super.cfg_thresh    = ucp_proto_rndv_cfg_thresh(context, rndv_modes),
        .super.cfg_priority  = 80,
        .super.min_length    = 0,
        .super.max_length    = max_length,
        .super.min_iov       = 1,
        .super.min_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.min_zcopy),
        .super.max_frag_offs = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.max_zcopy),
        .super.max_iov_offs  = ucs_offsetof(uct_iface_attr_t, cap.put.max_iov),
        .super.send_op       = UCT_EP_OP_PUT_ZCOPY,
        .super.memtype_op    = memtype_op,
        .super.flags         = flags | UCP_PROTO_COMMON_INIT_FLAG_RECV_ZCOPY |
                               UCP_PROTO_COMMON_INIT_FLAG_REMOTE_ACCESS |
                               UCP_PROTO_COMMON_INIT_FLAG_MIN_FRAG,
        .super.exclude_map   = 0,
        .super.reg_mem_info  = *reg_mem_info,
        .max_lanes           = context->config.ext.max_rndv_lanes,
        .initial_reg_md_map  = initial_reg_md_map,
        .first.tl_cap_flags  = UCT_IFACE_FLAG_PUT_ZCOPY,
        .first.lane_type     = UCP_LANE_TYPE_RMA_BW,
        .middle.tl_cap_flags = UCT_IFACE_FLAG_PUT_ZCOPY,
        .middle.lane_type    = UCP_LANE_TYPE_RMA_BW,
        .super.hdr_size      = 0,
        .opt_align_offs      = ucs_offsetof(uct_iface_attr_t,
                                            cap.put.opt_zcopy_align),
    };
    const uct_iface_attr_t *iface_attr;
    ucp_lane_index_t lane_idx, lane;
    ucp_proto_rndv_put_priv_t rpriv;
    int send_atp, use_fence;
    ucp_proto_perf_t *perf;
    ucs_status_t status;
    unsigned atp_map;

    if ((init_params->select_param->dt_class != UCP_DATATYPE_CONTIG) ||
        !ucp_proto_rndv_op_check(init_params, UCP_OP_ID_RNDV_SEND,
                                 support_ppln) ||
        !ucp_proto_common_init_check_err_handling(&params.super)) {
        return;
    }

    status = ucp_proto_rndv_bulk_init(&params, UCP_PROTO_RNDV_PUT_DESC,
                                      UCP_PROTO_RNDV_ATP_NAME, &perf,
                                      &rpriv.bulk);
    if (status != UCS_OK) {
        return;
    }

    send_atp = !ucp_proto_rndv_init_params_is_ppln_frag(init_params);

    /* Check which lanes support sending ATP */
    atp_map = 0;
    for (lane_idx = 0; lane_idx < rpriv.bulk.mpriv.num_lanes; ++lane_idx) {
        lane       = rpriv.bulk.mpriv.lanes[lane_idx].super.lane;
        iface_attr = ucp_proto_common_get_iface_attr(init_params, lane);
        if (((iface_attr->cap.flags & UCT_IFACE_FLAG_AM_SHORT) &&
             (iface_attr->cap.am.max_short >= atp_size)) ||
            ((iface_attr->cap.flags & UCT_IFACE_FLAG_AM_BCOPY) &&
             (iface_attr->cap.am.max_bcopy >= atp_size))) {
            atp_map |= UCS_BIT(lane);
        }
    }

    /* Use fence only if all lanes support sending ATP and flush is not forced
     */
    use_fence = send_atp && !context->config.ext.rndv_put_force_flush &&
                (rpriv.bulk.mpriv.lane_map == atp_map);

    /* All lanes can send ATP - invalidate am_lane, to use mpriv->lanes.
     * Otherwise, would need to flush all lanes and send ATP on:
     * - All lanes supporting ATP send. This ensures that data is flushed
     *   remotely (i.e. resides in the target buffer), which may not be the case
     *   with IB transports. An alternative would be to pass
     *   UCT_FLUSH_FLAG_REMOTE to uct_ep_flush, but using this flag increases
     *   UCP worker address size.
     *   TODO: Consider calling UCT ep flush with remote flag when/if address
     *   size is not an issue anymore.
     * - Control lane if none of the lanes support sending ATP
     */
    if (use_fence) {
        /* Send fence followed by ATP on all lanes */
        rpriv.bulk.super.lane = UCP_NULL_LANE;
        rpriv.put_comp_cb     = comp_cb;
        rpriv.atp_comp_cb     = NULL;
        rpriv.stage_after_put = UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP;
        rpriv.flush_map       = 0;
        rpriv.atp_map         = rpriv.bulk.mpriv.lane_map;
    } else {
        /* Flush all lanes and send ATP on all supporting lanes (or control lane
         * otherwise) */
        if (send_atp) {
            rpriv.put_comp_cb =
                    ucp_proto_rndv_put_common_flush_completion_send_atp;
            rpriv.atp_comp_cb = comp_cb;
            rpriv.atp_map     = (atp_map == 0) ?
                                UCS_BIT(rpriv.bulk.super.lane) : atp_map;
        } else {
            rpriv.put_comp_cb = comp_cb;
            rpriv.atp_comp_cb = NULL;
            rpriv.atp_map     = 0;
        }
        rpriv.stage_after_put = UCP_PROTO_RNDV_PUT_STAGE_FLUSH;
        rpriv.flush_map       = rpriv.bulk.mpriv.lane_map;
        ucs_assert(rpriv.flush_map != 0);
    }

    if (send_atp) {
        ucs_assert(rpriv.atp_map != 0);
    }
    rpriv.atp_num_lanes = ucs_popcount(rpriv.atp_map);
    rpriv.stat_counter  = stat_counter;

    ucp_proto_select_add_proto(&params.super.super, params.super.cfg_thresh,
                               params.super.cfg_priority, perf, &rpriv,
                               UCP_PROTO_MULTI_EXTENDED_PRIV_SIZE(&rpriv,
                                                                  bulk.mpriv));
}

static const char *
ucp_proto_rndv_put_common_query(const ucp_proto_query_params_t *params,
                                ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_put_priv_t *rpriv     = params->priv;
    ucp_proto_query_params_t bulk_query_params = {
        .proto         = params->proto,
        .priv          = &rpriv->bulk,
        .worker        = params->worker,
        .select_param  = params->select_param,
        .ep_config_key = params->ep_config_key,
        .msg_length    = params->msg_length
    };

    ucp_proto_rndv_bulk_query(&bulk_query_params, attr);

    if (rpriv->atp_map == 0) {
        return UCP_PROTO_RNDV_PUT_DESC;
    } else if (rpriv->flush_map != 0) {
        return "flushed " UCP_PROTO_RNDV_PUT_DESC;
    } else {
        return "fenced " UCP_PROTO_RNDV_PUT_DESC;
    }
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_put_zcopy_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;
    size_t max_payload;
    uct_iov_t iov;

    max_payload = ucp_proto_rndv_bulk_max_payload_align(req, &rpriv->bulk,
                                                        lpriv, lane_shift);
    ucp_datatype_iter_next_iov(&req->send.state.dt_iter, max_payload,
                               lpriv->super.md_index,
                               UCS_BIT(UCP_DATATYPE_CONTIG), next_iter, &iov,
                               1);
    return ucp_proto_rndv_put_common_send(req, lpriv, &iov,
                                          &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_rndv_put_zcopy_send_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    return ucp_proto_multi_zcopy_progress(
            req, &rpriv->bulk.mpriv, ucp_proto_rndv_put_common_request_init,
            UCT_MD_MEM_ACCESS_LOCAL_READ, UCS_BIT(UCP_DATATYPE_CONTIG),
            ucp_proto_rndv_put_zcopy_send_func,
            ucp_proto_rndv_put_common_data_sent, rpriv->put_comp_cb);
}

static void
ucp_proto_rndv_put_zcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_memory_info_t reg_mem_info = {
        .type    = init_params->select_param->mem_type,
        .sys_dev = init_params->select_param->sys_dev
    };

    ucp_proto_rndv_put_common_probe(
            init_params, UCS_BIT(UCP_RNDV_MODE_PUT_ZCOPY), SIZE_MAX,
            UCT_EP_OP_LAST,
            UCP_PROTO_COMMON_INIT_FLAG_SEND_ZCOPY |
            UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING,
            0, ucp_proto_rndv_put_zcopy_completion, 0,
            UCP_WORKER_STAT_RNDV_PUT_ZCOPY, &reg_mem_info);
}

static void
ucp_proto_rndv_put_zcopy_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    const char *put_desc;

    put_desc = ucp_proto_rndv_put_common_query(params, attr);
    ucs_snprintf_safe(attr->desc, sizeof(attr->desc), "%s %s",
                      UCP_PROTO_ZCOPY_DESC, put_desc);
}

static ucs_status_t ucp_proto_rndv_put_zcopy_reset(ucp_request_t *req)
{
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;

    if (req->send.rndv.put.atp_count == rpriv->atp_num_lanes) {
        /* Sent all ATPs so the iterator should be at the end */
        ucs_assertv_always(ucp_datatype_iter_is_end(&req->send.state.dt_iter),
                           "req=%p offset=%zu length=%zu", req,
                           req->send.state.dt_iter.offset,
                           req->send.state.dt_iter.length);
    } else {
        /* Last ATP was not sent yet or length was less than number of lanes -
           in both cases, each sent ATP acknowledged 1 byte. */
        ucp_datatype_iter_seek(&req->send.state.dt_iter,
                               req->send.rndv.put.atp_count,
                               UCS_BIT(UCP_DATATYPE_CONTIG));
    }

    req->flags &= ~UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    return UCS_OK;
}

ucp_proto_t ucp_rndv_put_zcopy_proto = {
    .name     = "rndv/put/zcopy",
    .desc     = NULL,
    .flags    = 0,
    .probe    = ucp_proto_rndv_put_zcopy_probe,
    .query    = ucp_proto_rndv_put_zcopy_query,
    .progress = {
        [UCP_PROTO_RNDV_PUT_ZCOPY_STAGE_SEND] = ucp_proto_rndv_put_zcopy_send_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_FLUSH]      = ucp_proto_rndv_put_common_flush_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_ATP]        = ucp_proto_rndv_put_common_atp_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP] = ucp_proto_rndv_put_common_fenced_atp_progress,
    },
    .abort    = ucp_proto_request_zcopy_abort,
    .reset    = ucp_proto_rndv_put_zcopy_reset
};


static void ucp_proto_rndv_put_mtype_pack_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);
    const ucp_proto_rndv_put_priv_t *rpriv;

    ucp_trace_req(req, "mtype_pack_completion mdesc %p", req->send.rndv.mdesc);

    rpriv = req->send.proto_config->priv;
    ucp_proto_completion_init(&req->send.state.uct_comp, rpriv->put_comp_cb);
    ucp_proto_request_set_stage(req, UCP_PROTO_RNDV_PUT_MTYPE_STAGE_SEND);
    ucp_request_send(req);
}

static UCS_F_ALWAYS_INLINE ucs_status_t ucp_proto_rndv_put_mtype_send_func(
        ucp_request_t *req, const ucp_proto_multi_lane_priv_t *lpriv,
        ucp_datatype_iter_t *next_iter, ucp_lane_index_t *lane_shift)
{
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;
    uct_iov_t iov;

    ucp_proto_rndv_mtype_next_iov(req, &rpriv->bulk, lpriv, next_iter, &iov);
    return ucp_proto_rndv_put_common_send(req, lpriv, &iov,
                                          &req->send.state.uct_comp);
}

static ucs_status_t
ucp_proto_rndv_put_mtype_copy_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req                     = ucs_container_of(uct_req,
                                                              ucp_request_t,
                                                              send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv = req->send.proto_config->priv;
    ucs_status_t status;

    ucs_assert(!(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED));

    status = ucp_proto_rndv_mtype_request_init(req, rpriv->bulk.frag_mem_type);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    ucp_proto_rndv_put_common_request_init(req);
    req->flags |= UCP_REQUEST_FLAG_PROTO_INITIALIZED;
    ucp_proto_rndv_mdesc_mtype_copy(req, uct_ep_get_zcopy,
                                    ucp_proto_rndv_put_mtype_pack_completion,
                                    "in from");

    return UCS_OK;
}

static ucs_status_t
ucp_proto_rndv_put_mtype_send_progress(uct_pending_req_t *uct_req)
{
    ucp_request_t *req = ucs_container_of(uct_req, ucp_request_t, send.uct);
    const ucp_proto_rndv_put_priv_t *rpriv;

    ucs_assert(req->flags & UCP_REQUEST_FLAG_PROTO_INITIALIZED);

    rpriv = req->send.proto_config->priv;
    return ucp_proto_multi_progress(req, &rpriv->bulk.mpriv,
                                    ucp_proto_rndv_put_mtype_send_func,
                                    ucp_proto_rndv_put_common_data_sent,
                                    UCS_BIT(UCP_DATATYPE_CONTIG));
}

static void ucp_proto_rndv_put_mtype_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "rndv_put_mtype_completion");
    ucs_mpool_put(req->send.rndv.mdesc);
    ucp_proto_rndv_put_common_complete(req);
}

static void ucp_proto_rndv_put_mtype_frag_completion(uct_completion_t *uct_comp)
{
    ucp_request_t *req = ucs_container_of(uct_comp, ucp_request_t,
                                          send.state.uct_comp);

    ucp_trace_req(req, "rndv_put_mtype_frag_completion");
    ucs_mpool_put(req->send.rndv.mdesc);
    ucp_proto_rndv_ppln_send_frag_complete(req, 1);
}

static void
ucp_proto_rndv_put_mtype_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_context_t *context = init_params->worker->context;
    uct_completion_callback_t comp_cb;
    ucp_md_map_t mdesc_md_map;
    ucs_status_t status;
    size_t frag_size;
    unsigned flags;
    ucp_md_index_t UCS_V_UNUSED dummy_md_id;
    ucp_memory_info_t frag_mem_info;

    if (init_params->rkey_config_key == NULL) {
        return;
    }

    /* Can initialize only the same fragment type as received in RTR
     * because pipeline protocols assume that both peers use the same
     * fragment sizes (and they are different for different memory types by
     * default). */
    frag_mem_info.type = init_params->rkey_config_key->mem_type;

    status = ucp_proto_rndv_mtype_init(init_params, frag_mem_info.type,
                                       &mdesc_md_map, &frag_size);
    if (status != UCS_OK) {
        return;
    }

    status = ucp_mm_get_alloc_md_index(context, frag_mem_info.type,
                                       &dummy_md_id,
                                       &frag_mem_info.sys_dev);
    if (status != UCS_OK) {
        return;
    }

    flags = context->config.ext.rndv_errh_ppln_enable ?
            UCP_PROTO_COMMON_INIT_FLAG_ERR_HANDLING : 0;

    if (ucp_proto_rndv_init_params_is_ppln_frag(init_params)) {
        comp_cb = ucp_proto_rndv_put_mtype_frag_completion;
    } else {
        comp_cb = ucp_proto_rndv_put_mtype_completion;
    }

    ucp_proto_rndv_put_common_probe(
            init_params, UCS_BIT(UCP_RNDV_MODE_PUT_PIPELINE), frag_size,
            UCT_EP_OP_GET_ZCOPY, flags, mdesc_md_map, comp_cb, 1,
            UCP_WORKER_STAT_RNDV_PUT_MTYPE_ZCOPY, &frag_mem_info);
}

static void
ucp_proto_rndv_put_mtype_query(const ucp_proto_query_params_t *params,
                               ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_put_priv_t *rpriv = params->priv;
    const char *put_desc;

    put_desc = ucp_proto_rndv_put_common_query(params, attr);
    ucp_proto_rndv_mtype_query_desc(params, rpriv->bulk.frag_mem_type, attr,
                                    put_desc);
}

ucp_proto_t ucp_rndv_put_mtype_proto = {
    .name     = "rndv/put/mtype",
    .desc     = NULL,
    .flags    = 0,
    .probe    = ucp_proto_rndv_put_mtype_probe,
    .query    = ucp_proto_rndv_put_mtype_query,
    .progress = {
        [UCP_PROTO_RNDV_PUT_MTYPE_STAGE_COPY] = ucp_proto_rndv_put_mtype_copy_progress,
        [UCP_PROTO_RNDV_PUT_MTYPE_STAGE_SEND] = ucp_proto_rndv_put_mtype_send_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_FLUSH]      = ucp_proto_rndv_put_common_flush_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_ATP]        = ucp_proto_rndv_put_common_atp_progress,
        [UCP_PROTO_RNDV_PUT_STAGE_FENCED_ATP] = ucp_proto_rndv_put_common_fenced_atp_progress,
    },
    .abort    = ucp_proto_rndv_stub_abort,
    .reset    = (ucp_request_reset_func_t)ucp_proto_reset_fatal_not_implemented
};
