/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_failover.h"

#include <ucp/core/ucp_context.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/proto/proto_common.inl>
#include <ucp/proto/proto_init.h>
#include <ucp/proto/proto_single.h>
#include <ucp/wireup/wireup_ep.h>
#include <uct/base/uct_iface.h>
#include <ucs/debug/log.h>

#include <limits.h>
#include <string.h>


static int
ucp_proto_failover_replay_op_supported(const uct_ep_op_info_t *op_info)
{
    const uint64_t data_mask = UCT_EP_OP_INFO_FIELD_DATA;

    switch (op_info->operation) {
    case UCT_EP_OP_AM_BCOPY:
        /* Extract normalizes inline AM WQEs to AM_BCOPY. AM_SHORT carries a
         * separate 64-bit header and is intentionally not replayed here. */
        return ucs_test_all_flags(op_info->field_mask,
                                  UCT_EP_OP_INFO_FIELD_AM | data_mask);
    case UCT_EP_OP_PUT_SHORT:
    case UCT_EP_OP_PUT_BCOPY:
        return ucs_test_all_flags(op_info->field_mask,
                                  UCT_EP_OP_INFO_FIELD_RMA | data_mask);
    default:
        return 0;
    }
}

ucs_status_t
ucp_proto_failover_replay_op_create(const uct_ep_op_info_t *op_info,
                                    ucp_proto_failover_replay_op_t **replay_op_p)
{
    ucp_proto_failover_replay_op_t *op;
    size_t length;

    if (!ucp_proto_failover_replay_op_supported(op_info)) {
        return UCS_ERR_UNSUPPORTED;
    }

    length = op_info->inline_data.length;
    if ((length > 0) && (op_info->inline_data.buffer == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    op = ucs_malloc(sizeof(*op) + length, "failover_replay_op");
    if (op == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    op->req  = NULL;
    op->info = *op_info;
    if (length > 0) {
        memcpy(op->data, op_info->inline_data.buffer, length);
        op->info.inline_data.buffer = op->data;
    } else {
        op->info.inline_data.buffer = NULL;
    }

    *replay_op_p = op;
    return UCS_OK;
}

void ucp_proto_failover_replay_op_destroy(ucp_proto_failover_replay_op_t *op)
{
    if (op->req != NULL) {
        if (!(op->req->flags & UCP_REQUEST_FLAG_COMPLETED)) {
            ucp_datatype_iter_cleanup(&op->req->send.state.dt_iter, 0,
                                      UCS_BIT(UCP_DATATYPE_CONTIG));
        }

        ucp_request_put(op->req);
    }

    ucs_free(op);
}

static ucp_operation_id_t
ucp_proto_failover_replay_op_id(const uct_ep_op_info_t *op_info)
{
    switch (op_info->operation) {
    case UCT_EP_OP_AM_BCOPY:
        return UCP_OP_ID_FAILOVER_AM_BCOPY;
    case UCT_EP_OP_PUT_SHORT:
        return UCP_OP_ID_FAILOVER_PUT_SHORT;
    case UCT_EP_OP_PUT_BCOPY:
        return UCP_OP_ID_FAILOVER_PUT_BCOPY;
    default:
        return UCP_OP_ID_LAST;
    }
}

static size_t ucp_proto_failover_pack(void *dest, void *arg)
{
    const uct_ep_op_info_t *op_info = arg;
    size_t length                   = op_info->inline_data.length;

    if (length > 0) {
        memcpy(dest, op_info->inline_data.buffer, length);
    }

    return length;
}

static ucs_status_t ucp_proto_failover_bcopy_status(ssize_t packed_size)
{
    if (ucs_unlikely(packed_size < 0)) {
        return (ucs_status_t)packed_size;
    }

    return UCS_OK;
}

static unsigned ucp_proto_failover_uct_flags(const uct_ep_op_info_t *op_info)
{
    return (op_info->field_mask & UCT_EP_OP_INFO_FIELD_FLAGS) ? op_info->flags :
                                                                0;
}

static ucs_status_t
ucp_proto_failover_am_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    const uct_ep_op_info_t *op_info      = req->send.failover.op_info;
    ssize_t packed_size;

    packed_size = uct_ep_am_bcopy(ucp_ep_get_lane(req->send.ep,
                                                  spriv->super.lane),
                                  op_info->am.am_id, ucp_proto_failover_pack,
                                  (void*)op_info,
                                  ucp_proto_failover_uct_flags(op_info));

    return ucp_proto_failover_bcopy_status(packed_size);
}

static ucs_status_t
ucp_proto_failover_put_short_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    const uct_ep_op_info_t *op_info      = req->send.failover.op_info;

    return uct_ep_put_short(ucp_ep_get_lane(req->send.ep, spriv->super.lane),
                            op_info->inline_data.buffer,
                            op_info->inline_data.length,
                            op_info->rma.remote_addr, op_info->rma.rkey);
}

static ucs_status_t
ucp_proto_failover_put_bcopy_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_single_priv_t *spriv = req->send.proto_config->priv;
    const uct_ep_op_info_t *op_info      = req->send.failover.op_info;
    ssize_t packed_size;

    packed_size = uct_ep_put_bcopy(ucp_ep_get_lane(req->send.ep,
                                                   spriv->super.lane),
                                   ucp_proto_failover_pack, (void*)op_info,
                                   op_info->rma.remote_addr, op_info->rma.rkey);

    return ucp_proto_failover_bcopy_status(packed_size);
}

static ucp_lane_map_t
ucp_proto_failover_exclude_map(const ucp_proto_init_params_t *init_params,
                               int same_md)
{
    const ucp_ep_config_key_t *key = init_params->ep_config_key;
    ucp_context_h context          = init_params->worker->context;
    ucp_lane_index_t failed_lane =
            init_params->select_param->op.failover.failed_lane;
    ucp_rsc_index_t failed_rsc;
    ucp_md_index_t failed_md;
    ucp_lane_map_t exclude_map = 0;
    ucp_rsc_index_t rsc_index;
    ucp_lane_index_t lane;

    if (failed_lane >= key->num_lanes) {
        return UCS_MASK(UCP_MAX_LANES);
    }

    exclude_map |= UCS_BIT(failed_lane);
    if (!same_md) {
        return exclude_map;
    }

    /* RMA replay stores the UCT rkey extracted from the failed WQE. Since UCP
     * does not have a ucp_rkey_h here to repack per-lane keys, only lanes on the
     * same MD can use that rkey safely. */
    failed_rsc = key->lanes[failed_lane].rsc_index;
    if (failed_rsc == UCP_NULL_RESOURCE) {
        return UCS_MASK(UCP_MAX_LANES);
    }

    failed_md = context->tl_rscs[failed_rsc].md_index;
    for (lane = 0; lane < key->num_lanes; ++lane) {
        rsc_index = key->lanes[lane].rsc_index;
        if ((rsc_index == UCP_NULL_RESOURCE) ||
            (context->tl_rscs[rsc_index].md_index != failed_md)) {
            exclude_map |= UCS_BIT(lane);
        }
    }

    return exclude_map;
}

static void
ucp_proto_failover_probe_common(const ucp_proto_init_params_t *init_params,
                                ucp_operation_id_t op_id,
                                ucp_lane_type_t lane_type,
                                uint64_t tl_cap_flags, ptrdiff_t max_frag_offs,
                                uct_ep_operation_t send_op, int same_md)
{
    ucp_proto_single_init_params_t params = {
        .super.super         = *init_params,
        .super.latency       = 0,
        .super.overhead      = 0,
        .super.cfg_thresh    = UCS_MEMUNITS_AUTO,
        .super.cfg_priority  = 0,
        .super.min_length    = 0,
        .super.max_length    = SIZE_MAX,
        .super.min_iov       = 0,
        .super.min_frag_offs = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.max_frag_offs = max_frag_offs,
        .super.max_iov_offs  = UCP_PROTO_COMMON_OFFSET_INVALID,
        .super.hdr_size      = 0,
        .super.send_op       = send_op,
        .super.memtype_op    = UCT_EP_OP_LAST,
        .super.flags         = UCP_PROTO_COMMON_INIT_FLAG_SINGLE_FRAG |
                               UCP_PROTO_COMMON_INIT_FLAG_FAILOVER,
        .super.exclude_map   = ucp_proto_failover_exclude_map(init_params,
                                                            same_md),
        .super.reg_mem_info  = ucp_mem_info_unknown,
        .lane_type           = lane_type,
        .tl_cap_flags        = tl_cap_flags,
        .tl_v2_cap_flags     = UCT_IFACE_FLAG_V2_QUERY_TOKEN
    };

    if (init_params->ep_config_key->err_mode !=
        UCP_ERR_HANDLING_MODE_FAILOVER) {
        return;
    }

    if (!ucp_proto_init_check_op(init_params, UCS_BIT(op_id)) ||
        !ucp_proto_is_short_supported(init_params->select_param)) {
        return;
    }

    ucp_proto_single_probe(&params);
}

static void
ucp_proto_failover_am_bcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_failover_probe_common(init_params, UCP_OP_ID_FAILOVER_AM_BCOPY,
                                    UCP_LANE_TYPE_AM, UCT_IFACE_FLAG_AM_BCOPY,
                                    ucs_offsetof(uct_iface_attr_t,
                                                 cap.am.max_bcopy),
                                    UCT_EP_OP_AM_BCOPY, 0);
}

static void
ucp_proto_failover_put_short_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_failover_probe_common(init_params, UCP_OP_ID_FAILOVER_PUT_SHORT,
                                    UCP_LANE_TYPE_RMA, UCT_IFACE_FLAG_PUT_SHORT,
                                    ucs_offsetof(uct_iface_attr_t,
                                                 cap.put.max_short),
                                    UCT_EP_OP_PUT_SHORT, 1);
}

static void
ucp_proto_failover_put_bcopy_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_failover_probe_common(init_params, UCP_OP_ID_FAILOVER_PUT_BCOPY,
                                    UCP_LANE_TYPE_RMA, UCT_IFACE_FLAG_PUT_BCOPY,
                                    ucs_offsetof(uct_iface_attr_t,
                                                 cap.put.max_bcopy),
                                    UCT_EP_OP_PUT_BCOPY, 1);
}

static void
ucp_proto_failover_replay_probe(const ucp_proto_init_params_t *init_params)
{
    switch (ucp_proto_select_op_id(init_params->select_param)) {
    case UCP_OP_ID_FAILOVER_AM_BCOPY:
        ucp_proto_failover_am_bcopy_probe(init_params);
        break;
    case UCP_OP_ID_FAILOVER_PUT_SHORT:
        ucp_proto_failover_put_short_probe(init_params);
        break;
    case UCP_OP_ID_FAILOVER_PUT_BCOPY:
        ucp_proto_failover_put_bcopy_probe(init_params);
        break;
    default:
        break;
    }
}

static void
ucp_proto_failover_select_param_init(ucp_proto_select_param_t *select_param,
                                     ucp_operation_id_t op_id,
                                     ucp_lane_index_t failed_lane)
{
    ucp_memory_info_t mem_info;

    ucp_memory_info_set_host(&mem_info);
    ucp_proto_select_param_init(select_param, op_id, 0, 0, UCP_DATATYPE_CONTIG,
                                &mem_info, 1);
    select_param->op.failover.failed_lane = failed_lane;
    select_param->op.failover.reserved    = 0;
}

static ucs_status_t
ucp_proto_failover_replay_op_request_init(ucp_ep_h ep,
                                          ucp_lane_index_t failed_lane,
                                          ucp_proto_failover_replay_op_t *op)
{
    ucp_operation_id_t op_id;
    ucp_proto_select_param_t select_param;
    ucp_worker_cfg_index_t rkey_cfg_index;
    ucp_proto_select_t *proto_select;
    ucp_memory_info_t mem_info;
    ucp_request_t *req;
    ucs_status_t status;
    size_t length;

    op_id = ucp_proto_failover_replay_op_id(&op->info);
    if (op_id == UCP_OP_ID_LAST) {
        return UCS_ERR_UNSUPPORTED;
    }

    proto_select = ucp_proto_select_get(ep->worker, ep->cfg_index,
                                        UCP_WORKER_CFG_INDEX_NULL,
                                        &rkey_cfg_index);
    if (proto_select == NULL) {
        return UCS_ERR_UNSUPPORTED;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    length                         = op->info.inline_data.length;
    req->status                    = UCS_INPROGRESS;
    req->flags                     = UCP_REQUEST_FLAG_PROTO_SEND;
    req->send.ep                   = ep;
    req->send.failover.op_info     = &op->info;
    req->send.failover.failed_lane = failed_lane;
    ucp_memory_info_set_host(&mem_info);
    ucp_datatype_iter_init_contig(&req->send.state.dt_iter, op->data, length,
                                  &mem_info);
    ucp_request_send_state_reset(req, NULL, UCP_REQUEST_SEND_PROTO_BCOPY_AM);

    ucp_proto_failover_select_param_init(&select_param, op_id, failed_lane);
    status = ucp_proto_request_lookup_proto(ep->worker, ep, req, proto_select,
                                            rkey_cfg_index, &select_param,
                                            length);
    if (status != UCS_OK) {
        ucp_datatype_iter_cleanup(&req->send.state.dt_iter, 0,
                                  UCS_BIT(UCP_DATATYPE_CONTIG));
        ucp_request_put(req);
        return status;
    }

    op->req = req;
    return UCS_OK;
}

ucs_status_t
ucp_proto_failover_replay_op_progress(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                                      ucp_proto_failover_replay_op_t *op)
{
    ucs_status_t status;

    if (op->req == NULL) {
        status = ucp_proto_failover_replay_op_request_init(ep, failed_lane, op);
        if (status != UCS_OK) {
            return status;
        }
    }

    status = op->req->send.proto_config->proto->progress[0](&op->req->send.uct);
    if (status == UCS_ERR_NO_RESOURCE) {
        return status;
    }

    ucp_datatype_iter_cleanup(&op->req->send.state.dt_iter, 0,
                              UCS_BIT(UCP_DATATYPE_CONTIG));
    ucp_request_complete_send(op->req, status);
    if (status == UCS_OK) {
        ucs_trace("ep %p: replayed failover op %d on lane %u", ep,
                  (int)op->info.operation,
                  ((const ucp_proto_single_priv_t*)
                   op->req->send.proto_config->priv)->super.lane);
    }

    return status;
}

static ucs_status_t ucp_proto_failover_replay_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    switch (ucp_proto_select_op_id(&req->send.proto_config->select_param)) {
    case UCP_OP_ID_FAILOVER_AM_BCOPY:
        return ucp_proto_failover_am_bcopy_progress(self);
    case UCP_OP_ID_FAILOVER_PUT_SHORT:
        return ucp_proto_failover_put_short_progress(self);
    case UCP_OP_ID_FAILOVER_PUT_BCOPY:
        return ucp_proto_failover_put_bcopy_progress(self);
    default:
        return UCS_ERR_UNSUPPORTED;
    }
}

ucp_proto_t ucp_failover_replay_proto = {
    .name     = "failover/replay",
    .desc     = "failover replay",
    .flags    = 0,
    .dt_mask  = UCS_BIT(UCP_DATATYPE_CONTIG),
    .probe    = ucp_proto_failover_replay_probe,
    .query    = ucp_proto_single_query,
    .progress = {ucp_proto_failover_replay_progress},
    .abort    = ucp_proto_request_bcopy_abort,
    .reset    = ucp_proto_request_bcopy_reset
};
