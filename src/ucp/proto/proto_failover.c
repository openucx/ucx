/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_failover.h"

#include <ucp/core/ucp_ep.inl>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_rkey.h>
#include <ucp/core/ucp_rkey.inl>
#include <ucp/rma/rma.h>
#include <ucs/debug/log.h>

#include <string.h>


static int
ucp_proto_failover_replay_op_supported(const uct_ep_op_info_t *op_info)
{
    const uint64_t data_mask = UCT_EP_OP_INFO_FIELD_DATA;

    if (!(op_info->field_mask & UCT_EP_OP_INFO_FIELD_OPERATION)) {
        return 0;
    }

    switch (op_info->operation) {
    case UCT_EP_OP_AM_BCOPY:
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

    length = op_info->data.length;
    if ((length > 0) && (op_info->data.buffer == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    op = ucs_malloc(sizeof(*op) + length, "failover_replay_op");
    if (op == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    op->req  = NULL;
    op->info = *op_info;
    if (length > 0) {
        memcpy(op->data, op_info->data.buffer, length);
        op->info.data.buffer = op->data;
    } else {
        op->info.data.buffer = NULL;
    }

    *replay_op_p = op;
    return UCS_OK;
}


void ucp_proto_failover_replay_op_destroy(ucp_proto_failover_replay_op_t *op,
                                          ucs_status_t status)
{
    if (op->req != NULL) {
        ucp_request_reset_super(op->req);
        ucp_request_put(op->req);
    }

    if ((op->info.field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op->info.comp != NULL)) {
        ucp_invoke_uct_completion(op->info.comp, status);
    }

    ucs_free(op);
}


static size_t ucp_proto_failover_pack(void *dest, void *arg)
{
    const uct_ep_op_info_t *op_info = arg;
    size_t length                   = op_info->data.length;

    if (length > 0) {
        memcpy(dest, op_info->data.buffer, length);
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


static unsigned ucp_proto_failover_am_flags(const uct_ep_op_info_t *op_info)
{
    return (op_info->field_mask & UCT_EP_OP_INFO_FIELD_AM_FLAGS) ?
                   op_info->am.flags :
                   0;
}


static ucp_lane_index_t
ucp_proto_failover_am_lane(ucp_ep_h ep, ucp_lane_index_t failed_lane)
{
    const ucp_ep_config_key_t *key = &ucp_ep_config(ep)->key;
    ucp_lane_index_t lane;

    if ((key->am_lane != UCP_NULL_LANE) && (key->am_lane != failed_lane)) {
        return key->am_lane;
    }

    for (lane = 0; lane < key->num_lanes; ++lane) {
        if ((lane != failed_lane) &&
            (key->lanes[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_AM))) {
            return lane;
        }
    }

    return UCP_NULL_LANE;
}


static ucp_lane_index_t
ucp_proto_failover_rma_lane(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                            ucp_rkey_h rkey, int is_bw, uct_rkey_t *tl_rkey_p)
{
    const ucp_ep_config_t *config = ucp_ep_config(ep);
    const ucp_ep_config_key_t *key = &config->key;
    const ucp_lane_index_t *lanes;
    ucp_md_index_t failed_md;
    ucp_lane_index_t lane;
    int prio;

    if (rkey != NULL) {
        lanes = is_bw ? key->rma_bw_lanes : key->rma_lanes;
        return ucp_rkey_find_rma_lane(ep->worker->context, config,
                                     UCS_MEMORY_TYPE_HOST, lanes, rkey,
                                     UCS_BIT(failed_lane), tl_rkey_p);
    }

    if ((failed_lane >= key->num_lanes) ||
        (key->lanes[failed_lane].rsc_index == UCP_NULL_RESOURCE)) {
        return UCP_NULL_LANE;
    }

    failed_md = key->lanes[failed_lane].dst_md_index;
    lanes     = is_bw ? key->rma_bw_lanes : key->rma_lanes;
    for (prio = 0; prio < UCP_MAX_LANES; ++prio) {
        lane = lanes[prio];
        if (lane == UCP_NULL_LANE) {
            break;
        }

        if ((lane == failed_lane) ||
            (key->lanes[lane].dst_md_index != failed_md)) {
            continue;
        }

        *tl_rkey_p = UCT_INVALID_RKEY;
        return lane;
    }

    return UCP_NULL_LANE;
}


static ucs_status_t
ucp_proto_failover_replay_op_request_init(ucp_ep_h ep,
                                          ucp_lane_index_t failed_lane,
                                          ucp_request_t *super_req,
                                          ucp_proto_failover_replay_op_t *op)
{
    ucp_rma_op_t *rma_op = NULL;
    ucp_request_t *req;
    ucp_lane_index_t lane;
    uct_rkey_t tl_rkey;
    int is_bw;

    if ((op->info.field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op->info.comp != NULL)) {
        rma_op = ucs_container_of(op->info.comp, ucp_rma_op_t, comp);
    }

    switch (op->info.operation) {
    case UCT_EP_OP_AM_BCOPY:
        lane = ucp_proto_failover_am_lane(ep, failed_lane);
        tl_rkey = 0;
        break;
    case UCT_EP_OP_PUT_SHORT:
        is_bw = 0;
        lane  = ucp_proto_failover_rma_lane(ep, failed_lane,
                                           rma_op ? rma_op->rkey : NULL, is_bw,
                                           &tl_rkey);
        if ((lane != UCP_NULL_LANE) && (rma_op == NULL)) {
            tl_rkey = op->info.rma.rkey;
        }
        break;
    case UCT_EP_OP_PUT_BCOPY:
        is_bw = 1;
        lane  = ucp_proto_failover_rma_lane(ep, failed_lane,
                                           rma_op ? rma_op->rkey : NULL, is_bw,
                                           &tl_rkey);
        if ((lane != UCP_NULL_LANE) && (rma_op == NULL)) {
            tl_rkey = op->info.rma.rkey;
        }
        break;
    default:
        return UCS_ERR_UNSUPPORTED;
    }

    if (lane == UCP_NULL_LANE) {
        return UCS_ERR_UNREACHABLE;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->status                    = UCS_INPROGRESS;
    req->flags                     = 0;
    req->send.ep                   = ep;
    req->send.lane                 = lane;
    req->send.failover.op_info     = &op->info;
    req->send.failover.failed_lane = failed_lane;
    req->send.failover.tl_rkey     = tl_rkey;
    ucp_request_set_super(req, super_req);

    op->req = req;
    return UCS_OK;
}


static ucs_status_t
ucp_proto_failover_replay_am_bcopy(ucp_request_t *req)
{
    const uct_ep_op_info_t *op_info = req->send.failover.op_info;

    return ucp_proto_failover_bcopy_status(
            uct_ep_am_bcopy(ucp_ep_get_lane(req->send.ep, req->send.lane),
                            op_info->am.am_id, ucp_proto_failover_pack,
                            (void*)op_info,
                            ucp_proto_failover_am_flags(op_info)));
}


static ucs_status_t
ucp_proto_failover_replay_put_short(ucp_request_t *req)
{
    const uct_ep_op_info_t *op_info = req->send.failover.op_info;

    return uct_ep_put_short(ucp_ep_get_lane(req->send.ep, req->send.lane),
                            op_info->data.buffer, op_info->data.length,
                            op_info->rma.remote_addr,
                            req->send.failover.tl_rkey);
}


static ucs_status_t
ucp_proto_failover_replay_put_bcopy(ucp_request_t *req)
{
    const uct_ep_op_info_t *op_info = req->send.failover.op_info;
    ssize_t packed_size;

    if ((op_info->field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op_info->comp != NULL)) {
        packed_size = uct_ep_put_bcopy_ft(
                ucp_ep_get_lane(req->send.ep, req->send.lane),
                ucp_proto_failover_pack, (void*)op_info,
                op_info->rma.remote_addr, req->send.failover.tl_rkey,
                op_info->comp);
    } else {
        packed_size = uct_ep_put_bcopy(
                ucp_ep_get_lane(req->send.ep, req->send.lane),
                ucp_proto_failover_pack, (void*)op_info,
                op_info->rma.remote_addr, req->send.failover.tl_rkey);
    }

    return ucp_proto_failover_bcopy_status(packed_size);
}


ucs_status_t
ucp_proto_failover_replay_op_progress(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                                      ucp_request_t *super_req,
                                      ucp_proto_failover_replay_op_t *op)
{
    ucp_request_t *req;
    ucs_status_t status;

    if (op->req == NULL) {
        status = ucp_proto_failover_replay_op_request_init(ep, failed_lane,
                                                           super_req, op);
        if (status != UCS_OK) {
            return status;
        }
    }

    req = op->req;
    switch (op->info.operation) {
    case UCT_EP_OP_AM_BCOPY:
        status = ucp_proto_failover_replay_am_bcopy(req);
        break;
    case UCT_EP_OP_PUT_SHORT:
        status = ucp_proto_failover_replay_put_short(req);
        break;
    case UCT_EP_OP_PUT_BCOPY:
        status = ucp_proto_failover_replay_put_bcopy(req);
        break;
    default:
        status = UCS_ERR_UNSUPPORTED;
        break;
    }

    if (status == UCS_ERR_NO_RESOURCE) {
        return status;
    }

    if (status == UCS_OK) {
        if ((op->info.operation == UCT_EP_OP_PUT_BCOPY) &&
            (op->info.field_mask & UCT_EP_OP_INFO_FIELD_COMP)) {
            op->info.comp = NULL;
        }

        ucs_trace("ep %p: replayed failover op %d on lane %u", ep,
                  (int)op->info.operation, req->send.lane);
    }

    return status;
}
