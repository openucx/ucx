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
#include <ucp/core/ucp_ep_failover.h>
#include <ucp/core/ucp_request.inl>
#include <ucp/core/ucp_worker.h>
#include <ucp/proto/proto_common.h>
#include <ucp/wireup/wireup_ep.h>
#include <ucs/debug/log.h>

#include <string.h>


/* Contiguous payload accessors matching onto uct_ep_op_info nested layout. */
static void *
ucp_proto_failover_op_data_buffer(const uct_ep_op_info_t *op_info)
{
    switch (op_info->operation) {
    case UCT_EP_OP_AM_SHORT:
    case UCT_EP_OP_AM_BCOPY:
        return op_info->am.payload.data.buffer;
    default:
        return NULL;
    }
}

static size_t
ucp_proto_failover_op_data_length(const uct_ep_op_info_t *op_info)
{
    switch (op_info->operation) {
    case UCT_EP_OP_AM_SHORT:
    case UCT_EP_OP_AM_BCOPY:
        return op_info->am.payload.data.length;
    default:
        return 0;
    }
}

static void
ucp_proto_failover_op_set_data_buffer(uct_ep_op_info_t *op_info, void *buffer)
{
    switch (op_info->operation) {
    case UCT_EP_OP_AM_SHORT:
    case UCT_EP_OP_AM_BCOPY:
        op_info->am.payload.data.buffer = buffer;
        break;
    default:
        break;
    }
}


/* Only AM short and AM bcopy are re-posted from extracted WQEs. RMA and zcopy
 * operations keep their user buffers, so they are recovered by restarting the
 * owning UCP request instead. */
static int
ucp_proto_failover_replay_op_supported(const uct_ep_op_info_t *op_info)
{
    if (!(op_info->field_mask & UCT_EP_OP_INFO_FIELD_OPERATION) ||
        !(op_info->field_mask & UCT_EP_OP_INFO_FIELD_AM)) {
        return 0;
    }

    switch (op_info->operation) {
    case UCT_EP_OP_AM_SHORT:
        return ucs_test_all_flags(op_info->am.field_mask,
                                  UCT_EP_OP_INFO_AM_FIELD_AM_ID |
                                  UCT_EP_OP_INFO_AM_FIELD_HEADER_VALUE |
                                  UCT_EP_OP_INFO_AM_FIELD_PAYLOAD_DATA);
    case UCT_EP_OP_AM_BCOPY:
        return ucs_test_all_flags(op_info->am.field_mask,
                                  UCT_EP_OP_INFO_AM_FIELD_AM_ID |
                                  UCT_EP_OP_INFO_AM_FIELD_PAYLOAD_DATA);
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
    void *src;

    if (!ucp_proto_failover_replay_op_supported(op_info)) {
        return UCS_ERR_UNSUPPORTED;
    }

    length = ucp_proto_failover_op_data_length(op_info);
    src    = ucp_proto_failover_op_data_buffer(op_info);
    if ((length > 0) && (src == NULL)) {
        return UCS_ERR_INVALID_PARAM;
    }

    op = ucs_malloc(sizeof(*op) + length, "failover_replay_op");
    if (op == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    op->req  = NULL;
    op->info = *op_info;
    if (length > 0) {
        memcpy(op->data, src, length);
        ucp_proto_failover_op_set_data_buffer(&op->info, op->data);
    } else {
        ucp_proto_failover_op_set_data_buffer(&op->info, NULL);
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
        op->req = NULL;
    }

    if ((op->info.field_mask & UCT_EP_OP_INFO_FIELD_COMP) &&
        (op->info.comp != NULL)) {
        ucp_invoke_uct_completion(op->info.comp, status);
        op->info.comp = NULL;
    }

    ucs_free(op);
}


static size_t ucp_proto_failover_pack(void *dest, void *arg)
{
    const uct_ep_op_info_t *op_info = arg;
    size_t length                   = ucp_proto_failover_op_data_length(op_info);
    void *src                       = ucp_proto_failover_op_data_buffer(op_info);

    if (length > 0) {
        memcpy(dest, src, length);
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
    return (op_info->am.field_mask & UCT_EP_OP_INFO_AM_FIELD_FLAGS) ?
                   op_info->am.flags :
                   0;
}


static int
ucp_proto_failover_lane_is_usable(ucp_ep_h ep, ucp_lane_index_t lane,
                                  uint64_t required_flags)
{
    const ucp_ep_config_key_t *key = &ucp_ep_config(ep)->key;
    ucp_rsc_index_t rsc_index;
    const uct_iface_attr_t *attr;
    uct_ep_h uct_ep;

    if ((lane == UCP_NULL_LANE) || (lane >= key->num_lanes)) {
        return 0;
    }

    if (key->lanes[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_FAILED)) {
        return 0;
    }

    rsc_index = key->lanes[lane].rsc_index;
    if (rsc_index == UCP_NULL_RESOURCE) {
        return 0;
    }

    uct_ep = ucp_ep_get_lane(ep, lane);
    if (uct_ep == NULL) {
        return 0;
    }

    /* Recovering wireup proxy is usable: pending_add holds the request until
     * the lane is ready. */
    if (ucp_wireup_ep_test(uct_ep)) {
        return 1;
    }

    if (ucp_is_uct_ep_failed(uct_ep)) {
        return 0;
    }

    attr = ucp_worker_iface_get_attr(ep->worker, rsc_index);
    return (attr->cap.flags & required_flags) == required_flags;
}


static ucp_lane_index_t
ucp_proto_failover_am_lane(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                           uint64_t required_flags)
{
    const ucp_ep_config_key_t *key = &ucp_ep_config(ep)->key;
    ucp_lane_index_t lane;

    if (ucp_proto_failover_lane_is_usable(ep, key->am_lane, required_flags) &&
        (key->am_lane != failed_lane)) {
        return key->am_lane;
    }

    for (lane = 0; lane < key->num_lanes; ++lane) {
        if ((lane == failed_lane) ||
            !(key->lanes[lane].lane_types & UCS_BIT(UCP_LANE_TYPE_AM))) {
            continue;
        }

        if (ucp_proto_failover_lane_is_usable(ep, lane, required_flags)) {
            return lane;
        }
    }

    /* Fall back to the recovering failed lane itself (wireup pending holds
     * the op until that lane is rebuilt). */
    if (ucp_proto_failover_lane_is_usable(ep, failed_lane, required_flags)) {
        return failed_lane;
    }

    return UCP_NULL_LANE;
}


static ucs_status_t
ucp_proto_failover_replay_select_lane(ucp_request_t *req)
{
    const uct_ep_op_info_t *op_info = req->send.failover.op_info;
    ucp_ep_h ep                     = req->send.ep;
    ucp_lane_index_t failed_lane    = req->send.failover.failed_lane;
    ucp_lane_index_t lane;

    switch (op_info->operation) {
    case UCT_EP_OP_AM_SHORT:
        lane = ucp_proto_failover_am_lane(ep, failed_lane,
                                          UCT_IFACE_FLAG_AM_SHORT);
        break;
    case UCT_EP_OP_AM_BCOPY:
        lane = ucp_proto_failover_am_lane(ep, failed_lane,
                                          UCT_IFACE_FLAG_AM_BCOPY);
        break;
    default:
        return UCS_ERR_UNSUPPORTED;
    }

    if (lane == UCP_NULL_LANE) {
        return UCS_ERR_UNREACHABLE;
    }

    req->send.lane = lane;
    return UCS_OK;
}


static ucs_status_t
ucp_proto_failover_replay_am_short(ucp_request_t *req)
{
    const uct_ep_op_info_t *op_info = req->send.failover.op_info;

    return uct_ep_am_short(ucp_ep_get_lane(req->send.ep, req->send.lane),
                           op_info->am.am_id, op_info->am.header.value,
                           ucp_proto_failover_op_data_buffer(op_info),
                           ucp_proto_failover_op_data_length(op_info));
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


static void
ucp_proto_failover_replay_finish(ucp_request_t *req, ucs_status_t status)
{
    ucp_proto_failover_replay_op_t *op =
            ucs_container_of(req->send.failover.op_info,
                             ucp_proto_failover_replay_op_t, info);
    ucp_ep_h ep                        = req->send.ep;
    ucp_lane_index_t failed_lane       = req->send.failover.failed_lane;

    op->req = NULL;
    ucp_request_reset_super(req);
    ucp_request_put(req);
    ucp_proto_failover_replay_op_destroy(op, status);
    ucp_ep_failover_replay_completed(ep, failed_lane, status);
}


void ucp_proto_failover_replay_abort(ucp_request_t *req, ucs_status_t status)
{
    ucp_proto_failover_replay_finish(req, status);
}


ucs_status_t ucp_proto_failover_replay_progress(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucs_status_t status;
    const uct_ep_op_info_t *op_info;

    status = ucp_proto_failover_replay_select_lane(req);
    if (status == UCS_ERR_UNREACHABLE) {
        /* Lane not ready yet; keep the request for another progress attempt. */
        return UCS_ERR_NO_RESOURCE;
    } else if (status != UCS_OK) {
        ucp_proto_failover_replay_finish(req, status);
        return UCS_OK;
    }

    op_info = req->send.failover.op_info;
    switch (op_info->operation) {
    case UCT_EP_OP_AM_SHORT:
        status = ucp_proto_failover_replay_am_short(req);
        break;
    case UCT_EP_OP_AM_BCOPY:
        status = ucp_proto_failover_replay_am_bcopy(req);
        break;
    default:
        status = UCS_ERR_UNSUPPORTED;
        break;
    }

    if (status == UCS_ERR_NO_RESOURCE) {
        return status;
    }

    if (status == UCS_OK) {
        ucs_trace("ep %p: replayed failover op %d on lane %u", req->send.ep,
                  (int)op_info->operation, req->send.lane);
    }

    ucp_proto_failover_replay_finish(req, status);
    return UCS_OK;
}


ucs_status_t
ucp_proto_failover_replay_op_start(ucp_ep_h ep, ucp_lane_index_t failed_lane,
                                   ucp_request_t *super_req,
                                   ucp_proto_failover_replay_op_t *op)
{
    ucp_request_t *req;
    ucs_status_t status;

    /* Probe lane availability before allocating the request so we can retry
     * cleanly when recovery has not installed wireup proxies yet. */
    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->status                    = UCS_INPROGRESS;
    req->flags                     = 0;
    req->send.ep                   = ep;
    req->send.lane                 = UCP_NULL_LANE;
    req->send.uct.func             = ucp_proto_failover_replay_progress;
    req->send.failover.op_info     = &op->info;
    req->send.failover.failed_lane = failed_lane;
    req->send.state.uct_comp.func  = NULL;
    ucp_request_set_super(req, super_req);
    op->req = req;

    status = ucp_proto_failover_replay_select_lane(req);
    if (status == UCS_ERR_UNREACHABLE) {
        /* Recovery still installing proxies / no live capable lane yet. */
        op->req = NULL;
        ucp_request_reset_super(req);
        ucp_request_put(req);
        return UCS_ERR_NO_RESOURCE;
    } else if (status != UCS_OK) {
        op->req = NULL;
        ucp_request_reset_super(req);
        ucp_request_put(req);
        return status;
    }

    ucp_request_send(req);
    return UCS_OK;
}
