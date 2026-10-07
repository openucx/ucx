/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2026. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_repost.h"
#include "proto_debug.h"
#include "proto_perf.h"
#include "proto_common.inl"

#include <ucp/core/ucp_ep.inl>

#include <string.h>


/* Operations outstanding purge reports.
 * The lane must support all of them since we have single protocol for reposting. */
#define UCP_PROTO_REPOST_TL_CAP_FLAGS \
    (UCT_IFACE_FLAG_AM_SHORT  | UCT_IFACE_FLAG_AM_BCOPY  | \
     UCT_IFACE_FLAG_PUT_SHORT | UCT_IFACE_FLAG_PUT_BCOPY | \
     UCT_IFACE_FLAG_PUT_ZCOPY)


typedef struct {
    ucp_lane_index_t lane;
} ucp_proto_repost_priv_t;


typedef struct {
    const void *buffer;
    size_t     length;
} ucp_proto_repost_pack_t;


static size_t ucp_proto_repost_pack(void *dest, void *arg)
{
    ucp_proto_repost_pack_t *pack = arg;

    if (pack->length > 0) {
        memcpy(dest, pack->buffer, pack->length);
    }

    return pack->length;
}

static void ucp_proto_repost_probe(const ucp_proto_init_params_t *params)
{
    ucp_proto_perf_factors_t perf_factors = UCP_PROTO_PERF_FACTORS_INITIALIZER;
    ucp_lane_index_t lane                 = UCP_NULL_LANE;
    ucp_lane_index_t num_lanes;
    ucp_proto_repost_priv_t rpriv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    UCS_STATIC_ASSERT(UCP_OP_ID_REPOST < UCP_PROTO_SELECT_OP_FLAGS_BASE);

    if (params->select_param->op_id_flags != UCP_OP_ID_REPOST) {
        return;
    }

    num_lanes = ucp_proto_common_find_lanes(params, 0, UCP_LANE_TYPE_LAST,
                                            UCP_PROTO_REPOST_TL_CAP_FLAGS, 0,
                                            1, 0, NULL, &lane);
    rpriv.lane = (num_lanes == 0) ? UCP_NULL_LANE : lane;

    status = ucp_proto_perf_create("repost", &perf);
    if (status != UCS_OK) {
        return;
    }

    /* Finite cost, so this instance wins over the reconfig fallback. */
    perf_factors[UCP_PROTO_PERF_FACTOR_LOCAL_TL] = UCS_LINEAR_FUNC_ZERO;
    status = ucp_proto_perf_add_funcs(
            perf, 0, SIZE_MAX, perf_factors,
            ucp_proto_perf_node_new_data("repost", ""), NULL);
    if (status != UCS_OK) {
        ucp_proto_perf_destroy(perf);
        return;
    }

    ucp_proto_select_add_proto(params, 0, 0, perf, &rpriv, sizeof(rpriv));
}

static void
ucp_proto_repost_query(const ucp_proto_query_params_t *params,
                       ucp_proto_query_attr_t *attr)
{
    ucp_proto_default_query(params, attr);
}

static ucs_status_t
ucp_proto_repost_copy_data(ucp_request_t *req, void **buffer_p, size_t length)
{
    void *copy;

    if (length == 0) {
        return UCS_OK;
    }

    if (*buffer_p == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    copy = ucs_malloc(length, "ucp_proto_repost_data");
    if (copy == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memcpy(copy, *buffer_p, length);
    *buffer_p                = copy;
    req->send.repost.buffer  = copy;
    return UCS_OK;
}

static ucs_status_t ucp_proto_repost_copy_iov(ucp_request_t *req)
{
    uct_ep_op_info_t *info = &req->send.repost.info;
    size_t iovcnt          = info->rma.payload.zcopy.iovcnt;
    const uct_iov_t *iov   = info->rma.payload.zcopy.iov;
    uct_iov_t *copy;

    if (iovcnt == 0) {
        return UCS_OK;
    }

    if (iov == NULL) {
        return UCS_ERR_INVALID_PARAM;
    }

    copy = ucs_malloc(iovcnt * sizeof(*copy), "ucp_proto_repost_iov");
    if (copy == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    memcpy(copy, iov, iovcnt * sizeof(*copy));
    info->rma.payload.zcopy.iov = copy;
    req->send.repost.buffer     = copy;
    return UCS_OK;
}


/* Descriptor and short/bcopy bytes are valid only for the purge callback. */
static ucs_status_t
ucp_proto_repost_save(ucp_request_t *req, const uct_ep_op_info_t *op_info)
{
    uct_ep_op_info_t *info = &req->send.repost.info;

    *info                   = *op_info;
    req->send.repost.buffer = NULL;

    switch (info->operation) {
    case UCT_EP_OP_AM_SHORT:
    case UCT_EP_OP_AM_BCOPY:
        return ucp_proto_repost_copy_data(req, &info->am.payload.data.buffer,
                                          info->am.payload.data.length);
    case UCT_EP_OP_PUT_SHORT:
    case UCT_EP_OP_PUT_BCOPY:
        return ucp_proto_repost_copy_data(req, &info->rma.payload.data.buffer,
                                          info->rma.payload.data.length);
    case UCT_EP_OP_PUT_ZCOPY:
        return ucp_proto_repost_copy_iov(req);
    default:
        return UCS_ERR_UNSUPPORTED;
    }
}


static void ucp_proto_repost_release(ucp_request_t *req, ucs_status_t status)
{
    if (req->send.repost.buffer != NULL) {
        ucs_free(req->send.repost.buffer);
        req->send.repost.buffer = NULL;
    }

    ucp_request_complete_send(req, status);
}

static uct_completion_t *ucp_proto_repost_origin_comp(const ucp_request_t *req)
{
    const uct_ep_op_info_t *info = &req->send.repost.info;

    return (info->field_mask & UCT_EP_OP_INFO_FIELD_COMP) ? info->comp : NULL;
}


static void ucp_proto_repost_completion(uct_completion_t *self)
{
    ucp_request_t *req       = ucs_container_of(self, ucp_request_t,
                                                send.state.uct_comp);
    uct_completion_t *origin = ucp_proto_repost_origin_comp(req);

    req->send.repost.info.comp = NULL;
    if (origin != NULL) {
        ucp_invoke_uct_completion(origin, self->status);
    }

    ucp_proto_repost_release(req, self->status);
}


static void ucp_proto_repost_abort(ucp_request_t *req, ucs_status_t status)
{
    if (ucp_proto_repost_origin_comp(req) != NULL) {
        ucp_invoke_uct_completion(&req->send.state.uct_comp, status);
        return;
    }

    ucp_proto_repost_release(req, status);
}

static ucs_status_t ucp_proto_repost_bcopy_status(ssize_t packed)
{
    return (packed >= 0) ? UCS_OK : (ucs_status_t)packed;
}

static ucs_status_t ucp_proto_repost_progress(uct_pending_req_t *self)
{
    ucp_request_t *req                   = ucs_container_of(self, ucp_request_t,
                                                            send.uct);
    const ucp_proto_repost_priv_t *rpriv = req->send.proto_config->priv;
    const uct_ep_op_info_t *info         = &req->send.repost.info;
    ucp_proto_repost_pack_t pack;
    uct_completion_t *comp;
    ucs_status_t status;
    uct_ep_h uct_ep;
    ssize_t packed;

    if (rpriv->lane == UCP_NULL_LANE) {
        ucp_proto_repost_abort(req, UCS_ERR_UNREACHABLE);
        return UCS_OK;
    }

    uct_ep = ucp_ep_get_lane(req->send.ep, rpriv->lane);
    if (uct_ep == NULL) {
        ucp_proto_repost_abort(req, UCS_ERR_UNREACHABLE);
        return UCS_OK;
    }

    comp = (ucp_proto_repost_origin_comp(req) != NULL) ?
            &req->send.state.uct_comp : NULL;
    switch (info->operation) {
    case UCT_EP_OP_AM_SHORT:
        status = uct_ep_am_short(uct_ep, info->am.am_id, info->am.header.value,
                                 info->am.payload.data.buffer,
                                 info->am.payload.data.length);
        break;
    case UCT_EP_OP_AM_BCOPY:
        pack.buffer = info->am.payload.data.buffer;
        pack.length = info->am.payload.data.length;
        packed      = uct_ep_am_bcopy(uct_ep, info->am.am_id,
                                      ucp_proto_repost_pack, &pack,
                                      info->am.flags);
        status      = ucp_proto_repost_bcopy_status(packed);
        break;
    case UCT_EP_OP_PUT_SHORT:
        status = uct_ep_put_short(uct_ep, info->rma.payload.data.buffer,
                                  info->rma.payload.data.length,
                                  info->rma.remote_addr, info->rma.rkey);
        break;
    case UCT_EP_OP_PUT_BCOPY:
        pack.buffer = info->rma.payload.data.buffer;
        pack.length = info->rma.payload.data.length;
        packed      = uct_ep_put_bcopy(uct_ep, ucp_proto_repost_pack, &pack,
                                       info->rma.remote_addr, info->rma.rkey);
        status      = ucp_proto_repost_bcopy_status(packed);
        break;
    case UCT_EP_OP_PUT_ZCOPY:
        status = uct_ep_put_zcopy(uct_ep, info->rma.payload.zcopy.iov,
                                  info->rma.payload.zcopy.iovcnt,
                                  info->rma.remote_addr, info->rma.rkey, comp);
        break;
    default:
        ucp_proto_repost_abort(req, UCS_ERR_UNSUPPORTED);
        return UCS_OK;
    }

    if (status == UCS_ERR_NO_RESOURCE) {
        req->send.lane = rpriv->lane;
        return UCS_ERR_NO_RESOURCE;
    }

    /* Transport invokes the request completion only after UCS_INPROGRESS. */
    if ((status == UCS_INPROGRESS) &&
        (ucp_proto_repost_origin_comp(req) != NULL)) {
        return UCS_OK;
    }

    ucp_proto_repost_abort(req, (status == UCS_INPROGRESS) ? UCS_OK : status);
    return UCS_OK;
}

static ucs_status_t ucp_proto_repost_reset(ucp_request_t *req)
{
    return UCS_OK;
}


ucp_proto_t ucp_repost_proto = {
    .name     = "repost",
    .desc     = "repost an undelivered operation",
    .flags    = UCP_PROTO_FLAG_INVALID,
    .dt_mask  = UCP_DT_MASK_ALL,
    .probe    = ucp_proto_repost_probe,
    .query    = ucp_proto_repost_query,
    .progress = {ucp_proto_repost_progress},
    .abort    = ucp_proto_repost_abort,
    .reset    = ucp_proto_repost_reset
};


static const ucp_proto_config_t *
ucp_proto_repost_config(ucp_ep_h ep)
{
    ucp_proto_select_param_t select_param;
    ucp_proto_select_elem_t *select_elem;
    const ucp_proto_threshold_elem_t *thresh;

    memset(&select_param, 0, sizeof(select_param));
    select_param.op_id_flags = UCP_OP_ID_REPOST;
    select_param.dt_class    = UCP_DATATYPE_CONTIG;
    select_param.mem_type    = UCS_MEMORY_TYPE_HOST;
    select_param.sg_count    = 1;

    select_elem = ucp_proto_select_lookup_slow(
            ep->worker, &ucp_ep_config(ep)->proto_select, 0, ep->cfg_index,
            UCP_WORKER_CFG_INDEX_NULL, &select_param);
    if (select_elem == NULL) {
        return NULL;
    }

    thresh = ucp_proto_thresholds_search_slow(select_elem->thresholds, 0);
    if ((thresh == NULL) ||
        (thresh->proto_config.proto != &ucp_repost_proto)) {
        return NULL;
    }

    return &thresh->proto_config;
}

ucs_status_t
ucp_proto_repost_submit(ucp_ep_h ep, const uct_ep_op_info_t *op_info)
{
    const ucp_proto_config_t *config = ucp_proto_repost_config(ep);
    const ucp_proto_repost_priv_t *rpriv;
    ucp_request_t *req;
    ucs_status_t status;

    if (config == NULL) {
        return UCS_ERR_UNREACHABLE;
    }

    rpriv = config->priv;
    if ((rpriv->lane == UCP_NULL_LANE) ||
        (ucp_ep_get_lane(ep, rpriv->lane) == NULL)) {
        return UCS_ERR_UNREACHABLE;
    }

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                     = UCP_REQUEST_FLAG_PROTO_SEND |
                                     UCP_REQUEST_FLAG_RELEASED;
    req->send.ep                   = ep;
    req->send.state.dt_iter.offset = 0;
    req->send.state.dt_iter.length = 0;
    ucp_proto_request_set_proto(req, config, 0);

    status = ucp_proto_repost_save(req, op_info);
    if (status != UCS_OK) {
        ucp_request_put(req);
        return status;
    }

    if (ucp_proto_repost_origin_comp(req) != NULL) {
        ucp_proto_completion_init(&req->send.state.uct_comp,
                                  ucp_proto_repost_completion);
    }

    ucp_request_send(req);
    return UCS_OK;
}
