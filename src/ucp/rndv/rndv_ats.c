/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_rndv.inl"


static void ucp_proto_rndv_ats_probe(const ucp_proto_init_params_t *init_params)
{
    ucp_proto_common_init_params_t params;
    ucp_proto_rndv_ack_priv_t priv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    params              = ucp_proto_common_init_params(init_params);
    params.cfg_priority = 80;

    /* This protocols supports either a regular rendezvous receive but without
     * data, or a rendezvous receive which should ignore the data.
     * In either case, we just need to send an ATS.
     */
    if (ucp_proto_init_check_op(init_params, UCS_BIT(UCP_OP_ID_RNDV_RECV))) {
        params.max_length = 0;
    } else if (ucp_proto_init_check_op(init_params,
                                       UCS_BIT(UCP_OP_ID_RNDV_RECV_DROP))) {
        params.max_length = SIZE_MAX;
    } else {
        return;
    }

    status = ucp_proto_rndv_ack_init(&params, UCP_PROTO_RNDV_ATS_NAME, 0, &perf,
                                     &priv);
    if ((status != UCS_OK) || (perf == NULL)) {
        return;
    }

    ucp_proto_select_add_proto(&params.super, params.cfg_thresh,
                               params.cfg_priority, perf, &priv, sizeof(priv));
}

static void
ucp_proto_rndv_ats_abort(ucp_request_t *request, ucs_status_t status)
{
    ucp_request_get_super(request)->status = status;
    ucp_proto_rndv_ats_complete(request);
}

static void ucp_proto_rndv_ats_query(const ucp_proto_query_params_t *params,
                                     ucp_proto_query_attr_t *attr)
{
    const ucp_proto_rndv_ack_priv_t *apriv = params->priv;

    ucp_proto_default_query(params, attr);
    attr->lane_map |= UCS_BIT(apriv->lane);
}

ucp_proto_t ucp_rndv_ats_proto = {
    .name     = "rndv/ats",
    .desc     = "no data fetch",
    .flags    = 0,
    .probe    = ucp_proto_rndv_ats_probe,
    .query    = ucp_proto_rndv_ats_query,
    .progress = {ucp_proto_rndv_ats_progress},
    .abort    = ucp_proto_rndv_ats_abort,
    .reset    = (ucp_request_reset_func_t)ucs_empty_function_return_success
};
