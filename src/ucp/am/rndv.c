/**
 * Copyright (C) 2022, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "ucp_am.inl"

#include <ucp/proto/proto_single.inl>
#include <ucp/rndv/proto_rndv.inl>


static size_t ucp_am_rndv_rts_pack(void *dest, void *arg)
{
    ucp_request_t *req          = arg;
    ucp_rndv_rts_hdr_t *rts_hdr = dest;
    size_t rts_size;

    ucp_am_fill_header(ucp_am_hdr_from_rts(rts_hdr), req);
    rts_hdr->opcode = UCP_RNDV_RTS_AM;
    rts_size        = ucp_proto_rndv_rts_pack(req, rts_hdr, sizeof(*rts_hdr));
    ucp_am_pack_user_header(UCS_PTR_BYTE_OFFSET(rts_hdr, rts_size), req);

    return rts_size + req->send.msg_proto.am.header_length;
}

UCS_PROFILE_FUNC(ucs_status_t, ucp_am_rndv_proto_progress, (self),
                 uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    const ucp_proto_rndv_ctrl_priv_t *rpriv;
    ucs_status_t status;
    size_t max_rts_size;

    rpriv  = req->send.proto_config->priv;
    status = UCS_PROFILE_CALL(ucp_proto_rndv_rts_request_init, req);
    if (status != UCS_OK) {
        ucp_proto_request_abort(req, status);
        return UCS_OK;
    }

    max_rts_size = sizeof(ucp_rndv_rts_hdr_t) + rpriv->packed_rkey_size +
                   req->send.msg_proto.am.header_length;
    return UCS_PROFILE_CALL(ucp_proto_am_bcopy_single_progress, req,
                            UCP_AM_ID_RNDV_RTS, rpriv->lane,
                            ucp_am_rndv_rts_pack, req, max_rts_size, NULL);
}

ucs_status_t ucp_am_rndv_rts_init(const ucp_proto_init_params_t *init_params)
{
    if (!ucp_am_check_init_params(init_params, UCP_AM_OP_ID_MASK_ALL,
                                  UCP_PROTO_SELECT_OP_FLAG_AM_EAGER) ||
        (init_params->select_param->dt_class != UCP_DATATYPE_CONTIG)) {
        return UCS_ERR_UNSUPPORTED;
    }

    return ucp_proto_rndv_rts_init(init_params);
}

ucp_proto_t ucp_am_rndv_proto = {
    .name     = "am/rndv",
    .desc     = NULL,
    .flags    = 0,
    .init     = ucp_am_rndv_rts_init,
    .query    = ucp_proto_rndv_rts_query,
    .progress = {ucp_am_rndv_proto_progress},
    .abort    = ucp_proto_rndv_rts_abort
};
