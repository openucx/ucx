/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
* Copyright (C) Los Alamos National Security, LLC. 2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "rma.h"

#include <ucp/proto/proto_am.inl>


static ucs_status_t ucp_rma_basic_progress_put(uct_pending_req_t *self)
{
    ucp_request_t *req              = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep                    = req->send.ep;
    ucp_rkey_h rkey                 = req->send.rma.rkey;
    ucp_lane_index_t lane           = req->send.lane;
    ucp_ep_rma_config_t *rma_config = &ucp_ep_config(ep)->rma[lane];
    ucs_status_t status;
    ssize_t packed_len;

    ucs_assert(rkey->cache.ep_cfg_index == ep->cfg_index);
    ucs_assert(rkey->cache.rma_lane == lane);

    if (req->send.length <= ucp_ep_config(ep)->bcopy_thresh) {
        packed_len = ucs_min(req->send.length, rma_config->max_put_short);
        status = UCS_PROFILE_CALL(uct_ep_put_short,
                                  ep->uct_eps[lane],
                                  req->send.buffer,
                                  packed_len,
                                  req->send.rma.remote_addr,
                                  rkey->cache.rma_rkey);
    } else if (ucs_likely(req->send.length < rma_config->put_zcopy_thresh)) {
        ucp_memcpy_pack_context_t pack_ctx;
        pack_ctx.src    = req->send.buffer;
        pack_ctx.length = ucs_min(req->send.length, rma_config->max_put_bcopy);
        packed_len = UCS_PROFILE_CALL(uct_ep_put_bcopy,
                                      ep->uct_eps[lane],
                                      ucp_memcpy_pack,
                                      &pack_ctx,
                                      req->send.rma.remote_addr,
                                      rkey->cache.rma_rkey);
        status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
    } else {
        uct_iov_t iov;

        /* TODO: leave last fragment for bcopy */
        packed_len = ucs_min(req->send.length, rma_config->max_put_zcopy);
        /* TODO: use ucp_dt_iov_copy_uct */
        iov.buffer = (void *)req->send.buffer;
        iov.length = packed_len;
        iov.count  = 1;
        iov.memh   = req->send.state.dt.dt.contig.memh[0];

        status = UCS_PROFILE_CALL(uct_ep_put_zcopy,
                                  ep->uct_eps[lane],
                                  &iov, 1,
                                  req->send.rma.remote_addr,
                                  rkey->cache.rma_rkey,
                                  &req->send.state.uct_comp);
        ucp_request_send_state_advance(req, NULL, UCP_REQUEST_SEND_PROTO_RMA,
                                       status);
    }

    return ucp_rma_request_advance(req, packed_len, status);
}

static ucs_status_t ucp_rma_basic_progress_get(uct_pending_req_t *self)
{
    ucp_request_t *req              = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_ep_t *ep                    = req->send.ep;
    ucp_rkey_h rkey                 = req->send.rma.rkey;
    ucp_lane_index_t lane           = req->send.lane;
    ucp_ep_rma_config_t *rma_config = &ucp_ep_config(ep)->rma[lane];
    ucs_status_t status;
    size_t frag_length;

    ucs_assert(rkey->cache.ep_cfg_index == ep->cfg_index);
    ucs_assert(rkey->cache.rma_lane == lane);

    if (ucs_likely(req->send.length < rma_config->get_zcopy_thresh)) {
        frag_length = ucs_min(rma_config->max_get_bcopy, req->send.length);
        status = UCS_PROFILE_CALL(uct_ep_get_bcopy,
                                  ep->uct_eps[lane],
                                  (uct_unpack_callback_t)memcpy,
                                  (void*)req->send.buffer,
                                  frag_length,
                                  req->send.rma.remote_addr,
                                  rkey->cache.rma_rkey,
                                  &req->send.state.uct_comp);
    } else {
        uct_iov_t iov;
        frag_length = ucs_min(req->send.length, rma_config->max_get_zcopy);
        iov.buffer  = (void *)req->send.buffer;
        iov.length  = frag_length;
        iov.count   = 1;
        iov.memh    = req->send.state.dt.dt.contig.memh[0];

        status = UCS_PROFILE_CALL(uct_ep_get_zcopy,
                                  ep->uct_eps[lane],
                                  &iov, 1,
                                  req->send.rma.remote_addr,
                                  rkey->cache.rma_rkey,
                                  &req->send.state.uct_comp);
    }

    if (status == UCS_INPROGRESS) {
        ucp_request_send_state_advance(req, 0, UCP_REQUEST_SEND_PROTO_RMA,
                                       UCS_INPROGRESS);
    }

    return ucp_rma_request_advance(req, frag_length, status);
}

ucp_rma_proto_t ucp_rma_basic_proto = {
    .name         = "basic_rma",
    .progress_put = ucp_rma_basic_progress_put,
    .progress_get = ucp_rma_basic_progress_get
};
