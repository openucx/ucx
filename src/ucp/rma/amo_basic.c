/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "amo.inl"


static ucs_status_t ucp_amo_basic_progress_post(uct_pending_req_t *self)
{
    ucp_request_t *req    = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rkey_h rkey       = req->send.amo.rkey;

    req->send.lane = rkey->cache.amo_lane;
    return ucp_amo_progress_post(req, rkey->cache.amo_rkey);
}

ucs_status_t ucp_amo_basic_progress_fetch(uct_pending_req_t *self)
{
    ucp_request_t *req    = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rkey_h rkey       = req->send.amo.rkey;

    req->send.lane = rkey->cache.amo_lane;
    return ucp_amo_progress_fetch(req, rkey->cache.amo_rkey);
}

ucp_amo_proto_t ucp_amo_basic_proto = {
    .name           = "basic_amo",
    .progress_fetch = ucp_amo_basic_progress_fetch,
    .progress_post  = ucp_amo_basic_progress_post
};
