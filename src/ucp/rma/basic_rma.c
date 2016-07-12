/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (c) UT-Battelle, LLC. 2015. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucp/core/ucp_mm.h>

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_worker.h>
#include <ucp/core/ucp_context.h>
#include <ucp/dt/dt_contig.h>

#include <ucp/core/ucp_request.inl>
#include <ucs/datastruct/mpool.inl>
#include <ucp/core/ucp_ep.inl>


#define UCP_RMA_CHECK_PARAMS(_buffer, _length) \
    if ((_length) == 0) { \
        return UCS_OK; \
    } \
    if (ENABLE_PARAMS_CHECK && ((_buffer) == NULL)) { \
        return UCS_ERR_INVALID_PARAM; \
    }

ucs_status_t ucp_put(ucp_ep_h ep, const void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_ep_rma_config_t *rma_config;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;
    ssize_t packed_len;
    ucp_lane_index_t lane;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    /* Loop until all message has been sent.
     * We re-check the configuration on every iteration, because it can be
     * changed by transport switch.
     */
    for (;;) {
        UCP_EP_RESOLVE_RKEY_RMA(ep, rkey, lane, uct_rkey, rma_config);
        if (length <= rma_config->max_put_short) {
            status = uct_ep_put_short(ep->uct_eps[lane], buffer, length,
                                      remote_addr, uct_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                break;
            }
        } else {
            if (length <= ucp_ep_config(ep)->bcopy_thresh) {
                frag_length = ucs_min(length, rma_config->max_put_short);
                status = uct_ep_put_short(ep->uct_eps[lane], buffer, frag_length,
                                          remote_addr, uct_rkey);
            } else {
                ucp_memcpy_pack_context_t pack_ctx;
                pack_ctx.src    = buffer;
                pack_ctx.length = frag_length =
                                ucs_min(length, rma_config->max_put_bcopy);
                packed_len = uct_ep_put_bcopy(ep->uct_eps[lane], ucp_memcpy_pack,
                                              &pack_ctx, remote_addr, uct_rkey);
                status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
            }
            if (ucs_likely(status == UCS_OK)) {
                length      -= frag_length;
                if (length == 0) {
                    break;
                }

                buffer      += frag_length;
                remote_addr += frag_length;
            } else if (status != UCS_ERR_NO_RESOURCE) {
                break;
            }
        }
        ucp_worker_progress(ep->worker);
    }

    return status;
}

static ucs_status_t ucp_progress_put_nbi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rkey_h rkey    = req->send.rma.rkey;
    ucp_ep_t *ep       = req->send.ep;
    ucp_ep_rma_config_t *rma_config;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    ssize_t packed_len;

    UCP_EP_RESOLVE_RKEY_RMA(ep, rkey, req->send.lane, uct_rkey, rma_config);

    if (req->send.length <= ep->worker->context->config.ext.bcopy_thresh) {
        /* Should be replaced with bcopy */
        packed_len = ucs_min(req->send.length, rma_config->max_put_short);
        status = uct_ep_put_short(ep->uct_eps[req->send.lane],
                                  req->send.buffer,
                                  packed_len,
                                  req->send.rma.remote_addr,
                                  uct_rkey);
    } else {
        /* We don't do it right now, but in future we have to add
         * an option to use zcopy
         */
        ucp_memcpy_pack_context_t pack_ctx;
        pack_ctx.src    = req->send.buffer;
        pack_ctx.length = ucs_min(req->send.length, rma_config->max_put_bcopy);
        packed_len = uct_ep_put_bcopy(ep->uct_eps[req->send.lane],
                                      ucp_memcpy_pack,
                                      &pack_ctx,
                                      req->send.rma.remote_addr,
                                      uct_rkey);
        status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
    }

    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        req->send.length -= packed_len;
        if (req->send.length == 0) {
            ucp_request_put(req);
            return UCS_OK;
        }

        req->send.buffer += packed_len;
        req->send.rma.remote_addr += packed_len;
        return UCS_INPROGRESS;
    } else {
        return status;
    }
}

static ucs_status_t
ucp_rma_start_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                  uint64_t remote_addr, ucp_rkey_h rkey,
                  uct_pending_callback_t cb)
{
    ucp_request_t *req;

    req = ucp_request_get(ep->worker);
    if (req == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    req->flags                = UCP_REQUEST_FLAG_RELEASED; /* Implicit release */
    req->send.ep              = ep;
    req->send.buffer          = buffer;
    req->send.length          = length;
    req->send.rma.remote_addr = remote_addr;
    req->send.rma.rkey        = rkey;
    req->send.uct.func        = cb;
#if ENABLE_ASSERT
    req->send.cb              = NULL;
    req->send.lane            = UCP_NULL_LANE;
#endif
    return ucp_request_start_send(req);
}

ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_ep_rma_config_t *rma_config;
    ucp_lane_index_t lane;
    uct_rkey_t uct_rkey;
    ucs_status_t status;

    UCP_RMA_CHECK_PARAMS(buffer, length);
    UCP_EP_RESOLVE_RKEY_RMA(ep, rkey, lane, uct_rkey, rma_config);

    /* Fast path for a single short message */
    if (length <= rma_config->max_put_short) {
        status = uct_ep_put_short(ep->uct_eps[lane], buffer, length, remote_addr,
                                  uct_rkey);
        if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
            /* Return on error or success */
            return status;
        }
    }

    ucp_rma_start_nbi(ep, buffer, length, remote_addr, rkey,
                        ucp_progress_put_nbi);
    return UCS_INPROGRESS;
}

ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_ep_rma_config_t *rma_config;
    uct_completion_t comp;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;
    ucp_lane_index_t lane;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    comp.count = 1;

    for (;;) {
        UCP_EP_RESOLVE_RKEY_RMA(ep, rkey, lane, uct_rkey, rma_config);

        /* Push out all fragments, and request completion only for the last
         * fragment.
         */
        frag_length = ucs_min(rma_config->max_get_bcopy, length);
        status = uct_ep_get_bcopy(ep->uct_eps[lane], (uct_unpack_callback_t)memcpy,
                                  (void*)buffer, frag_length, remote_addr,
                                  uct_rkey, &comp);
        if (ucs_likely(status == UCS_OK)) {
            goto posted;
        } else if (status == UCS_INPROGRESS) {
            ++comp.count;
            goto posted;
        } else if (status == UCS_ERR_NO_RESOURCE) {
            goto retry;
        } else {
            return status;
        }

posted:
        length      -= frag_length;
        if (length == 0) {
            break;
        }

        buffer      += frag_length;
        remote_addr += frag_length;
retry:
        ucp_worker_progress(ep->worker);
    }

    /* coverity[loop_condition] */
    while (comp.count > 1) {
        ucp_worker_progress(ep->worker);
    }
    return UCS_OK;
}

static ucs_status_t ucp_progress_get_nbi(uct_pending_req_t *self)
{
    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);
    ucp_rkey_h rkey    = req->send.rma.rkey;
    ucp_ep_t *ep       = req->send.ep;
    ucp_ep_rma_config_t *rma_config;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;

    UCP_EP_RESOLVE_RKEY_RMA(ep, rkey, req->send.lane, uct_rkey, rma_config);

    frag_length = ucs_min(rma_config->max_get_bcopy, req->send.length);
    status = uct_ep_get_bcopy(ep->uct_eps[req->send.lane],
                              (uct_unpack_callback_t)memcpy,
                              (void*)req->send.buffer,
                              frag_length,
                              req->send.rma.remote_addr,
                              uct_rkey,
                              NULL);
    if ((status == UCS_OK) || (status == UCS_INPROGRESS)) {
        /* Get was initiated */
        req->send.length          -= frag_length;
        req->send.buffer          += frag_length;
        req->send.rma.remote_addr += frag_length;
        if (req->send.length == 0) {
            /* Get was posted */
            ucp_request_put(req);
            return UCS_OK;
        } else {
            return UCS_INPROGRESS;
        }
    } else {
        return status;
    }
}

ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucp_ep_rma_config_t *rma_config;
    ucp_lane_index_t lane;
    uct_rkey_t uct_rkey;
    ucs_status_t status;

    UCP_RMA_CHECK_PARAMS(buffer, length);
    UCP_EP_RESOLVE_RKEY_RMA(ep, rkey, lane, uct_rkey, rma_config);

    if (length <= rma_config->max_get_bcopy) {
        status = uct_ep_get_bcopy(ep->uct_eps[lane],
                                  (uct_unpack_callback_t)memcpy,
                                  (void*)buffer,
                                  length,
                                  remote_addr,
                                  uct_rkey,
                                  NULL);
        if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
            /* Return on error or success */
            return status;
        }
    }

    ucp_rma_start_nbi(ep, buffer, length, remote_addr, rkey,
                      ucp_progress_get_nbi);
    return UCS_INPROGRESS;
}

ucs_status_t ucp_worker_fence(ucp_worker_h worker)
{
    unsigned rsc_index;
    ucs_status_t status;

    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index] == NULL) {
            continue;
        }

        status = uct_iface_fence(worker->ifaces[rsc_index], 0);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

ucs_status_t ucp_worker_flush(ucp_worker_h worker)
{
    unsigned rsc_index;

    while (worker->stub_pend_count > 0) {
        ucp_worker_progress(worker);
    }

    /* TODO flush in parallel */
    for (rsc_index = 0; rsc_index < worker->context->num_tls; ++rsc_index) {
        if (worker->ifaces[rsc_index] == NULL) {
            continue;
        }

        while (uct_iface_flush(worker->ifaces[rsc_index], 0, NULL) != UCS_OK) {
            ucp_worker_progress(worker);
        }
    }

    return UCS_OK;
}

ucs_status_t ucp_ep_flush(ucp_ep_h ep)
{
    ucp_lane_index_t lane;
    ucs_status_t status;

    for (lane = 0; lane < ucp_ep_num_lanes(ep); ++lane) {
        for (;;) {
            status = uct_ep_flush(ep->uct_eps[lane], 0, NULL);
            if (status == UCS_OK) {
                break;
            } else if ((status != UCS_INPROGRESS) && (status != UCS_ERR_NO_RESOURCE)) {
                return status;
            }
            ucp_worker_progress(ep->worker);
        }
    }
    return UCS_OK;
}

