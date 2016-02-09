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
#include <ucs/datastruct/mpool.inl>


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
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;
    ssize_t packed_len;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    /* Loop until all message has been sent.
     * We re-check the configuration on every iteration, because it can be
     * changed by transport switch.
     */
    for (;;) {
        if (length <= ucp_ep_config(ep)->put.max_short) {
            status = uct_ep_put_short(ep->uct_ep, buffer, length, remote_addr,
                                      uct_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                break;
            }
        } else {
            if (length <= ucp_ep_config(ep)->put.bcopy_thresh) {
                frag_length = ucs_min(length, ucp_ep_config(ep)->put.max_short);
                status = uct_ep_put_short(ep->uct_ep, buffer, frag_length, remote_addr,
                                          uct_rkey);
            } else {
                ucp_memcpy_pack_context_t pack_ctx;
                pack_ctx.src    = buffer;
                pack_ctx.length = frag_length =
                                ucs_min(length, ucp_ep_config(ep)->put.max_bcopy);
                packed_len = uct_ep_put_bcopy(ep->uct_ep, ucp_memcpy_pack, &pack_ctx,
                                              remote_addr, uct_rkey);
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
    ucs_status_t status;
    ssize_t packed_len;

    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    ucp_ep_t *ep = req->send.ep;
    uct_rkey_t uct_rkey = UCP_RKEY_LOOKUP(ep, req->send.rma.rkey);

    for (;;) {
        if (req->send.length <= ep->worker->context->config.ext.bcopy_thresh) {
            /* Should be replaced with bcopy */
            packed_len = ucs_min(req->send.length, ucp_ep_config(ep)->put.max_short);
            status = uct_ep_put_short(ep->uct_ep,
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
            pack_ctx.length =
                ucs_min(req->send.length, ucp_ep_config(ep)->put.max_bcopy);
            packed_len = uct_ep_put_bcopy(ep->uct_ep,
                                          ucp_memcpy_pack,
                                          &pack_ctx,
                                          req->send.rma.remote_addr,
                                          uct_rkey);
            status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
        }

        if (ucs_likely(status == UCS_OK || status == UCS_INPROGRESS)) {
            req->send.length -= packed_len;
            if (req->send.length == 0) {
                ucp_request_complete(req, void);
                break;
            }

            req->send.buffer += packed_len;
            req->send.rma.remote_addr += packed_len;
        } else {
            break;
        }
    }

    return status;
}

static UCS_F_ALWAYS_INLINE
void ucp_add_pending_rma(ucp_request_t *req, ucp_ep_h ep, const void *buffer,
                         size_t length, uint64_t remote_addr, ucp_rkey_h rkey,
                         uct_pending_callback_t cb)
{
    req->send.ep = ep;
    req->send.buffer = buffer;
    req->send.length = length;
    req->send.rma.remote_addr = remote_addr;
    req->send.rma.rkey = rkey;
    req->send.uct.func = cb;
    req->flags = UCP_REQUEST_FLAG_RELEASED;
    ucp_ep_add_pending(ep, ep->uct_ep, req);
}

ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    ssize_t packed_len;
    ucp_request_t *req;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        if (length <= ucp_ep_config(ep)->put.max_short) {
            /* Fast path for a single short message */
            status = uct_ep_put_short(ep->uct_ep, buffer, length, remote_addr,
                                      uct_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                /* Return on error or success */
                break;
            } else {
                /* Out of resources - adding request for later schedule */
                req = ucs_mpool_get_inline(&ep->worker->req_mp);
                if (req == NULL) {
                    status = UCS_ERR_NO_MEMORY;
                    break;
                }
                ucp_add_pending_rma(req, ep, buffer, length, remote_addr, rkey,
                                    ucp_progress_put_nbi);
                status = UCS_INPROGRESS;
                break;
            }
        } else {
            /* Fragmented put */
            if (length <= ucp_ep_config(ep)->put.bcopy_thresh) {
                /* TBD: Should be replaced with bcopy */
                packed_len = ucs_min(length, ucp_ep_config(ep)->put.max_short);
                status = uct_ep_put_short(ep->uct_ep, buffer, packed_len,
                                          remote_addr, uct_rkey);
            } else {
                /* TBD: Use z-copy */
                ucp_memcpy_pack_context_t pack_ctx;
                pack_ctx.src    = buffer;
                pack_ctx.length =
                    ucs_min(length, ucp_ep_config(ep)->put.max_bcopy);
                packed_len = uct_ep_put_bcopy(ep->uct_ep, ucp_memcpy_pack, &pack_ctx,
                                              remote_addr, uct_rkey);
                status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
            }

            if (ucs_likely(status == UCS_OK || status == UCS_INPROGRESS)) {
                length -= packed_len;
                if (length == 0) {
                    /* Put is completed - return success */
                    break;
                }

                buffer += packed_len;
                remote_addr += packed_len;
            } else if (status == UCS_ERR_NO_RESOURCE) {
                /* Out of resources - adding request for later schedule */
                req = ucs_mpool_get_inline(&ep->worker->req_mp);
                if (req == NULL) {
                    status = UCS_ERR_NO_MEMORY;
                    break;
                }
                ucp_add_pending_rma(req, ep, buffer, length, remote_addr, rkey,
                                    ucp_progress_put_nbi);
                status = UCS_INPROGRESS;
                break;
            } else {
                /* Return - Error occured */
                break;
            }
        }
    }

    return status;
}

ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey)
{
    uct_completion_t comp;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    comp.count = 1;

    for (;;) {

        /* Push out all fragments, and request completion only for the last
         * fragment.
         */
        frag_length = ucs_min(ucp_ep_config(ep)->get.max_bcopy, length);
        status = uct_ep_get_bcopy(ep->uct_ep, (uct_unpack_callback_t)memcpy,
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
    ucs_status_t status;
    size_t frag_length;

    ucp_request_t *req = ucs_container_of(self, ucp_request_t, send.uct);

    ucp_ep_t *ep = req->send.ep;
    ucp_rkey_h rkey = req->send.rma.rkey;
    uct_rkey_t uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        frag_length = ucs_min(ucp_ep_config(ep)->get.max_bcopy, req->send.length);
        status = uct_ep_get_bcopy(ep->uct_ep,
                                  (uct_unpack_callback_t)memcpy,
                                  (void*)req->send.buffer,
                                  frag_length,
                                  req->send.rma.remote_addr,
                                  uct_rkey,
                                  NULL);
        if (ucs_likely(status == UCS_OK || status == UCS_INPROGRESS)) {
            /* Get was initiated */
            req->send.length -= frag_length;
            req->send.buffer += frag_length;
            req->send.rma.remote_addr += frag_length;
            if (req->send.length == 0) {
                /* Get was posted */
                ucp_request_complete(req, void);
                status = UCS_OK;
                break;
            }
        } else {
            /* Error - abort */
            break;
        }
    }

    return status;
}

ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_t status;
    size_t frag_length;

    uct_rkey_t uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    UCP_RMA_CHECK_PARAMS(buffer, length);
    uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        frag_length = ucs_min(ucp_ep_config(ep)->get.max_bcopy, length);
        status = uct_ep_get_bcopy(ep->uct_ep,
                                  (uct_unpack_callback_t)memcpy,
                                  (void*)buffer,
                                  frag_length,
                                  remote_addr,
                                  uct_rkey,
                                  NULL);
        if (ucs_likely(status == UCS_OK || status == UCS_INPROGRESS)) {
            /* Get was initiated */
            length -= frag_length;
            buffer += frag_length;
            remote_addr += frag_length;
            if (length == 0) {
                break;
            }
        } else if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
            /* Out of resources - adding request for later schedule */
            ucp_request_t *req;
            req = ucs_mpool_get_inline(&ep->worker->req_mp);
            if (req == NULL) {
                /* can't allocate memory for request - abort */
                status = UCS_ERR_NO_MEMORY;
                break;
            }
            ucp_add_pending_rma(req, ep, buffer, length, remote_addr,
                                rkey, ucp_progress_get_nbi);

            /* Mark it as in progress */
            status = UCS_INPROGRESS;
            break;
        } else {
            /* Error */
            break;
        }
    }

    return status;
}

ucs_status_t ucp_worker_fence(ucp_worker_h worker)
{
    return UCS_ERR_UNSUPPORTED;
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

        while (uct_iface_flush(worker->ifaces[rsc_index]) != UCS_OK) {
            ucp_worker_progress(worker);
        }
    }

    return UCS_OK;
}
