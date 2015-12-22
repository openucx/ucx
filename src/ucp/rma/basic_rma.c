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
        if (length <= ep->config.max_short_put) {
            status = uct_ep_put_short(ep->uct_ep, buffer, length, remote_addr,
                                      uct_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                break;
            }
        } else {
            if (length <= ep->worker->context->config.bcopy_thresh) {
                frag_length = ucs_min(length, ep->config.max_short_put);
                status = uct_ep_put_short(ep->uct_ep, buffer, frag_length, remote_addr,
                                          uct_rkey);
            } else {
                ucp_memcpy_pack_context_t pack_ctx;
                pack_ctx.src    = buffer;
                pack_ctx.length = frag_length =
                                ucs_min(length, ep->config.max_bcopy_put);
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
    size_t frag_length;
    ssize_t packed_len;

    ucp_request_t *req = ucs_container_of(self, ucp_request_t, uct);

    ucp_ep_t   *ep         = req->rma.ep;
    const void *buffer     = req->rma.buffer;
    uint64_t   remote_addr = req->rma.remote_addr;
    size_t     length      = req->rma.length;
    ucp_rkey_h rkey        = req->rma.rkey;
    uct_rkey_t uct_rkey    = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        if (length <= ep->worker->context->config.bcopy_thresh) {
            /* Should be replaced with bcopy */
            frag_length = ucs_min(length, ep->config.max_short_put);
            status = uct_ep_put_short(ep->uct_ep, buffer, frag_length, remote_addr,
                                      uct_rkey);
        } else {
            /* We don't do it right now, but in future we have to add
             * an option to use zcopy
             */
            ucp_memcpy_pack_context_t pack_ctx;
            pack_ctx.src    = buffer;
            pack_ctx.length = frag_length =
                ucs_min(length, ep->config.max_bcopy_put);
            packed_len = uct_ep_put_bcopy(ep->uct_ep, ucp_memcpy_pack, &pack_ctx,
                                          remote_addr, uct_rkey);
            status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
        }

        if (ucs_likely(status == UCS_OK)) {
            length -= frag_length;
            if (length == 0) {
                break;
            }

            buffer += frag_length;
            remote_addr += frag_length;
        } else if (status != UCS_ERR_NO_RESOURCE) {
            break;
        } else {
            status = UCS_INPROGRESS;
            break;
        }
    }

    if (length == 0) {
        /* Make sure that we mark is as a released */
        ucp_request_complete(req, req->cb.send, UCS_OK);
    }

    return status;
}

#define ADD_PENDING_RMA(_req, _ep, _buf, _len, _raddr, _rkey, _cb) \
    do { \
        (_req)->rma.ep           = _ep; \
        (_req)->rma.buffer       = _buf; \
        (_req)->rma.length       = _len; \
        (_req)->rma.remote_addr  = _raddr; \
        (_req)->rma.rkey         = _rkey; \
        (_req)->uct.func         = _cb; \
        (_req)->flags            = UCP_REQUEST_FLAG_COMPLETED; \
        ucp_ep_add_pending((_ep)->uct_ep, (_req)); \
    } while (0);

ucs_status_t ucp_put_nbi(ucp_ep_h ep, const void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;
    ssize_t packed_len;
    ucp_request_t *req;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        if (length <= ep->config.max_short_put) {
            /* Fast path for a single short message */
            status = uct_ep_put_short(ep->uct_ep, buffer, length, remote_addr,
                                      uct_rkey);
            if (ucs_likely(status != UCS_ERR_NO_RESOURCE)) {
                /* Rerurn on error or success */
                break;
            } else {
                /* Out of resources - adding request for later schedule */
                req = ucs_mpool_get_inline(&ep->worker->req_mp);
                if (req == NULL) {
                    status = UCS_ERR_NO_MEMORY;
                    break;
                }
                ADD_PENDING_RMA(req, ep, buffer, length, remote_addr, rkey
                                , ucp_progress_put_nbi);
                status = UCS_INPROGRESS;
                break;
            }
        } else {
            /* Fragmented put */
            if (length <= ep->worker->context->config.bcopy_thresh) {
                /* TBD: Should be replaced with bcopy */
                frag_length = ucs_min(length, ep->config.max_short_put);
                status = uct_ep_put_short(ep->uct_ep, buffer, frag_length, remote_addr,
                                          uct_rkey);
            } else {
                /* TBD: Use z-copy */
                ucp_memcpy_pack_context_t pack_ctx;
                pack_ctx.src    = buffer;
                pack_ctx.length = frag_length =
                    ucs_min(length, ep->config.max_bcopy_put);
                packed_len = uct_ep_put_bcopy(ep->uct_ep, ucp_memcpy_pack, &pack_ctx,
                                              remote_addr, uct_rkey);
                status = (packed_len > 0) ? UCS_OK : (ucs_status_t)packed_len;
            }
            if (ucs_likely(status == UCS_OK)) {
                length      -= frag_length;
                if (length == 0) {
                    /* Put is completed - return success */
                    break;
                }

                buffer      += frag_length;
                remote_addr += frag_length;
            } else if (status == UCS_ERR_NO_RESOURCE) {
                /* Out of resources - adding request for later schedule */
                req = ucs_mpool_get_inline(&ep->worker->req_mp);
                if (req == NULL) {
                    return UCS_ERR_NO_MEMORY;
                }
                ADD_PENDING_RMA(req, ep, buffer, length, remote_addr, rkey
                                , ucp_progress_put_nbi);
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
        frag_length = ucs_min(ep->config.max_bcopy_get, length);
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

static ucs_status_t ucp_progress_get_nbi(uct_pending_req_t *self);

#define UCP_ADD_PENDING 1
#define UCP_NO_PENDING 0

static UCS_F_ALWAYS_INLINE int ucp_get_post_single(ucp_ep_h ep,
                                                   const void *buffer,
                                                   size_t length,
                                                   uint64_t remote_addr,
                                                   uct_rkey_t uct_rkey,
                                                   ucp_rkey_h ucp_rkey,
                                                   int add_to_pending)
{
    uct_completion_t comp;
    ucs_status_t status;
    size_t frag_length;
    ucp_request_t *req;

    comp.count = 1;

    frag_length = ucs_min(ep->config.max_bcopy_get, length);
    status = uct_ep_get_bcopy(ep->uct_ep, (uct_unpack_callback_t)memcpy,
                              (void*)buffer, frag_length, remote_addr,
                              uct_rkey, &comp);
    if (ucs_unlikely(status == UCS_ERR_NO_RESOURCE)) {
        if (add_to_pending == UCP_ADD_PENDING) {
            /* Out of resources - adding request for later schedule */
            req = ucs_mpool_get_inline(&ep->worker->req_mp);
            if (req == NULL) {
                /* can't allocate memory for request - abort */
                return UCS_ERR_NO_MEMORY;
            }
            ADD_PENDING_RMA(req, ep, buffer, length, remote_addr, ucp_rkey,
                            ucp_progress_get_nbi);
        }
        return status;
    } else if (ucs_likely(status == UCS_OK || status == UCS_INPROGRESS)) {
        length      -= frag_length;
        buffer      += frag_length;
        remote_addr += frag_length;
        return length;
    }

    /* Error */
    return status;
}

static ucs_status_t ucp_progress_get_nbi(uct_pending_req_t *self)
{
    int ret;

    ucp_request_t *req = ucs_container_of(self, ucp_request_t, uct);

    ucp_ep_t   *ep         = req->rma.ep;
    const void       *buffer     = req->rma.buffer;
    uint64_t   remote_addr = req->rma.remote_addr;
    size_t     length      = req->rma.length;
    ucp_rkey_h rkey        = req->rma.rkey;
    uct_rkey_t uct_rkey    = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        ret = ucp_get_post_single(ep, buffer, length, remote_addr, uct_rkey,
                                  rkey, UCP_NO_PENDING);
        if (ret == 0) {
            /* Make sure that we mark is as a released */
            ucp_request_complete(req, req->cb.send, UCS_OK);
            break;
        } else if (ret < 0) {
            break;
        }
    }
    return ret;
}


ucs_status_t ucp_get_nbi(ucp_ep_h ep, void *buffer, size_t length,
                         uint64_t remote_addr, ucp_rkey_h rkey)
{
    uct_rkey_t uct_rkey;
    int ret;

    UCP_RMA_CHECK_PARAMS(buffer, length);
    uct_rkey = UCP_RKEY_LOOKUP(ep, rkey);

    for (;;) {
        ret = ucp_get_post_single(ep, buffer, length, remote_addr, uct_rkey,
                                  rkey, UCP_ADD_PENDING);
        if (ret <= 0) {
            break;
        }
    }

    return ret;
}

ucs_status_t ucp_worker_fence(ucp_worker_h worker)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_worker_flush(ucp_worker_h worker)
{
    unsigned rsc_index;

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
