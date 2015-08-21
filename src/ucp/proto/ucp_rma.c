/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "ucp_int.h"


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

    UCP_RMA_CHECK_PARAMS(buffer, length);

    uct_rkey = UCP_RMA_RKEY_LOOKUP(ep, rkey);

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
                frag_length = ucs_min(length, ep->config.max_bcopy_put);
                status = uct_ep_put_bcopy(ep->uct_ep, (uct_pack_callback_t)memcpy,
                                          (void*)buffer, frag_length, remote_addr,
                                          uct_rkey);
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

ucs_status_t ucp_get(ucp_ep_h ep, void *buffer, size_t length,
                     uint64_t remote_addr, ucp_rkey_h rkey)
{
    uct_completion_t comp;
    ucs_status_t status;
    uct_rkey_t uct_rkey;
    size_t frag_length;

    UCP_RMA_CHECK_PARAMS(buffer, length);

    uct_rkey = UCP_RMA_RKEY_LOOKUP(ep, rkey);

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

ucs_status_t ucp_fence(ucp_worker_h worker)
{
    return UCS_ERR_UNSUPPORTED;
}

ucs_status_t ucp_flush(ucp_worker_h worker)
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
