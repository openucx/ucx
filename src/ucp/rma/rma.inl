/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_RMA_INL_
#define UCP_RMA_INL_

#include "rma.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_request.inl>
#include <ucs/debug/log.h>


static UCS_F_ALWAYS_INLINE ucs_status_ptr_t
ucp_rma_send_request_cb(ucp_request_t *req, ucp_send_callback_t cb)
{
    ucs_status_t status = ucp_request_send(req);

    if (req->flags & UCP_REQUEST_FLAG_COMPLETED) {
        ucs_trace_req("releasing send request %p, returning status %s", req,
                      ucs_status_string(status));
        ucs_mpool_put(req);
        return UCS_STATUS_PTR(status);
    }

    ucs_trace_req("returning request %p, status %s", req,
                  ucs_status_string(status));
    ucp_request_set_callback(req, send.cb, cb);
    return req + 1;
}

static inline ucs_status_t ucp_rma_wait(ucp_worker_h worker, void *user_req,
                                        const char *op_name)
{
    ucs_status_t status;
    ucp_request_t *req;

    if (ucs_likely(user_req == NULL)) {
        return UCS_OK;
    } else if (ucs_unlikely(UCS_PTR_IS_ERR(user_req))) {
        ucs_warn("%s failed: %s", op_name,
                 ucs_status_string(UCS_PTR_STATUS(user_req)));
        return UCS_PTR_STATUS(user_req);
    } else {
        req = (ucp_request_t*)user_req - 1;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(user_req);
        } while (!(req->flags & UCP_REQUEST_FLAG_COMPLETED));
        status = req->status;
        ucp_request_release(user_req);
        return status;
    }
}

static inline void ucp_ep_rma_remote_request_sent(ucp_ep_t *ep)
{
    ++ucp_ep_flush_state(ep)->send_sn;
    ++ep->worker->flush_ops_count;
}

static inline void ucp_ep_rma_remote_request_completed(ucp_ep_t *ep)
{
    ucp_ep_flush_state_t *flush_state = ucp_ep_flush_state(ep);
    ucp_request_t *req;

    --ep->worker->flush_ops_count;
    ++flush_state->cmpl_sn;

    ucs_queue_for_each_extract(req, &flush_state->reqs, send.flush.queue,
                               UCS_CIRCULAR_COMPARE32(req->send.flush.cmpl_sn,
                                                      <= ,flush_state->cmpl_sn)) {
        ucp_ep_flush_remote_completed(req);
    }
}

#ifdef __aarch64__

/* On aarch64, the default memcpy() uses byte-wise copy for length < 16 bytes,
 * which may copy inconsistent values from memory written by DMA operation.
 * With RMA operations, we want to provide a consistent view of memory state,
 * at least for basic data types, so use short/int/long instructions when possible.
 */
static inline void ucp_rma_memcpy(void *dst, const void *src, size_t length)
{
    /* check length and pointer alignment in one shot by ORing their bits
     * we also check that length<15 by comparing high-order bits to 0, therefore
     * need to zero-out the high-order pointer bits as well. */
    uintptr_t ptr_align_check = (((uintptr_t)dst | (uintptr_t)src) & 0xf) | length;

    if (ptr_align_check == 8) {
        /* length and pointers are 8-byte aligned, length < 16 */
        *(uint64_t*)dst = *(const uint64_t*)src;
    } else if (!(ptr_align_check & ~(8|4))) {
        /* length and pointers are 4-byte aligned, length < 16 */
        UCS_STATIC_ASSERT(sizeof(uint32_t) == 4);
        for (; length > 0; length -= 4) {
            *(uint32_t*)dst = *(uint32_t*)src;
            src += 4;
            dst += 4;
        }
    } else if (!(ptr_align_check & ~(8|4|2))) {
        /* length and pointers are 2-byte aligned, length < 16 */
        UCS_STATIC_ASSERT(sizeof(uint16_t) == 2);
        for (; length > 0; length -= 2) {
            *(uint16_t*)dst = *(uint16_t*)src;
            src += 2;
            dst += 2;
         }
    } else {
        /* fallback to system memcpy() */
        memcpy(dst, src, length);
    }
}

#else
#define ucp_rma_memcpy  memcpy
#endif

#endif
