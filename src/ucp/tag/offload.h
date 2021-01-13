/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_OFFLOAD_H_
#define UCP_TAG_OFFLOAD_H_

#include <ucp/tag/eager.h>
#include <ucp/dt/dt_contig.h>
#include <ucp/core/ucp_request.h>
#include <ucs/datastruct/queue.h>


enum {
    UCP_TAG_OFFLOAD_CANCEL_FORCE = UCS_BIT(0)
};

/**
 * Header for unexpected rendezvous
 */
typedef struct {
    uint64_t       ep_id;        /* Endpoint ID */
    uint64_t       req_id;       /* Request ID */
    uint8_t        md_index;     /* md index */
} UCS_S_PACKED ucp_tag_offload_unexp_rndv_hdr_t;


/**
 * Header for sync send acknowledgment
 */
typedef struct {
    uint64_t          ep_id;
    ucp_tag_t         sender_tag;
} UCS_S_PACKED ucp_offload_ssend_hdr_t;


/**
 * Header for multi-fragmented sync send acknowledgment
 * (carried by last fragment)
 */
typedef struct {
    ucp_eager_middle_hdr_t    super;
    ucp_offload_ssend_hdr_t   ssend_ack;
} UCS_S_PACKED ucp_offload_last_ssend_hdr_t;


extern const ucp_request_send_proto_t ucp_tag_offload_proto;
extern const ucp_request_send_proto_t ucp_tag_offload_sync_proto;

ucs_status_t ucp_tag_offload_rndv_zcopy(uct_pending_req_t *self);

ucs_status_t ucp_tag_offload_sw_rndv(uct_pending_req_t *self);

void ucp_tag_offload_cancel_rndv(ucp_request_t *req);

ucs_status_t ucp_tag_offload_start_rndv(ucp_request_t *sreq);

ucs_status_t ucp_tag_offload_unexp_eager(void *arg, void *data, size_t length,
                                         unsigned flags, uct_tag_t stag,
                                         uint64_t imm, void **context);


ucs_status_t ucp_tag_offload_unexp_rndv(void *arg, unsigned flags, uint64_t stag,
                                        const void *hdr, unsigned hdr_length,
                                        uint64_t remote_addr, size_t length,
                                        const void *rkey_buf);

void ucp_tag_offload_cancel(ucp_worker_t *worker, ucp_request_t *req,
                            unsigned mode);

int ucp_tag_offload_post(ucp_request_t *req, ucp_request_queue_t *req_queue);

void ucp_tag_offload_sync_send_ack(ucp_worker_h worker, uintptr_t ep_ptr,
                                   ucp_tag_t stag, uint16_t recv_flags);

/**
 * @brief Activate tag offload interface
 *
 * @param [in]  wiface   UCP worker interface.
 */
void ucp_tag_offload_iface_activate(ucp_worker_iface_t *wiface);

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_try_post(ucp_worker_t *worker, ucp_request_t *req,
                         ucp_request_queue_t *req_queue)
{
    if (ucs_unlikely(req->recv.length >= worker->tm.offload.thresh)) {
        if (ucp_tag_offload_post(req, req_queue)) {
            return;
        }
    }

    ++worker->tm.expected.sw_all_count;
    ++req_queue->sw_count;
    req_queue->block_count += !!(req->flags & UCP_REQUEST_FLAG_BLOCK_OFFLOAD);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_try_cancel(ucp_worker_t *worker, ucp_request_t *req,
                           unsigned mode)
{
    if (ucs_unlikely(req->flags & UCP_REQUEST_FLAG_OFFLOADED)) {
        ucp_tag_offload_cancel(worker, req, mode);
    }
}

/**
 * @brief Handle tag offload unexpected message
 *
 * The routine activates tag offload interface if it the first unexpected
 * message received on this interface. Also it maintains hash of tags, if
 * more than one interface is active. Then, when expected receive request needs
 * to be offloaded, the corresponding offload-capable interface is retrieved
 * from the hash.
 *
 * @note Hash key is a tag masked with 'tag_sender_mask', because it needs to
 *       identify a particular sender, rather than every single tag.
 *
 * @note Tag is added to the hash table for messages bigger than TM_THRESH.
 *       Smaller messages are not supposed to be matched in HW, thus no need
 *       to waste time on hashing for them.
 *
 *
 * @param [in]  wiface        UCP worker interface.
 * @param [in]  tag           Tag of the arrived unexpected message.
 */
static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_unexp(ucp_worker_iface_t *wiface, ucp_tag_t tag, size_t length)
{
    ucp_worker_t *worker = wiface->worker;
    ucp_tag_t tag_key;
    khiter_t hash_it;
    int ret;

    ++wiface->proxy_recv_count;

    if (ucs_unlikely(!(wiface->flags & UCP_WORKER_IFACE_FLAG_OFFLOAD_ACTIVATED))) {
        ucp_tag_offload_iface_activate(wiface);
    }

    /* Need to hash all tags of messages arriving to offload-capable interface
       if more than one interface is activated on the worker. This is needed to
       avoid unwanted postings of receive buffers (those, which are expected to
       arrive from offload incapable iface) to the HW. */
    if (ucs_unlikely((length >= worker->tm.offload.thresh) &&
                     (worker->num_active_ifaces > 1))) {
        tag_key = worker->context->config.tag_sender_mask & tag;
        hash_it = kh_get(ucp_tag_offload_hash, &worker->tm.offload.tag_hash,
                         tag_key);
        if (ucs_likely(hash_it != kh_end(&worker->tm.offload.tag_hash))) {
            return;
        }

        hash_it = kh_put(ucp_tag_offload_hash, &worker->tm.offload.tag_hash,
                         tag_key, &ret);
        ucs_assertv((ret == 1) || (ret == 2), "ret=%d", ret);
        kh_value(&worker->tm.offload.tag_hash, hash_it) = wiface;
    }
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_request_check_flags(ucp_request_t *req)
{
    ucs_assert(!ucp_ep_use_indirect_id(req->send.ep) &&
               !(req->flags & UCP_REQUEST_FLAG_IN_PTR_MAP));
}

#endif
