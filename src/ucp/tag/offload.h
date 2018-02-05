/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_OFFLOAD_H_
#define UCP_TAG_OFFLOAD_H_

#include <ucp/dt/dt_contig.h>
#include <ucp/core/ucp_request.h>
#include <ucp/proto/proto.h>
#include <ucs/datastruct/queue.h>


enum {
    UCP_TAG_OFFLOAD_CANCEL_FORCE = UCS_BIT(0),
    UCP_TAG_OFFLOAD_CANCEL_DEREG = UCS_BIT(1)
};

/**
 * Header for unexpected rendezvous
 */
typedef struct {
    uint64_t       sender_uuid;  /* Sender worker uuid */
    uintptr_t      reqptr;       /* Request pointer */
    uint8_t        md_index;     /* md index */
} UCS_S_PACKED ucp_tag_offload_unexp_rndv_hdr_t;


/**
 * Header for sync send acknowledgment
 */
typedef struct {
    uint64_t          sender_uuid;
    ucp_tag_t         sender_tag;
} UCS_S_PACKED ucp_offload_ssend_hdr_t;


extern const ucp_proto_t ucp_tag_offload_proto;

extern const ucp_proto_t ucp_tag_offload_sync_proto;

ucs_status_t ucp_tag_offload_rndv_zcopy(uct_pending_req_t *self);

ucs_status_t ucp_tag_offload_sw_rndv(uct_pending_req_t *self);

void ucp_tag_offload_cancel_rndv(ucp_request_t *req);

ucs_status_t ucp_tag_offload_start_rndv(ucp_request_t *sreq);

void ucp_tag_offload_eager_sync_send_ack(ucp_worker_h worker,
                                         uint64_t sender_uuid,
                                         ucp_tag_t sender_tag);

ucs_status_t ucp_tag_offload_unexp_eager(void *arg, void *data, size_t length,
                                         unsigned flags, uct_tag_t stag, uint64_t imm);


ucs_status_t ucp_tag_offload_unexp_rndv(void *arg, unsigned flags, uint64_t stag,
                                        const void *hdr, unsigned hdr_length,
                                        uint64_t remote_addr, size_t length,
                                        const void *rkey_buf);

void ucp_tag_offload_cancel(ucp_worker_t *worker, ucp_request_t *req, unsigned mode);

int ucp_tag_offload_post(ucp_request_t *req, ucp_request_queue_t *req_queue);

/**
 * @brief Activate tag offload interface
 *
 * @param [in]  wiface   UCP worker interface.
 *
 * @return 0 - if tag offloading is disabled in the configuration
 *         1 - wiface interface is activated (if it was inactive before)
 */
int ucp_tag_offload_iface_activate(ucp_worker_iface_t *wiface);

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
ucp_tag_offload_try_cancel(ucp_worker_t *worker, ucp_request_t *req, unsigned mode)
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
 * from the hash. Having just one offload-capable interface is supposed to be
 * a fast path, because it matches homogeneous cluster configurations. So, no
 * hashing is done, while only one offload-capable interface is active.
 *
 * @note Hash key is a tag masked with 'tag_sender_mask', because it needs to
 *       identify a particular sender, rather than every single tag.
 *
 * @param [in]  wiface        UCP worker interface.
 * @param [in]  tag           Tag of the arrived unexpected message.
 */
static UCS_F_ALWAYS_INLINE void
ucp_tag_offload_unexp(ucp_worker_iface_t *wiface, ucp_tag_t tag)
{
    ucp_worker_t *worker = wiface->worker;
    ucp_tag_t tag_key;
    khiter_t hash_it;
    int ret;

    ++wiface->proxy_recv_count;

    if (ucs_unlikely(!(wiface->flags & UCP_WORKER_IFACE_FLAG_OFFLOAD_ACTIVATED))) {
        if (!ucp_tag_offload_iface_activate(wiface)) {
            return;
        }
    }

    if (ucs_unlikely(worker->tm.offload.num_ifaces > 1)) {
        tag_key = worker->context->config.tag_sender_mask & tag;
        hash_it = kh_put(ucp_tag_offload_hash, &worker->tm.offload.tag_hash,
                         tag_key, &ret);

        /* khash returns 1 or 2 if key is not present and value can be set */
        if (ret > 0) {
            kh_value(&worker->tm.offload.tag_hash, hash_it) = wiface;
        }
    }
}


#endif
