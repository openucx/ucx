/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_MATCH_INL_
#define UCP_TAG_MATCH_INL_

#include "tag_match.h"

#include <ucp/core/ucp_request.h>
#include <ucp/dt/dt.h>
#include <ucs/debug/log.h>
#include <ucs/datastruct/queue.h>
#include <ucs/datastruct/mpool.inl>
#include <inttypes.h>


/* Hash size is a prime number just below 1024. Prime number for even distribution,
 * and small enough to fit L1 cache. */
#define UCP_TAG_MATCH_HASH_SIZE     1021



#define ucp_tag_log_match(_recv_tag, _recv_len,_req, _exp_tag, _exp_tag_mask, \
                          _offset, _title) \
    ucs_trace_req("matched tag %"PRIx64" len %zu to %s request %p offset %zu " \
                  "with tag %"PRIx64"/%"PRIx64, (_recv_tag), (size_t)(_recv_len), \
                  (_title), (_req), (size_t)(_offset), (_exp_tag), (_exp_tag_mask))


static UCS_F_ALWAYS_INLINE
int ucp_tag_is_match(ucp_tag_t tag, ucp_tag_t exp_tag, ucp_tag_t tag_mask)
{
    /* The bits in which expected and actual tag differ, should not fall
     * inside the mask.
     */
    return ((tag ^ exp_tag) & tag_mask) == 0;
}

static UCS_F_ALWAYS_INLINE
int ucp_tag_recv_is_match(ucp_tag_t recv_tag, unsigned recv_flags,
                          ucp_tag_t exp_tag, ucp_tag_t tag_mask,
                          size_t offset, ucp_tag_t curr_tag)
{
    /*
     * For first fragment, we search a matching request
     * For subsequent fragments, we search for a request with exact same tag,
     * which would also mean it arrives from the same sender.
     */
    return (((offset == 0) && (recv_flags & UCP_RECV_DESC_FLAG_FIRST) &&
              ucp_tag_is_match(recv_tag, exp_tag, tag_mask)) ||
            (!(offset == 0) && !(recv_flags & UCP_RECV_DESC_FLAG_FIRST) &&
              (recv_tag == curr_tag)));
}

static UCS_F_ALWAYS_INLINE size_t
ucp_tag_match_calc_hash(ucp_tag_t tag)
{
    /* Compute two 32-bit modulo and combine their result */
    return ((uint32_t)tag % UCP_TAG_MATCH_HASH_SIZE) ^
           ((uint32_t)(tag >> 32) % UCP_TAG_MATCH_HASH_SIZE);
}

static UCS_F_ALWAYS_INLINE ucs_queue_head_t*
ucp_tag_exp_get_queue_for_tag(ucp_tag_match_t *tm, ucp_tag_t tag)
{
    return &tm->expected.hash[ucp_tag_match_calc_hash(tag)];
}

static UCS_F_ALWAYS_INLINE ucs_queue_head_t*
ucp_tag_exp_get_queue(ucp_tag_match_t *tm, ucp_tag_t tag, ucp_tag_t tag_mask)
{
    if (tag_mask == UCP_TAG_MASK_FULL) {
        return ucp_tag_exp_get_queue_for_tag(tm, tag);
    } else {
        return &tm->expected.wildcard;
    }
}

static UCS_F_ALWAYS_INLINE ucs_queue_head_t*
ucp_tag_exp_get_req_queue(ucp_tag_match_t *tm, ucp_request_t *req)
{
    return ucp_tag_exp_get_queue(tm, req->recv.tag, req->recv.tag_mask);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_exp_push(ucp_tag_match_t *tm, ucs_queue_head_t *queue, ucp_request_t *req)
{
    req->recv.sn = tm->expected.sn++;
    ucs_queue_push(queue, &req->recv.queue);
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_exp_add(ucp_tag_match_t *tm, ucp_request_t *req)
{
    ucp_tag_exp_push(tm, ucp_tag_exp_get_req_queue(tm, req), req);
}

static UCS_F_ALWAYS_INLINE ucp_request_t *
ucp_tag_exp_search(ucp_tag_match_t *tm, ucp_tag_t recv_tag, size_t recv_len,
                   unsigned recv_flags)
{
    ucs_queue_head_t *queue;
    ucs_queue_iter_t iter;
    ucp_request_t *req;

    if (ucs_unlikely(!ucs_queue_is_empty(&tm->expected.wildcard))) {
        queue = ucp_tag_exp_get_queue_for_tag(tm, recv_tag);
        return ucp_tag_exp_search_all(tm, queue, recv_tag, recv_len, recv_flags);
    }

    /* fast path - wildcard queue is empty, search only the specific queue */
    queue = ucp_tag_exp_get_queue_for_tag(tm, recv_tag);
    ucs_queue_for_each_safe(req, iter, queue, recv.queue) {
        req = ucs_container_of(*iter, ucp_request_t, recv.queue);
        ucs_trace_data("checking req %p tag %"PRIx64"/%"PRIx64" with recv_tag %"PRIx64,
                       req, req->recv.tag, req->recv.tag_mask, recv_tag);
        if (ucp_tag_recv_is_match(recv_tag, recv_flags, req->recv.tag,
                                  req->recv.tag_mask, req->recv.state.offset,
                                  req->recv.info.sender_tag))
        {
            ucp_tag_log_match(recv_tag, recv_len, req, req->recv.tag,
                              req->recv.tag_mask, req->recv.state.offset, "expected");
            if (recv_flags & UCP_RECV_DESC_FLAG_LAST) {
                ucs_queue_del_iter(queue, iter);
            }
            return req;
        }
    }
    return NULL;
}

static UCS_F_ALWAYS_INLINE ucp_tag_t ucp_rdesc_get_tag(ucp_recv_desc_t *rdesc)
{
    return ((ucp_tag_hdr_t*)(rdesc + 1))->tag;
}

static UCS_F_ALWAYS_INLINE ucs_list_link_t*
ucp_tag_unexp_get_list_for_tag(ucp_tag_match_t *tm, ucp_tag_t tag)
{
    return &tm->unexpected.hash[ucp_tag_match_calc_hash(tag)];
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_unexp_remove(ucp_recv_desc_t *rdesc)
{
    ucs_list_del(&rdesc->list[UCP_RDESC_HASH_LIST]);
    ucs_list_del(&rdesc->list[UCP_RDESC_ALL_LIST] );
}

static UCS_F_ALWAYS_INLINE ucs_status_t
ucp_tag_unexp_recv(ucp_tag_match_t *tm, ucp_worker_h worker, void *data,
                   size_t length, unsigned am_flags, uint16_t hdr_len,
                   uint16_t flags)
{
    ucp_recv_desc_t *rdesc = (ucp_recv_desc_t *)data - 1;
    ucs_list_link_t *hash_list;
    ucs_status_t status;

    if (ucs_unlikely(am_flags & UCT_CB_FLAG_DESC)) {
        /* desc==data is slowpath */
        rdesc->flags = flags | UCP_RECV_DESC_FLAG_UCT_DESC;
        status = UCS_INPROGRESS;
    } else {
        rdesc = (ucp_recv_desc_t*)ucs_mpool_get_inline(&worker->am_mp);
        if (rdesc == NULL) {
            ucs_error("ucp recv descriptor is not allocated");
            return UCS_ERR_NO_MEMORY;
        }

        rdesc->flags = flags;
        memcpy(rdesc + 1, data, length);
        status = UCS_OK;
    }

    ucs_trace_req("unexp recv %c%c%c%c%c tag %"PRIx64" length %zu desc %p",
                  (flags & UCP_RECV_DESC_FLAG_FIRST) ? 'f' : '-',
                  (flags & UCP_RECV_DESC_FLAG_LAST)  ? 'l' : '-',
                  (flags & UCP_RECV_DESC_FLAG_EAGER) ? 'e' : '-',
                  (flags & UCP_RECV_DESC_FLAG_SYNC)  ? 's' : '-',
                  (flags & UCP_RECV_DESC_FLAG_RNDV)  ? 'r' : '-',
                  ucp_rdesc_get_tag(rdesc), length - hdr_len, rdesc);

    rdesc->length  = length;
    rdesc->hdr_len = hdr_len;
    hash_list = ucp_tag_unexp_get_list_for_tag(tm, ucp_rdesc_get_tag(rdesc));
    ucs_list_add_tail(hash_list,           &rdesc->list[UCP_RDESC_HASH_LIST]);
    ucs_list_add_tail(&tm->unexpected.all, &rdesc->list[UCP_RDESC_ALL_LIST]);
    return status;
}

static UCS_F_ALWAYS_INLINE void
ucp_tag_unexp_desc_release(ucp_recv_desc_t *rdesc)
{
    ucs_trace_req("release receive descriptor %p", rdesc);
    if (ucs_unlikely(rdesc->flags & UCP_RECV_DESC_FLAG_UCT_DESC)) {
        uct_iface_release_desc(rdesc); /* uct desc is slowpath */
    } else {
        ucs_mpool_put_inline(rdesc);
    }
}

#endif
