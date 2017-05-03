/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tag_match.inl"


ucs_status_t ucp_tag_match_init(ucp_tag_match_t *tm)
{
    size_t hash_size, bucket;

    hash_size = ucs_roundup_pow2(UCP_TAG_MATCH_HASH_SIZE);

    tm->expected.sn   = 0;
    ucs_queue_head_init(&tm->expected.wildcard);

    tm->expected.hash = ucs_malloc(sizeof(*tm->expected.hash) * hash_size,
                                   "ucp_tm_exp_hash");
    if (tm->expected.hash == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    for (bucket = 0; bucket < hash_size; ++bucket) {
        ucs_queue_head_init(&tm->expected.hash[bucket]);
    }

    ucs_queue_head_init(&tm->unexpected);
    ucs_queue_head_init(&tm->offload_ifaces);
    tm->post_thresh = SIZE_MAX;
    return UCS_OK;
}

void ucp_tag_match_cleanup(ucp_tag_match_t *tm)
{
    ucs_free(tm->expected.hash);
}

int ucp_tag_unexp_is_empty(ucp_tag_match_t *tm)
{
    return ucs_queue_is_empty(&tm->unexpected);
}

void ucp_tag_exp_remove(ucp_tag_match_t *tm, ucp_request_t *req)
{
    ucs_queue_head_t *queue = ucp_tag_exp_get_queue(tm, req);
    ucs_queue_iter_t iter;
    ucp_request_t *qreq;

    ucs_queue_for_each_safe(qreq, iter, queue, recv.queue) {
        if (qreq == req) {
            ucs_queue_del_iter(queue, iter);
            return;
        }
    }

    ucs_bug("expected request not found");
}

static inline uint64_t ucp_tag_exp_req_seq(ucs_queue_iter_t iter)
{
    return (*iter == NULL) ? ULONG_MAX :
                    ucs_container_of(*iter, ucp_request_t, recv.queue)->recv.sn;
}

ucp_request_t*
ucp_tag_exp_search_all(ucp_tag_match_t *tm, ucs_queue_head_t *hash_queue,
                       ucp_tag_t recv_tag, size_t recv_len, unsigned recv_flags)
{
    ucs_queue_head_t *queue;
    ucs_queue_iter_t hash_iter, wild_iter, *iter;
    uint64_t hash_sn, wild_sn, *sn_p;
    ucp_request_t *req;

    *hash_queue->ptail           = NULL;
    *tm->expected.wildcard.ptail = NULL;

    hash_iter = ucs_queue_iter_begin(hash_queue);
    wild_iter = ucs_queue_iter_begin(&tm->expected.wildcard);

    hash_sn = ucp_tag_exp_req_seq(hash_iter);
    wild_sn = ucp_tag_exp_req_seq(wild_iter);

    while (hash_sn != wild_sn) {
        if (hash_sn < wild_sn) {
            iter  = &hash_iter;
            sn_p  = &hash_sn;
            queue = hash_queue;
        } else {
            iter  = &wild_iter;
            sn_p  = &wild_sn;
            queue = &tm->expected.wildcard;
        }

        req = ucs_container_of(**iter, ucp_request_t, recv.queue);
        if (ucp_tag_recv_is_match(recv_tag, recv_flags, req->recv.tag,
                                  req->recv.tag_mask, req->recv.state.offset,
                                  req->recv.info.sender_tag))
        {
            ucp_tag_log_match(recv_tag, recv_len, req, req->recv.tag,
                              req->recv.tag_mask, req->recv.state.offset, "expected");
            if (recv_flags & UCP_RECV_DESC_FLAG_LAST) {
                ucs_queue_del_iter(queue, *iter);
            }
            return req;
        }

        *iter = ucs_queue_iter_next(*iter);
        *sn_p = ucp_tag_exp_req_seq(*iter);
    }

    ucs_assertv((hash_sn == ULONG_MAX) && (wild_sn == ULONG_MAX),
                "hash_seq=%lu wild_seq=%lu", hash_sn, wild_sn);
    ucs_assert(ucs_queue_iter_end(hash_queue, hash_iter));
    ucs_assert(ucs_queue_iter_end(&tm->expected.wildcard, wild_iter));
    return NULL;
}
