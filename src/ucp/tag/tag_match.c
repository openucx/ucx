/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tag_match.inl"
#include <ucp/tag/offload.h>


ucs_status_t ucp_tag_match_init(ucp_tag_match_t *tm)
{
    size_t hash_size, bucket;

    hash_size = ucs_roundup_pow2(UCP_TAG_MATCH_HASH_SIZE);

    tm->expected.sn           = 0;
    tm->expected.sw_all_count = 0;
    ucs_queue_head_init(&tm->expected.wildcard.queue);
    ucs_list_head_init(&tm->unexpected.all);

    tm->expected.hash = ucs_malloc(sizeof(*tm->expected.hash) * hash_size,
                                   "ucp_tm_exp_hash");
    if (tm->expected.hash == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    tm->unexpected.hash = ucs_malloc(sizeof(*tm->unexpected.hash) * hash_size,
                                     "ucp_tm_unexp_hash");
    if (tm->unexpected.hash == NULL) {
        ucs_free(tm->expected.hash);
        return UCS_ERR_NO_MEMORY;
    }

    for (bucket = 0; bucket < hash_size; ++bucket) {
        tm->expected.hash[bucket].sw_count    = 0;
        tm->expected.hash[bucket].block_count = 0;
        ucs_queue_head_init(&tm->expected.hash[bucket].queue);
        ucs_list_head_init(&tm->unexpected.hash[bucket]);
    }

    kh_init_inplace(ucp_tag_frag_hash, &tm->frag_hash);
    ucs_queue_head_init(&tm->offload.sync_reqs);
    kh_init_inplace(ucp_tag_offload_hash, &tm->offload.tag_hash);
    tm->offload.thresh       = SIZE_MAX;
    tm->offload.zcopy_thresh = SIZE_MAX;
    tm->offload.iface        = NULL;
    tm->offload.num_ifaces   = 0;
    tm->am.message_id        = ucs_generate_uuid(0);
    return UCS_OK;
}

void ucp_tag_match_cleanup(ucp_tag_match_t *tm)
{
    kh_destroy_inplace(ucp_tag_offload_hash, &tm->offload.tag_hash);
    kh_destroy_inplace(ucp_tag_frag_hash, &tm->frag_hash);
    ucs_free(tm->unexpected.hash);
    ucs_free(tm->expected.hash);
}

int ucp_tag_unexp_is_empty(ucp_tag_match_t *tm)
{
    return ucs_list_is_empty(&tm->unexpected.all);
}

void ucp_tag_exp_remove(ucp_tag_match_t *tm, ucp_request_t *req)
{
    ucp_request_queue_t *req_queue = ucp_tag_exp_get_req_queue(tm, req);
    ucs_queue_iter_t iter;
    ucp_request_t *qreq;

    ucs_queue_for_each_safe(qreq, iter, &req_queue->queue, recv.queue) {
        if (qreq == req) {
            ucp_tag_offload_try_cancel(req->recv.worker, req, 0);
            ucp_tag_exp_delete(req, tm, req_queue, iter);
            return;
        }
    }

    ucs_bug("expected request not found");
}

static inline uint64_t ucp_tag_exp_req_seq(ucs_queue_iter_t iter)
{
    return (*iter == NULL) ? ULONG_MAX :
           ucs_container_of(*iter, ucp_request_t, recv.queue)->recv.tag.sn;
}

ucp_request_t*
ucp_tag_exp_search_all(ucp_tag_match_t *tm, ucp_request_queue_t *req_queue,
                       ucp_tag_t tag)
{
    ucs_queue_head_t *hash_queue = &req_queue->queue;
    ucp_request_queue_t *queue;
    ucs_queue_iter_t hash_iter, wild_iter, *iter;
    uint64_t hash_sn, wild_sn, *sn_p;
    ucp_request_t *req;

    *hash_queue->ptail                 = NULL;
    *tm->expected.wildcard.queue.ptail = NULL;

    hash_iter = ucs_queue_iter_begin(hash_queue);
    wild_iter = ucs_queue_iter_begin(&tm->expected.wildcard.queue);

    hash_sn = ucp_tag_exp_req_seq(hash_iter);
    wild_sn = ucp_tag_exp_req_seq(wild_iter);

    while (hash_sn != wild_sn) {
        if (hash_sn < wild_sn) {
            iter  = &hash_iter;
            sn_p  = &hash_sn;
            queue = req_queue;
        } else {
            iter  = &wild_iter;
            sn_p  = &wild_sn;
            queue = &tm->expected.wildcard;
        }

        req = ucs_container_of(**iter, ucp_request_t, recv.queue);
        if (ucp_tag_is_match(tag, req->recv.tag.tag, req->recv.tag.tag_mask)) {
            ucs_trace_req("matched received tag %"PRIx64" to req %p", tag, req);
            ucp_tag_exp_delete(req, tm, queue, *iter);
            return req;
        }

        *iter = ucs_queue_iter_next(*iter);
        *sn_p = ucp_tag_exp_req_seq(*iter);
    }

    ucs_assertv((hash_sn == ULONG_MAX) && (wild_sn == ULONG_MAX),
                "hash_seq=%lu wild_seq=%lu", hash_sn, wild_sn);
    ucs_assert(ucs_queue_iter_end(hash_queue, hash_iter));
    ucs_assert(ucs_queue_iter_end(&tm->expected.wildcard.queue, wild_iter));
    return NULL;
}

void ucp_tag_frag_list_process_queue(ucp_tag_match_t *tm, ucp_request_t *req,
                                     uint64_t msg_id UCS_STATS_ARG(int counter_idx))
{
    ucp_eager_middle_hdr_t *hdr;
    ucp_tag_frag_match_t *matchq;
    ucp_recv_desc_t *rdesc;
    ucs_status_t status;
    khiter_t iter;
    int ret;

    iter   = kh_put(ucp_tag_frag_hash, &tm->frag_hash, msg_id, &ret);
    matchq = &kh_value(&tm->frag_hash, iter);
    if (ret == 0) {
        status = UCS_INPROGRESS;
        ucs_assert(ucp_tag_frag_match_is_unexp(matchq));
        ucs_queue_for_each_extract(rdesc, &matchq->unexp_q, tag_frag_queue,
                                   status == UCS_INPROGRESS) {
            UCS_STATS_UPDATE_COUNTER(req->recv.worker->stats, counter_idx, 1);
            hdr    = (void*)(rdesc + 1);
            status = ucp_tag_recv_request_process_rdesc(req, rdesc, hdr->offset);
        }
        ucs_assert(ucs_queue_is_empty(&matchq->unexp_q));

        /* if we completed the request, delete hash entry */
        if (status != UCS_INPROGRESS) {
            kh_del(ucp_tag_frag_hash, &tm->frag_hash, iter);
        }
    }

    /* request not completed, put it on the hash */
    ucp_tag_frag_hash_init_exp(matchq, req);
}
