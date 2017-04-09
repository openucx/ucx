/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tag_match.inl"


ucs_status_t ucp_tag_match_init(ucp_tag_match_t *tm)
{
    ucs_queue_head_init(&tm->expected);
    ucs_queue_head_init(&tm->unexpected);
    return UCS_OK;
}

void ucp_tag_match_cleanup(ucp_tag_match_t *tm)
{
}

int ucp_tag_unexp_is_empty(ucp_tag_match_t *tm)
{
    return ucs_queue_is_empty(&tm->unexpected);
}

void ucp_tag_exp_remove(ucp_tag_match_t *tm, ucp_request_t *req)
{
    ucs_queue_iter_t iter;
    ucp_request_t *qreq;

    ucs_queue_for_each_safe(qreq, iter, &tm->expected, recv.queue) {
        if (qreq == req) {
            ucs_queue_del_iter(&tm->expected, iter);
            return;
        }
    }

    ucs_bug("expected request not found");
}
