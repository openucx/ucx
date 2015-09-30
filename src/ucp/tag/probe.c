/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "match.h"
#include "eager.h"
#include "rndv.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/datastruct/queue.h>


static UCS_F_ALWAYS_INLINE ucp_recv_desc_t*
ucp_tag_probe_search(ucp_context_h context, ucp_tag_t tag, uint64_t tag_mask,
                     ucp_tag_recv_info_t *info, int remove)
{
    ucp_recv_desc_t *rdesc;
    ucp_tag_hdr_t *hdr;
    ucs_queue_iter_t iter;
    ucp_tag_t recv_tag;
    unsigned flags;

    ucs_queue_for_each_safe(rdesc, iter, &context->tag.unexpected, queue) {
        hdr      = (void*)(rdesc + 1);
        recv_tag = hdr->tag;
        flags    = rdesc->flags;
        ucs_trace_req("searching for %"PRIx64"/%"PRIx64"checking desc %p %"PRIx64"/%x",
                      tag, tag_mask, rdesc, recv_tag, flags);
        if ((flags & UCP_RECV_DESC_FLAG_FIRST) &&
            ucp_tag_is_match(recv_tag, tag, tag_mask))
        {
            ucp_tag_log_match(recv_tag, NULL, tag, tag_mask, 0, "probe");

            info->sender_tag = hdr->tag;
            if (flags & UCP_RECV_DESC_FLAG_EAGER) {
                info->length = ucp_eager_total_len(ucs_container_of(hdr, ucp_eager_hdr_t, super),
                                                   flags,
                                                   rdesc->length - sizeof(*hdr));
            } else {
                info->length = ucp_rndv_total_len(ucs_container_of(hdr, ucp_rts_hdr_t, super));
            }

            if (remove) {
                ucs_queue_del_iter(&context->tag.unexpected, iter);
            }
            return rdesc;
        }
    }

    return NULL;
}

ucs_status_t ucp_tag_probe_nb(ucp_worker_h worker, ucp_tag_t tag,
                              ucp_tag_t tag_mask, ucp_tag_recv_info_t *info)
{
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc;

    ucs_trace_req("probe_nb tag %"PRIx64"/%"PRIx64, tag, tag_mask);
    ucp_worker_progress(worker);
    rdesc = ucp_tag_probe_search(context, tag, tag_mask, info, 0);
    return (rdesc != NULL) ? UCS_OK : UCS_ERR_NO_MESSAGE;
}

ucs_status_ptr_t ucp_tag_msg_probe_nb(ucp_worker_h worker, ucp_tag_t tag,
                                      ucp_tag_t tag_mask,
                                      ucp_tag_recv_info_t *info)
{
    ucp_context_h context = worker->context;
    ucp_recv_desc_t *rdesc;

    ucs_trace_req("msg_probe_nb tag %"PRIx64"/%"PRIx64, tag, tag_mask);
    ucp_worker_progress(worker);
    rdesc = ucp_tag_probe_search(context, tag, tag_mask, info, 1);
    ucs_trace_req("msg_probe_nb returning %p", rdesc);
    return (rdesc != NULL) ? rdesc : UCS_STATUS_PTR(UCS_ERR_NO_MESSAGE);
}
