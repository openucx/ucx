/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "eager.h"
#include "rndv.h"
#include "tag_match.inl"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_worker.h>
#include <ucs/datastruct/queue.h>


static UCS_F_ALWAYS_INLINE ucp_recv_desc_t*
ucp_tag_probe_search(ucp_worker_h worker, ucp_tag_t tag, uint64_t tag_mask,
                     ucp_tag_recv_info_t *info, int remove)
{
    ucp_recv_desc_t *rdesc;
    ucp_tag_hdr_t *hdr;
    ucp_tag_t recv_tag;
    unsigned flags;

    ucs_list_for_each(rdesc, &worker->tm.unexpected.all,
                      tag_list[UCP_RDESC_ALL_LIST]) {
        hdr      = (void*)(rdesc + 1);
        recv_tag = hdr->tag;
        flags    = rdesc->flags;
        ucs_trace_req("searching for %"PRIx64"/%"PRIx64"checking desc %p %"PRIx64"/%x",
                      tag, tag_mask, rdesc, recv_tag, flags);
        if ((flags & UCP_RECV_DESC_FLAG_FIRST) &&
            ucp_tag_is_match(recv_tag, tag, tag_mask))
        {
            ucp_tag_log_match(recv_tag, rdesc->length - rdesc->payload_offset,
                              NULL, tag, tag_mask, 0, "probe");

            info->sender_tag = hdr->tag;
            if (flags & UCP_RECV_DESC_FLAG_EAGER) {
                info->length = ucp_eager_total_len(ucs_container_of(hdr, ucp_eager_hdr_t, super),
                                                   flags, rdesc->length - rdesc->payload_offset);
            } else {
                info->length = ucp_rndv_total_len(ucs_container_of(hdr, ucp_rndv_rts_hdr_t, super));
            }

            if (remove) {
                /* Prevent the receive descriptor, and any fragments after it,
                 * from being matched by receive requests.
                 */
                rdesc->flags &= ~UCP_RECV_DESC_FLAG_FIRST;
            }
            return rdesc;
        }
    }

    return NULL;
}

ucp_tag_message_h ucp_tag_probe_nb(ucp_worker_h worker, ucp_tag_t tag,
                                   ucp_tag_t tag_mask, int remove,
                                   ucp_tag_recv_info_t *info)
{
    ucp_context_h UCS_V_UNUSED context = worker->context;
    ucp_recv_desc_t *ret;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    ucs_trace_req("probe_nb tag %"PRIx64"/%"PRIx64, tag, tag_mask);
    ret = ucp_tag_probe_search(worker, tag, tag_mask, info, remove);

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);

    return ret;
}
