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


ucp_tag_message_h ucp_tag_probe_nb(ucp_worker_h worker, ucp_tag_t tag,
                                   ucp_tag_t tag_mask, int remove,
                                   ucp_tag_recv_info_t *info)
{
    ucp_context_h UCS_V_UNUSED context = worker->context;
    ucp_recv_desc_t *rdesc;
    uint16_t flags;

    UCP_THREAD_CS_ENTER_CONDITIONAL(&worker->mt_lock);

    ucs_trace_req("probe_nb tag %"PRIx64"/%"PRIx64" remove=%d", tag, tag_mask,
                  remove);

    rdesc = ucp_tag_unexp_search(&worker->tm, tag, tag_mask, remove, "probe");
    if (rdesc != NULL) {
        flags            = rdesc->flags;
        info->sender_tag = ucp_rdesc_get_tag(rdesc);

        if (flags & UCP_RECV_DESC_FLAG_EAGER_ONLY) {
            info->length = rdesc->length - rdesc->payload_offset;
        } else if (flags & UCP_RECV_DESC_FLAG_EAGER) {
            info->length = ((ucp_eager_first_hdr_t*)(rdesc + 1))->total_len;
        } else {
            ucs_assert(flags & UCP_RECV_DESC_FLAG_RNDV);
            info->length = ((ucp_rndv_rts_hdr_t*)(rdesc + 1))->size;
        }
    }

    UCP_THREAD_CS_EXIT_CONDITIONAL(&worker->mt_lock);

    return rdesc;
}
