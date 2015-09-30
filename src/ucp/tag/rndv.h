/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_RNDV_H_
#define UCP_TAG_RNDV_H_

#include "match.h"

#include <ucp/api/ucp.h>
#include <ucp/core/ucp_request.h>


/*
 * Rendezvous RTS
 */
typedef struct {
    ucp_tag_hdr_t             super;
    size_t                    total_len;
} UCS_S_PACKED ucp_rts_hdr_t;



ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *req);


void ucp_rndv_unexp_match(ucp_recv_desc_t *rdesc, ucp_request_t *req,
                          void *buffer, size_t count, ucp_datatype_t datatype);


static inline size_t ucp_rndv_total_len(ucp_rts_hdr_t *hdr)
{
    return hdr->total_len;
}

#endif
