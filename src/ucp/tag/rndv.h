/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_RNDV_H_
#define UCP_TAG_RNDV_H_

#include <ucp/api/ucp.h>

#include <ucp/core/ucp_request.h>


ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *req);


void ucp_rndv_unexp_match(ucp_recv_desc_t *rdesc, ucp_request_t *req,
                          void *buffer, size_t count, ucp_datatype_t datatype);

#endif
