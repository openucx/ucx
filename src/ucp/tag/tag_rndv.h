/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_RNDV_H_
#define UCP_TAG_RNDV_H_

#include <ucp/rndv/rndv.h>


ucs_status_t ucp_tag_send_start_rndv(ucp_request_t *req);

void ucp_tag_rndv_matched(ucp_worker_h worker, ucp_request_t *req,
                          const ucp_rndv_rts_hdr_t *rndv_rts_hdr);

ucs_status_t ucp_tag_rndv_process_rts(ucp_worker_h worker,
                                      ucp_rndv_rts_hdr_t *rts_hdr,
                                      size_t length, unsigned tl_flags);

size_t ucp_tag_rndv_rts_pack(void *dest, void *arg);

#endif
