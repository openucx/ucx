/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_MATCH_H_
#define UCP_TAG_MATCH_H_

#include <ucp/api/ucp_def.h>
#include <ucp/core/ucp_types.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/sys/compiler_def.h>


/**
 * Tag-match header
 */
typedef struct {
    ucp_tag_t                 tag;
} UCS_S_PACKED ucp_tag_hdr_t;


/**
 * Tag-matching context
 */
typedef struct ucp_tag_match {
    ucs_queue_head_t          expected;   /* Expected requests */
    ucs_queue_head_t          unexpected; /* Unexpected received descriptors */
} ucp_tag_match_t;


ucs_status_t ucp_tag_match_init(ucp_tag_match_t *tm);

void ucp_tag_match_cleanup(ucp_tag_match_t *tm);

void ucp_tag_exp_remove(ucp_tag_match_t *tm, ucp_request_t *req);

int ucp_tag_unexp_is_empty(ucp_tag_match_t *tm);

#endif
