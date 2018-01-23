/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_TAG_MATCH_H_
#define UCP_TAG_MATCH_H_

#include <ucp/api/ucp_def.h>
#include <ucp/core/ucp_types.h>
#include <ucs/datastruct/queue_types.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/stats/stats.h>


#define UCP_TAG_MASK_FULL     0xffffffffffffffffUL  /* All 1-s */


KHASH_INIT(ucp_tag_offload_hash, ucp_tag_t, ucp_worker_iface_t *, 1,
           kh_int64_hash_func, kh_int64_hash_equal);


/**
 * Tag-match header
 */
typedef struct {
    ucp_tag_t                 tag;
} UCS_S_PACKED ucp_tag_hdr_t;


/**
 * Queue of expected requests
 */
typedef struct {
    ucs_queue_head_t      queue;       /* Requests queue */
    unsigned              sw_count;    /* Number of requests in this queue which
                                          are not posted to offload */
    unsigned              block_count; /* Number of requests which can't be
                                          posted to offload. */
} ucp_request_queue_t;


/**
 * Hash table entry for tag message fragments
 */
typedef union {
    ucs_queue_head_t      unexp_q;    /* Queue of unexpected descriptors */
    ucp_request_t         *exp_req;   /* Expected request */
} ucp_tag_frag_match_t;


KHASH_INIT(ucp_tag_frag_hash, uint64_t, ucp_tag_frag_match_t, 1,
           kh_int64_hash_func, kh_int64_hash_equal);


/**
 * Tag-matching context
 */
typedef struct ucp_tag_match {

    /* Expected queue */
    struct {
        ucp_request_queue_t   wildcard;   /* Expected wildcard requests */
        ucp_request_queue_t   *hash;      /* Hash table of expected non-wild tags */
        uint64_t              sn;
        unsigned              sw_all_count; /* Number of all expected requests which
                                               are not posted to offload */
    } expected;

    /* Unexpected queue */
    struct {
        ucs_list_link_t       all;        /* Linked list of all tags */
        ucs_list_link_t       *hash;      /* Hash table of unexpected tags */
    } unexpected;

    /* Hash for fragment assembly, the key is a globally unique tag message id */
    khash_t(ucp_tag_frag_hash) frag_hash;

    /* Tag offload fields */
    struct {
        ucs_queue_head_t      sync_reqs;        /* Outgoing sync send requests */
        khash_t(ucp_tag_offload_hash) tag_hash; /* Hash table of offload ifaces */
        ucp_worker_iface_t    *iface;           /* Active offload iface (relevant if num_ifaces
                                                   is 1, otherwise hash should be used) */
        size_t                thresh;           /* Minimal receive buffer size to be
                                                   used with tag-matching offload. */
        size_t                zcopy_thresh;     /* Minimal size of user-provided
                                                   receive buffer to be passed
                                                   directly to tag-matching offload
                                                   on the transport. Buffers smaller
                                                   than this threshold would either
                                                   bounce to UCP internal buffers,
                                                   or not be used with tag-matching
                                                   offload at all, according to
                                                   'thresh' configuration. */
        unsigned              num_ifaces;       /* Number of active offload
                                                   capable interfaces */
    } offload;

    struct {
        uint64_t              message_id;       /* Unique ID for active messages */
    } am;

} ucp_tag_match_t;


ucs_status_t ucp_tag_match_init(ucp_tag_match_t *tm);

void ucp_tag_match_cleanup(ucp_tag_match_t *tm);

void ucp_tag_exp_remove(ucp_tag_match_t *tm, ucp_request_t *req);

int ucp_tag_unexp_is_empty(ucp_tag_match_t *tm);

ucp_request_t*
ucp_tag_exp_search_all(ucp_tag_match_t *tm, ucp_request_queue_t *req_queue,
                       ucp_tag_t tag);

void ucp_tag_frag_list_process_queue(ucp_tag_match_t *tm, ucp_request_t *req,
                                     uint64_t msg_id
                                     UCS_STATS_ARG(int counter_idx));

#endif
