/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_EP_MATCH_H_
#define UCP_EP_MATCH_H_

#include <ucp/core/ucp_types.h>
#include <ucs/datastruct/khash.h>
#include <ucs/datastruct/list.h>


/*
 * Structure to embed in a UCP endpoint to support matching with remote endpoints
 */
typedef struct {
    uint64_t                  dest_uuid;     /* Destination worker UUID */
    ucs_list_link_t           list;          /* List entry into endpoint
                                                matching structure */
} ucp_ep_match_t;


/**
 * Endpoint-to-endpoint matching entry - allows *ordered* matching of endpoints
 * between a pair of workers.
 * The expected/unexpected lists are *not* circular
 */
typedef struct ucp_ep_match_entry {
    ucs_list_link_t          exp_ep_q;        /* Endpoints created by API and not
                                                 connected to remote endpoint */
    ucs_list_link_t          unexp_ep_q;      /* Endpoints created internally as
                                                 connected a to remote endpoints,
                                                 but not provided to user yet */
    ucp_ep_conn_sn_t         next_conn_sn;    /* Sequence number of matching
                                                 endpoints, since UCT may provide
                                                 wireup messages which were sent
                                                 on different endpoint out-of-order */
} ucp_ep_match_entry_t;


__KHASH_TYPE(ucp_ep_match, uint64_t, ucp_ep_match_entry_t)


/* Context for matching endpoints */
typedef struct {
    khash_t(ucp_ep_match)    hash;
} ucp_ep_match_ctx_t;


void ucp_ep_match_init(ucp_ep_match_ctx_t *match_ctx);

void ucp_ep_match_cleanup(ucp_ep_match_ctx_t *match_ctx);

ucp_ep_conn_sn_t ucp_ep_match_get_next_sn(ucp_ep_match_ctx_t *match_ctx,
                                          uint64_t dest_uuid);

void ucp_ep_match_insert_exp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                             ucp_ep_h ep);

void ucp_ep_match_insert_unexp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                               ucp_ep_h ep);

ucp_ep_h ucp_ep_match_retrieve_exp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                                   ucp_ep_conn_sn_t conn_sn);

ucp_ep_h ucp_ep_match_retrieve_unexp(ucp_ep_match_ctx_t *ep_conn, uint64_t dest_uuid,
                                     ucp_ep_conn_sn_t conn_sn);

void ucp_ep_match_remove_ep(ucp_ep_match_ctx_t *ep_conn, ucp_ep_h ep);


#endif
