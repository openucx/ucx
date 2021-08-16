/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCP_EP_MATCH_H_
#define UCP_EP_MATCH_H_

#include <ucp/core/ucp_types.h>
#include <ucs/datastruct/conn_match.h>


/**
 * Maximal value for EP connection sequence number
 */
#define UCP_EP_MATCH_CONN_SN_MAX    ((ucp_ep_match_conn_sn_t)-1)


/**
 * UCP EP connection matching sequence number
 */
typedef uint16_t ucp_ep_match_conn_sn_t;


/**
 * Object that represents matching with remote endpoints
 */
typedef struct {
    uint64_t                  dest_uuid;         /* Destination worker UUID */
    ucs_conn_match_elem_t     conn_match;        /* Connection matching object */
} ucp_ep_match_elem_t;


extern const ucs_conn_match_ops_t ucp_ep_match_ops;


ucp_ep_match_conn_sn_t ucp_ep_match_get_sn(ucp_worker_h worker,
                                           uint64_t dest_uuid);


int ucp_ep_match_insert(ucp_worker_h worker, ucp_ep_h ep, uint64_t dest_uuid,
                        ucp_ep_match_conn_sn_t conn_sn,
                        ucs_conn_match_queue_type_t conn_queue_type);


ucp_ep_h ucp_ep_match_retrieve(ucp_worker_h worker, uint64_t dest_uuid,
                               ucp_ep_match_conn_sn_t conn_sn,
                               ucs_conn_match_queue_type_t conn_queue_type);


void ucp_ep_match_remove_ep(ucp_worker_h worker, ucp_ep_h ep);


#endif
