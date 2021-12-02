/**
 * Copyright (C) Mellanox Technologies Ltd. 2018-2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif


#include "ep_match.h"

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>


static const ucp_ep_ext_gen_t *
ucp_ep_ext_gen_from_conn_match(const ucs_conn_match_elem_t *conn_match)
{
    return ucs_container_of(conn_match, ucp_ep_ext_gen_t,
                            ep_match.conn_match);
}

static const void *
ucp_ep_match_get_address(const ucs_conn_match_elem_t *conn_match)
{
    const ucp_ep_ext_gen_t *ep_ext = ucp_ep_ext_gen_from_conn_match(conn_match);
    return &ep_ext->ep_match.dest_uuid;
}

static ucs_conn_sn_t
ucp_ep_match_get_conn_sn(const ucs_conn_match_elem_t *conn_match)
{
    return (ucs_conn_sn_t)
           ucp_ep_from_ext_gen((ucp_ep_ext_gen_t*)
                               ucp_ep_ext_gen_from_conn_match(
                                   conn_match))->conn_sn;
}

static const char *
ucp_ep_match_address_str(const ucs_conn_match_ctx_t *conn_match_ctx,
                         const void *address, char *str, size_t max_size)
{
    ucs_snprintf_zero(str, max_size, "%"PRIu64, *(uint64_t*)address);
    return str;
}

const ucs_conn_match_ops_t ucp_ep_match_ops = {
    .get_address = ucp_ep_match_get_address,
    .get_conn_sn = ucp_ep_match_get_conn_sn,
    .address_str = ucp_ep_match_address_str,
    .purge_cb    = NULL
};

ucp_ep_match_conn_sn_t ucp_ep_match_get_sn(ucp_worker_h worker,
                                           uint64_t dest_uuid)
{
    return ucs_conn_match_get_next_sn(&worker->conn_match_ctx, &dest_uuid);
}

int ucp_ep_match_insert(ucp_worker_h worker, ucp_ep_h ep, uint64_t dest_uuid,
                        ucp_ep_match_conn_sn_t conn_sn,
                        ucs_conn_match_queue_type_t conn_queue_type)
{
    ucs_assert((conn_queue_type == UCS_CONN_MATCH_QUEUE_UNEXP) ||
               !(ep->flags & UCP_EP_FLAG_REMOTE_ID));
    /* EP matching is not used in CM flow */
    ucs_assert(!ucp_ep_has_cm_lane(ep));

    /* NOTE: protect union */
    ucs_assert(!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX));
    ucp_ep_flush_state_invalidate(ep);

    ucp_ep_ext_gen(ep)->ep_match.dest_uuid = dest_uuid;

    if (ucs_conn_match_insert(&worker->conn_match_ctx, &dest_uuid,
                              (ucs_conn_sn_t)conn_sn,
                              &ucp_ep_ext_gen(ep)->ep_match.conn_match,
                              conn_queue_type)) {
        ucp_ep_update_flags(ep, UCP_EP_FLAG_ON_MATCH_CTX, 0);
        return 1;
    }

    /* EP was not added to EP matching, make EP's flush state valid */
    ucp_ep_flush_state_reset(ep);
    return 0;
}

ucp_ep_h ucp_ep_match_retrieve(ucp_worker_h worker, uint64_t dest_uuid,
                               ucp_ep_match_conn_sn_t conn_sn,
                               ucs_conn_match_queue_type_t conn_queue_type)
{
    ucp_ep_flags_t UCS_V_UNUSED exp_ep_flags = UCP_EP_FLAG_ON_MATCH_CTX;
    ucs_conn_match_elem_t *conn_match;
    ucp_ep_h ep;

    if (conn_queue_type == UCS_CONN_MATCH_QUEUE_UNEXP) {
        exp_ep_flags |= UCP_EP_FLAG_REMOTE_ID;
    }

    conn_match = ucs_conn_match_get_elem(&worker->conn_match_ctx, &dest_uuid,
                                         (ucs_conn_sn_t)conn_sn,
                                         conn_queue_type, 1);
    if (conn_match == NULL) {
        return NULL;
    }

    ep = ucp_ep_from_ext_gen(ucs_container_of(conn_match, ucp_ep_ext_gen_t,
                                              ep_match.conn_match));

    /* EP matching is not used in CM flow */
    ucs_assert(!ucp_ep_has_cm_lane(ep));
    ucs_assertv(ucs_test_all_flags(ep->flags, exp_ep_flags),
                "ep=%p flags=0x%x exp_flags=0x%x", ep, ep->flags,
                exp_ep_flags);
    ucp_ep_update_flags(ep, 0, UCP_EP_FLAG_ON_MATCH_CTX);
    ucp_ep_flush_state_reset(ep);

    return ep;
}

void ucp_ep_match_remove_ep(ucp_worker_h worker, ucp_ep_h ep)
{
    if (!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX)) {
        return;
    }

    ucs_assert(ep->conn_sn != UCP_EP_MATCH_CONN_SN_MAX);

    ucs_conn_match_remove_elem(&worker->conn_match_ctx,
                               &ucp_ep_ext_gen(ep)->ep_match.conn_match,
                               (ep->flags & UCP_EP_FLAG_REMOTE_ID) ?
                               UCS_CONN_MATCH_QUEUE_UNEXP :
                               UCS_CONN_MATCH_QUEUE_EXP);

    ucp_ep_update_flags(ep, 0, UCP_EP_FLAG_ON_MATCH_CTX);
    /* Reset the endpoint's flush state to make it valid in case of discarding
     * the endpoint during error handling. The flush state will be used to
     * complete remote RMA requests during purging requests */
    ucp_ep_flush_state_reset(ep);
}
