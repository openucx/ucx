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


const void *ucp_ep_match_get_address(const ucs_conn_match_elem_t *conn_match)
{
    const ucp_ep_ext_gen_t *ep_ext = ucs_container_of(conn_match,
                                                      ucp_ep_ext_gen_t,
                                                      ep_match.conn_match);
    return &ep_ext->ep_match.dest_uuid;
}

ucs_conn_sn_t ucp_ep_match_get_conn_sn(const ucs_conn_match_elem_t *conn_match)
{
    return (ucs_conn_sn_t)ucp_ep_from_ext_gen((ucp_ep_ext_gen_t*)
                                              ucs_container_of(conn_match,
                                                               ucp_ep_ext_gen_t,
                                                               ep_match.conn_match))->conn_sn;
}

const char* ucp_ep_match_address_str(const void *address,
                                     char *str, size_t max_size)
{
    size_t required_size = snprintf(NULL, 0, "%zu", *(uint64_t*)address);

    if (max_size < required_size) {
        ucs_fatal("size of the string (%zu) is not enough to be filled"
                  " by the address (%zu), required - %zu",
                  max_size, *(uint64_t*)address, required_size);
    }

    ucs_snprintf_zero(str, max_size, "%zu", *(uint64_t*)address);
    return str;
}

const ucs_conn_match_ops_t ucp_ep_match_ops = {
    .get_address = ucp_ep_match_get_address,
    .get_conn_sn = ucp_ep_match_get_conn_sn,
    .address_str = ucp_ep_match_address_str
};

ucp_ep_match_conn_sn_t ucp_ep_match_get_sn(ucp_worker_h worker,
                                           uint64_t dest_uuid)
{
    ucs_conn_sn_t conn_sn;

    conn_sn = ucs_conn_match_get_next_sn(&worker->conn_match_ctx, &dest_uuid);
    if (conn_sn >= UCP_EP_MATCH_CONN_SN_MAX) {
        ucs_fatal("connection ID reached the maximal possible value - %u",
                  UCP_EP_MATCH_CONN_SN_MAX);
    }

    return conn_sn;
}

void ucp_ep_match_insert(ucp_worker_h worker, ucp_ep_h ep, uint64_t dest_uuid,
                         ucp_ep_match_conn_sn_t conn_sn, int is_exp)
{
    ucs_assert(!is_exp || !(ep->flags & UCP_EP_FLAG_DEST_EP));
    /* NOTE: protect union */
    ucs_assert(!(ep->flags & (UCP_EP_FLAG_ON_MATCH_CTX |
                              UCP_EP_FLAG_FLUSH_STATE_VALID |
                              UCP_EP_FLAG_LISTENER)));
    ep->flags                             |= UCP_EP_FLAG_ON_MATCH_CTX;
    ucp_ep_ext_gen(ep)->ep_match.dest_uuid = dest_uuid;

    ucs_conn_match_insert(&worker->conn_match_ctx, &dest_uuid,
                          (ucs_conn_sn_t)conn_sn,
                          &ucp_ep_ext_gen(ep)->ep_match.conn_match, is_exp);
}

ucp_ep_h ucp_ep_match_retrieve(ucp_worker_h worker, uint64_t dest_uuid,
                               ucp_ep_match_conn_sn_t conn_sn, int is_exp)
{
        ucp_ep_flags_t UCS_V_UNUSED exp_ep_flags = UCP_EP_FLAG_ON_MATCH_CTX |
                                                   (!is_exp ? UCP_EP_FLAG_DEST_EP : 0);
        ucs_conn_match_elem_t *conn_match;
        ucp_ep_h ep;

        conn_match = ucs_conn_match_retrieve(&worker->conn_match_ctx, &dest_uuid,
                                             (ucs_conn_sn_t)conn_sn, is_exp);
        if (conn_match == NULL) {
            return NULL;
        }

        ep = ucp_ep_from_ext_gen(ucs_container_of(conn_match, ucp_ep_ext_gen_t,
                                                  ep_match.conn_match));

        ucs_assertv(ucs_test_all_flags(ep->flags, exp_ep_flags),
                    "ep=%p flags=0x%x exp_flags=0x%x", ep, ep->flags,
                    exp_ep_flags);
        ep->flags &= ~UCP_EP_FLAG_ON_MATCH_CTX;
        return ep;
}

void ucp_ep_match_remove_ep(ucp_worker_h worker, ucp_ep_h ep)
{
    if (!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX)) {
        return;
    }

    ucs_conn_match_remove_elem(&worker->conn_match_ctx,
                               &ucp_ep_ext_gen(ep)->ep_match.conn_match,
                               !(ep->flags & UCP_EP_FLAG_DEST_EP));

    ep->flags &= ~UCP_EP_FLAG_ON_MATCH_CTX;
}
