/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "conn_match.h"

#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/memtrack.h>
#include <ucs/algorithm/crc.h>


/**
 * Maximal length of address
 */
#define UCS_CONN_MATCH_ADDRESS_STR_MAX    128


struct ucs_conn_match_peer {
    ucs_hlist_head_t         conn_q[UCS_CONN_MATCH_QUEUE_LAST]; /* Connection queues */
    ucs_conn_sn_t            next_conn_sn;                      /* Sequence number of matching
                                                                   connections, since wireup messages
                                                                   used for connection establishment
                                                                   procedure which were sent on different
                                                                   connections could be provided
                                                                   out-of-order */
    size_t                   address_length;                    /* Length of the addresses used for the
                                                                   connection between peers */
    char                     address[0];
};

static UCS_F_ALWAYS_INLINE khint_t
ucs_conn_match_peer_hash(ucs_conn_match_peer_t *peer)
{
    return ucs_crc32(0, &peer->address, peer->address_length);
}

static UCS_F_ALWAYS_INLINE int
ucs_conn_match_peer_equal(ucs_conn_match_peer_t *peer1,
                          ucs_conn_match_peer_t *peer2)
{
    return (peer1->address_length == peer2->address_length) &&
           !memcmp(&peer1->address, &peer2->address, peer1->address_length);
}

KHASH_IMPL(ucs_conn_match, ucs_conn_match_peer_t*, char, 0,
           ucs_conn_match_peer_hash, ucs_conn_match_peer_equal);


const static char *ucs_conn_match_queue_title[] = {
    [UCS_CONN_MATCH_QUEUE_EXP]   = "expected",
    [UCS_CONN_MATCH_QUEUE_UNEXP] = "unexpected"
};


void ucs_conn_match_init(ucs_conn_match_ctx_t *conn_match_ctx,
                         size_t address_length,
                         const ucs_conn_match_ops_t *ops)
{
    kh_init_inplace(ucs_conn_match, &conn_match_ctx->hash);
    conn_match_ctx->address_length = address_length;
    conn_match_ctx->ops            = *ops;
}

void ucs_conn_match_cleanup(ucs_conn_match_ctx_t *conn_match_ctx)
{
    char address_str[UCS_CONN_MATCH_ADDRESS_STR_MAX];
    ucs_conn_match_peer_t *peer;
    unsigned i;

    kh_foreach_key(&conn_match_ctx->hash, peer, {
        for (i = 0; i < UCS_CONN_MATCH_QUEUE_LAST; i++) {
            if (!ucs_hlist_is_empty(&peer->conn_q[i])) {
                ucs_diag("match_ctx %p: %s queue is not empty for %s address",
                         conn_match_ctx,
                         ucs_conn_match_queue_title[i],
                         conn_match_ctx->ops.address_str(&peer->address, address_str,
                                                         UCS_CONN_MATCH_ADDRESS_STR_MAX));
            }
        }

        ucs_free(peer);
    })
    kh_destroy_inplace(ucs_conn_match, &conn_match_ctx->hash);
}

static ucs_conn_match_peer_t*
ucs_conn_match_peer_alloc(ucs_conn_match_ctx_t *conn_match_ctx,
                          const void *address)
{
    char address_str[UCS_CONN_MATCH_ADDRESS_STR_MAX];
    ucs_conn_match_peer_t *peer;

    peer = ucs_calloc(1, sizeof(*peer), "conn match peer");
    if (peer == NULL) {
        ucs_fatal("match_ctx %p: failed to allocate memory for %s address",
                  conn_match_ctx,
                  conn_match_ctx->ops.address_str(address, address_str,
                                                  UCS_CONN_MATCH_ADDRESS_STR_MAX));
    }

    peer->address_length = conn_match_ctx->address_length;
    memcpy(&peer->address, address, peer->address_length);

    return peer;
}

static ucs_conn_match_peer_t*
ucs_conn_match_get_conn(ucs_conn_match_ctx_t *conn_match_ctx,
                        const void *address)
{
    char address_str[UCS_CONN_MATCH_ADDRESS_STR_MAX];
    ucs_conn_match_peer_t *peer;
    khiter_t iter;
    int ret;

    peer = ucs_conn_match_peer_alloc(conn_match_ctx, address);
    iter = kh_put(ucs_conn_match, &conn_match_ctx->hash, peer, &ret);
    if (ucs_unlikely(ret == UCS_KH_PUT_FAILED)) {
        ucs_free(peer);
        ucs_fatal("match_ctx %p: kh_put failed for %s",
                  conn_match_ctx,
                  conn_match_ctx->ops.address_str(address, address_str,
                                                  UCS_CONN_MATCH_ADDRESS_STR_MAX));
    }

    if (ret == UCS_KH_PUT_KEY_PRESENT) {
        ucs_free(peer);
        return kh_key(&conn_match_ctx->hash, iter);
    }

    /* initialize match list on first use */
    peer->next_conn_sn = 0;
    ucs_hlist_head_init(&peer->conn_q[UCS_CONN_MATCH_QUEUE_EXP]);
    ucs_hlist_head_init(&peer->conn_q[UCS_CONN_MATCH_QUEUE_UNEXP]);

    return peer;
}

ucs_conn_sn_t ucs_conn_match_get_next_sn(ucs_conn_match_ctx_t *conn_match_ctx,
                                         const void *address)
{
    ucs_conn_match_peer_t *peer = ucs_conn_match_get_conn(conn_match_ctx,
                                                          address);
    return peer->next_conn_sn++;
}

void ucs_conn_match_insert(ucs_conn_match_ctx_t *conn_match_ctx,
                           const void *address, ucs_conn_sn_t conn_sn,
                           ucs_conn_match_elem_t *conn_match,
                           ucs_conn_match_queue_type_t conn_queue_type)
{
    ucs_conn_match_peer_t *peer = ucs_conn_match_get_conn(conn_match_ctx,
                                                          address);
    ucs_hlist_head_t *head      = &peer->conn_q[conn_queue_type];
    char UCS_V_UNUSED address_str[UCS_CONN_MATCH_ADDRESS_STR_MAX];

    ucs_hlist_add_tail(head, &conn_match->list);
    ucs_trace("match_ctx %p: conn_match %p added as %s address %s conn_sn %zu",
              conn_match_ctx, conn_match,
              ucs_conn_match_queue_title[conn_queue_type],
              conn_match_ctx->ops.address_str(address, address_str,
                                              UCS_CONN_MATCH_ADDRESS_STR_MAX),
              conn_sn);
}

ucs_conn_match_elem_t *
ucs_conn_match_retrieve(ucs_conn_match_ctx_t *conn_match_ctx,
                        const void *address, ucs_conn_sn_t conn_sn,
                        ucs_conn_match_queue_type_t conn_queue_type)
{
    char UCS_V_UNUSED address_str[UCS_CONN_MATCH_ADDRESS_STR_MAX];
    ucs_conn_match_peer_t *peer;
    ucs_conn_match_elem_t *elem;
    ucs_hlist_head_t *head;
    khiter_t iter;

    peer = ucs_conn_match_peer_alloc(conn_match_ctx, address);
    iter = kh_get(ucs_conn_match, &conn_match_ctx->hash, peer);
    ucs_free(peer);
    if (iter == kh_end(&conn_match_ctx->hash)) {
        goto notfound; /* no hash entry */
    }

    peer = kh_key(&conn_match_ctx->hash, iter);
    head = &peer->conn_q[conn_queue_type];

    ucs_hlist_for_each(elem, head, list) {
        if (conn_match_ctx->ops.get_conn_sn(elem) == conn_sn) {
            ucs_hlist_del(head, &elem->list);
            ucs_trace("match_ctx %p: matched %s conn_match %p by address %s conn_sn %zu",
                      conn_match_ctx, ucs_conn_match_queue_title[conn_queue_type], elem,
                      conn_match_ctx->ops.address_str(address, address_str,
                                                      UCS_CONN_MATCH_ADDRESS_STR_MAX),
                      conn_sn);
            return elem;
        }
    }

notfound:
    ucs_trace("match_ctx %p: %s address %s conn_sn %zu not found",
              conn_match_ctx, ucs_conn_match_queue_title[conn_queue_type],
              conn_match_ctx->ops.address_str(address, address_str,
                                              UCS_CONN_MATCH_ADDRESS_STR_MAX),
              conn_sn);
    return NULL;
}

void ucs_conn_match_remove_elem(ucs_conn_match_ctx_t *conn_match_ctx,
                                ucs_conn_match_elem_t *elem,
                                ucs_conn_match_queue_type_t conn_queue_type)
{
    const void *address = conn_match_ctx->ops.get_address(elem);
    char UCS_V_UNUSED address_str[UCS_CONN_MATCH_ADDRESS_STR_MAX];
    ucs_conn_match_peer_t *peer;
    ucs_hlist_head_t *head;
    khiter_t iter;

    peer = ucs_conn_match_peer_alloc(conn_match_ctx, address);
    iter = kh_get(ucs_conn_match, &conn_match_ctx->hash, peer);
    if (iter == kh_end(&conn_match_ctx->hash)) {
        ucs_fatal("match_ctx %p: conn_match %p address %s conn_sn %zu "
                  "wasn't found in hash", conn_match_ctx, elem,
                  conn_match_ctx->ops.address_str(&address, address_str,
                                                  UCS_CONN_MATCH_ADDRESS_STR_MAX),
                  conn_match_ctx->ops.get_conn_sn(elem));
    }

    ucs_free(peer);

    peer = kh_key(&conn_match_ctx->hash, iter);
    head = &peer->conn_q[conn_queue_type];

    ucs_hlist_del(head, &elem->list);
    ucs_trace("match_ctx %p: remove %s conn_match %p address %s conn_sn %zu)",
              conn_match_ctx, ucs_conn_match_queue_title[conn_queue_type],
              elem, conn_match_ctx->ops.address_str(&address, address_str,
                                                    UCS_CONN_MATCH_ADDRESS_STR_MAX),
              conn_match_ctx->ops.get_conn_sn(elem));
}
