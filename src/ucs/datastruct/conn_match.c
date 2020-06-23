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


static UCS_F_ALWAYS_INLINE uint32_t
ucs_conn_match_addr_hash(ucs_conn_match_addr_t conn_match_addr)
{
    return ucs_crc32(0, (const void*)conn_match_addr.addr,
                     conn_match_addr.length);
}

static UCS_F_ALWAYS_INLINE int
ucs_conn_match_addr_equal(ucs_conn_match_addr_t conn_match_addr1,
                          ucs_conn_match_addr_t conn_match_addr2)
{
    ucs_assert(conn_match_addr1.length == conn_match_addr2.length);
    return !memcmp(conn_match_addr1.addr, conn_match_addr2.addr,
                   conn_match_addr1.length);
}

__KHASH_IMPL(ucs_conn_match, static UCS_F_MAYBE_UNUSED inline,
             ucs_conn_match_addr_t, ucs_conn_match_entry_t, 1,
             ucs_conn_match_addr_hash, ucs_conn_match_addr_equal);


const char *ucs_conn_match_queue_title[] = {
    [UCS_CONN_MATCH_QUEUE_EXP]   = "expected",
    [UCS_CONN_MATCH_QUEUE_UNEXP] = "unexpected"
};


void ucs_conn_match_init(ucs_conn_match_ctx_t *conn_match_ctx,
                         const ucs_conn_match_ops_t *ops)
{
    kh_init_inplace(ucs_conn_match, &conn_match_ctx->hash);
    conn_match_ctx->ops = *ops;
    
}

void ucs_conn_match_cleanup(ucs_conn_match_ctx_t *conn_match_ctx)
{
    char dest_addr_str[UCS_CONN_MATCH_ADDR_STR_MAX];
    ucs_conn_match_entry_t entry;
    ucs_conn_match_addr_t conn_match_addr;

    kh_foreach(&conn_match_ctx->hash, conn_match_addr, entry, {
        if (!ucs_hlist_is_empty(&entry.conn_q[UCS_CONN_MATCH_QUEUE_EXP])) {
            ucs_warn("match_ctx %p: addr %s expected queue is not empty",
                     conn_match_ctx,
                     conn_match_ctx->ops.addr_str(&conn_match_addr, dest_addr_str,
                                                  UCS_CONN_MATCH_ADDR_STR_MAX));
        }
        if (!ucs_hlist_is_empty(&entry.conn_q[UCS_CONN_MATCH_QUEUE_UNEXP])) {
            ucs_warn("match_ctx %p: addr %s unexpected queue is not empty",
                     conn_match_ctx,
                     conn_match_ctx->ops.addr_str(&conn_match_addr, dest_addr_str,
                                                  UCS_CONN_MATCH_ADDR_STR_MAX));
        }

        ucs_free(conn_match_addr.addr);
    })
    kh_destroy_inplace(ucs_conn_match, &conn_match_ctx->hash);
}

static ucs_conn_match_entry_t*
ucs_conn_match_entry_get(ucs_conn_match_ctx_t *conn_match_ctx,
                         const ucs_conn_match_addr_t *dest_addr)
{
    char dest_addr_str[UCS_CONN_MATCH_ADDR_STR_MAX];
    ucs_conn_match_entry_t *entry;
    ucs_conn_match_addr_t *key;
    khiter_t iter;
    int ret;

    iter  = kh_put(ucs_conn_match, &conn_match_ctx->hash,
                   *dest_addr, &ret);
    entry = &kh_value(&conn_match_ctx->hash, iter);

    if (ucs_unlikely(ret == UCS_KH_PUT_FAILED)) {
        ucs_fatal("match_ctx %p: kh_put failed for %s",
                  conn_match_ctx,
                  conn_match_ctx->ops.addr_str(dest_addr, dest_addr_str,
                                               UCS_CONN_MATCH_ADDR_STR_MAX));
    } else if (ret != UCS_KH_PUT_KEY_PRESENT) {
        /* initialize match list on first use */
        entry->next_conn_sn = 0;
        ucs_hlist_head_init(&entry->conn_q[UCS_CONN_MATCH_QUEUE_EXP]);
        ucs_hlist_head_init(&entry->conn_q[UCS_CONN_MATCH_QUEUE_UNEXP]);

        key = &kh_key(&conn_match_ctx->hash, iter);
        key->addr = ucs_malloc(key->length, "conn match address");
        if (key->addr == NULL) {
            ucs_fatal("match_ctx %p: failed to allocate memory for %s address",
                      conn_match_ctx,
                      conn_match_ctx->ops.addr_str(dest_addr, dest_addr_str,
                                                   UCS_CONN_MATCH_ADDR_STR_MAX));
        }
        ucs_assert(key->length == dest_addr->length);
        memcpy(key->addr, dest_addr->addr, key->length);
    }

    return entry;
}

ucs_conn_sn_t ucs_conn_match_get_next_sn(ucs_conn_match_ctx_t *conn_match_ctx,
                                         const ucs_conn_match_addr_t *dest_addr)
{
    ucs_conn_match_entry_t *entry = ucs_conn_match_entry_get(conn_match_ctx,
                                                             dest_addr);
    return entry->next_conn_sn++;
}

void ucs_conn_match_insert(ucs_conn_match_ctx_t *conn_match_ctx,
                           const ucs_conn_match_addr_t *dest_addr,
                           ucs_conn_sn_t conn_sn,
                           ucs_conn_match_t *conn_match,
                           ucs_conn_match_queue_type_t conn_queue_type)
{
    ucs_conn_match_entry_t *entry = ucs_conn_match_entry_get(conn_match_ctx,
                                                             dest_addr);
    ucs_hlist_head_t *head        = &entry->conn_q[conn_queue_type];
    char dest_addr_str[UCS_CONN_MATCH_ADDR_STR_MAX];

    ucs_hlist_add_tail(head, &conn_match->list);
    ucs_trace("match_ctx %p: conn_match %p added as %s addr %s conn_sn %u",
              conn_match_ctx, conn_match,
              ucs_conn_match_queue_title[conn_queue_type],
              conn_match_ctx->ops.addr_str(dest_addr, dest_addr_str,
                                           UCS_CONN_MATCH_ADDR_STR_MAX),
              conn_sn);
}

ucs_conn_match_t *ucs_conn_match_retrieve(ucs_conn_match_ctx_t *conn_match_ctx,
                                          const ucs_conn_match_addr_t *dest_addr,
                                          ucs_conn_sn_t conn_sn,
                                          ucs_conn_match_queue_type_t conn_queue_type)
{
    char dest_addr_str[UCS_CONN_MATCH_ADDR_STR_MAX];
    ucs_conn_match_entry_t *entry;
    ucs_conn_match_t *conn_match;
    ucs_hlist_head_t *head;
    khiter_t iter;

    iter = kh_get(ucs_conn_match, &conn_match_ctx->hash, *dest_addr);
    if (iter == kh_end(&conn_match_ctx->hash)) {
        goto notfound; /* no hash entry */
    }

    entry = &kh_value(&conn_match_ctx->hash, iter);
    head  = &entry->conn_q[conn_queue_type];

    ucs_hlist_for_each(conn_match, head, list) {
        if (conn_match_ctx->ops.get_conn_sn(conn_match) == conn_sn) {
            ucs_hlist_del(head, &conn_match->list);
            ucs_trace("match_ctx %p: matched %s conn_match %p by addr %s conn_sn %u",
                      conn_match_ctx, ucs_conn_match_queue_title[conn_queue_type],
                      conn_match,
                      conn_match_ctx->ops.addr_str(dest_addr, dest_addr_str,
                                                   UCS_CONN_MATCH_ADDR_STR_MAX),
                      conn_sn);
            return conn_match;
        }
    }

notfound:
    ucs_trace("match_ctx %p: %s addr %s conn_sn %u not found",
              conn_match_ctx, ucs_conn_match_queue_title[conn_queue_type],
              conn_match_ctx->ops.addr_str(dest_addr, dest_addr_str,
                                           UCS_CONN_MATCH_ADDR_STR_MAX),
              conn_sn);
    return NULL;
}

void ucs_conn_match_remove_conn(ucs_conn_match_ctx_t *conn_match_ctx,
                                ucs_conn_match_t *conn_match,
                                ucs_conn_match_queue_type_t conn_queue_type)
{
    ucs_conn_match_addr_t dest_addr;
    char dest_addr_str[UCS_CONN_MATCH_ADDR_STR_MAX];
    ucs_conn_match_entry_t *entry;
    ucs_hlist_head_t *head;
    khiter_t iter;

    conn_match_ctx->ops.get_addr(conn_match, &dest_addr);

    iter = kh_get(ucs_conn_match, &conn_match_ctx->hash, dest_addr);
    ucs_assertv(iter != kh_end(&conn_match_ctx->hash),
                "conn_match %p not found in hash", conn_match);
    entry = &kh_value(&conn_match_ctx->hash, iter);
    head  = &entry->conn_q[conn_queue_type];

    ucs_hlist_del(head, &conn_match->list);
    ucs_trace("match_ctx %p: remove %s conn_match %p addr %s conn_sn %u)",
              conn_match_ctx, ucs_conn_match_queue_title[conn_queue_type],
              conn_match, conn_match_ctx->ops.addr_str(&dest_addr, dest_addr_str,
                                                       UCS_CONN_MATCH_ADDR_STR_MAX),
              conn_match_ctx->ops.get_conn_sn(conn_match));
}
