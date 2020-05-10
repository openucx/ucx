/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/wireup/ep_match.h>
#include <inttypes.h>


__KHASH_IMPL(ucp_ep_match, static UCS_F_MAYBE_UNUSED inline, uint64_t,
             ucp_ep_match_entry_t, 1, kh_int64_hash_func, kh_int64_hash_equal);

void ucp_ep_match_init(ucp_ep_match_ctx_t *match_ctx)
{
    kh_init_inplace(ucp_ep_match, &match_ctx->hash);
}

void ucp_ep_match_cleanup(ucp_ep_match_ctx_t *match_ctx)
{
    ucp_ep_match_entry_t entry;
    uint64_t dest_uuid;

    kh_foreach(&match_ctx->hash, dest_uuid, entry, {
        if (!ucs_hlist_is_empty(&entry.exp_ep_q)) {
            ucs_warn("match_ctx %p: uuid 0x%"PRIx64" expected queue is not empty",
                     match_ctx, dest_uuid);
        }
        if (!ucs_hlist_is_empty(&entry.unexp_ep_q)) {
            ucs_warn("match_ctx %p: uuid 0x%"PRIx64" unexpected queue is not empty",
                     match_ctx, dest_uuid);
        }
    })
    kh_destroy_inplace(ucp_ep_match, &match_ctx->hash);
}

static ucp_ep_match_entry_t*
ucp_ep_match_entry_get(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid)
{
    ucp_ep_match_entry_t *entry;
    khiter_t iter;
    int ret;

    iter  = kh_put(ucp_ep_match, &match_ctx->hash, dest_uuid, &ret);
    entry = &kh_value(&match_ctx->hash, iter);

    if (ret != 0) {
        /* initialize match list on first use */
        entry->next_conn_sn    = 0;
        ucs_hlist_head_init(&entry->exp_ep_q);
        ucs_hlist_head_init(&entry->unexp_ep_q);
    }

    return entry;
}

ucp_ep_conn_sn_t ucp_ep_match_get_next_sn(ucp_ep_match_ctx_t *match_ctx,
                                          uint64_t dest_uuid)
{
    ucp_ep_match_entry_t *entry = ucp_ep_match_entry_get(match_ctx, dest_uuid);
    return entry->next_conn_sn++;
}

static void ucp_ep_match_insert_common(ucp_ep_match_ctx_t *match_ctx,
                                       ucs_hlist_head_t *head, ucp_ep_h ep,
                                       uint64_t dest_uuid, const char *title)
{
    /* NOTE: protect union */
    ucs_assert(!(ep->flags & (UCP_EP_FLAG_ON_MATCH_CTX |
                              UCP_EP_FLAG_FLUSH_STATE_VALID |
                              UCP_EP_FLAG_LISTENER)));

    ucs_hlist_add_tail(head, &ucp_ep_ext_gen(ep)->ep_match.list);
    ep->flags                              |= UCP_EP_FLAG_ON_MATCH_CTX;
    ucp_ep_ext_gen(ep)->ep_match.dest_uuid  = dest_uuid;
    ucs_trace("match_ctx %p: ep %p added as %s uuid 0x%"PRIx64" conn_sn %d",
              match_ctx, ep, title, dest_uuid, ep->conn_sn);
}

void ucp_ep_match_insert_exp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                             ucp_ep_h ep)
{
    ucp_ep_match_entry_t *entry = ucp_ep_match_entry_get(match_ctx, dest_uuid);

    ucs_assert(!(ep->flags & UCP_EP_FLAG_DEST_EP));
    ucp_ep_match_insert_common(match_ctx, &entry->exp_ep_q, ep, dest_uuid,
                               "expected");
}

void ucp_ep_match_insert_unexp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                               ucp_ep_h ep)
{
    ucp_ep_match_entry_t *entry = ucp_ep_match_entry_get(match_ctx, dest_uuid);

    ucp_ep_match_insert_common(match_ctx, &entry->unexp_ep_q, ep, dest_uuid,
                               "unexpected");
}

static ucp_ep_h
ucp_ep_match_retrieve_common(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                             ucp_ep_conn_sn_t conn_sn, int is_exp,
                             ucp_ep_flags_t exp_ep_flags, const char *title)
{
    ucp_ep_match_entry_t *entry;
    ucp_ep_ext_gen_t *ep_ext;
    ucs_hlist_head_t *head;
    khiter_t iter;
    ucp_ep_h ep;

    iter = kh_get(ucp_ep_match, &match_ctx->hash, dest_uuid);
    if (iter == kh_end(&match_ctx->hash)) {
        goto notfound; /* no hash entry */
    }

    entry = &kh_value(&match_ctx->hash, iter);
    head  = is_exp ? &entry->exp_ep_q : &entry->unexp_ep_q;

    ucs_hlist_for_each(ep_ext, head, ep_match.list) {
        ep = ucp_ep_from_ext_gen(ep_ext);
        if (ep->conn_sn == conn_sn) {
            ucs_hlist_del(head, &ep_ext->ep_match.list);
            ucs_trace("match_ctx %p: matched %s ep %p by uuid 0x%"PRIx64" conn_sn %d",
                      match_ctx, title, ep, dest_uuid, conn_sn);
            ucs_assertv(ucs_test_all_flags(ep->flags,
                                           exp_ep_flags | UCP_EP_FLAG_ON_MATCH_CTX),
                        "ep=%p flags=0x%x exp_flags=0x%x", ep, ep->flags,
                        exp_ep_flags);
            ep->flags &= ~UCP_EP_FLAG_ON_MATCH_CTX;
            return ep;
        }
    }

notfound:
    ucs_trace("match_ctx %p: %s uuid 0x%"PRIx64" conn_sn %d not found",
              match_ctx, title, dest_uuid, conn_sn);
    return NULL;
}

ucp_ep_h ucp_ep_match_retrieve_exp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                                   ucp_ep_conn_sn_t conn_sn)
{
    return ucp_ep_match_retrieve_common(match_ctx, dest_uuid, conn_sn, 1, 0,
                                        "expected");
}

ucp_ep_h ucp_ep_match_retrieve_unexp(ucp_ep_match_ctx_t *match_ctx, uint64_t dest_uuid,
                                     ucp_ep_conn_sn_t conn_sn)
{
    return ucp_ep_match_retrieve_common(match_ctx, dest_uuid, conn_sn, 0,
                                        UCP_EP_FLAG_DEST_EP, "unexpected");
}

void ucp_ep_match_remove_ep(ucp_ep_match_ctx_t *match_ctx, ucp_ep_h ep)
{
    ucp_ep_ext_gen_t *ep_ext = ucp_ep_ext_gen(ep);
    ucp_ep_match_entry_t *entry;
    khiter_t iter;

    if (!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX)) {
        return;
    }

    iter = kh_get(ucp_ep_match, &match_ctx->hash, ep_ext->ep_match.dest_uuid);
    ucs_assertv(iter != kh_end(&match_ctx->hash), "ep %p not found in hash", ep);
    entry = &kh_value(&match_ctx->hash, iter);

    if (ep->flags & UCP_EP_FLAG_DEST_EP) {
        ucs_trace("match_ctx %p: remove unexpected ep %p", match_ctx, ep);
        ucs_hlist_del(&entry->unexp_ep_q, &ep_ext->ep_match.list);
    } else {
        ucs_trace("match_ctx %p: remove expected ep %p", match_ctx, ep);
        ucs_hlist_del(&entry->exp_ep_q, &ep_ext->ep_match.list);
    }
    ep->flags &= ~UCP_EP_FLAG_ON_MATCH_CTX;
}
