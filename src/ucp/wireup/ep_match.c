/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucp/core/ucp_ep.h>
#include <ucp/core/ucp_ep.inl>
#include <ucp/wireup/ep_match.h>
#include <inttypes.h>


__KHASH_IMPL(ucp_ep_match, static UCS_F_MAYBE_UNUSED inline, uint64_t,
             ucp_ep_match_entry_t, 1, kh_int64_hash_func, kh_int64_hash_equal);


#define ucp_ep_match_list_for_each(_elem, _head, _member) \
    for (_elem = ucs_container_of((_head)->next, typeof(*_elem), _member); \
         &(_elem)->_member != NULL; \
         _elem = ucs_container_of((_elem)->_member.next, typeof(*_elem), _member))

static inline void ucp_ep_match_list_add_tail(ucs_list_link_t *head,
                                              ucs_list_link_t *elem)
{
    ucs_list_link_t *last;

    last       = head->prev;
    elem->next = NULL;
    head->prev = elem;

    if (last == NULL) {
        elem->prev = NULL;
        head->next = elem;
    } else {
        elem->prev = last;
        last->next = elem;
    }
}

static inline void ucp_ep_match_list_del(ucs_list_link_t *head,
                                         ucs_list_link_t *elem)
{
    (elem->prev ? elem->prev : head)->next = elem->next;
    (elem->next ? elem->next : head)->prev = elem->prev;
}

void ucp_ep_match_init(ucp_ep_match_ctx_t *match_ctx)
{
    kh_init_inplace(ucp_ep_match, &match_ctx->hash);
}

void ucp_ep_match_cleanup(ucp_ep_match_ctx_t *match_ctx)
{
    ucp_ep_match_entry_t entry;
    uint64_t dest_uuid;

    kh_foreach(&match_ctx->hash, dest_uuid, entry, {
        if (entry.exp_ep_q.next != NULL) {
            ucs_warn("match_ctx %p: uuid 0x%"PRIx64" expected queue is not empty",
                     match_ctx, dest_uuid);
        }
        if (entry.unexp_ep_q.next != NULL) {
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
        entry->exp_ep_q.next   = NULL;
        entry->exp_ep_q.prev   = NULL;
        entry->unexp_ep_q.next = NULL;
        entry->unexp_ep_q.prev = NULL;
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
                                       ucs_list_link_t *list, ucp_ep_h ep,
                                       uint64_t dest_uuid, const char *title)
{
    ucs_assert(!(ep->flags & UCP_EP_FLAG_ON_MATCH_CTX));
    ucp_ep_match_list_add_tail(list, &ucp_ep_ext_gen(ep)->ep_match.list);
    ep->flags                              |= UCP_EP_FLAG_ON_MATCH_CTX;
    ucp_ep_ext_gen(ep)->ep_match.dest_uuid  = dest_uuid;
    ucs_trace("match_ctx %p: ep %p added as %s uuid 0x%"PRIx64" conn_sn %d",
              match_ctx, ep, title, dest_uuid,
              ucp_ep_ext_proto(ep)->conn.conn_sn);
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
    ucs_list_link_t *list;
    ucp_ep_ext_gen_t *ep_ext;
    khiter_t iter;
    ucp_ep_h ep;

    iter = kh_get(ucp_ep_match, &match_ctx->hash, dest_uuid);
    if (iter == kh_end(&match_ctx->hash)) {
        goto notfound; /* no hash entry */
    }

    entry = &kh_value(&match_ctx->hash, iter);
    list  = is_exp ? &entry->exp_ep_q : &entry->unexp_ep_q;
    ucp_ep_match_list_for_each(ep_ext, list, ep_match.list) {
        ep = ucp_ep_from_ext_gen(ep_ext);
        if (ucp_ep_ext_proto(ep)->conn.conn_sn == conn_sn) {
            ucp_ep_match_list_del(list, &ep_ext->ep_match.list);
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
        ucp_ep_match_list_del(&entry->unexp_ep_q, &ep_ext->ep_match.list);
    } else {
        ucs_trace("match_ctx %p: remove expected ep %p", match_ctx, ep);
        ucp_ep_match_list_del(&entry->exp_ep_q, &ep_ext->ep_match.list);
    }
    ep->flags &= ~UCP_EP_FLAG_ON_MATCH_CTX;
}
