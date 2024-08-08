/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2016. ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/type/spinlock.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/datastruct/array.h>
#include <ucs/datastruct/hlist.h>
#include <ucs/datastruct/khash.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/debug_int.h>
#include <ucs/sys/sys.h>

#include "callbackq.h"


/* Reserve one slot for proxy callback */
#define UCS_CALLBACKQ_FAST_MAX (UCS_CALLBACKQ_FAST_COUNT - 1)

/*
 * Callback element in the spill array
 */
typedef struct ucs_callbackq_spill_elem {
    ucs_callbackq_elem_t super;
    int                  id;
} ucs_callbackq_spill_elem_t;

/*
 * One-shot element in the hash table
 */
typedef struct ucs_callbackq_oneshot_elem {
    ucs_callbackq_elem_t super;
    ucs_hlist_link_t     hlist;
} ucs_callbackq_oneshot_elem_t;

#define ucs_callbackq_oneshot_key_hash(_key) \
    kh_int64_hash_func((int64_t)(_key))

/* Hash map of progress callbacks */
KHASH_INIT(ucs_callbackq_oneshot_elems, ucs_callbackq_key_t, ucs_hlist_head_t,
           1, ucs_callbackq_oneshot_key_hash, kh_int64_hash_equal);

struct ucs_callbackq_priv {
    /* Protects adding / removing */
    ucs_recursive_spinlock_t                          lock;

    /* IDs of fast-path callbacks */
    int                                               fast_ids[UCS_CALLBACKQ_FAST_COUNT];

    /* Number of fast-path elements */
    unsigned                                          num_fast_elems;

    /* Mask of which fast-path elements should be removed */
    uint64_t                                          fast_remove_mask;

    /* Callback ID to index mapping */
    ucs_array_s(unsigned, unsigned)                   idxs;

    /* Index of first free item in the freelist */
    int                                               free_idx_id;

    /* Array of slow path elements */
    ucs_array_s(unsigned, ucs_callbackq_spill_elem_t) spill_elems;

    /* Hash map of oneshot path elements, by key */
    khash_t(ucs_callbackq_oneshot_elems)              oneshot_elems;

    /* ID of oneshot-path proxy in fast-path array */
    int                                               proxy_cb_id;
};


static void ucs_callbackq_proxy_enable(ucs_callbackq_t *cbq);
static void ucs_callbackq_proxy_disable(ucs_callbackq_t *cbq);

static void ucs_callbackq_enter(ucs_callbackq_t *cbq)
{
    ucs_recursive_spin_lock(&cbq->priv->lock);
    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, 1);
}

static void ucs_callbackq_leave(ucs_callbackq_t *cbq)
{
    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, -1);
    ucs_recursive_spin_unlock(&cbq->priv->lock);
}

static void ucs_callbackq_fast_elem_set(ucs_callbackq_t *cbq, unsigned idx,
                                        ucs_callback_t cb, void *arg, int id)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_elem_t *elem = &cbq->fast_elems[idx];

    elem->cb            = cb;
    elem->arg           = arg;
    priv->fast_ids[idx] = id;
}

static void ucs_callbackq_elem_reset(ucs_callbackq_t *cbq, unsigned idx)
{
    /* Set 'cbq' as the callback argument, in case we install the proxy callback
       in this location */
    ucs_callbackq_fast_elem_set(cbq, idx, NULL, cbq, UCS_CALLBACKQ_ID_NULL);
}

/*
 * @param [in]  id  ID to release in the lookup array.
 * @return index which this ID used to hold.
 */
unsigned ucs_callbackq_put_id(ucs_callbackq_t *cbq, int id)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned idx, *idx_elem;

    ucs_trace_func("cbq=%p id=%d", cbq, id);

    ucs_assert(id != UCS_CALLBACKQ_ID_NULL);

    idx_elem          = &ucs_array_elem(&priv->idxs, id);
    idx               = *idx_elem; /* Retrieve the index */
    *idx_elem         = priv->free_idx_id; /* Add ID to free-list head */
    priv->free_idx_id = id;                /* Update free-list head */

    return idx;
}

/**
 * @param [in]  idx  Index to save in the lookup array.
 * @return unique ID which holds index 'idx'.
 */
static int ucs_callbackq_get_id(ucs_callbackq_t *cbq, unsigned idx)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    int id;

    ucs_trace_func("cbq=%p idx=%u", cbq, idx);

    if (priv->free_idx_id == UCS_CALLBACKQ_ID_NULL) {
        id = ucs_array_length(&priv->idxs);
        ucs_array_append(
                &priv->idxs,
                ucs_fatal("callback queue %p: could not grow indexes array",
                          cbq));
    } else {
        /* Get free ID from the list and update free-list head */
        id                = priv->free_idx_id;
        priv->free_idx_id = ucs_array_elem(&priv->idxs, id);
    }

    /* Install provided idx to array */
    ucs_array_elem(&priv->idxs, id) = idx;
    return id;
}

static unsigned ucs_callbackq_get_fast_idx(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned idx;

    ucs_trace_func("cbq=%p num_fast_elems=%u", cbq, priv->num_fast_elems);

    idx = priv->num_fast_elems++;
    ucs_assertv(idx < UCS_CALLBACKQ_FAST_COUNT, "idx=%u", idx);

    return idx;
}

static int
ucs_callbackq_fast_elem_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg)
{
    unsigned idx;
    int id;

    ucs_trace_func("cbq=%p cb=%s arg=%p", cbq, ucs_debug_get_symbol_name(cb),
                   arg);

    idx = ucs_callbackq_get_fast_idx(cbq);
    id  = ucs_callbackq_get_id(cbq, idx);
    ucs_callbackq_fast_elem_set(cbq, idx, cb, arg, id);
    return id;
}

/* Should be called from dispatch thread only */
static void ucs_callbackq_fast_elems_purge(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_elem_t *replace_elem;
    unsigned idx, replace_idx;
    int replace_id;

    ucs_trace_func("cbq=%p fast_remove_mask=0x%" PRIx64, cbq,
                   priv->fast_remove_mask);
    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, 1);

    ucs_assertv(priv->num_fast_elems >= ucs_popcount(priv->fast_remove_mask),
                "num_fast_elems=%u fast_remove_mask=0x%" PRIx64,
                priv->num_fast_elems, priv->fast_remove_mask);

    while (priv->fast_remove_mask) {
        ucs_assert(priv->num_fast_elems > 0);

        idx                     = ucs_ilog2(priv->fast_remove_mask);
        replace_idx             = --priv->num_fast_elems;
        priv->fast_remove_mask &= ~UCS_BIT(idx);

        if (idx == replace_idx) {
            /* Removing last element */
            ucs_callbackq_elem_reset(cbq, replace_idx);
            continue;
        }

        /*
         * Since we iterate using ilog2 - from high bit to low - if we are not
         * removing the last element, it means the last element is not part of
         * the remove mask
         */
        ucs_assert(!(priv->fast_remove_mask & UCS_BIT(replace_idx)));

        replace_id = priv->fast_ids[replace_idx];
        ucs_assert(replace_id != UCS_CALLBACKQ_ID_NULL);
        ucs_trace_func("cbq=%p replace fast idx=%u by idx=%u id=%d", cbq, idx,
                       replace_idx, replace_id);

        /* Replace removed element by last element */
        replace_elem = &cbq->fast_elems[replace_idx];
        ucs_callbackq_fast_elem_set(cbq, idx, replace_elem->cb,
                                    replace_elem->arg, replace_id);
        ucs_array_elem(&priv->idxs, replace_id) = idx;
        ucs_callbackq_elem_reset(cbq, replace_idx);
    }

    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, -1);
}

static int
ucs_callbackq_spill_elem_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_spill_elem_t *elem;
    unsigned idx;
    int id;

    ucs_trace_func("cbq=%p cb=%s arg=%p", cbq, ucs_debug_get_symbol_name(cb),
                   arg);

    idx  = ucs_array_length(&priv->spill_elems);
    elem = ucs_array_append(
            &priv->spill_elems,
            ucs_fatal("callbackq %p: failed to allocate spill array", cbq));
    id   = ucs_callbackq_get_id(cbq, idx + UCS_CALLBACKQ_FAST_COUNT);

    elem->super.cb  = cb;
    elem->super.arg = arg;
    elem->id        = id;

    ucs_callbackq_proxy_enable(cbq);
    return id;
}

static void *ucs_callbackq_spill_elem_remove(ucs_callbackq_t *cbq, unsigned idx)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_spill_elem_t *elem;

    ucs_assertv(idx < ucs_array_length(&priv->spill_elems), "idx=%u length=%u",
                idx, ucs_array_length(&priv->spill_elems));
    elem     = &ucs_array_elem(&priv->spill_elems, idx);
    elem->id = UCS_CALLBACKQ_ID_NULL;

    return elem->super.arg;
}

/* Should be called from dispatch thread only */
static void ucs_callbackq_spill_elems_purge(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_spill_elem_t *src_elem;
    unsigned src_idx, dst_idx;

    ucs_trace_func("cbq=%p", cbq);

    /*
     * Copy valid elements from src_idx to dst_idx, essentially rebuilding the
     * array of elements in-place, keeping only the valid ones.
     * As an optimization, if no elements are actually removed, then src_idx will
     * always be equal to dst_idx, so nothing will be actually copied/moved.
     */
    dst_idx = 0;
    for (src_idx = 0; src_idx < ucs_array_length(&priv->spill_elems);
         ++src_idx) {
        src_elem = &ucs_array_elem(&priv->spill_elems, src_idx);
        if (src_elem->id != UCS_CALLBACKQ_ID_NULL) {
            ucs_assert(dst_idx <= src_idx);
            if (dst_idx != src_idx) {
                ucs_array_elem(&priv->idxs, src_elem->id) =
                        dst_idx + UCS_CALLBACKQ_FAST_COUNT;
                ucs_array_elem(&priv->spill_elems, dst_idx) = *src_elem;
            }
            ++dst_idx;
        }
    }

    ucs_array_set_length(&priv->spill_elems, dst_idx);
}

static void ucs_callbackq_oneshot_elems_free(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_oneshot_elem_t *oneshot_elem;
    ucs_hlist_head_t hlist;

    kh_foreach_value(&priv->oneshot_elems, hlist, {
        ucs_hlist_for_each_extract(oneshot_elem, &hlist, hlist) {
            ucs_free(oneshot_elem);
        }
    });
    kh_clear(ucs_callbackq_oneshot_elems, &priv->oneshot_elems);
}

static int ucs_callback_is_proxy_needed(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;

    return !ucs_array_is_empty(&priv->spill_elems) ||
           (kh_size(&priv->oneshot_elems) > 0) || priv->fast_remove_mask;
}

/* Promote a spill element to fast-path array */
static void ucs_callbackq_spill_elem_promote(ucs_callbackq_t *cbq, unsigned idx)
{
    ucs_callbackq_priv_t *priv       = cbq->priv;
    ucs_callbackq_spill_elem_t *elem = &ucs_array_elem(&priv->spill_elems, idx);
    unsigned fast_idx;

    ucs_trace_func("cbq=%p idx=%u elem->id=%d", cbq, idx, elem->id);

    fast_idx = ucs_callbackq_get_fast_idx(cbq);
    ucs_callbackq_fast_elem_set(cbq, fast_idx, elem->super.cb, elem->super.arg,
                                elem->id);
    ucs_array_elem(&priv->idxs, elem->id) = fast_idx;
    ucs_callbackq_spill_elem_remove(cbq, idx);
}

/* Lock must be held */
static unsigned ucs_callbackq_spill_elems_dispatch(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned num_spill_elems   = ucs_array_length(&priv->spill_elems);
    unsigned count             = 0;
    ucs_callbackq_spill_elem_t *spill_elem;
    ucs_callbackq_elem_t tmp_elem;
    unsigned spill_elem_idx;

    /* Execute and update spill callbacks by num_oneshot_elems copy to avoid
     * infinite loop if callback adds another one */
    for (spill_elem_idx = 0; spill_elem_idx < num_spill_elems;
         ++spill_elem_idx) {
        spill_elem = &ucs_array_elem(&priv->spill_elems, spill_elem_idx);
        if (spill_elem->id == UCS_CALLBACKQ_ID_NULL) {
            continue;
        }

        tmp_elem = spill_elem->super;
        if (priv->num_fast_elems < UCS_CALLBACKQ_FAST_MAX) {
            ucs_callbackq_spill_elem_promote(cbq, spill_elem_idx);
        }

        ucs_callbackq_leave(cbq);

        /* Execute callback without lock */
        count += tmp_elem.cb(tmp_elem.arg);

        ucs_callbackq_enter(cbq);
    }

    return count;
}

/* Lock must be held */
static unsigned ucs_callbackq_oneshot_elems_dispatch(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned count             = 0;
    ucs_callbackq_oneshot_elem_t *oneshot_elem;
    unsigned key_idx, num_keys;
    struct {
        void   *key;
        size_t count; /* How many callbacks should be called, at most */
    } * keys;
    ucs_hlist_head_t *hlist;
    khiter_t khiter;
    void *key;

    num_keys = kh_size(&priv->oneshot_elems);
    if (num_keys == 0) {
        return 0;
    }

    keys = ucs_malloc(sizeof(*keys) * num_keys, "ucs_callbackq_keys");
    if (keys == NULL) {
        ucs_fatal("callbackq %p: failed to allocate oneshot key array", cbq);
    }

    key_idx = 0;
    kh_foreach_key(&priv->oneshot_elems, key, {
        khiter = kh_get(ucs_callbackq_oneshot_elems, &priv->oneshot_elems, key);
        hlist  = &kh_value(&priv->oneshot_elems, khiter);
        keys[key_idx].key   = key;
        keys[key_idx].count = ucs_hlist_length(hlist);
        ++key_idx;
    })
    ucs_assertv(key_idx == num_keys, "key_idx=%u num_keys=%u", key_idx,
                num_keys);

    key_idx = 0;
    while (key_idx < num_keys) {
        khiter = kh_get(ucs_callbackq_oneshot_elems, &priv->oneshot_elems,
                        keys[key_idx].key);
        if (khiter == kh_end(&priv->oneshot_elems)) {
            ++key_idx; /* Not found, move to next key */
            continue;
        }

        hlist = &kh_value(&priv->oneshot_elems, khiter);
        if (ucs_hlist_is_empty(hlist)) {
            kh_del(ucs_callbackq_oneshot_elems, &priv->oneshot_elems, khiter);
            ++key_idx; /* Empty list, remove and move to next key */
            continue;
        }

        if (keys[key_idx].count == 0) {
            /* Should not call any more callbacks from this key. This avoids
               an infinite loop in case a callback adds more callbacks. */
            ++key_idx;
            continue;
        }

        /* Extract a single oneshot element from the list and dispatch it.
         * Dispatching the elements one-by-one allows callbacks to remove other
         * callbacks from the list.
         */
        --keys[key_idx].count;
        oneshot_elem = ucs_hlist_extract_head_elem(hlist,
                                                   ucs_callbackq_oneshot_elem_t,
                                                   hlist);
        ucs_callbackq_leave(cbq);

        count += oneshot_elem->super.cb(oneshot_elem->super.arg);
        ucs_free(oneshot_elem);

        ucs_callbackq_enter(cbq);
    }

    ucs_free(keys);
    return count;
}

static unsigned ucs_callbackq_proxy_callback(void *arg)
{
    ucs_callbackq_t *cbq = arg;
    unsigned count;

    ucs_trace_poll("cbq=%p proxy_callback", cbq);

    ucs_callbackq_enter(cbq);

    count = ucs_callbackq_spill_elems_dispatch(cbq) +
            ucs_callbackq_oneshot_elems_dispatch(cbq);

    /* Remove remaining callbacks */
    ucs_callbackq_fast_elems_purge(cbq);
    ucs_callbackq_spill_elems_purge(cbq);

    /* Disable this proxy if no more work to do */
    if (!ucs_callback_is_proxy_needed(cbq)) {
        ucs_callbackq_proxy_disable(cbq);
    }

    ucs_callbackq_leave(cbq);

    return count;
}

static void ucs_callbackq_proxy_enable(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;

    ucs_trace_func("cbq=%p", cbq);

    ucs_assertv(ucs_callback_is_proxy_needed(cbq), "cbq=%p", cbq);

    if (priv->proxy_cb_id != UCS_CALLBACKQ_ID_NULL) {
        /* Already enabled */
        return;
    }

    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, 1);
    priv->proxy_cb_id =
            ucs_callbackq_fast_elem_add(cbq, ucs_callbackq_proxy_callback, cbq);
    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, -1);
}

static void ucs_callbackq_proxy_disable(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned idx;

    ucs_trace_func("cbq=%p proxy_cb_id=%d", cbq, priv->proxy_cb_id);

    if (priv->proxy_cb_id == UCS_CALLBACKQ_ID_NULL) {
        /* Already disabled */
        return;
    }

    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, 1);

    idx                     = ucs_callbackq_put_id(cbq, priv->proxy_cb_id);
    priv->proxy_cb_id       = UCS_CALLBACKQ_ID_NULL;
    priv->fast_remove_mask |= UCS_BIT(idx);
    ucs_callbackq_fast_elems_purge(cbq);

    ucs_log_indent_level(UCS_LOG_LEVEL_TRACE_FUNC, -1);
}

static void
ucs_callbackq_elem_show(const char *title, const ucs_callbackq_elem_t *elem)
{
    ucs_diag("%s: cb %s (%p) arg %p", title,
             ucs_debug_get_symbol_name(elem->cb), elem->cb, elem->arg);
}

static void ucs_callbackq_show_remaining_elems(ucs_callbackq_t *cbq)
{
    const char *diag_log_str   = ", increase log level to diag for details";
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_oneshot_elem_t *oneshot_elem;
    ucs_callbackq_spill_elem_t *spill_elem;
    ucs_callbackq_elem_t *fast_elem;
    ucs_hlist_head_t hlist;
    unsigned total_elems;

    total_elems = priv->num_fast_elems + ucs_array_length(&priv->spill_elems);
    kh_foreach_value(&priv->oneshot_elems, hlist, {
        ucs_hlist_for_each(oneshot_elem, &hlist, hlist) {
            ++total_elems;
        }
    })
    if (total_elems == 0) {
        return;
    }

    ucs_warn("callbackq %p: %d callback%s not removed%s", cbq, total_elems,
             (total_elems > 1) ? "s were" : " was",
             ucs_log_is_enabled(UCS_LOG_LEVEL_DIAG) ? "" : diag_log_str);
    if (!ucs_log_is_enabled(UCS_LOG_LEVEL_DIAG)) {
        return;
    }

    ucs_log_indent(1);

    ucs_carray_for_each(fast_elem, cbq->fast_elems, priv->num_fast_elems) {
        ucs_callbackq_elem_show("fast-path", fast_elem);
    }
    ucs_array_for_each(spill_elem, &priv->spill_elems) {
        ucs_callbackq_elem_show("spill", &spill_elem->super);
    }
    kh_foreach_value(&priv->oneshot_elems, hlist, {
        ucs_hlist_for_each(oneshot_elem, &hlist, hlist) {
            ucs_callbackq_elem_show("one-shot", &oneshot_elem->super);
        }
    })

    ucs_log_indent(-1);
}

ucs_status_t ucs_callbackq_init(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv;
    unsigned idx;

    priv = ucs_malloc(sizeof(*priv), "ucs_callbackq_priv");
    if (priv == NULL) {
        ucs_error("failed to allocate ucs_callbackq_priv");
        return UCS_ERR_NO_MEMORY;
    }

    ucs_recursive_spinlock_init(&priv->lock, 0);
    ucs_array_init_dynamic(&priv->idxs);
    ucs_array_init_dynamic(&priv->spill_elems);
    kh_init_inplace(ucs_callbackq_oneshot_elems, &priv->oneshot_elems);
    priv->num_fast_elems   = 0;
    priv->fast_remove_mask = 0;
    priv->free_idx_id      = UCS_CALLBACKQ_ID_NULL;
    priv->proxy_cb_id      = UCS_CALLBACKQ_ID_NULL;
    cbq->priv              = priv;

    for (idx = 0; idx < UCS_CALLBACKQ_FAST_COUNT; ++idx) {
        ucs_callbackq_elem_reset(cbq, idx);
    }
    cbq->fast_elems[UCS_CALLBACKQ_FAST_COUNT].cb = NULL;

    return UCS_OK;
}

void ucs_callbackq_cleanup(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = cbq->priv;

    ucs_callbackq_fast_elems_purge(cbq);
    ucs_callbackq_spill_elems_purge(cbq);
    ucs_callbackq_proxy_disable(cbq);
    ucs_callbackq_show_remaining_elems(cbq);

    ucs_callbackq_oneshot_elems_free(cbq);
    kh_destroy_inplace(ucs_callbackq_oneshot_elems, &priv->oneshot_elems);
    ucs_array_cleanup_dynamic(&priv->spill_elems);
    ucs_array_cleanup_dynamic(&priv->idxs);

    ucs_free(priv);
}

int ucs_callbackq_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    int id;

    ucs_trace_func("cbq=%p cb=%s arg=%p", cbq, ucs_debug_get_symbol_name(cb),
                   arg);

    ucs_callbackq_enter(cbq);

    if (ucs_likely(priv->num_fast_elems < UCS_CALLBACKQ_FAST_MAX)) {
        id = ucs_callbackq_fast_elem_add(cbq, cb, arg);
    } else {
        id = ucs_callbackq_spill_elem_add(cbq, cb, arg);
    }

    ucs_callbackq_leave(cbq);
    return id;
}

void *ucs_callbackq_remove(ucs_callbackq_t *cbq, int id)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned idx;
    void *cb_arg;

    ucs_trace_func("cbq=%p id=%d", cbq, id);

    ucs_callbackq_enter(cbq);

    idx = ucs_callbackq_put_id(cbq, id);
    if (idx < UCS_CALLBACKQ_FAST_COUNT) {
        ucs_assertv(idx < priv->num_fast_elems, "idx=%u num_fast_elems=%u", idx,
                    priv->num_fast_elems);
        cb_arg                  = cbq->fast_elems[idx].arg;
        priv->fast_remove_mask |= UCS_BIT(idx);
        ucs_callbackq_fast_elems_purge(cbq);
    } else {
        cb_arg = ucs_callbackq_spill_elem_remove(
                cbq, idx - UCS_CALLBACKQ_FAST_COUNT);
    }

    ucs_callbackq_leave(cbq);

    return cb_arg;
}

int ucs_callbackq_add_safe(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg)
{
    int id;

    ucs_trace_func("cbq=%p cb=%s arg=%p", cbq, ucs_debug_get_symbol_name(cb),
                   arg);

    ucs_callbackq_enter(cbq);

    /* Add callback to spill elems, and it may be upgraded to fast-path later by
     * the proxy callback. It's not safe to add to fast_elems directly.
     */
    id = ucs_callbackq_spill_elem_add(cbq, cb, arg);

    ucs_callbackq_leave(cbq);
    return id;
}

void *ucs_callbackq_remove_safe(ucs_callbackq_t *cbq, int id)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    unsigned idx;
    void *cb_arg;

    ucs_trace_func("cbq=%p id=%d", cbq, id);

    ucs_callbackq_enter(cbq);

    idx = ucs_callbackq_put_id(cbq, id);
    if (idx < UCS_CALLBACKQ_FAST_COUNT) {
        UCS_STATIC_ASSERT(UCS_CALLBACKQ_FAST_MAX <= 64);
        ucs_assertv(idx < priv->num_fast_elems, "idx=%u num_fast_elems=%u", idx,
                    priv->num_fast_elems);
        /* Make sure user callback will not be called in case we try to dispatch
           the removed fast-path element before the proxy callback had a chance
           to clean it up. */
        cb_arg                  = cbq->fast_elems[idx].arg;
        cbq->fast_elems[idx].cb = (ucs_callback_t)ucs_empty_function_return_zero;
        priv->fast_remove_mask |= UCS_BIT(idx);
        ucs_callbackq_proxy_enable(cbq);
    } else {
        cb_arg = ucs_callbackq_spill_elem_remove(
                cbq, idx - UCS_CALLBACKQ_FAST_COUNT);
    }

    ucs_callbackq_leave(cbq);

    return cb_arg;
}

void ucs_callbackq_add_oneshot(ucs_callbackq_t *cbq, ucs_callbackq_key_t key,
                               ucs_callback_t cb, void *arg)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_oneshot_elem_t *elem;
    ucs_hlist_head_t *hlist;
    khiter_t khiter;
    int khret;

    ucs_trace_func("cbq=%p key=%p cb=%s arg=%p", cbq, key,
                   ucs_debug_get_symbol_name(cb), arg);

    ucs_callbackq_enter(cbq);

    elem = ucs_malloc(sizeof(*elem), "ucs_callbackq_oneshot_elem");
    if (elem == NULL) {
        ucs_fatal("callbackq %p: failed to allocate oneshot element", cbq);
    }

    elem->super.cb  = cb;
    elem->super.arg = arg;

    khiter = kh_put(ucs_callbackq_oneshot_elems, &priv->oneshot_elems, key,
                    &khret);
    if ((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
        (khret == UCS_KH_PUT_BUCKET_CLEAR)) {
        hlist = &kh_value(&priv->oneshot_elems, khiter);
        ucs_hlist_head_init(hlist);
    } else if (khret == UCS_KH_PUT_KEY_PRESENT) {
        hlist = &kh_value(&priv->oneshot_elems, khiter);
    } else {
        ucs_fatal("callbackq %p: failed to insert oneshot element (khret=%d)",
                  cbq, khret);
    }

    ucs_hlist_add_tail(hlist, &elem->hlist);
    ucs_callbackq_proxy_enable(cbq);

    ucs_callbackq_leave(cbq);
}

void ucs_callbackq_remove_oneshot(ucs_callbackq_t *cbq, ucs_callbackq_key_t key,
                                  ucs_callbackq_predicate_t pred, void *arg)
{
    ucs_callbackq_priv_t *priv = cbq->priv;
    ucs_callbackq_oneshot_elem_t *elem, *telem;
    ucs_hlist_head_t *hlist, thead;
    khiter_t khiter;

    ucs_trace_func("cbq=%p key=%p pred=%s arg=%p", cbq, key,
                   ucs_debug_get_symbol_name(pred), arg);

    ucs_callbackq_enter(cbq);

    khiter = kh_get(ucs_callbackq_oneshot_elems, &priv->oneshot_elems, key);
    if (khiter == kh_end(&priv->oneshot_elems)) {
        goto out;
    }

    hlist = &kh_value(&priv->oneshot_elems, khiter);
    ucs_hlist_for_each_safe(elem, telem, hlist, &thead, hlist) {
        if (pred(&elem->super, arg)) {
            ucs_hlist_del(hlist, &elem->hlist);
            ucs_free(elem);
        }
    }

    if (ucs_hlist_is_empty(hlist)) {
        kh_del(ucs_callbackq_oneshot_elems, &priv->oneshot_elems, khiter);
    }

out:
    ucs_callbackq_leave(cbq);
}
