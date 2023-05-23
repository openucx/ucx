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
#include <ucs/datastruct/hlist.h>
#include <ucs/datastruct/khash.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/debug_int.h>
#include <ucs/sys/sys.h>

#include "callbackq.h"


#define UCS_CALLBACKQ_IDX_FLAG_SLOW  0x80000000u
#define UCS_CALLBACKQ_IDX_MASK       0x7fffffffu
#define UCS_CALLBACKQ_FAST_MAX       (UCS_CALLBACKQ_FAST_COUNT - 1)

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

typedef struct ucs_callbackq_priv {
    ucs_recursive_spinlock_t lock;           /**< Protects adding / removing */

    ucs_callbackq_elem_t     *slow_elems;    /**< Array of slow-path elements */
    unsigned                 num_slow_elems; /**< Number of slow-path elements */
    unsigned                 max_slow_elems; /**< Maximal number of slow-path elements */
    int                      slow_proxy_id;  /**< ID of slow-path proxy in fast-path array.
                                                  keep track while this moves around. */

    uint64_t                 fast_remove_mask; /**< Mask of which fast-path elements
                                                    should be removed */
    unsigned                 num_fast_elems; /**< Number of fast-path elements */

    /** Hash map of oneshot path elements, by key */
    khash_t(ucs_callbackq_oneshot_elems) oneshot_elems;

    /* Lookup table for callback IDs. This allows moving callbacks around in
     * the arrays, while the user can always use a single ID to remove the
     * callback in O(1).
     */
    int                      free_idx_id;    /**< Index of first free item in the list */
    int                      num_idxs;       /**< Size of idxs array */
    unsigned                 *idxs;          /**< ID-to-index lookup */

} ucs_callbackq_priv_t;


static unsigned ucs_callbackq_slow_proxy(void *arg);

static inline ucs_callbackq_priv_t* ucs_callbackq_priv(ucs_callbackq_t *cbq)
{
    UCS_STATIC_ASSERT(sizeof(cbq->priv) == sizeof(ucs_callbackq_priv_t));
    return (void*)cbq->priv;
}

static void ucs_callbackq_enter(ucs_callbackq_t *cbq)
{
    ucs_recursive_spin_lock(&ucs_callbackq_priv(cbq)->lock);
}

static void ucs_callbackq_leave(ucs_callbackq_t *cbq)
{
    ucs_recursive_spin_unlock(&ucs_callbackq_priv(cbq)->lock);
}

static void ucs_callbackq_elem_reset(ucs_callbackq_t *cbq,
                                     ucs_callbackq_elem_t *elem)
{
    elem->cb    = NULL;
    elem->arg   = cbq;
    elem->id    = UCS_CALLBACKQ_ID_NULL;
    elem->flags = 0;
}

static void *ucs_callbackq_array_grow(ucs_callbackq_t *cbq, void *ptr,
                                      size_t elem_size, int count,
                                      int *new_count, const char *alloc_name)
{
    void *new_ptr;

    if (count == 0) {
        *new_count = ucs_get_page_size() / elem_size;
    } else {
        *new_count = count * 2;
    }

    new_ptr = ucs_sys_realloc(ptr, elem_size * count, elem_size * *new_count);
    if (new_ptr == NULL) {
        ucs_fatal("cbq %p: could not allocate memory for %s", cbq, alloc_name);
    }
    return new_ptr;
}

static void ucs_callbackq_array_free(void *ptr, size_t elem_size, int count)
{
    ucs_sys_free(ptr, elem_size * count);
}

/*
 * @param [in]  id  ID to release in the lookup array.
 * @return index which this ID used to hold.
 */
int ucs_callbackq_put_id(ucs_callbackq_t *cbq, int id)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx_with_flag;

    ucs_trace_func("cbq=%p id=%d", cbq, id);

    ucs_assert(id != UCS_CALLBACKQ_ID_NULL);

    idx_with_flag     = priv->idxs[id];    /* Retrieve the index */
    priv->idxs[id]    = priv->free_idx_id; /* Add ID to free-list head */
    priv->free_idx_id = id;                /* Update free-list head */

    return idx_with_flag;
}

int ucs_callbackq_put_id_noflag(ucs_callbackq_t *cbq, int id)
{
    return ucs_callbackq_put_id(cbq, id) & UCS_CALLBACKQ_IDX_MASK;
}

/**
 * @param [in]  idx  Index to save in the lookup array.
 * @return unique ID which holds index 'idx'.
 */
int ucs_callbackq_get_id(ucs_callbackq_t *cbq, unsigned idx)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    int new_num_idxs;
    int id;

    ucs_trace_func("cbq=%p idx=%u", cbq, idx);

    if (priv->free_idx_id == UCS_CALLBACKQ_ID_NULL) {
        priv->idxs = ucs_callbackq_array_grow(cbq, priv->idxs, sizeof(*priv->idxs),
                                              priv->num_idxs, &new_num_idxs,
                                              "indexes");

        /* Add new items to free-list */
        for (id = priv->num_idxs; id < new_num_idxs; ++id) {
            priv->idxs[id]    = priv->free_idx_id;
            priv->free_idx_id = id;
        }

        priv->num_idxs = new_num_idxs;
    }

    id = priv->free_idx_id;             /* Get free ID from the list */
    ucs_assert(id != UCS_CALLBACKQ_ID_NULL);
    priv->free_idx_id = priv->idxs[id]; /* Update free-list head */
    priv->idxs[id]    = idx;            /* Install provided idx to array */
    return id;
}

static unsigned ucs_callbackq_get_fast_idx(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx;

    idx = priv->num_fast_elems++;
    ucs_assert(idx < UCS_CALLBACKQ_FAST_COUNT);
    return idx;
}

static int ucs_callbackq_add_fast(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                  void *arg, unsigned flags)
{
    unsigned idx;
    int id;

    ucs_trace_func("cbq=%p cb=%s arg=%p flags=%u", cbq,
                   ucs_debug_get_symbol_name(cb), arg, flags);

    ucs_assert(!(flags & UCS_CALLBACKQ_FLAG_ONESHOT));

    idx = ucs_callbackq_get_fast_idx(cbq);
    id  = ucs_callbackq_get_id(cbq, idx);
    cbq->fast_elems[idx].cb    = cb;
    cbq->fast_elems[idx].arg   = arg;
    cbq->fast_elems[idx].flags = flags;
    cbq->fast_elems[idx].id    = id;
    return id;
}

/* should be called from dispatch thread only */
static void ucs_callbackq_remove_fast(ucs_callbackq_t *cbq, unsigned idx)
{
    ucs_callbackq_priv_t *priv     = ucs_callbackq_priv(cbq);
    ucs_callbackq_elem_t *dst_elem = &cbq->fast_elems[idx];
    unsigned last_idx;
    int id;

    ucs_trace_func("cbq=%p idx=%u", cbq, idx);

    ucs_assert(priv->num_fast_elems > 0);
    last_idx = --priv->num_fast_elems;

    /* replace removed with last */
    *dst_elem = cbq->fast_elems[last_idx];
    ucs_callbackq_elem_reset(cbq, &cbq->fast_elems[last_idx]);

    if (priv->fast_remove_mask & UCS_BIT(last_idx)) {
        /* replaced by marked-for-removal element, still need to remove 'idx' */
        ucs_assert(priv->fast_remove_mask & UCS_BIT(idx));
        priv->fast_remove_mask &= ~UCS_BIT(last_idx);
    } else {
        /* replaced by a live element, remove from the mask and update 'idxs' */
        priv->fast_remove_mask &= ~UCS_BIT(idx);
        if (last_idx != idx) {
            id = dst_elem->id;
            ucs_assert(id != UCS_CALLBACKQ_ID_NULL);
            priv->idxs[id] = idx;
        }
    }
}

/* should be called from dispatch thread only */
static void ucs_callbackq_purge_fast(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx;

    ucs_trace_func("cbq=%p map=0x%"PRIx64, cbq, priv->fast_remove_mask);

    /* Remove fast-path callbacks marked for removal */
    while (priv->fast_remove_mask) {
        idx = ucs_ffs64(priv->fast_remove_mask);
        ucs_callbackq_remove_fast(cbq, idx);
    }
}

static void ucs_callbackq_enable_proxy(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx;
    int id;

    ucs_trace_func("cbq=%p", cbq);

    if (priv->slow_proxy_id != UCS_CALLBACKQ_ID_NULL) {
        return;
    }

    ucs_assertv((priv->num_slow_elems > 0) || priv->fast_remove_mask ||
                (kh_size(&priv->oneshot_elems) > 0),
                "cbq=%p num_slow_elems=%u fast_remove_mask=0x%" PRIx64
                " oneshot_elems=%u",
                cbq, priv->num_slow_elems, priv->fast_remove_mask,
                kh_size(&priv->oneshot_elems));

    idx = ucs_callbackq_get_fast_idx(cbq);
    id  = ucs_callbackq_get_id(cbq, idx);

    ucs_assert(cbq->fast_elems[idx].arg == cbq);
    cbq->fast_elems[idx].cb    = ucs_callbackq_slow_proxy;
    cbq->fast_elems[idx].flags = 0;
    cbq->fast_elems[idx].id    = id;
    /* Avoid writing 'arg' because the dispatching thread may not see it in case
     * of weak memory ordering. Instead, 'arg' is reset to 'cbq' for all free and
     * removed elements, from the main thread.
     */

    priv->slow_proxy_id        = id;
}

static void ucs_callbackq_disable_proxy(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx;

    ucs_trace_func("cbq=%p slow_proxy_id=%d", cbq, priv->slow_proxy_id);

    ucs_assert(priv->fast_remove_mask == 0);

    if (priv->slow_proxy_id == UCS_CALLBACKQ_ID_NULL) {
        return;
    }

    idx = ucs_callbackq_put_id(cbq, priv->slow_proxy_id);
    ucs_callbackq_remove_fast(cbq, idx);
    priv->slow_proxy_id = UCS_CALLBACKQ_ID_NULL;
}

static int ucs_callbackq_add_slow(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                  void *arg, unsigned flags)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    ucs_callbackq_elem_t *new_slow_elems;
    int new_max_slow_elems;
    unsigned idx;
    int id;

    ucs_trace_func("cbq=%p cb=%s arg=%p flags=%u", cbq,
                   ucs_debug_get_symbol_name(cb), arg, flags);

    /* Grow slow-path array if needed */
    if (priv->num_slow_elems >= priv->max_slow_elems) {
        new_slow_elems = ucs_callbackq_array_grow(cbq, priv->slow_elems,
                                                  sizeof(*priv->slow_elems),
                                                  priv->max_slow_elems,
                                                  &new_max_slow_elems,
                                                  "slow_elems");
        for (idx = priv->max_slow_elems; idx < new_max_slow_elems; ++idx) {
            ucs_callbackq_elem_reset(cbq, &new_slow_elems[idx]);
        }

        priv->max_slow_elems = new_max_slow_elems;
        priv->slow_elems     = new_slow_elems;
    }

    /* Add slow-path element to the queue */
    idx = priv->num_slow_elems++;
    id  = ucs_callbackq_get_id(cbq, idx | UCS_CALLBACKQ_IDX_FLAG_SLOW);
    priv->slow_elems[idx].cb    = cb;
    priv->slow_elems[idx].arg   = arg;
    priv->slow_elems[idx].flags = flags;
    priv->slow_elems[idx].id    = id;

    ucs_callbackq_enable_proxy(cbq);
    return id;
}

static void ucs_callbackq_remove_slow(ucs_callbackq_t *cbq, unsigned idx)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);

    ucs_trace_func("cbq=%p idx=%u", cbq, idx);

    /* Mark for removal by ucs_callbackq_purge_slow() */
    ucs_callbackq_elem_reset(cbq, &priv->slow_elems[idx]);
}

static void ucs_callbackq_purge_slow(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    ucs_callbackq_elem_t *src_elem;
    unsigned src_idx, dst_idx;

    ucs_trace_func("cbq=%p", cbq);

    /*
     * Copy valid elements from src_idx to dst_idx, essentially rebuilding the
     * array of elements in-place, keeping only the valid ones.
     * As an optimization, if no elements are actually removed, then src_idx will
     * always be equal to dst_idx, so nothing will be actually copied/moved.
     */
    dst_idx = 0;
    for (src_idx = 0; src_idx < priv->num_slow_elems; ++src_idx) {
        src_elem = &priv->slow_elems[src_idx];
        if (src_elem->id != UCS_CALLBACKQ_ID_NULL) {
            ucs_assert(dst_idx <= src_idx);
            if (dst_idx != src_idx) {
                priv->idxs[src_elem->id]  = dst_idx | UCS_CALLBACKQ_IDX_FLAG_SLOW;
                priv->slow_elems[dst_idx] = *src_elem;
            }
            ++dst_idx;
        }
    }

    priv->num_slow_elems = dst_idx;
}

/* Lock must be held */
static unsigned ucs_callbackq_oneshot_elems_dispatch(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
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

static unsigned ucs_callbackq_slow_proxy(void *arg)
{
    ucs_callbackq_t      *cbq  = arg;
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned num_slow_elems    = priv->num_slow_elems;
    unsigned count             = 0;
    ucs_callbackq_elem_t *elem;
    unsigned UCS_V_UNUSED removed_idx;
    unsigned slow_idx, fast_idx;
    ucs_callbackq_elem_t tmp_elem;

    ucs_trace_poll("cbq=%p", cbq);

    ucs_callbackq_enter(cbq);

    /* Execute and update slow-path callbacks by num_slow_elems copy to avoid
     * infinite loop if callback adds another one */
    for (slow_idx = 0; slow_idx < num_slow_elems; ++slow_idx) {
        elem = &priv->slow_elems[slow_idx];
        if (elem->id == UCS_CALLBACKQ_ID_NULL) {
            continue;
        }

        tmp_elem = *elem;
        if (elem->flags & UCS_CALLBACKQ_FLAG_FAST) {
            ucs_assert(!(elem->flags & UCS_CALLBACKQ_FLAG_ONESHOT));
            if (priv->num_fast_elems < UCS_CALLBACKQ_FAST_MAX) {
                fast_idx = ucs_callbackq_get_fast_idx(cbq);
                cbq->fast_elems[fast_idx] = *elem;
                priv->idxs[elem->id]      = fast_idx;
                ucs_callbackq_remove_slow(cbq, slow_idx);
            }
        } else if (elem->flags & UCS_CALLBACKQ_FLAG_ONESHOT) {
            removed_idx = ucs_callbackq_put_id_noflag(cbq, elem->id);
            ucs_assert(removed_idx == slow_idx);
            ucs_callbackq_remove_slow(cbq, slow_idx);
        }

        ucs_callbackq_leave(cbq);

        count += tmp_elem.cb(tmp_elem.arg); /* Execute callback without lock */

        ucs_callbackq_enter(cbq);
    }

    count += ucs_callbackq_oneshot_elems_dispatch(cbq);

    ucs_callbackq_purge_fast(cbq);
    ucs_callbackq_purge_slow(cbq);

    /* Disable this proxy if no more work to do */
    if (!priv->fast_remove_mask && (priv->num_slow_elems == 0) &&
        (kh_size(&priv->oneshot_elems) == 0)) {
        ucs_callbackq_disable_proxy(cbq);
    }

    ucs_callbackq_leave(cbq);

    return count;
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
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    ucs_callbackq_oneshot_elem_t *oneshot_elem;
    ucs_callbackq_elem_t *elem;
    ucs_hlist_head_t hlist;
    unsigned total_elems;

    total_elems = priv->num_fast_elems + priv->num_slow_elems;
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

    ucs_carray_for_each(elem, cbq->fast_elems, priv->num_fast_elems) {
        ucs_callbackq_elem_show("fast-path", elem);
    }
    ucs_carray_for_each(elem, priv->slow_elems, priv->num_slow_elems) {
        ucs_callbackq_elem_show("slow", elem);
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
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx;

    for (idx = 0; idx < UCS_CALLBACKQ_FAST_COUNT + 1; ++idx) {
        ucs_callbackq_elem_reset(cbq, &cbq->fast_elems[idx]);
    }

    ucs_recursive_spinlock_init(&priv->lock, 0);
    kh_init_inplace(ucs_callbackq_oneshot_elems, &priv->oneshot_elems);
    priv->slow_elems        = NULL;
    priv->num_slow_elems    = 0;
    priv->max_slow_elems    = 0;
    priv->slow_proxy_id     = UCS_CALLBACKQ_ID_NULL;
    priv->fast_remove_mask  = 0;
    priv->num_fast_elems    = 0;
    priv->free_idx_id       = UCS_CALLBACKQ_ID_NULL;
    priv->num_idxs          = 0;
    priv->idxs              = NULL;
    return UCS_OK;
}

void ucs_callbackq_cleanup(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);

    ucs_callbackq_purge_fast(cbq);
    ucs_callbackq_disable_proxy(cbq);
    ucs_callbackq_purge_slow(cbq);
    ucs_callbackq_show_remaining_elems(cbq);

    kh_destroy_inplace(ucs_callbackq_oneshot_elems, &priv->oneshot_elems);
    ucs_callbackq_array_free(priv->slow_elems, sizeof(*priv->slow_elems),
                             priv->max_slow_elems);
    ucs_callbackq_array_free(priv->idxs, sizeof(*priv->idxs), priv->num_idxs);
}

int ucs_callbackq_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg,
                      unsigned flags)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    int id;

    ucs_callbackq_enter(cbq);

    ucs_trace_func("cbq=%p cb=%s arg=%p flags=%u", cbq,
                   ucs_debug_get_symbol_name(cb), arg, flags);

    if ((flags & UCS_CALLBACKQ_FLAG_FAST) &&
        (priv->num_fast_elems < UCS_CALLBACKQ_FAST_MAX))
    {
        id = ucs_callbackq_add_fast(cbq, cb, arg, flags);
    } else {
        id = ucs_callbackq_add_slow(cbq, cb, arg, flags);
    }

    ucs_callbackq_leave(cbq);
    return id;
}

void ucs_callbackq_remove(ucs_callbackq_t *cbq, int id)
{
    unsigned idx_with_flag, idx;

    ucs_callbackq_enter(cbq);

    ucs_trace_func("cbq=%p id=%d", cbq, id);

    ucs_callbackq_purge_fast(cbq);

    idx_with_flag = ucs_callbackq_put_id(cbq, id);
    idx           = idx_with_flag & UCS_CALLBACKQ_IDX_MASK;

    if (idx_with_flag & UCS_CALLBACKQ_IDX_FLAG_SLOW) {
        ucs_callbackq_remove_slow(cbq, idx);
    } else {
        ucs_callbackq_remove_fast(cbq, idx);
    }

    ucs_callbackq_leave(cbq);
}

int ucs_callbackq_add_safe(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg,
                           unsigned flags)
{
    int id;

    ucs_callbackq_enter(cbq);

    ucs_trace_func("cbq=%p cb=%s arg=%p flags=%u", cbq,
                   ucs_debug_get_symbol_name(cb), arg, flags);

    /* Add callback to slow-path, and it may be upgraded to fast-path later by
     * the proxy callback. It's not safe to add fast-path callback directly
     * from this context.
     */
    id = ucs_callbackq_add_slow(cbq, cb, arg, flags);

    ucs_callbackq_leave(cbq);
    return id;
}

void ucs_callbackq_remove_safe(ucs_callbackq_t *cbq, int id)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx_with_flag, idx;

    ucs_callbackq_enter(cbq);

    ucs_trace_func("cbq=%p id=%d", cbq, id);

    idx_with_flag = ucs_callbackq_put_id(cbq, id);
    idx           = idx_with_flag & UCS_CALLBACKQ_IDX_MASK;

    if (idx_with_flag & UCS_CALLBACKQ_IDX_FLAG_SLOW) {
        ucs_callbackq_remove_slow(cbq, idx);
    } else {
        UCS_STATIC_ASSERT(UCS_CALLBACKQ_FAST_MAX <= 64);
        ucs_assert(idx < priv->num_fast_elems);
        priv->fast_remove_mask |= UCS_BIT(idx);
        cbq->fast_elems[idx].id = UCS_CALLBACKQ_ID_NULL; /* for assertion */
        ucs_callbackq_enable_proxy(cbq);
    }

    ucs_callbackq_leave(cbq);
}

void ucs_callbackq_remove_if(ucs_callbackq_t *cbq, ucs_callbackq_predicate_t pred,
                             void *arg)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    ucs_callbackq_elem_t *elem;
    unsigned idx;

    ucs_callbackq_enter(cbq);

    ucs_trace_func("cbq=%p", cbq);

    ucs_callbackq_purge_fast(cbq);

    /* Mark matched fast-path elements to be removed  */
    for (elem = cbq->fast_elems; elem->cb != NULL; ++elem) {
        if (pred(elem, arg)) {
            ucs_callbackq_remove_safe(cbq, elem->id);
        }
    }

    /* Purge fast-path elements marked for removal.
     * Elements are collected and then removed to suppress Coverity warning
     * about using the element's argument after freeing it, Coverity wrongly
     * assumes that reusing the same element for the next element could be
     * harmful */
    ucs_callbackq_purge_fast(cbq);

    /* Remove slow-path elements */
    for (elem = priv->slow_elems;
         elem < (priv->slow_elems + priv->num_slow_elems); ++elem) {
        if (pred(elem, arg)) {
            idx = ucs_callbackq_put_id_noflag(cbq, elem->id);
            ucs_assert(idx == (elem - priv->slow_elems));
            ucs_callbackq_remove_slow(cbq, idx);
        }
    }

    ucs_callbackq_leave(cbq);
}

void ucs_callbackq_add_oneshot(ucs_callbackq_t *cbq, ucs_callbackq_key_t key,
                               ucs_callback_t cb, void *arg)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
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
    ucs_callbackq_enable_proxy(cbq);

    ucs_callbackq_leave(cbq);
}

void ucs_callbackq_remove_oneshot(ucs_callbackq_t *cbq, ucs_callbackq_key_t key,
                                  ucs_callbackq_predicate_t pred, void *arg)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
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
