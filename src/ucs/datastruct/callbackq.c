/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/type/spinlock.h>
#include <ucs/arch/atomic.h>
#include <ucs/arch/bitops.h>
#include <ucs/async/async.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/debug.h>
#include <ucs/sys/sys.h>

#include "callbackq.h"


#define UCS_CALLBACKQ_IDX_FLAG_SLOW  0x80000000u
#define UCS_CALLBACKQ_IDX_MASK       0x7fffffffu
#define UCS_CALLBACKQ_FAST_MAX       (UCS_CALLBACKQ_FAST_COUNT - 1)


typedef struct ucs_callbackq_priv {
    ucs_spinlock_t         lock;           /**< Protects adding / removing */

    ucs_callbackq_elem_t   *slow_elems;    /**< Array of slow-path elements */
    unsigned               num_slow_elems; /**< Number of slow-path elements */
    unsigned               max_slow_elems; /**< Maximal number of slow-path elements */
    unsigned               slow_idx;       /**< Iterator over slow-path elements */
    int                    slow_proxy_id;  /**< ID of slow-path proxy in fast-path array.
                                                keep track while this moves around. */

    uint64_t               fast_remove_mask; /**< Mask of which fast-path elements
                                                  should be removed */
    unsigned               num_fast_elems; /**< Number of fast-path elements */

    /* Lookup table for callback IDs. This allows moving callbacks around in
     * the arrays, while the user can always use a single ID to remove the
     * callback in O(1).
     */
    int                    free_idx_id;    /**< Index of first free item in the list */
    int                    num_idxs;       /**< Size of idxs array */
    unsigned               *idxs;          /**< ID-to-index lookup */

} ucs_callbackq_priv_t;


static unsigned ucs_callbackq_slow_proxy(void *arg);

static inline ucs_callbackq_priv_t* ucs_callbackq_priv(ucs_callbackq_t *cbq)
{
    UCS_STATIC_ASSERT(sizeof(cbq->priv) == sizeof(ucs_callbackq_priv_t));
    return (void*)cbq->priv;
}

static void ucs_callbackq_enter(ucs_callbackq_t *cbq)
{
    ucs_spin_lock(&ucs_callbackq_priv(cbq)->lock);
}

static void ucs_callbackq_leave(ucs_callbackq_t *cbq)
{
    ucs_spin_unlock(&ucs_callbackq_priv(cbq)->lock);
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

static void ucs_callbackq_remove_common(ucs_callbackq_t *cbq,
                                        ucs_callbackq_elem_t *elems,
                                        unsigned idx, unsigned last_idx,
                                        unsigned idx_slow_flag,
                                        uint64_t *remove_mask)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    int id;

    ucs_trace_func("cbq=%p idx=%u last_idx=%u slow_flag=0x%x", cbq, idx,
                   last_idx, idx_slow_flag);

    ucs_assert(idx <= last_idx);

    /* replace removed with last */
    elems[idx] = elems[last_idx];
    ucs_callbackq_elem_reset(cbq, &elems[last_idx]);

    if (*remove_mask & UCS_BIT(last_idx)) {
        /* replaced by marked-for-removal element, still need to remove 'idx' */
        *remove_mask &= ~UCS_BIT(last_idx);
    } else {
        /* replaced by a live element, remove from the mask and update 'idxs' */
        *remove_mask &= ~UCS_BIT(idx);
        if (last_idx != idx) {
            id = elems[idx].id;
            ucs_assert(id != UCS_CALLBACKQ_ID_NULL);
            priv->idxs[id] = idx | idx_slow_flag;
        }
    }
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
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned last_idx;

    ucs_trace_func("cbq=%p idx=%u", cbq, idx);

    ucs_assert(priv->num_fast_elems > 0);
    last_idx = --priv->num_fast_elems;
    ucs_callbackq_remove_common(cbq, cbq->fast_elems, idx, last_idx, 0,
                                &priv->fast_remove_mask);
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

    ucs_assert((priv->num_slow_elems > 0) || priv->fast_remove_mask);

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
    unsigned last_idx;
    uint64_t dummy = 0;

    ucs_trace_func("cbq=%p idx=%u", cbq, idx);

    /* When the slow-path proxy callback sees there are no more elements, it
     * will disable itself.
     */
    ucs_assert(priv->num_slow_elems > 0);
    last_idx = --priv->num_slow_elems;
    ucs_callbackq_remove_common(cbq, priv->slow_elems, idx, last_idx,
                                UCS_CALLBACKQ_IDX_FLAG_SLOW, &dummy);

    /* Make the slow-path iterator go over the element we moved from the end of
     * the array, otherwise it would be skipped. */
    if (idx <= priv->slow_idx) {
        priv->slow_idx = idx;
    }
}

static unsigned ucs_callbackq_slow_proxy(void *arg)
{
    ucs_callbackq_t      *cbq  = arg;
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    ucs_callbackq_elem_t *elem;
    unsigned UCS_V_UNUSED removed_idx;
    unsigned slow_idx, fast_idx;
    ucs_callbackq_elem_t tmp_elem;
    unsigned count = 0;

    ucs_trace_poll("cbq=%p", cbq);

    ucs_callbackq_enter(cbq);

    /* Execute and update slow-path callbacks */
    while ( (slow_idx = priv->slow_idx) < priv->num_slow_elems ) {
        elem = &priv->slow_elems[slow_idx];
        priv->slow_idx++; /* Increment slow_idx here to give the remove functions
                             an opportunity to rewind it */

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

    priv->slow_idx = 0;

    ucs_callbackq_purge_fast(cbq);

    /* Disable this proxy if no more work to do */
    if (!priv->fast_remove_mask && (priv->num_slow_elems == 0)) {
        ucs_callbackq_disable_proxy(cbq);
    }

    ucs_callbackq_leave(cbq);

    return count;
}

ucs_status_t ucs_callbackq_init(ucs_callbackq_t *cbq)
{
    ucs_callbackq_priv_t *priv = ucs_callbackq_priv(cbq);
    unsigned idx;

    for (idx = 0; idx < UCS_CALLBACKQ_FAST_COUNT + 1; ++idx) {
        ucs_callbackq_elem_reset(cbq, &cbq->fast_elems[idx]);
    }

    ucs_spinlock_init(&priv->lock);
    priv->slow_elems        = NULL;
    priv->num_slow_elems    = 0;
    priv->max_slow_elems    = 0;
    priv->slow_idx          = 0;
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

    ucs_callbackq_disable_proxy(cbq);

    if ((priv->num_fast_elems) > 0 || (priv->num_slow_elems > 0)) {
        ucs_warn("%d fast-path and %d slow-path callbacks remain in the queue",
                 priv->num_fast_elems, priv->num_slow_elems);
    }

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

    /* remote fast-path elements  */
    elem = cbq->fast_elems;
    while (elem->cb != NULL) {
        if (pred(elem, arg)) {
            idx = ucs_callbackq_put_id_noflag(cbq, elem->id);
            ucs_assert(idx == (elem - cbq->fast_elems));
            ucs_callbackq_remove_fast(cbq, idx);
        } else {
            ++elem;
       }
    }

    /* remote slow-path elements */
    elem = priv->slow_elems;
    while (elem < priv->slow_elems + priv->num_slow_elems) {
        if (pred(elem, arg)) {
            idx = ucs_callbackq_put_id_noflag(cbq, elem->id);
            ucs_assert(idx == (elem - priv->slow_elems));
            ucs_callbackq_remove_slow(cbq, idx);
        } else {
            ++elem;
       }
    }

    ucs_callbackq_leave(cbq);
}
