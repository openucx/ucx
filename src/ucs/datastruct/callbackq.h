/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CALLBACKQ_H
#define UCS_CALLBACKQ_H

#include <ucs/datastruct/list_types.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/type/status.h>
#include <stddef.h>
#include <stdint.h>

BEGIN_C_DECLS

/*
 * Thread-safe callback queue:
 *  - only one thread can dispatch
 *  - any thread can add and remove
 *  - add/remove operations are O(1)
 */

#define UCS_CALLBACKQ_FAST_COUNT   7     /* Max. number of fast-path callbacks */
#define UCS_CALLBACKQ_ID_NULL      (-1)  /* Invalid callback identifier */


/*
 * Forward declarations
 */
typedef struct ucs_callbackq       ucs_callbackq_t;
typedef struct ucs_callbackq_elem  ucs_callbackq_elem_t;


/**
 * Callback which can be placed in a queue.
 *
 * @param [in] arg  User-defined argument for the callback.
 *
 * @return Count of how much "work" was done by the callback. For example, zero
 *         means that no work was done, and any nonzero value means that something
 *         was done.
 */
typedef unsigned (*ucs_callback_t)(void *arg);


/**
 * Callback queue element predicate.
 *
 * @param [in] elem  Callback queue element to check.
 * @param [in] arg   User-defined argument.
 *
 * @return Predicate result value - nonzero means "true", zero means "false".
 */
typedef int (*ucs_callbackq_predicate_t)(const ucs_callbackq_elem_t *elem,
                                         void *arg);


/**
 * Callback flags
 */
enum ucs_callbackq_flags {
    UCS_CALLBACKQ_FLAG_FAST        = UCS_BIT(0), /**< Fast-path (best effort) */
    UCS_CALLBACKQ_FLAG_ONESHOT     = UCS_BIT(1)  /**< Call the callback only once
                                                      (cannot be used with FAST) */
};


/**
 * Callback queue element.
 */
struct ucs_callbackq_elem {
    ucs_callback_t                 cb;       /**< Callback function */
    void                           *arg;     /**< Function argument */
    unsigned                       flags;    /**< Callback flags */
    int                            id;       /**< Callback id */
};


/**
 * A queue of callback to execute
 */
struct ucs_callbackq {
    /**
     * Array of fast-path element, the last is reserved as a sentinel to mark
     * array end.
     */
    ucs_callbackq_elem_t           fast_elems[UCS_CALLBACKQ_FAST_COUNT + 1];

    /**
     * Private data, which we don't want to expose in API to avoid pulling
     * more header files
     */
    char                           priv[72];
};


/**
 * Initialize the callback queue.
 *
 * @param  [in] cbq      Callback queue to initialize.
 */
ucs_status_t ucs_callbackq_init(ucs_callbackq_t *cbq);


/**
 * Clean up the callback queue and release associated memory.
 *
 * @param  [in] cbq      Callback queue to clean up.
 */
void ucs_callbackq_cleanup(ucs_callbackq_t *cbq);


/**
 * Add a callback to the queue.
 * This is *not* safe to call while another thread might be dispatching callbacks.
 * However, it can be used from the dispatch context (e.g a callback may use this
 * function to add another callback).
 *
 * @param  [in] cbq      Callback queue to add the callback to.
 * @param  [in] cb       Callback to add.
 * @param  [in] arg      User-defined argument for the callback.
 * @param  [in] flags    Flags for the callback, from  @ref ucs_callbackq_flags.
 *
 * @return Unique identifier of the callback in the queue.
 */
int ucs_callbackq_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg,
                      unsigned flags);


/**
 * Remove a callback from the queue immediately.
 * This is *not* safe to call while another thread might be dispatching callbacks.
 * However, it can be used from the dispatch context (e.g a callback may use this
 * function to remove itself or another callback). In this case, the callback may
 * still be dispatched once after this function returned.
 *
 * @param  [in] cbq      Callback queue to remove the callback from.
 * @param  [in] id       Callback identifier to remove.
 */
void ucs_callbackq_remove(ucs_callbackq_t *cbq, int id);


/**
 * Add a callback to the queue.
 * This can be used from any context and any thread, including but not limited to:
 * - A callback can add another callback.
 * - A thread can add a callback while another thread is dispatching callbacks.
 *
 * @param  [in] cbq      Callback queue to add the callback to.
 * @param  [in] cb       Callback to add.
 * @param  [in] arg      User-defined argument for the callback.
 * @param  [in] flags    Flags for the callback, from  @ref ucs_callbackq_flags.
 *
 * @return Unique identifier of the callback in the queue.
 */
int ucs_callbackq_add_safe(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg,
                           unsigned flags);


/**
 * Remove a callback from the queue in a safe but lazy fashion. The callback will
 * be removed at some point in the near future.
 * This can be used from any context and any thread, including but not limited to:
 * - A callback can remove another callback or itself.
 * - A thread can't remove a callback while another thread is dispatching callbacks.
 *
 * @param  [in] cbq      Callback queue to remove the callback from.
 * @param  [in] id       Callback identifier to remove.
 */
void ucs_callbackq_remove_safe(ucs_callbackq_t *cbq, int id);


/**
 * Remove all callbacks from the queue for which the given predicate returns
 * "true" (nonzero) value.
 * This is *not* safe to call while another thread might be dispatching callbacks.
 * However, it can be used from the dispatch context (e.g a callback may use this
 * function to remove itself or another callback). In this case, the callback may
 * still be dispatched once after this function returned.
 *
 * @param  [in] cbq       Callback queue.
 * @param  [in] pred      Predicate to check candidates for removal.
 * @param  [in] arg       User-defined argument for the predicate.
 */
void ucs_callbackq_remove_if(ucs_callbackq_t *cbq, ucs_callbackq_predicate_t pred,
                             void *arg);


/**
 * Dispatch callbacks from the callback queue.
 * Must be called from single thread only.
 *
 * @param  [in] cbq      Callback queue to dispatch callbacks from.

 * @return Sum of all return values from the dispatched callbacks.
 */
static inline unsigned ucs_callbackq_dispatch(ucs_callbackq_t *cbq)
{
    ucs_callbackq_elem_t *elem;
    ucs_callback_t cb;
    unsigned count;

    count = 0;
    for (elem = cbq->fast_elems; (cb = elem->cb) != NULL; ++elem) {
        count += cb(elem->arg);
    }
    return count;
}

END_C_DECLS

#endif
