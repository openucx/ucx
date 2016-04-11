/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CALLBACKQ_H
#define UCS_CALLBACKQ_H

#include <ucs/arch/cpu.h> /* for memory load fence */
#include <ucs/async/async_fwd.h>
#include <stdint.h>

/*
 *
 * Thread-safe callback queue:
 *  - only one thread can dispatch
 *  - any thread can add and remove
 *  - callbacks are reference-counted
 */


/*
 * Forward declarations
 */
typedef struct ucs_callbackq         ucs_callbackq_t;
typedef struct ucs_callbackq_elem    ucs_callbackq_elem_t;
typedef void                         (*ucs_callback_t)(void *arg);


/**
 * Callback queue element.
 */
struct ucs_callbackq_elem {
    ucs_callback_t                   cb;       /**< Callback function */
    void                             *arg;     /**< Function argument */
    volatile uint32_t                refcount; /**< Reference count */
};


/**
 * A queue of callback to execute
 */
struct ucs_callbackq {
    ucs_callbackq_elem_t             *start;   /**< Iteration start pointer */
    ucs_callbackq_elem_t             *end;     /**< Iteration end pointer */
    ucs_callbackq_elem_t             *ptr;     /**< Array of elements */
    size_t                           size;     /**< Array size */
    char                             priv[24]; /**< Private data, which we don't want
                                                    to expose in API to avoid
                                                    pulling more header files */
};


/**
 * Iterate over all elements in the callback queue.
 * This should be done only from one thread at a time.
 */
#define ucs_callbackq_for_each(_elem, _cbq) \
    for (_elem = (_cbq)->start, \
             ({ ucs_memory_cpu_load_fence(); 1; }); \
         _elem < (_cbq)->end; \
         ++_elem)


/**
 * Initialize the callback queue.
 *
 * @param  [in] cbq      Callback queue to initialize.
 * @param  [in] size     Callback queue size.
 * @param  [in] async    If != NULL, make calling add/remove from this async
 *                       context deadlock-free.
 *
 * @note The callback queue currently does not expand beyond the size defined
 *       during initialization time. More callbacks *cannot* be added.
 */
ucs_status_t ucs_callbackq_init(ucs_callbackq_t *cbq, size_t size,
                                ucs_async_context_t *async);


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
 * function to add reference to itself or add another callback).
 *
 * If the pair (cb, arg) already exists, it is not added, but its reference count
 * is incremented.
 *
 * Complexity: O(n)
 *
 * @param  [in] cbq      Callback queue to add the callback to.
 * @param  [in] cb       Callback to add.
 * @param  [in] arg      User-defined argument for the callback.
 */
void ucs_callbackq_add(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg);


/**
 * Remove a callback from the queue immediately.
 * This is *not* safe to call while another thread might be dispatching callbacks.
 * However, it can be used from the dispatch context (e.g a callback may use this
 * function to remove itself or another callback).
 *
 * Complexity: O(n)
 *
 * If the pair (cb, arg) has a reference count > 1, the reference count is
 * decremented by 1, and the callback is not removed.
 *
 * @param  [in] cbq      Callback queue to remove the callback from.
 * @param  [in] cb       Callback to remove.
 * @param  [in] arg      User-defined argument for the callback.
 *
 * @return UCS_ERR_NO_ELEM if element does not exist.
 */
ucs_status_t ucs_callbackq_remove(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                  void *arg);


/**
 * Remove a callback from the queue immediately and completely.
 * This is *not* safe to call while another thread might be dispatching callbacks.
 * Same as @ref ucs_callbackq_remove_sync, except it removes the callback even
 * if its reference count is > 1, and guarantees the next time the callback
 * queue is dispatched (on this thread), this callback will not be called, assuming
 * it is called after @ref ucs_callbackq_add_async.
 *
 * Complexity: O(n)
 *
 * @param  [in] cbq      Callback queue to remove the callback from.
 * @param  [in] cb       Callback to remove.
 * @param  [in] arg      User-defined argument for the callback.
 */
void ucs_callbackq_remove_all(ucs_callbackq_t *cbq, ucs_callback_t cb, void *arg);


/**
 * Add a callback to the queue.
 * This can be used from any context and any thread, including but not limited to:
 * - A callback can add another callback.
 * - A thread can add a callback while another thread is dispatching callbacks.
 *
 * If the pair (cb, arg) already exists, it is not added, but its reference count
 * is incremented.
 *
 * Complexity: O(n)
 *
 * @param  [in] cbq      Callback queue to add the callback to.
 * @param  [in] cb       Callback to add.
 * @param  [in] arg      User-defined argument for the callback.
 */
ucs_status_t ucs_callbackq_add_safe(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                    void *arg);


/**
 * Remove a callback from the queue in a lazy fashion. The callback will be
 * removed at some point in the near future.
 * This can be used from any context and any thread, including but not limited to:
 * - A callback can remove another callback or itself.
 * - A thread remove add a callback while another thread is dispatching callbacks.
 *
 * If the pair (cb, arg) has a reference count > 1, the reference count is
 * decremented by 1, and the callback is not removed.
 *
 * Complexity: O(n)
 *
 * @param  [in] cbq      Callback queue to remove the callback from.
 * @param  [in] cb       Callback to remove.
 * @param  [in] arg      User-defined argument for the callback.
 */
ucs_status_t ucs_callbackq_remove_safe(ucs_callbackq_t *cbq, ucs_callback_t cb,
                                       void *arg);


/**
 * Call all callbacks on the queue.
 * This should be done only from one thread at a time.
 *
 * Complexity: O(n)
 *
 * @param  [in] cbq      Callback queue whose elements to dispatch.
 */
static inline void ucs_callbackq_dispatch(ucs_callbackq_t *cbq)
{
    ucs_callbackq_elem_t *elem;

    ucs_callbackq_for_each(elem, cbq) {
        elem->cb(elem->arg);
    }
}

#endif
