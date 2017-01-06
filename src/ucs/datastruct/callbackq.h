/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2016.  ALL RIGHTS RESERVED.
* Copyright (C) ARM Ltd. 2016-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CALLBACKQ_H
#define UCS_CALLBACKQ_H

#include <ucs/async/async_fwd.h>
#include <ucs/datastruct/list.h>
#include <stdint.h>

/*
 *
 * Thread-safe callback queue:
 *  - only one thread can dispatch
 *  - any thread can add and remove
 *  - callbacks are reference-counted
 *  - every queue may contain one slow path element,
 *    which maintains its own list of callbacks
 */


/*
 * Forward declarations
 */
typedef struct ucs_callbackq            ucs_callbackq_t;
typedef struct ucs_callbackq_elem       ucs_callbackq_elem_t;
typedef struct ucs_callbackq_slow_elem  ucs_callbackq_slow_elem_t;
typedef void                            (*ucs_callback_t)(void *arg);
typedef void                            (*ucs_callback_slow_t)(ucs_callbackq_slow_elem_t *self);


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
    ucs_callbackq_elem_t             *start;    /**< Iteration start pointer */
    ucs_callbackq_elem_t             *end;      /**< Iteration end pointer
                                                     (end of last element) */
    ucs_callbackq_elem_t             *ptr;      /**< Array of elements */
    size_t                           size;      /**< Array size */
    ucs_list_link_t                  slow_path; /**< List of slow path callbacks */
    char                             priv[24];  /**< Private data, which we don't want
                                                     to expose in API to avoid
                                                     pulling more header files */
};


/**
 * Every ucs_callbackq instance may contain one slow path element. This
 * element represents a list of callbacks which is not limited in length.
 * Non performance-critical short tasks may be added to the slow path list
 * by means of the corresponding API.
 *
 * -----------
 * | | | |X| |
 * -----------
 *        |
 *     slow path list -> sp_elem1 -> sp_elem2 -> etc
 */
struct ucs_callbackq_slow_elem {
    ucs_callback_slow_t    cb;
    ucs_list_link_t        list;
};


/**
 * Initialize the callback queue.
 *
 * @param  [in] cbq      Callback queue to initialize.
 * @param  [in] size     Callback queue size.
 * @param  [in] async    If != NULL, make calling add/remove from this async
 *                       context deadlock-free.
 *
 * @note In general, calling add/remove from an async context, or a signal
 * handler, may cause a deadlock. However, if async != NULL is passed to this
 * function, it would be safe to use add/remove from this async context *only*.
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
 * Add a callback to the slow path list.
 * This can be used from any context and any thread, including but not limited to:
 * - A callback can add another callback.
 * - A thread can add a callback while another thread is dispatching callbacks.
 *
 * Important note: It is not allowed to add the same element more than once.
 *
 * Complexity: O(n) (n is number of regular callbackq elements)
 *
 * @param  [in] cbq      Callback queue to add the callback to.
 * @param  [in] elem     Slow path list element. The user is expected to
 *                       initialize the "cb" field.
 */
void ucs_callbackq_add_slow_path(ucs_callbackq_t *cbq,
                                 ucs_callbackq_slow_elem_t* elem);


/**
 * Remove a callback from the slow path list.
 * This is *not* safe to call while another thread might be dispatching callbacks.
 * However, it can be used from the dispatch context (e.g a callback may use this
 * function to remove itself).
 *
 * Important note: It is not allowed to pass the element which was not added
 * previously.
 *
 * Complexity: O(n) (n is number of regular callbackq elements)
 *
 * @param  [in] cbq      Callback queue to remove the callback from.
 * @param  [in] elem     Slow path list element, which was previously added by
 *                       ucs_callbackq_add_slow_path call.
 *
 */
void ucs_callbackq_remove_slow_path(ucs_callbackq_t *cbq,
                                    ucs_callbackq_slow_elem_t* elem);


/**
 * Remove all slow path elements with a given callback function from the list.
 *
 * @param [in]  cbq      Callback queue to remove the callbacks from.
 * @param [in]  cb       Callback function to search for.
 * @param [out] list     If != NULL, head of a list to which the removed
 *                       elements are added.
 *
 */
void ucs_callbackq_purge_slow_path(ucs_callbackq_t *cbq, ucs_callback_slow_t cb,
                                   ucs_list_link_t *list);

#endif
