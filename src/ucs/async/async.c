/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "async_int.h"

#include <ucs/arch/atomic.h>
#include <ucs/debug/debug.h>
#include <ucs/datastruct/hash.h>


#define UCS_ASYNC_TIMER_ID_MIN   10000000u
#define UCS_ASYNC_TIMER_ID_MAX   20000000u


/*
 * Hash table for all event and timer handlers.
 */
#define UCS_ASYNC_HANDLER_COMPARE(_h1, _h2)  ((_h1)->id - (_h2)->id)
#define UCS_ASYNC_HANDLER_HASH(_h)           ((_h)->id)
UCS_DEFINE_THREAD_SAFE_HASH(ucs_async_handler_t, next, 255,
                            UCS_ASYNC_HANDLER_COMPARE, UCS_ASYNC_HANDLER_HASH);

typedef struct ucs_async_global_context {
    ucs_hashed_ucs_async_handler_t handlers;
    volatile uint32_t              timer_id;
} ucs_async_global_context_t;


static ucs_async_global_context_t ucs_async_global_context = {
    .timer_id        = UCS_ASYNC_TIMER_ID_MIN
};


#define ucs_async_method_call(_mode, _func, ...) \
    ((_mode) == UCS_ASYNC_MODE_SIGNAL) ? ucs_async_signal_ops._func(__VA_ARGS__) : \
    ((_mode) == UCS_ASYNC_MODE_THREAD) ? ucs_async_thread_ops._func(__VA_ARGS__) : \
                                           ucs_async_poll_ops._func(__VA_ARGS__)

#define ucs_async_method_call_all(_func, ...) \
    { \
        ucs_async_signal_ops._func(__VA_ARGS__); \
        ucs_async_thread_ops._func(__VA_ARGS__); \
        ucs_async_poll_ops._func(__VA_ARGS__); \
    }


static ucs_status_t ucs_async_poll_init(ucs_async_context_t *async)
{
    async->poll_block = 0;
    return UCS_OK;
}

static ucs_async_ops_t ucs_async_poll_ops = {
    .init               = ucs_empty_function,
    .cleanup            = ucs_empty_function,
    .block              = ucs_empty_function,
    .unblock            = ucs_empty_function,
    .context_init       = ucs_async_poll_init,
    .context_try_block  = NULL,
    .context_unblock    = NULL,
    .add_event_fd       = ucs_empty_function_return_success,
    .remove_event_fd    = ucs_empty_function_return_success,
    .add_timer          = ucs_empty_function_return_success,
    .remove_timer       = ucs_empty_function_return_success,
};

static ucs_status_t ucs_async_dispatch_handler_cb(ucs_async_handler_t *handler,
                                                  void *arg)
{
    int from_async = (int)(uintptr_t)arg;
    ucs_async_context_t *async;
    ucs_async_mode_t mode;
    ucs_status_t status;

    mode  = handler->mode;
    async = handler->async;
    if (async == NULL) {
        ucs_trace_async("calling async handler %d", handler->id);
        handler->cb(handler->arg);
    } else if (!from_async) {
        ucs_trace_async("calling missed async handler %d", handler->id);
        ucs_async_method_call(mode, block);
        handler->missed = 0;
        handler->cb(handler->arg);
        ucs_async_method_call(mode, unblock);
    } else if (ucs_async_method_call(mode, context_try_block, async, from_async)) {
        ucs_trace_async("calling async handler %d", handler->id);
        handler->cb(handler->arg);
        ucs_async_method_call(mode, context_unblock, async);
    } else /* async != NULL */ {
        ucs_assert(from_async);
        ucs_trace_async("missed %d", handler->id);
        if (ucs_atomic_cswap32(&handler->missed, 0, 1) == 0) {
            status = ucs_mpmc_queue_push(&async->missed, handler->id);
            if (status != UCS_OK) {
                ucs_fatal("Failed to push to in async event queue: %s",
                          ucs_status_string(status));
            }
        }
    }
    return UCS_OK;
}

ucs_status_t ucs_async_dispatch_handler(int id, int from_async)
{
    ucs_async_handler_t search;
    ucs_status_t status;

    ucs_trace_func("id=%d from_async=%d", id, from_async);

    /*
     * Note:
     * This function is called while potentially holding a thread/signal lock,
     * and it grabs the handlers hash table lock.
     * Need to make sure that while holding hash table lock, no attempt is made
     * to modify thread/signal data structures.
     */

    search.id = id;
    status = ucs_hashed_ucs_async_handler_t_find(&ucs_async_global_context.handlers,
                                                 &search,
                                                 ucs_async_dispatch_handler_cb,
                                                 (void*)(uintptr_t)from_async);
    if ((status == UCS_ERR_NO_ELEM) && from_async) {
        ucs_trace_async("handler for %d not found - ignoring", id);
    } else if (status != UCS_OK){
        ucs_trace_async("failed to dispatch id %d: %s", id, ucs_status_string(status));
    }
    return status;
}

ucs_status_t ucs_async_context_init(ucs_async_context_t *async, ucs_async_mode_t mode)
{
    ucs_status_t status;

    ucs_trace_func("async=%p", async);

    status = ucs_mpmc_queue_init(&async->missed, ucs_global_opts.async_max_events);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_async_method_call(mode, context_init, async);
    if (status != UCS_OK) {
        goto err_free_miss_fds;
    }

    async->mode         = mode;
    async->num_handlers = 0;
    return UCS_OK;

err_free_miss_fds:
    ucs_mpmc_queue_cleanup(&async->missed);
err:
    return status;
}

void ucs_async_context_cleanup(ucs_async_context_t *async)
{
    ucs_trace_func("async=%p", async);

    if (async->num_handlers > 0) {
        ucs_warn("releasing async context with %d handlers", async->num_handlers);
    }
    ucs_mpmc_queue_cleanup(&async->missed);
}

static ucs_status_t ucs_async_add_handler(ucs_async_mode_t mode, int id,
                                          ucs_notifier_chain_func_t cb, void *arg,
                                          ucs_async_context_t *async)
{
    ucs_async_handler_t *handler;
    ucs_status_t status;

    /* If async context is given, it should have same mode */
    if ((async != NULL) && (async->mode != mode)) {
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* Limit amount of handlers per context */
    if (async != NULL) {
        if (ucs_atomic_fadd32(&async->num_handlers, +1) >= ucs_global_opts.async_max_events) {
            status = UCS_ERR_EXCEEDS_LIMIT;
            goto err_dec_num_handlers;
        }
    }

    handler = ucs_malloc(sizeof *handler, "async handler");
    if (handler == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    handler->id     = id;
    handler->mode   = mode;
    handler->cb     = cb;
    handler->arg    = arg;
    handler->async  = async;
    handler->missed = 0;
    ucs_async_method_call(mode, block);
    status = ucs_hashed_ucs_async_handler_t_add(&ucs_async_global_context.handlers,
                                                handler);
    ucs_async_method_call(mode, unblock);
    if (status != UCS_OK) {
        goto err_free;
    }

    return UCS_OK;

err_free:
    ucs_free(handler);
err_dec_num_handlers:
    if (async != NULL) {
        ucs_atomic_add32(&async->num_handlers, -1);
    }
err:
    return status;
}

static ucs_status_t ucs_async_remove_handler(int id,
                                             ucs_async_context_t **async_p,
                                             ucs_async_mode_t *mode_p)
{
    ucs_async_handler_t search, *handler;
    ucs_status_t status;

    search.id = id;
    ucs_async_method_call_all(block);
    status = ucs_hashed_ucs_async_handler_t_remove(&ucs_async_global_context.handlers,
                                                   &search, &handler);
    ucs_async_method_call_all(unblock);
    if (status != UCS_OK) {
        return status;
    }

    if (handler->async != NULL) {
        ucs_atomic_add32(&handler->async->num_handlers, -1);
    }

    if (async_p != NULL) {
        *async_p = handler->async;
    }
    if (mode_p != NULL) {
        *mode_p  = handler->mode;
    }
    ucs_free(handler);

    return UCS_OK;
}

ucs_status_t ucs_async_set_event_handler(ucs_async_mode_t mode, int event_fd,
                                         int events, ucs_notifier_chain_func_t cb,
                                         void *arg, ucs_async_context_t *async)
{
    ucs_status_t status;

    if (event_fd >= UCS_ASYNC_TIMER_ID_MIN) {
        /* File descriptor too large */
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err;
    }

    status = ucs_async_add_handler(mode, event_fd, cb, arg, async);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_async_method_call(mode, add_event_fd, async, event_fd, events);
    if (status != UCS_OK) {
        goto err_remove_handler;
    }

    ucs_debug("listening to async event fd %d events 0x%x mode %s", event_fd,
              events, ucs_async_mode_names[mode]);
    return UCS_OK;

err_remove_handler:
    ucs_async_remove_handler(event_fd, NULL, NULL);
err:
    return status;
}


ucs_status_t ucs_async_unset_event_handler(int event_fd)
{
    ucs_async_context_t *async;
    ucs_async_mode_t mode;
    ucs_status_t status;

    status = ucs_async_remove_handler(event_fd, &async, &mode);
    if (status != UCS_OK) {
        return status;
    }

    ucs_debug("removing async event fd %d", event_fd);
    return ucs_async_method_call(mode, remove_event_fd, async, event_fd);
}

ucs_status_t ucs_async_add_timer(ucs_async_mode_t mode, ucs_time_t interval,
                                 ucs_notifier_chain_func_t cb, void *arg,
                                 ucs_async_context_t *async, int *timer_id_p)
{
    ucs_status_t status;
    int timer_id;

    /* Search for unused timer ID */
    do {
        timer_id = ucs_atomic_fadd32(&ucs_async_global_context.timer_id, 1);
        if (timer_id >= UCS_ASYNC_TIMER_ID_MAX) {
            timer_id = UCS_ASYNC_TIMER_ID_MIN;
        }

        status = ucs_async_add_handler(mode, timer_id, cb, arg, async);
    } while (status == UCS_ERR_ALREADY_EXISTS);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_async_method_call(mode, add_timer, async, timer_id, interval);
    if (status != UCS_OK) {
        goto err_remove_handler;
    }

    *timer_id_p = timer_id;
    return UCS_OK;

err_remove_handler:
    ucs_async_remove_handler(timer_id, NULL, NULL);
err:
    return status;
}

ucs_status_t ucs_async_remove_timer(int timer_id)
{
    ucs_async_context_t *async;
    ucs_async_mode_t mode;
    ucs_status_t status;

    status = ucs_async_remove_handler(timer_id, &async, &mode);
    if (status != UCS_OK) {
        return status;
    }

    return ucs_async_method_call(mode, remove_timer, async, timer_id);
}

void __ucs_async_poll_missed(ucs_async_context_t *async)
{
    ucs_status_t status;
    uint32_t value;

    ucs_trace_async("miss handler");

    while (!ucs_mpmc_queue_is_empty(&async->missed)) {

        status = ucs_mpmc_queue_pull(&async->missed, &value);
        if (status == UCS_ERR_NO_PROGRESS) {
            /* TODO we should retry here if the code is change to check miss
             * only during ASYNC_UNBLOCK */
            break;
        }

        ucs_trace_async("handle missed event %d", value);
        ucs_async_method_call_all(block);
        ucs_async_dispatch_handler(value, 0);
        ucs_async_method_call_all(unblock);
    }
}

static ucs_status_t ucs_async_poll_iter_cb(ucs_async_handler_t *handler, void *arg)
{
    ucs_async_context_t *async = arg;

    if (((async == NULL) || (async == handler->async)) &&  /* Async context match */
        ((handler->async == NULL) || (handler->async->poll_block == 0))) /* Not blocked */
    {
        ucs_async_dispatch_handler_cb(handler, (void*)0);
    }
    return UCS_OK;
}

void ucs_async_poll(ucs_async_context_t *async)
{
    ucs_trace_poll("async=%p", async);

    ucs_hashed_ucs_async_handler_t_iter(&ucs_async_global_context.handlers,
                                        ucs_async_poll_iter_cb, async);
}

void ucs_async_global_init()
{
    ucs_hashed_ucs_async_handler_t_init(&ucs_async_global_context.handlers);
    ucs_async_method_call_all(init);
}

void ucs_async_global_cleanup()
{
    if (!ucs_hashed_ucs_async_handler_t_is_empty(&ucs_async_global_context.handlers)) {
        ucs_warn("async handler table is not empty during exit");
    }
    ucs_async_method_call_all(cleanup);
}
