/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2011.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "async_int.h"

#include <ucs/arch/atomic.h>
#include <ucs/debug/debug_int.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/stubs.h>


#define UCS_ASYNC_TIMER_ID_MIN          1000000u
#define UCS_ASYNC_TIMER_ID_MAX          2000000u

#define UCS_ASYNC_HANDLER_FMT           "%p [id=%d ref %d] %s()"
#define UCS_ASYNC_HANDLER_ARG(_h)       (_h), (_h)->id, (_h)->refcount, \
                                        ucs_debug_get_symbol_name((_h)->cb)

#define UCS_ASYNC_MISSED_QUEUE_SHIFT    32
#define UCS_ASYNC_MISSED_QUEUE_MASK     UCS_MASK(UCS_ASYNC_MISSED_QUEUE_SHIFT)

/* Hash table for all event and timer handlers */
KHASH_MAP_INIT_INT(ucs_async_handler, ucs_async_handler_t *);


typedef struct ucs_async_global_context {
    khash_t(ucs_async_handler)     handlers;
    pthread_rwlock_t               handlers_lock;
    volatile uint32_t              handler_id;
} ucs_async_global_context_t;


static ucs_async_global_context_t ucs_async_global_context = {
    .handlers_lock   = PTHREAD_RWLOCK_INITIALIZER,
    .handler_id      = UCS_ASYNC_TIMER_ID_MIN
};


#define ucs_async_method_call(_mode, _func, ...) \
    ((_mode) == UCS_ASYNC_MODE_SIGNAL)          ? ucs_async_signal_ops._func(__VA_ARGS__) : \
    ((_mode) == UCS_ASYNC_MODE_THREAD_SPINLOCK) ? ucs_async_thread_spinlock_ops._func(__VA_ARGS__) : \
    ((_mode) == UCS_ASYNC_MODE_THREAD_MUTEX)    ? ucs_async_thread_mutex_ops._func(__VA_ARGS__) : \
                                                  ucs_async_poll_ops._func(__VA_ARGS__)

#define ucs_async_method_call_all(_func, ...) \
    { \
        ucs_async_signal_ops._func(__VA_ARGS__); \
        ucs_async_thread_spinlock_ops._func(__VA_ARGS__); \
        ucs_async_thread_mutex_ops._func(__VA_ARGS__); \
        ucs_async_poll_ops._func(__VA_ARGS__); \
    }


static ucs_status_t ucs_async_poll_init(ucs_async_context_t *async)
{
    async->poll_block = 0;
    return UCS_OK;
}

static int ucs_async_poll_tryblock(ucs_async_context_t *async)
{
    return 1;
}

static ucs_async_ops_t ucs_async_poll_ops = {
    .init               = ucs_empty_function,
    .cleanup            = ucs_empty_function,
    .is_from_async      =
            (ucs_async_is_from_async_t)ucs_empty_function_return_zero,
    .block              = ucs_empty_function,
    .unblock            = ucs_empty_function,
    .context_init       = ucs_async_poll_init,
    .context_cleanup    = ucs_empty_function,
    .context_try_block  = ucs_async_poll_tryblock,
    .context_unblock    = ucs_empty_function,
    .add_event_fd       = (ucs_async_add_event_fd_t)ucs_empty_function_return_success,
    .remove_event_fd    = ucs_empty_function_return_success,
    .modify_event_fd    = (ucs_async_modify_event_fd_t)ucs_empty_function_return_success,
    .add_timer          = ucs_empty_function_return_success,
    .remove_timer       = ucs_empty_function_return_success,
};

static inline khiter_t ucs_async_handler_kh_get(int id)
{
    return kh_get(ucs_async_handler, &ucs_async_global_context.handlers, id);
}

static inline int ucs_async_handler_kh_is_end(khiter_t hash_it)
{
    return hash_it == kh_end(&ucs_async_global_context.handlers);
}

static inline uint64_t ucs_async_missed_event_pack(int id,
                                                   ucs_event_set_types_t events)
{
    return ((uint64_t)id << UCS_ASYNC_MISSED_QUEUE_SHIFT) | (uint32_t)events;
}

static inline void ucs_async_missed_event_unpack(uint64_t value, int *id_p,
                                                 int *events_p)
{
    *id_p     = value >> UCS_ASYNC_MISSED_QUEUE_SHIFT;
    *events_p = value & UCS_ASYNC_MISSED_QUEUE_MASK;
}

static void ucs_async_handler_hold(ucs_async_handler_t *handler)
{
    ucs_atomic_add32(&handler->refcount, 1);
}

/* incremented reference count and return the handler */
static ucs_async_handler_t *ucs_async_handler_get(int id)
{
    ucs_async_handler_t *handler;
    khiter_t hash_it;

    pthread_rwlock_rdlock(&ucs_async_global_context.handlers_lock);
    hash_it = ucs_async_handler_kh_get(id);
    if (ucs_async_handler_kh_is_end(hash_it)) {
        handler = NULL;
        goto out_unlock;
    }

    handler = kh_value(&ucs_async_global_context.handlers, hash_it);
    ucs_assert_always(handler->id == id);
    ucs_async_handler_hold(handler);

out_unlock:
    pthread_rwlock_unlock(&ucs_async_global_context.handlers_lock);
    return handler;
}

/* remove from hash and return the handler */
static ucs_async_handler_t *ucs_async_handler_extract(int id)
{
    ucs_async_handler_t *handler;
    khiter_t hash_it;

    pthread_rwlock_wrlock(&ucs_async_global_context.handlers_lock);
    hash_it = ucs_async_handler_kh_get(id);
    if (ucs_async_handler_kh_is_end(hash_it)) {
        ucs_debug("async handler [id=%d] not found in hash table", id);
        handler = NULL;
    } else {
        handler = kh_value(&ucs_async_global_context.handlers, hash_it);
        ucs_assert_always(handler->id == id);
        kh_del(ucs_async_handler, &ucs_async_global_context.handlers, hash_it);
        ucs_debug("removed async handler " UCS_ASYNC_HANDLER_FMT " from hash",
                  UCS_ASYNC_HANDLER_ARG(handler));
    }
    pthread_rwlock_unlock(&ucs_async_global_context.handlers_lock);

    return handler;
}

/* decrement reference count and release the handler if reached 0 */
static void ucs_async_handler_put(ucs_async_handler_t *handler)
{
    if (ucs_atomic_fsub32(&handler->refcount, 1) > 1) {
        return;
    }

    ucs_debug("release async handler " UCS_ASYNC_HANDLER_FMT,
              UCS_ASYNC_HANDLER_ARG(handler));
    ucs_free(handler);
}

/* add new handler to the table */
static ucs_status_t ucs_async_handler_add(int min_id, int max_id,
                                          ucs_async_handler_t *handler)
{
    khiter_t hash_it = 0;
    ucs_async_handler_t *handler_from_hash;
    int hash_extra_status;
    ucs_status_t status;
    int i, id;

    pthread_rwlock_wrlock(&ucs_async_global_context.handlers_lock);

    handler->id = -1;
    ucs_assert_always(handler->refcount == 1);

    /*
     * Search for an empty key in the range [min_id, max_id)
     * ucs_async_global_context.handler_id is used to generate "unique" keys.
     */
    for (i = min_id; i < max_id; ++i) {
        id = min_id + (ucs_atomic_fadd32(&ucs_async_global_context.handler_id, 1) %
                       (max_id - min_id));
        hash_it = kh_put(ucs_async_handler, &ucs_async_global_context.handlers,
                         id, &hash_extra_status);
        if (hash_extra_status == UCS_KH_PUT_FAILED) {
            ucs_error("Failed to add async handler " UCS_ASYNC_HANDLER_FMT
                      " to hash", UCS_ASYNC_HANDLER_ARG(handler));
            status = UCS_ERR_NO_MEMORY;
            goto out_unlock;
        } else if (hash_extra_status == UCS_KH_PUT_KEY_PRESENT) {
            if ((max_id - min_id) == 1) {
                handler_from_hash = kh_value(&ucs_async_global_context.handlers,
                                             hash_it);
                ucs_error("async handler %s() uses id %d,"
                          " new async handler %s couldn't use this id",
                          ucs_debug_get_symbol_name(handler_from_hash->cb), i,
                          ucs_debug_get_symbol_name(handler->cb));
                break;
            }
        } else {
            handler->id = id;
            ucs_assert(id != -1);
            break;
        }
    }

    if (handler->id == -1) {
        ucs_error("Cannot add async handler %s() - id range [%d..%d) is full",
                  ucs_debug_get_symbol_name(handler->cb), min_id, max_id);
        status = UCS_ERR_ALREADY_EXISTS;
        goto out_unlock;
    }

    ucs_assert_always(!ucs_async_handler_kh_is_end(hash_it));
    kh_value(&ucs_async_global_context.handlers, hash_it) = handler;
    ucs_debug("added async handler " UCS_ASYNC_HANDLER_FMT " to hash",
              UCS_ASYNC_HANDLER_ARG(handler));
    status = UCS_OK;

out_unlock:
    pthread_rwlock_unlock(&ucs_async_global_context.handlers_lock);
    return status;
}

static void ucs_async_handler_invoke(ucs_async_handler_t *handler,
                                     ucs_event_set_types_t events)
{
    ucs_trace_async("calling async handler " UCS_ASYNC_HANDLER_FMT,
                    UCS_ASYNC_HANDLER_ARG(handler));

    /* track call count to allow removing the handler synchronously from itself
     * the handler must always be called with async context blocked, so no need
     * for atomic operations here.
     */
    ucs_assert(handler->caller == UCS_ASYNC_PTHREAD_ID_NULL);
    handler->caller = pthread_self();
    handler->cb(handler->id, events, handler->arg);
    handler->caller = UCS_ASYNC_PTHREAD_ID_NULL;
}

static ucs_status_t ucs_async_handler_dispatch(ucs_async_handler_t *handler,
                                               ucs_event_set_types_t events)
{
    ucs_async_context_t *async;
    ucs_async_mode_t mode;
    ucs_status_t status;
    uint64_t value;

    mode  = handler->mode;
    async = handler->async;

    if (async == NULL) {
        ucs_async_handler_invoke(handler, events);
        return UCS_OK;
    }

    async->last_wakeup = ucs_get_time();
    if (ucs_async_method_call(mode, context_try_block, async)) {
        ucs_async_handler_invoke(handler, events);
        ucs_async_method_call(mode, context_unblock, async);
    } else {
        ucs_trace_async("missed " UCS_ASYNC_HANDLER_FMT ", last_wakeup %lu",
                        UCS_ASYNC_HANDLER_ARG(handler), async->last_wakeup);
        if (ucs_atomic_cswap32(&handler->missed, 0, 1) == 0) {
            /* save both the handler_id and events */
            value = ucs_async_missed_event_pack(handler->id, events);
            status = ucs_mpmc_queue_push(&async->missed, value);
            if (status != UCS_OK) {
                ucs_fatal("Failed to push event %d to miss queue: %s",
                          handler->id, ucs_status_string(status));
            }
        }
        return UCS_ERR_NO_PROGRESS;
    }
    return UCS_OK;
}

ucs_status_t ucs_async_dispatch_handlers(int *handler_ids, size_t count,
                                         ucs_event_set_types_t events)
{
    ucs_status_t status = UCS_OK, tmp_status;
    ucs_async_handler_t *handler;

    for (; count > 0; --count, ++handler_ids) {
        handler = ucs_async_handler_get(*handler_ids);
        if (handler == NULL) {
            ucs_trace_async("handler for %d not found - ignoring", *handler_ids);
            continue;
        }

        tmp_status = ucs_async_handler_dispatch(handler, events);
        if (tmp_status != UCS_OK) {
            status = tmp_status;
        }

        ucs_async_handler_put(handler);
    }
    return status;
}

ucs_status_t ucs_async_dispatch_timerq(ucs_timer_queue_t *timerq,
                                       ucs_time_t current_time)
{
    size_t max_timers, num_timers = 0;
    int *expired_timers;
    ucs_timer_t *timer;

    max_timers     = ucs_max(1, ucs_timerq_size(timerq));
    expired_timers = ucs_alloca(max_timers * sizeof(*expired_timers));

    ucs_timerq_for_each_expired(timer, timerq, current_time, {
        expired_timers[num_timers++] = timer->id;
        if (num_timers >= max_timers) {
            break; /* Keep timers which we don't have room for in the queue */
        }
    })

    return ucs_async_dispatch_handlers(expired_timers, num_timers,
                                       UCS_ASYNC_EVENT_DUMMY);
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
    async->last_wakeup  = ucs_get_time();
    return UCS_OK;

err_free_miss_fds:
    ucs_mpmc_queue_cleanup(&async->missed);
err:
    return status;
}

ucs_status_t ucs_async_context_create(ucs_async_mode_t mode,
                                      ucs_async_context_t **async_p)
{
    ucs_async_context_t *async;
    ucs_status_t status;

    async = ucs_malloc(sizeof(*async), "async context");
    if (async == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = ucs_async_context_init(async, mode);
    if (status != UCS_OK) {
        goto err_free_mem;
    }

    *async_p = async;
    return UCS_OK;

err_free_mem:
    ucs_free(async);
err:
    return status;
}

void ucs_async_context_cleanup(ucs_async_context_t *async)
{
    ucs_async_handler_t *handler;

    ucs_trace_func("async=%p", async);

    if (async->num_handlers > 0) {
        pthread_rwlock_rdlock(&ucs_async_global_context.handlers_lock);
        kh_foreach_value(&ucs_async_global_context.handlers, handler, {
            if (async == handler->async) {
                ucs_warn("async %p handler "UCS_ASYNC_HANDLER_FMT" not released",
                         async, UCS_ASYNC_HANDLER_ARG(handler));
            }
        });
        ucs_warn("releasing async context with %d handlers", async->num_handlers);
        pthread_rwlock_unlock(&ucs_async_global_context.handlers_lock);
    }

    ucs_async_method_call(async->mode, context_cleanup, async);
    ucs_mpmc_queue_cleanup(&async->missed);
}

void ucs_async_context_destroy(ucs_async_context_t *async)
{
    ucs_async_context_cleanup(async);
    ucs_free(async);
}

int ucs_async_is_from_async(const ucs_async_context_t *async)
{
    return ucs_async_method_call(async->mode, is_from_async);
}

static ucs_status_t
ucs_async_alloc_handler(int min_id, int max_id, ucs_async_mode_t mode,
                        ucs_event_set_types_t events, ucs_async_event_cb_t cb,
                        void *arg, ucs_async_context_t *async, int *id_p)
{
    ucs_async_handler_t *handler;
    ucs_status_t status;

    /* If async context is given, it should have same mode */
    if ((async != NULL) && (async->mode != mode)) {
        ucs_error("Async mode mismatch for handler %s(), "
                  "mode: %d async context mode: %d",
                  ucs_debug_get_symbol_name(cb), mode, async->mode);
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    /* Limit amount of handlers per context */
    if (async != NULL) {
        if (ucs_atomic_fadd32(&async->num_handlers, 1) >= ucs_global_opts.async_max_events) {
            status = UCS_ERR_EXCEEDS_LIMIT;
            goto err_dec_num_handlers;
        }
    }

    handler = ucs_malloc(sizeof *handler, "async handler");
    if (handler == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_dec_num_handlers;
    }

    handler->mode     = mode;
    handler->events   = events;
    handler->caller   = UCS_ASYNC_PTHREAD_ID_NULL;
    handler->cb       = cb;
    handler->arg      = arg;
    handler->async    = async;
    handler->missed   = 0;
    handler->refcount = 1;
    ucs_async_method_call(mode, block);
    status = ucs_async_handler_add(min_id, max_id, handler);
    ucs_async_method_call(mode, unblock);
    if (status != UCS_OK) {
        goto err_free;
    }

    ucs_assert((handler->id >= min_id) && (handler->id < max_id));
    *id_p = handler->id;
    return UCS_OK;

err_free:
    ucs_free(handler);
err_dec_num_handlers:
    if (async != NULL) {
        ucs_atomic_add32(&async->num_handlers, (uint32_t)-1);
    }
err:
    return status;
}

ucs_status_t ucs_async_set_event_handler(ucs_async_mode_t mode, int event_fd,
                                         ucs_event_set_types_t events,
                                         ucs_async_event_cb_t cb, void *arg,
                                         ucs_async_context_t *async)
{
    ucs_status_t status;
    int event_id;

    if (event_fd >= UCS_ASYNC_TIMER_ID_MIN) {
        /* File descriptor too large */
        status = UCS_ERR_EXCEEDS_LIMIT;
        goto err;
    }

    status = ucs_async_alloc_handler(event_fd, event_fd + 1, mode, events, cb,
                                     arg, async, &event_id);
    if (status != UCS_OK) {
        goto err;
    }
    ucs_assert(event_id == event_fd);

    status = ucs_async_method_call(mode, add_event_fd, async, event_fd, events);
    if (status != UCS_OK) {
        goto err_remove_handler;
    }

    ucs_debug("listening to async event fd %d events 0x%x mode %s", event_fd,
              events, ucs_async_mode_names[mode]);
    return UCS_OK;

err_remove_handler:
    ucs_async_remove_handler(event_fd, 1);
err:
    return status;
}

ucs_status_t ucs_async_add_timer(ucs_async_mode_t mode, ucs_time_t interval,
                                 ucs_async_event_cb_t cb, void *arg,
                                 ucs_async_context_t *async, int *timer_id_p)
{
    ucs_status_t status;
    int timer_id;

    status = ucs_async_alloc_handler(UCS_ASYNC_TIMER_ID_MIN, UCS_ASYNC_TIMER_ID_MAX,
                                     mode, 1, cb, arg, async, &timer_id);
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
    ucs_async_remove_handler(timer_id, 1);
err:
    return status;
}

ucs_status_t ucs_async_remove_handler(int id, int is_sync)
{
    ucs_async_handler_t *handler;
    ucs_status_t status;

    /* We can't find the async handle mode without taking a read lock, which in
     * turn may cause a deadlock if async handle is running. So we have to block
     * all modes.
     */
    ucs_async_method_call_all(block);
    handler = ucs_async_handler_extract(id);
    ucs_async_method_call_all(unblock);
    if (handler == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    ucs_debug("removing async handler " UCS_ASYNC_HANDLER_FMT,
              UCS_ASYNC_HANDLER_ARG(handler));
    if (handler->id >= UCS_ASYNC_TIMER_ID_MIN) {
        status = ucs_async_method_call(handler->mode, remove_timer,
                                       handler->async, handler->id);
    } else {
        status = ucs_async_method_call(handler->mode, remove_event_fd,
                                       handler->async, handler->id);
    }
    if (status != UCS_OK) {
        ucs_warn("failed to remove async handler " UCS_ASYNC_HANDLER_FMT " : %s",
                 UCS_ASYNC_HANDLER_ARG(handler), ucs_status_string(status));
    }

    if (handler->async != NULL) {
        ucs_atomic_add32(&handler->async->num_handlers, (uint32_t)-1);
    }

    if (is_sync) {
        int called = (pthread_self() == handler->caller);
        ucs_trace("waiting for " UCS_ASYNC_HANDLER_FMT " completion (called=%d)",
                  UCS_ASYNC_HANDLER_ARG(handler), called);
        while ((handler->refcount - called) > 1) {
            /* TODO use pthread_cond / futex to reduce CPU usage while waiting
             * for the async handler to complete */
            sched_yield();
        }
    }

    ucs_async_handler_put(handler);
    return UCS_OK;
}

ucs_status_t ucs_async_modify_handler(int fd, ucs_event_set_types_t events)
{
    ucs_async_handler_t *handler;
    ucs_status_t status;

    if (fd >= UCS_ASYNC_TIMER_ID_MIN) {
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_async_method_call_all(block);
    handler = ucs_async_handler_get(fd);
    ucs_async_method_call_all(unblock);

    if (handler == NULL) {
        return UCS_ERR_NO_ELEM;
    }

    handler->events = events;
    status = ucs_async_method_call(handler->mode, modify_event_fd,
                                   handler->async, fd, handler->events);
    ucs_async_handler_put(handler);

    return status;
}

void __ucs_async_poll_missed(ucs_async_context_t *async)
{
    ucs_async_handler_t *handler;
    int handler_id, events;
    ucs_status_t status;
    uint64_t value;

    ucs_trace_async("miss handler");

    while (!ucs_mpmc_queue_is_empty(&async->missed)) {

        status = ucs_mpmc_queue_pull(&async->missed, &value);
        if (status == UCS_ERR_NO_PROGRESS) {
            /* TODO we should retry here if the code is change to check miss
             * only during ASYNC_UNBLOCK */
            break;
        }

        ucs_async_method_call_all(block);
        UCS_ASYNC_BLOCK(async);

        ucs_async_missed_event_unpack(value, &handler_id, &events);
        handler = ucs_async_handler_get(handler_id);
        if (handler != NULL) {
            ucs_assert(handler->async == async);
            handler->missed = 0;
            ucs_async_handler_invoke(handler, events);
            ucs_async_handler_put(handler);
        }
        UCS_ASYNC_UNBLOCK(async);
        ucs_async_method_call_all(unblock);
    }
}

void ucs_async_poll(ucs_async_context_t *async)
{
    ucs_async_handler_t **handlers, *handler;
    size_t i, n;

    ucs_trace_poll("async=%p", async);

    pthread_rwlock_rdlock(&ucs_async_global_context.handlers_lock);
    handlers = ucs_alloca(kh_size(&ucs_async_global_context.handlers) * sizeof(*handlers));
    n = 0;
    kh_foreach_value(&ucs_async_global_context.handlers, handler, {
        if (((async == NULL) || (async == handler->async)) &&  /* Async context match */
            ((handler->async == NULL) || (handler->async->poll_block == 0)) && /* Not blocked */
            handler->events) /* Non-empty event set */
        {
            ucs_async_handler_hold(handler);
            handlers[n++] = handler;
        }
    });
    pthread_rwlock_unlock(&ucs_async_global_context.handlers_lock);

    for (i = 0; i < n; ++i) {
        /* dispatch the handler with all the registered events */
        ucs_async_handler_dispatch(handlers[i], handlers[i]->events);
        ucs_async_handler_put(handlers[i]);
    }
}

void ucs_async_global_init()
{
    int ret;

    ret = pthread_rwlock_init(&ucs_async_global_context.handlers_lock, NULL);
    if (ret) {
        ucs_fatal("pthread_rwlock_init() failed: %m");
    }

    kh_init_inplace(ucs_async_handler, &ucs_async_global_context.handlers);
    ucs_async_method_call_all(init);
}

void ucs_async_global_cleanup()
{
    int num_elems = kh_size(&ucs_async_global_context.handlers);
    if (num_elems != 0) {
        ucs_debug("async handler table is not empty during exit (contains %d elems)",
                  num_elems);
    }
    ucs_async_method_call_all(cleanup);
    kh_destroy_inplace(ucs_async_handler, &ucs_async_global_context.handlers);
    pthread_rwlock_destroy(&ucs_async_global_context.handlers_lock);
}
