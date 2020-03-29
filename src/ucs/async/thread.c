/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "thread.h"
#include "async_int.h"
#include "pipe.h"

#include <ucs/arch/atomic.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/stubs.h>
#include <ucs/sys/event_set.h>


#define UCS_ASYNC_EPOLL_MAX_EVENTS      16
#define UCS_ASYNC_EPOLL_MIN_TIMEOUT_MS  2.0


typedef struct ucs_async_thread {
    ucs_async_pipe_t    wakeup;
    ucs_sys_event_set_t *event_set;
    ucs_timer_queue_t   timerq;
    pthread_t           thread_id;
    int                 stop;
    uint32_t            refcnt;
} ucs_async_thread_t;


typedef struct ucs_async_thread_global_context {
    ucs_async_thread_t *thread;
    unsigned           use_count;
    pthread_mutex_t    lock;
} ucs_async_thread_global_context_t;


typedef struct ucs_async_thread_callback_arg {
    ucs_async_thread_t *thread;
    int                *is_missed;
} ucs_async_thread_callback_arg_t;


static ucs_async_thread_global_context_t ucs_async_thread_global_context = {
    .thread    = NULL,
    .use_count = 0,
    .lock      = PTHREAD_MUTEX_INITIALIZER
};


static void ucs_async_thread_hold(ucs_async_thread_t *thread)
{
    ucs_atomic_add32(&thread->refcnt, 1);
}

static void ucs_async_thread_put(ucs_async_thread_t *thread)
{
    if (ucs_atomic_fsub32(&thread->refcnt, 1) == 1) {
        ucs_event_set_cleanup(thread->event_set);
        ucs_async_pipe_destroy(&thread->wakeup);
        ucs_timerq_cleanup(&thread->timerq);
        ucs_free(thread);
    }
}

static void ucs_async_thread_ev_handler(void *callback_data, int event,
                                        void *arg)
{
    ucs_async_thread_callback_arg_t *cb_arg = (void*)arg;
    int fd                                  = (int)(uintptr_t)callback_data;
    ucs_status_t status;

    ucs_trace_async("ucs_async_thread_ev_handler(fd=%d, event=%d)",
                    fd, event);

    if (fd == ucs_async_pipe_rfd(&cb_arg->thread->wakeup)) {
        ucs_trace_async("progress thread woken up");
        ucs_async_pipe_drain(&cb_arg->thread->wakeup);
        return;
    }

    status = ucs_async_dispatch_handlers(&fd, 1, event);
    if (status == UCS_ERR_NO_PROGRESS) {
         *cb_arg->is_missed = 1;
    }
}

static void *ucs_async_thread_func(void *arg)
{
    ucs_async_thread_t *thread = arg;
    ucs_time_t last_time, curr_time, timer_interval, time_spent;
    int is_missed, timeout_ms;
    ucs_status_t status;
    unsigned num_events;
    ucs_async_thread_callback_arg_t cb_arg;

    is_missed        = 0;
    curr_time        = ucs_get_time();
    last_time        = ucs_get_time();
    cb_arg.thread    = thread;
    cb_arg.is_missed = &is_missed;

    while (!thread->stop) {
        num_events = ucs_min(UCS_ASYNC_EPOLL_MAX_EVENTS,
                             ucs_sys_event_set_max_wait_events);

        /* If we didn't get the lock, give other threads priority */
        if (is_missed) {
            sched_yield();
            is_missed = 0;
        }

        /* Wait until the remainder of current period */
        timer_interval = ucs_timerq_min_interval(&thread->timerq);
        if (timer_interval == UCS_TIME_INFINITY) {
            timeout_ms = -1;
        } else {
            time_spent = curr_time - last_time;
            timeout_ms = ucs_time_to_msec(timer_interval -
                                          ucs_min(time_spent, timer_interval));
        }

        status = ucs_event_set_wait(thread->event_set,
                                    &num_events, timeout_ms,
                                    ucs_async_thread_ev_handler,
                                    (void*)&cb_arg);
        if (UCS_STATUS_IS_ERR(status)) {
            ucs_fatal("ucs_event_set_wait() failed: %d", status);
        }

        /* Check timers */
        curr_time = ucs_get_time();
        if (curr_time - last_time > timer_interval) {
            status = ucs_async_dispatch_timerq(&thread->timerq, curr_time);
            if (status == UCS_ERR_NO_PROGRESS) {
                 is_missed = 1;
            }

            last_time = curr_time;
        }
    }

    ucs_async_thread_put(thread);
    return NULL;
}

static ucs_status_t ucs_async_thread_start(ucs_async_thread_t **thread_p)
{
    ucs_async_thread_t *thread;
    ucs_status_t status;
    int wakeup_rfd;
    int ret;

    ucs_trace_func("");

    pthread_mutex_lock(&ucs_async_thread_global_context.lock);
    if (ucs_async_thread_global_context.use_count++ > 0) {
        /* Thread already started */
        status = UCS_OK;
        goto out_unlock;
    }

    ucs_assert_always(ucs_async_thread_global_context.thread == NULL);

    thread = ucs_malloc(sizeof(*thread), "async_thread_context");
    if (thread == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    thread->stop   = 0;
    thread->refcnt = 1;

    status = ucs_timerq_init(&thread->timerq, "async_thread_timer");
    if (status != UCS_OK) {
        goto err_free;
    }

    status = ucs_async_pipe_create(&thread->wakeup);
    if (status != UCS_OK) {
        goto err_timerq_cleanup;
    }

    status = ucs_event_set_create(&thread->event_set);
    if (status != UCS_OK) {
        goto err_close_pipe;
    }

    /* Store file descriptor into void * storage without memory allocation. */
    wakeup_rfd    = ucs_async_pipe_rfd(&thread->wakeup);
    status = ucs_event_set_add(thread->event_set, wakeup_rfd,
                               UCS_EVENT_SET_EVREAD,
                               (void *)(uintptr_t)wakeup_rfd);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_free_event_set;
    }

    ret = pthread_create(&thread->thread_id, NULL, ucs_async_thread_func, thread);
    if (ret != 0) {
        ucs_error("pthread_create() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_free_event_set;
    }

    ucs_async_thread_global_context.thread = thread;
    status = UCS_OK;
    goto out_unlock;

err_free_event_set:
    ucs_event_set_cleanup(thread->event_set);
err_close_pipe:
    ucs_async_pipe_destroy(&thread->wakeup);
err_timerq_cleanup:
    ucs_timerq_cleanup(&thread->timerq);
err_free:
    ucs_free(thread);
err:
    --ucs_async_thread_global_context.use_count;
out_unlock:
    ucs_assert_always(ucs_async_thread_global_context.thread != NULL);
    *thread_p = ucs_async_thread_global_context.thread;
    pthread_mutex_unlock(&ucs_async_thread_global_context.lock);
    return status;
}

static void ucs_async_thread_stop()
{
    ucs_async_thread_t *thread = NULL;

    ucs_trace_func("");

    pthread_mutex_lock(&ucs_async_thread_global_context.lock);
    if (--ucs_async_thread_global_context.use_count == 0) {
        thread = ucs_async_thread_global_context.thread;
        ucs_async_thread_hold(thread);
        thread->stop = 1;
        ucs_async_pipe_push(&thread->wakeup);
        ucs_async_thread_global_context.thread = NULL;
    }
    pthread_mutex_unlock(&ucs_async_thread_global_context.lock);

    if (thread != NULL) {
        if (pthread_self() == thread->thread_id) {
            pthread_detach(thread->thread_id);
        } else {
            pthread_join(thread->thread_id, NULL);
        }
        ucs_async_thread_put(thread);
    }
}

static ucs_status_t ucs_async_thread_spinlock_init(ucs_async_context_t *async)
{
    return ucs_recursive_spinlock_init(&async->thread.spinlock, 0);
}

static void ucs_async_thread_spinlock_cleanup(ucs_async_context_t *async)
{
    ucs_status_t status;

    status = ucs_recursive_spinlock_destroy(&async->thread.spinlock);
    if (status != UCS_OK) {
        ucs_warn("ucs_recursive_spinlock_destroy() failed (%d)", status);
    }
}

static int ucs_async_thread_spinlock_try_block(ucs_async_context_t *async)
{
    return ucs_recursive_spin_trylock(&async->thread.spinlock);
}

static void ucs_async_thread_spinlock_unblock(ucs_async_context_t *async)
{
    ucs_recursive_spin_unlock(&async->thread.spinlock);
}

static ucs_status_t ucs_async_thread_mutex_init(ucs_async_context_t *async)
{
    pthread_mutexattr_t attr;
    int                 ret;

    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    ret = pthread_mutex_init(&async->thread.mutex, &attr);
    if (ret == 0) {
        return UCS_OK;
    }

    ucs_error("failed to initialize async lock: %s", strerror(ret));
    return UCS_ERR_INVALID_PARAM;
}

static void ucs_async_thread_mutex_cleanup(ucs_async_context_t *async)
{
    int ret = pthread_mutex_destroy(&async->thread.mutex);

    if (ret != 0) {
        ucs_warn("failed to destroy async lock: %s", strerror(ret));
    }
}

static ucs_status_t ucs_async_thread_add_event_fd(ucs_async_context_t *async,
                                                  int event_fd, int events)
{
    ucs_async_thread_t *thread;
    ucs_status_t status;

    status = ucs_async_thread_start(&thread);
    if (status != UCS_OK) {
        goto err;
    }

    /* Store file descriptor into void * storage without memory allocation. */
    status = ucs_event_set_add(thread->event_set, event_fd,
                               (ucs_event_set_type_t)events,
                               (void *)(uintptr_t)event_fd);
    if (status != UCS_OK) {
        status = UCS_ERR_IO_ERROR;
        goto err_removed;
    }

    ucs_async_pipe_push(&thread->wakeup);
    return UCS_OK;

err_removed:
    ucs_async_thread_stop();
err:
    return status;
}

static ucs_status_t ucs_async_thread_remove_event_fd(ucs_async_context_t *async,
                                                     int event_fd)
{
    ucs_async_thread_t *thread = ucs_async_thread_global_context.thread;
    ucs_status_t status;

    status = ucs_event_set_del(thread->event_set, event_fd);
    if (status != UCS_OK) {
        return status;
    }

    ucs_async_thread_stop();
    return UCS_OK;
}

static ucs_status_t ucs_async_thread_modify_event_fd(ucs_async_context_t *async,
                                                     int event_fd, int events)
{
    /* Store file descriptor into void * storage without memory allocation. */
    return ucs_event_set_mod(ucs_async_thread_global_context.thread->event_set,
                             event_fd, (ucs_event_set_type_t)events,
                             (void *)(uintptr_t)event_fd);
}

static int ucs_async_thread_mutex_try_block(ucs_async_context_t *async)
{
    return !pthread_mutex_trylock(&async->thread.mutex);
}

static void ucs_async_thread_mutex_unblock(ucs_async_context_t *async)
{
    (void)pthread_mutex_unlock(&async->thread.mutex);
}

static ucs_status_t ucs_async_thread_add_timer(ucs_async_context_t *async,
                                               ucs_time_t interval, int *timer_id_p)
{
    ucs_async_thread_t *thread;
    ucs_status_t status;

    if (ucs_time_to_msec(interval) == 0) {
        ucs_error("timer interval is too small (%.2f usec)", ucs_time_to_usec(interval));
        status = UCS_ERR_INVALID_PARAM;
        goto err;
    }

    status = ucs_async_thread_start(&thread);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_timerq_add(&thread->timerq, interval, timer_id_p);
    if (status != UCS_OK) {
        goto err_stop;
    }

    ucs_async_pipe_push(&thread->wakeup);
    return UCS_OK;

err_stop:
    ucs_async_thread_stop();
err:
    return status;
}

static ucs_status_t ucs_async_thread_remove_timer(ucs_async_context_t *async,
                                                  int timer_id)
{
    ucs_async_thread_t *thread = ucs_async_thread_global_context.thread;
    ucs_timerq_remove(&thread->timerq, timer_id);
    ucs_async_pipe_push(&thread->wakeup);
    ucs_async_thread_stop();
    return UCS_OK;
}

static void ucs_async_signal_global_cleanup()
{
    if (ucs_async_thread_global_context.thread != NULL) {
        ucs_debug("async thread still running (use count %u)",
                  ucs_async_thread_global_context.use_count);
    }
}

ucs_async_ops_t ucs_async_thread_spinlock_ops = {
    .init               = ucs_empty_function,
    .cleanup            = ucs_async_signal_global_cleanup,
    .block              = ucs_empty_function,
    .unblock            = ucs_empty_function,
    .context_init       = ucs_async_thread_spinlock_init,
    .context_cleanup    = ucs_async_thread_spinlock_cleanup,
    .context_try_block  = ucs_async_thread_spinlock_try_block,
    .context_unblock    = ucs_async_thread_spinlock_unblock,
    .add_event_fd       = ucs_async_thread_add_event_fd,
    .remove_event_fd    = ucs_async_thread_remove_event_fd,
    .modify_event_fd    = ucs_async_thread_modify_event_fd,
    .add_timer          = ucs_async_thread_add_timer,
    .remove_timer       = ucs_async_thread_remove_timer,
};

ucs_async_ops_t ucs_async_thread_mutex_ops = {
    .init               = ucs_empty_function,
    .cleanup            = ucs_async_signal_global_cleanup,
    .block              = ucs_empty_function,
    .unblock            = ucs_empty_function,
    .context_init       = ucs_async_thread_mutex_init,
    .context_cleanup    = ucs_async_thread_mutex_cleanup,
    .context_try_block  = ucs_async_thread_mutex_try_block,
    .context_unblock    = ucs_async_thread_mutex_unblock,
    .add_event_fd       = ucs_async_thread_add_event_fd,
    .remove_event_fd    = ucs_async_thread_remove_event_fd,
    .modify_event_fd    = ucs_async_thread_modify_event_fd,
    .add_timer          = ucs_async_thread_add_timer,
    .remove_timer       = ucs_async_thread_remove_timer,
};
