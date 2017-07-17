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
#include <ucs/sys/sys.h>


#define UCS_ASYNC_EPOLL_MAX_EVENTS      16
#define UCS_ASYNC_EPOLL_MIN_TIMEOUT_MS  2.0


typedef struct ucs_async_thread {
    ucs_async_pipe_t   wakeup;
    int                epfd;
    ucs_timer_queue_t  timerq;
    pthread_t          thread_id;
    int                stop;
    uint32_t           refcnt;
} ucs_async_thread_t;


typedef struct ucs_async_thread_global_context {
    ucs_async_thread_t *thread;
    unsigned           use_count;
    pthread_mutex_t    lock;
} ucs_async_thread_global_context_t;


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
    if (ucs_atomic_fadd32(&thread->refcnt, -1) == 1) {
        close(thread->epfd);
        ucs_async_pipe_destroy(&thread->wakeup);
        ucs_timerq_cleanup(&thread->timerq);
        ucs_free(thread);
    }
}

static void *ucs_async_thread_func(void *arg)
{
    ucs_async_thread_t *thread = arg;
    struct epoll_event events[UCS_ASYNC_EPOLL_MAX_EVENTS];
    ucs_time_t last_time, curr_time, timer_interval, time_spent;
    int i, nready, is_missed, timeout_ms;
    ucs_status_t status;
    int fd;

    is_missed  = 0;
    curr_time  = ucs_get_time();
    last_time  = ucs_get_time();

    while (!thread->stop) {

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
        nready = epoll_wait(thread->epfd, events, UCS_ASYNC_EPOLL_MAX_EVENTS,
                            timeout_ms);
        if ((nready < 0) && (errno != EINTR)) {
            ucs_fatal("epoll_wait() failed: %m");
        }
        ucs_trace_async("epoll_wait(epfd=%d, timeout=%d) returned %d",
                        thread->epfd, timeout_ms, nready);

        /* Check ready files */
        if (nready > 0) {
            for (i = 0; i < nready; ++i) {
                fd = events[i].data.fd;

                /* Check wakeup pipe */
                if (fd == ucs_async_pipe_rfd(&thread->wakeup)) {
                    ucs_trace_async("progress thread woken up");
                    ucs_async_pipe_drain(&thread->wakeup);
                    continue;
                }

                status = ucs_async_dispatch_handlers(&fd, 1);
                if (status == UCS_ERR_NO_PROGRESS) {
                    is_missed = 1;
                }
            }
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
    struct epoll_event event;
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

    status = ucs_timerq_init(&thread->timerq);
    if (status != UCS_OK) {
        goto err_free;
    }

    status = ucs_async_pipe_create(&thread->wakeup);
    if (status != UCS_OK) {
        goto err_timerq_cleanup;
    }

    /* Create epoll set the thread will wait on */
    thread->epfd = epoll_create(1);
    if (thread->epfd < 0) {
        ucs_error("epoll_create() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_close_pipe;
    }

    /* Add wakeup pipe to epoll set */
    wakeup_rfd    = ucs_async_pipe_rfd(&thread->wakeup);
    memset(&event, 0, sizeof(event));
    event.events  = EPOLLIN;
    event.data.fd = wakeup_rfd;
    ret = epoll_ctl(thread->epfd, EPOLL_CTL_ADD, wakeup_rfd, &event);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, ADD, fd=%d) failed: %m",thread->epfd,
                  wakeup_rfd);
        status = UCS_ERR_IO_ERROR;
        goto err_close_epfd;
    }

    ret = pthread_create(&thread->thread_id, NULL, ucs_async_thread_func, thread);
    if (ret != 0) {
        ucs_error("pthread_create() returned %d: %m", ret);
        status = UCS_ERR_IO_ERROR;
        goto err_close_epfd;
    }

    ucs_async_thread_global_context.thread = thread;
    status = UCS_OK;
    goto out_unlock;

err_close_epfd:
    close(thread->epfd);
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

static ucs_status_t ucs_async_thread_init(ucs_async_context_t *async)
{
#if !(NVALGRIND)
    pthread_mutexattr_t attr;
    int ret;

    if (RUNNING_ON_VALGRIND) {
        pthread_mutexattr_init(&attr);
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
        ret = pthread_mutex_init(&async->thread.mutex, &attr);
        if (ret != 0) {
            ucs_error("Failed to initialize lock: %s", strerror(ret));
            return UCS_ERR_INVALID_PARAM;
        }

        return UCS_OK;
    } else
#endif
        return ucs_spinlock_init(&async->thread.spinlock);
}

static ucs_status_t ucs_async_thread_add_event_fd(ucs_async_context_t *async,
                                                  int event_fd, int events)
{
    ucs_async_thread_t *thread;
    struct epoll_event event;
    ucs_status_t status;
    int ret;

    status = ucs_async_thread_start(&thread);
    if (status != UCS_OK) {
        goto err;
    }

    memset(&event, 0, sizeof(event));
    event.events  = events;
    event.data.fd = event_fd;
    ret = epoll_ctl(thread->epfd, EPOLL_CTL_ADD, event_fd, &event);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, ADD, fd=%d) failed: %m", thread->epfd,
                  event_fd);
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
    int ret;

    ret = epoll_ctl(thread->epfd, EPOLL_CTL_DEL, event_fd, NULL);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, DEL, fd=%d) failed: %m", thread->epfd,
                  event_fd);
        return UCS_ERR_INVALID_PARAM;
    }

    ucs_async_thread_stop();
    return UCS_OK;
}

static ucs_status_t ucs_async_thread_modify_event_fd(ucs_async_context_t *async,
                                                     int event_fd, int events)
{
    ucs_async_thread_t *thread = ucs_async_thread_global_context.thread;
    struct epoll_event event;
    int ret;

    memset(&event, 0, sizeof(event));
    event.events  = events;
    event.data.fd = event_fd;
    ret = epoll_ctl(thread->epfd, EPOLL_CTL_MOD, event_fd, &event);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, ADD, fd=%d) failed: %m", thread->epfd,
                  event_fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

static int ucs_async_thread_try_block(ucs_async_context_t *async)
{
    return
#if !(NVALGRIND)
    (RUNNING_ON_VALGRIND) ?
        (pthread_mutex_trylock(&async->thread.mutex) == 0) :
#endif
        ucs_spin_trylock(&async->thread.spinlock);
}

static void ucs_async_thread_unblock(ucs_async_context_t *async)
{
    UCS_ASYNC_THREAD_UNBLOCK(async);
}

static ucs_status_t ucs_async_thread_add_timer(ucs_async_context_t *async,
                                               int timer_id, ucs_time_t interval)
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

    status = ucs_timerq_add(&thread->timerq, timer_id, interval);
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
        ucs_info("async thread still running (use count %d)",
                 ucs_async_thread_global_context.use_count);
    }
}

ucs_async_ops_t ucs_async_thread_ops = {
    .init               = ucs_empty_function,
    .cleanup            = ucs_async_signal_global_cleanup,
    .block              = ucs_empty_function,
    .unblock            = ucs_empty_function,
    .context_init       = ucs_async_thread_init,
    .context_try_block  = ucs_async_thread_try_block,
    .context_unblock    = ucs_async_thread_unblock,
    .add_event_fd       = ucs_async_thread_add_event_fd,
    .remove_event_fd    = ucs_async_thread_remove_event_fd,
    .modify_event_fd    = ucs_async_thread_modify_event_fd,
    .add_timer          = ucs_async_thread_add_timer,
    .remove_timer       = ucs_async_thread_remove_timer,
};
