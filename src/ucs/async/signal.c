/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "signal.h"
#include "async_int.h"

#include <ucs/datastruct/list.h>
#include <ucs/datastruct/hash.h>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <ucs/time/timerq.h>
#include <signal.h>


/*
 * Per-thread system timer and software timer queue.
 */
typedef struct ucs_async_signal_timerq ucs_async_signal_timerq_t;
struct ucs_async_signal_timerq {
    pid_t                      tid;          /* Thread ID */
    timer_t                    sys_timer_id; /* System timer ID */
    ucs_timer_queue_t          timerq;       /* Queue of timers for the thread */
    ucs_async_signal_timerq_t  *next;
};


/*
 * Hash table of per-thread timer queues.
 */
#define UCS_ASYNC_SIGNAL_TIMERQ_COMPARE(_t1, _t2)  ((int)(_t1)->tid - (int)(_t2)->tid)
#define UCS_ASYNC_SIGNAL_TIMERQ_HASH(_t)           ((int)(_t)->tid)
UCS_DEFINE_THREAD_SAFE_HASH(ucs_async_signal_timerq_t, next, 37,
                            UCS_ASYNC_SIGNAL_TIMERQ_COMPARE,
                            UCS_ASYNC_SIGNAL_TIMERQ_HASH)


typedef struct ucs_async_signal_timer_info {
    int                        timer_id;
    ucs_time_t                 interval;
} ucs_async_signal_timer_info_t;


static struct {
    struct sigaction                     prev_sighandler; /* Previous signal handler */
    volatile uint32_t                    event_count;     /* Number of events in use */
    ucs_hashed_ucs_async_signal_timerq_t timers;          /* Hash of all threads */
} ucs_async_signal_global_context = {
    .event_count = 0,
};


/**
 * In signal mode, we allow user to manipulate events only from the same thread.
 * Otherwise, we'd get into big synchronization issues.
 */
#define UCS_ASYNC_SIGNAL_CHECK_THREAD(_async) \
    if (ucs_get_tid() != __ucs_async_signal_context_tid(_async)) { \
        ucs_error("cannot manipulate signal async from different thread"); \
        return UCS_ERR_UNREACHABLE; \
    }


/**
 * @return To which thread the async context should deliver events to.
 */
static pid_t __ucs_async_signal_context_tid(ucs_async_context_t *async)
{
    static pid_t pid = -1;

    if (pid == -1) {
        pid = getpid();
    }
    return (async == NULL) ? pid : async->signal.tid;;
}

static ucs_status_t
ucs_async_signal_set_fd_owner(pid_t dest_tid, int fd)
{
#if HAVE_DECL_F_SETOWN_EX
    struct f_owner_ex owner;

    owner.type = F_OWNER_TID;
    owner.pid  = dest_tid;

    ucs_trace_async("fcntl(F_SETOWN, fd=%d, tid=%d)", fd, dest_tid);
    if (0 > fcntl(fd, F_SETOWN_EX, &owner)) {
        ucs_error("fcntl F_SETOWN_EX failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
#else
    if (dest_tid != getpid()) {
        ucs_error("Cannot use signaled events to threads without F_SETOWN_EX support");
        return UCS_ERR_UNSUPPORTED;
    }

    if (0 > fcntl(fd, F_SETOWN, dest_tid)) {
        ucs_error("fcntl F_SETOWN failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
#endif
}

static ucs_status_t
ucs_async_signal_sys_timer_create(pid_t tid, timer_t *sys_timer_id)
{
    struct sigevent ev;
    timer_t timer;
    int ret;

    ucs_trace_func("tid=%d", tid);

    /* Create timer signal */
    memset(&ev, 0, sizeof(ev));
    ev.sigev_notify           = SIGEV_THREAD_ID;
    ev.sigev_signo            = ucs_global_opts.async_signo;
    ev.sigev_value.sival_int = tid; /* parameter to timer */
    ev._sigev_un._tid        = tid; /* target thread */
    ret = timer_create(CLOCK_REALTIME, &ev, &timer);
    if (ret < 0) {
        ucs_error("failed to create an interval timer: %m");
        return UCS_ERR_INVALID_PARAM;
    }

    *sys_timer_id = timer;
    return UCS_OK;
}

static ucs_status_t
ucs_async_signal_sys_timer_set_interval(timer_t sys_timer_id, ucs_time_t interval)
{
    struct itimerspec its;
    int ret;

    ucs_trace_func("sys_timer_id=%p interval=%.2f usec", sys_timer_id,
                   ucs_time_to_usec(interval));

    /* Modify the timer to have the desired accuracy */
    ucs_sec_to_timespec(ucs_time_to_sec(interval), &its.it_interval);
    its.it_value      = its.it_interval;
    ret = timer_settime(sys_timer_id, 0, &its, NULL);
    if (ret < 0) {
        ucs_error("failed to set the interval for the interval timer: %m");
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void ucs_async_signal_sys_timer_delete(timer_t sys_timer_id)
{
    int ret;

    ucs_trace_func("sys_timer_id=%p", sys_timer_id);

    ret = timer_delete(sys_timer_id);
    if (ret < 0) {
        ucs_warn("failed to remove the timer: %m");
    }
}

static ucs_status_t
ucs_async_signal_dispatch_timerq(ucs_async_signal_timerq_t *timerq, void *arg)
{
    ucs_time_t current_time = ucs_get_time();
    ucs_timer_t *timer;

    ucs_trace_func("");

    ucs_timerq_for_each_expired(timer, &timerq->timerq, current_time) {
        ucs_async_dispatch_handler(timer->id, 1);
    }
    return UCS_OK;
}

static void ucs_async_signal_handler(int signo, siginfo_t *siginfo, void *arg)
{
    ucs_async_signal_timerq_t search;

    ucs_assert(signo == ucs_global_opts.async_signo);

    /* Check event code */
    switch (siginfo->si_code) {
    case SI_TIMER:
        search.tid = siginfo->si_int;
        ucs_assert(siginfo->si_int == ucs_get_tid());
        ucs_trace_async("timer signal on thread %d", search.tid);
        ucs_hashed_ucs_async_signal_timerq_t_find(&ucs_async_signal_global_context.timers,
                                                  &search,
                                                  ucs_async_signal_dispatch_timerq,
                                                  NULL);
        return;
    case POLL_IN:
    case POLL_OUT:
    case POLL_HUP:
    case POLL_ERR:
    case POLL_MSG:
    case POLL_PRI:
        ucs_trace_async("async signal handler called for fd %d", siginfo->si_fd);
        ucs_async_dispatch_handler(siginfo->si_fd, 1);
        return;
    default:
        ucs_warn("signal handler called with unexpected event code %d, ignoring",
                 siginfo->si_code);
        return;
    }
}

static void ucs_async_signal_allow(int allow)
{
    sigset_t sigset;

    ucs_trace_func("enable=%d tid=%d", allow, ucs_get_tid());

    sigemptyset(&sigset);
    sigaddset(&sigset, ucs_global_opts.async_signo);
    pthread_sigmask(allow ? SIG_UNBLOCK : SIG_BLOCK, &sigset, NULL);
}

static void ucs_async_signal_block_all()
{
    if (ucs_async_signal_global_context.event_count > 0) {
        ucs_async_signal_allow(0);
    }
}

static void ucs_async_signal_unblock_all()
{
    if (ucs_async_signal_global_context.event_count > 0) {
        ucs_async_signal_allow(1);
    }
}

static ucs_status_t ucs_async_signal_install_handler()
{
    struct sigaction new_action;
    int ret;

    ucs_trace_func("");

    if (ucs_atomic_fadd32(&ucs_async_signal_global_context.event_count, +1) == 0) {
        /* Set our signal handler */
        new_action.sa_sigaction = ucs_async_signal_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags    = SA_RESTART|SA_SIGINFO;
        new_action.sa_restorer = NULL;
        ret = sigaction(ucs_global_opts.async_signo, &new_action,
                        &ucs_async_signal_global_context.prev_sighandler);
        if (ret < 0) {
            ucs_error("failed to set a handler for signal %d: %m",
                      ucs_global_opts.async_signo);
            ucs_atomic_fadd32(&ucs_async_signal_global_context.event_count, -1);
            return UCS_ERR_INVALID_PARAM;
        }

        ucs_trace_async("installed signal handler for %s",
                        ucs_signal_names[ucs_global_opts.async_signo]);
    }

    return UCS_OK;
}

static void ucs_async_signal_uninstall_handler()
{
    int ret;

    ucs_trace_func("");

    if (ucs_atomic_fadd32(&ucs_async_signal_global_context.event_count, -1) == 1) {
        ret = sigaction(ucs_global_opts.async_signo,
                        &ucs_async_signal_global_context.prev_sighandler, NULL);
        if (ret < 0) {
            ucs_warn("failed to restore the async signal handler: %m");
        }

        ucs_trace_async("uninstalled signal handler for %s",
                        ucs_signal_names[ucs_global_opts.async_signo]);
    }
}

static ucs_status_t ucs_async_signal_init(ucs_async_context_t *async)
{
    async->signal.block_count = 0;
    async->signal.tid         = ucs_get_tid();
    async->signal.pthread     = pthread_self();
    async->signal.timer       = NULL;
    return UCS_OK;
}

static ucs_status_t ucs_async_signal_add_event_fd(ucs_async_context_t *async,
                                                  int event_fd, int events)
{
    ucs_status_t status;
    pid_t tid;

    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    status = ucs_async_signal_install_handler();
    if (status != UCS_OK) {
        goto err;
    }

    /* Send signal when fd is ready */
    ucs_trace_async("fcntl(F_STSIG, fd=%d, sig=%s)", event_fd,
                    ucs_signal_names[ucs_global_opts.async_signo]);
    if (0 > fcntl(event_fd, F_SETSIG, ucs_global_opts.async_signo)) {
        ucs_error("fcntl F_SETSIG failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_remove_handler;
    }

    /* Send the signal to the desired thread */
    tid = __ucs_async_signal_context_tid(async);
    status = ucs_async_signal_set_fd_owner(tid, event_fd);
    if (status != UCS_OK) {
        goto err_remove_handler;
    }

    /* Allow async events on the file descriptor */
    ucs_trace_async("fcntl(O_ASYNC, fd=%d)", event_fd);
    status = ucs_sys_fcntl_modfl(event_fd, O_ASYNC, 0);
    if (status != UCS_OK) {
        ucs_error("fcntl F_SETFL failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_remove_handler;
    }

    return UCS_OK;

err_remove_handler:
    ucs_async_signal_uninstall_handler();
err:
    return status;
}

static ucs_status_t ucs_async_signal_remove_event_fd(ucs_async_context_t *async, int event_fd)
{
    ucs_status_t status;

    ucs_trace_func("event_fd=%d", event_fd);

    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    ucs_async_signal_allow(0);
    status = ucs_sys_fcntl_modfl(event_fd, 0, O_ASYNC);
    ucs_async_signal_allow(1);

    ucs_async_signal_uninstall_handler();
    return status;
}

static int ucs_async_signal_try_block(ucs_async_context_t *async, int from_async)
{
    if (from_async && (async->signal.block_count > 0)) {
        return 0;
    }

    UCS_ASYNC_SIGNAL_BLOCK(async);
    return 1;
}

static void ucs_async_signal_unblock(ucs_async_context_t *async)
{
    UCS_ASYNC_SIGNAL_UNBLOCK(async);
}

static ucs_status_t ucs_async_signal_timerq_add_timer(ucs_async_signal_timerq_t *timerq,
                                                      ucs_async_signal_timer_info_t *timer_info)
{
    ucs_status_t status;

    status = ucs_timerq_add(&timerq->timerq, timer_info->timer_id, timer_info->interval);
    if (status != UCS_OK) {
        goto err;
    }

    status = ucs_async_signal_sys_timer_set_interval(timerq->sys_timer_id,
                                                 ucs_timerq_min_interval(&timerq->timerq));
    if (status != UCS_OK) {
        goto err_remove;
    }

    return UCS_OK;

err_remove:
    ucs_timerq_remove(&timerq->timerq, timer_info->timer_id);
err:
    return status;
}

static ucs_status_t
ucs_async_signal_alloc_timerq_cb(ucs_async_signal_timerq_t *search, void *arg,
                                 ucs_async_signal_timerq_t **elem)
{
    ucs_async_signal_timerq_t *timerq;
    ucs_status_t status;

    timerq = ucs_malloc(sizeof *timerq, "async signal timer");
    if (timerq == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    status = ucs_async_signal_sys_timer_create(search->tid, &timerq->sys_timer_id);
    if (status != UCS_OK) {
        goto err_free;
    }

    timerq->tid = search->tid;
    ucs_timerq_init(&timerq->timerq);

    status = ucs_async_signal_timerq_add_timer(timerq, arg);
    if (status != UCS_OK) {
        goto err_timer_delete;
    }

    *elem = timerq;
    return UCS_OK;

err_timer_delete:
    ucs_async_signal_sys_timer_delete(timerq->sys_timer_id);
err_free:
    ucs_free(timerq);
err:
    return status;
}

static ucs_status_t
ucs_async_signal_timerq_exists_cb(ucs_async_signal_timerq_t *timerq, void *arg)
{
    return ucs_async_signal_timerq_add_timer(timerq, arg);
}

static ucs_status_t
ucs_async_signal_remove_timer_cb(ucs_async_signal_timerq_t *timerq, void *arg)
{
    ucs_async_signal_timer_info_t *timer_info = arg;

    ucs_timerq_remove(&timerq->timerq, timer_info->timer_id);
    if (ucs_timerq_is_empty(&timerq->timerq)) {
        /* Remove the timer queue because it became empty */
        ucs_async_signal_sys_timer_delete(timerq->sys_timer_id);
        ucs_timerq_cleanup(&timerq->timerq);
        return UCS_OK;
    }

    return UCS_ERR_NO_PROGRESS; /* Do not remove yet */
}

static ucs_status_t ucs_async_signal_add_timer(ucs_async_context_t *async,
                                               int timer_id, ucs_time_t interval)
{
    ucs_async_signal_timer_info_t timer_info;
    ucs_async_signal_timerq_t search;
    ucs_status_t status;

    ucs_trace_func("async=%p interval=%.2fus timer_id=%d",
                   async, ucs_time_to_usec(interval), timer_id);

    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    /* Must install signal handler before arming the timer */
    status = ucs_async_signal_install_handler();
    if (status != UCS_OK) {
        goto err;
    }

    ucs_async_signal_allow(0);
    search.tid          = __ucs_async_signal_context_tid(async);
    timer_info.timer_id = timer_id;
    timer_info.interval = interval;
    status = ucs_hashed_ucs_async_signal_timerq_t_add_if(&ucs_async_signal_global_context.timers,
                                                         &search,
                                                         ucs_async_signal_alloc_timerq_cb,
                                                         ucs_async_signal_timerq_exists_cb,
                                                         &timer_info);
    ucs_async_signal_allow(1);
    if (status != UCS_OK) {
        goto err_uninstall_handler;
    }

    return UCS_OK;

err_uninstall_handler:
    ucs_async_signal_uninstall_handler();
err:
    return status;
}

static ucs_status_t ucs_async_signal_remove_timer(ucs_async_context_t *async,
                                                  int timer_id)
{
    ucs_async_signal_timer_info_t timer_info;
    ucs_async_signal_timerq_t search;
    ucs_async_signal_timerq_t *timerq;
    ucs_status_t status;

    ucs_trace_func("async=%p timer_id=%d", async, timer_id);

    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    search.tid          = __ucs_async_signal_context_tid(async);
    timer_info.timer_id = timer_id;
    ucs_async_signal_allow(0);
    status = ucs_hashed_ucs_async_signal_timerq_t_remove_if(&ucs_async_signal_global_context.timers,
                                                            &search,
                                                            ucs_async_signal_remove_timer_cb,
                                                            &timer_info,
                                                            &timerq);
    ucs_async_signal_allow(1);

    if (status == UCS_OK) {
        free(timerq);
    } else if (status != UCS_ERR_NO_PROGRESS) {
        return status; /* Timer not found */
    }

    /* Maybe timer queue was not removed, but it's still OK */
    ucs_async_signal_uninstall_handler();
    return UCS_OK;
}

static void ucs_async_signal_global_init()
{
    ucs_hashed_ucs_async_signal_timerq_t_init(&ucs_async_signal_global_context.timers);
}

static void ucs_async_signal_global_cleanup()
{
    if (ucs_async_signal_global_context.event_count != 0) {
        ucs_warn("signal handler not removed (%d events remaining)",
                 ucs_async_signal_global_context.event_count);
    }
}

ucs_async_ops_t ucs_async_signal_ops = {
    .init               = ucs_async_signal_global_init,
    .cleanup            = ucs_async_signal_global_cleanup,
    .block              = ucs_async_signal_block_all,
    .unblock            = ucs_async_signal_unblock_all,
    .context_init       = ucs_async_signal_init,
    .context_try_block  = ucs_async_signal_try_block,
    .context_unblock    = ucs_async_signal_unblock,
    .add_event_fd       = ucs_async_signal_add_event_fd,
    .remove_event_fd    = ucs_async_signal_remove_event_fd,
    .add_timer          = ucs_async_signal_add_timer,
    .remove_timer       = ucs_async_signal_remove_timer,
};

