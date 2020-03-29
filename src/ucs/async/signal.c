/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "signal.h"
#include "async_int.h"

#include <ucs/arch/atomic.h>
#include <ucs/datastruct/list.h>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/sys.h>
#include <signal.h>

#define UCS_SIGNAL_MAX_TIMERQS  64

/*
 * Per-thread system timer and software timer queue. We can dispatch timers only
 * on the same thread which added them.
 */
typedef struct ucs_async_signal_timer {
    pid_t                      tid;          /* Thread ID */
    timer_t                    sys_timer_id; /* System timer ID */
    ucs_timer_queue_t          timerq;       /* Queue of timers for the thread */
} ucs_async_signal_timer_t;


static struct {
    struct sigaction            prev_sighandler;       /* Previous signal handler */
    int                         event_count;           /* Number of events in use */
    pthread_mutex_t             event_lock;            /* Lock for adding/removing events */
    pthread_mutex_t             timers_lock;           /* Lock for timers array */
    ucs_async_signal_timer_t    timers[UCS_SIGNAL_MAX_TIMERQS];/* Array of all threads */
} ucs_async_signal_global_context = {
    .event_count = 0,
    .event_lock  = PTHREAD_MUTEX_INITIALIZER,
    .timers_lock = PTHREAD_MUTEX_INITIALIZER,
    .timers      = {{ .tid = 0 }}
};


/**
 * In signal mode, we allow user to manipulate events only from the same thread.
 * Otherwise, we'd get into big synchronization issues.
 */
#define UCS_ASYNC_SIGNAL_CHECK_THREAD(_async) \
    if (ucs_get_tid() != ucs_async_signal_context_tid(_async)) { \
        ucs_error("cannot manipulate signal async from different thread"); \
        return UCS_ERR_UNREACHABLE; \
    }


/**
 * @return To which thread the async context should deliver events to.
 */
static pid_t ucs_async_signal_context_tid(ucs_async_context_t *async)
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

    ucs_trace_async("fcntl(F_SETOWN_EX, fd=%d, tid=%d)", fd, dest_tid);
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
ucs_async_signal_sys_timer_create(int uid, pid_t tid, timer_t *sys_timer_id)
{
    struct sigevent ev;
    timer_t timer;
    int ret;

    ucs_trace_func("tid=%d", tid);

    /* Create timer signal */
    memset(&ev, 0, sizeof(ev));
    ev.sigev_notify          = SIGEV_THREAD_ID;
    ev.sigev_signo           = ucs_global_opts.async_signo;
    ev.sigev_value.sival_int = uid; /* user parameter to timer */
#if defined(HAVE_SIGEVENT_SIGEV_UN_TID)
    ev._sigev_un._tid        = tid; /* target thread */
#elif defined(HAVE_SIGEVENT_SIGEV_NOTIFY_THREAD_ID)
    ev.sigev_notify_thread_id = tid; /* target thread */
#else
#error "Port me"
#endif
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

    ucs_trace_async("removed system timer %p", sys_timer_id);
}

static ucs_status_t ucs_async_signal_dispatch_timer(int uid)
{
    ucs_async_signal_timer_t *timer = &ucs_async_signal_global_context.timers[uid];

    ucs_assertv_always((uid >= 0) && (uid < UCS_SIGNAL_MAX_TIMERQS), "uid=%d", uid);

    /* No need to take lock - remove operation blocks signals on the same thread */
    if (timer->tid != ucs_get_tid()) {
        return UCS_OK;
    }

    return ucs_async_dispatch_timerq(&timer->timerq, ucs_get_time());
}

static inline int ucs_signal_map_to_events(int si_code)
{
    int events;

    switch (si_code) {
    case POLL_IN:
    case POLL_MSG:
    case POLL_PRI:
        events = UCS_EVENT_SET_EVREAD;
        return events;
    case POLL_OUT:
        events = UCS_EVENT_SET_EVWRITE;
        return events;
    case POLL_HUP:
    case POLL_ERR:
        events = UCS_EVENT_SET_EVERR;
        return events;
    default:
        ucs_warn("unexpected si_code %d", si_code);
        return UCS_ASYNC_EVENT_DUMMY;
    }
}

static void ucs_async_signal_handler(int signo, siginfo_t *siginfo, void *arg)
{
    ucs_assert(signo == ucs_global_opts.async_signo);

    /* Check event code */
    switch (siginfo->si_code) {
    case SI_TIMER:
        ucs_trace_async("timer signal uid=%d", siginfo->si_value.sival_int);
        ucs_async_signal_dispatch_timer(siginfo->si_value.sival_int);
        return;
    case POLL_IN:
    case POLL_OUT:
    case POLL_HUP:
    case POLL_ERR:
    case POLL_MSG:
    case POLL_PRI:
        ucs_trace_async("async signal handler called for fd %d", siginfo->si_fd);
        ucs_async_dispatch_handlers(&siginfo->si_fd, 1,
                                    ucs_signal_map_to_events(siginfo->si_code));
        return;
    default:
        ucs_warn("signal handler called with unexpected event code %d, ignoring",
                 siginfo->si_code);
        return;
    }
}

static void ucs_async_signal_allow(int allow)
{
    sigset_t sig_set;

    ucs_trace_func("enable=%d tid=%d", allow, ucs_get_tid());

    sigemptyset(&sig_set);
    sigaddset(&sig_set, ucs_global_opts.async_signo);
    pthread_sigmask(allow ? SIG_UNBLOCK : SIG_BLOCK, &sig_set, NULL);
}

static void ucs_async_signal_block_all()
{
    pthread_mutex_lock(&ucs_async_signal_global_context.event_lock);
    if (ucs_async_signal_global_context.event_count > 0) {
        ucs_async_signal_allow(0);
    }
    pthread_mutex_unlock(&ucs_async_signal_global_context.event_lock);
}

static void ucs_async_signal_unblock_all()
{
    pthread_mutex_lock(&ucs_async_signal_global_context.event_lock);
    if (ucs_async_signal_global_context.event_count > 0) {
        ucs_async_signal_allow(1);
    }
    pthread_mutex_unlock(&ucs_async_signal_global_context.event_lock);
}

static ucs_status_t ucs_async_signal_install_handler()
{
    struct sigaction new_action;
    int ret;

    ucs_trace_func("event_count=%d", ucs_async_signal_global_context.event_count);

    pthread_mutex_lock(&ucs_async_signal_global_context.event_lock);
    if (ucs_async_signal_global_context.event_count == 0) {
        /* Set our signal handler */
        new_action.sa_sigaction = ucs_async_signal_handler;
        sigemptyset(&new_action.sa_mask);
        new_action.sa_flags    = SA_RESTART|SA_SIGINFO;
#if HAVE_SIGACTION_SA_RESTORER
        new_action.sa_restorer = NULL;
#endif
        ret = sigaction(ucs_global_opts.async_signo, &new_action,
                        &ucs_async_signal_global_context.prev_sighandler);
        if (ret < 0) {
            ucs_error("failed to set a handler for signal %d: %m",
                      ucs_global_opts.async_signo);
            pthread_mutex_unlock(&ucs_async_signal_global_context.event_lock);
            return UCS_ERR_INVALID_PARAM;
        }

        ucs_trace_async("installed signal handler for %s",
                        ucs_signal_names[ucs_global_opts.async_signo]);
    }
    ++ucs_async_signal_global_context.event_count;
    pthread_mutex_unlock(&ucs_async_signal_global_context.event_lock);

    return UCS_OK;
}

static void fatal_sighandler(int signo, siginfo_t *siginfo, void *arg)
{
    ucs_fatal("got timer signal");
}

static void ucs_async_signal_uninstall_handler()
{
    struct sigaction new_action;
    int ret;

    ucs_trace_func("event_count=%d", ucs_async_signal_global_context.event_count);

    pthread_mutex_lock(&ucs_async_signal_global_context.event_lock);
    if (--ucs_async_signal_global_context.event_count == 0) {
        new_action = ucs_async_signal_global_context.prev_sighandler;
        new_action.sa_sigaction = fatal_sighandler;
        ret = sigaction(ucs_global_opts.async_signo, &new_action, NULL);
        if (ret < 0) {
            ucs_warn("failed to restore the async signal handler: %m");
        }

        ucs_trace_async("uninstalled signal handler for %s",
                        ucs_signal_names[ucs_global_opts.async_signo]);
    }
    pthread_mutex_unlock(&ucs_async_signal_global_context.event_lock);
}

static ucs_status_t ucs_async_signal_init(ucs_async_context_t *async)
{
    async->signal.block_count = 0;
    async->signal.tid         = ucs_get_tid();
    async->signal.pthread     = pthread_self();
    async->signal.timer       = NULL;
    return UCS_OK;
}

static void ucs_async_signal_cleanup(ucs_async_context_t *async)
{
    if (async->signal.block_count > 0) {
        ucs_warn("destroying async signal context with block_count %d",
                 async->signal.block_count);
    }
}

static ucs_status_t ucs_async_signal_modify_event_fd(ucs_async_context_t *async,
                                                     int event_fd, int events)
{
    ucs_status_t status;
    int add, rm;

    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    if (events) {
        add = O_ASYNC; /* Enable notifications */
        rm  = 0;
    } else {
        add = 0;       /* Disable notifications */
        rm  = O_ASYNC;
    }

    ucs_trace_async("fcntl(fd=%d, add=0x%x, remove=0x%x)", event_fd, add, rm);
    status = ucs_sys_fcntl_modfl(event_fd, add, rm);
    if (status != UCS_OK) {
        ucs_error("fcntl F_SETFL failed: %m");
        return UCS_ERR_IO_ERROR;
    }

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
    tid = ucs_async_signal_context_tid(async);
    status = ucs_async_signal_set_fd_owner(tid, event_fd);
    if (status != UCS_OK) {
        goto err_remove_handler;
    }

    /* Set async events on the file descriptor */
    status = ucs_async_signal_modify_event_fd(async, event_fd, events);
    if (status != UCS_OK) {
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

static int ucs_async_signal_try_block(ucs_async_context_t *async)
{
    if (async->signal.block_count > 0) {
        return 0;
    }

    UCS_ASYNC_SIGNAL_BLOCK(async);
    return 1;
}

static void ucs_async_signal_unblock(ucs_async_context_t *async)
{
    UCS_ASYNC_SIGNAL_UNBLOCK(async);
}

static void ucs_timer_reset_if_empty(ucs_async_signal_timer_t *timer)
{
    if (ucs_timerq_is_empty(&timer->timerq)) {
        ucs_async_signal_sys_timer_delete(timer->sys_timer_id);
        ucs_timerq_cleanup(&timer->timerq);
        timer->tid = 0;
    }
}

/* Add a timer, possible initializing the timerq */
static ucs_status_t
ucs_async_signal_timerq_add_timer(ucs_async_signal_timer_t *timer, int tid,
                                  ucs_time_t interval, int *timer_id_p)
{
    ucs_time_t sys_interval;
    ucs_status_t status;
    int uid;

    if (timer->tid == 0) {
        timer->tid = tid;
        ucs_timerq_init(&timer->timerq, "async_signal");

        uid = (timer - ucs_async_signal_global_context.timers);
        status = ucs_async_signal_sys_timer_create(uid, timer->tid,
                                                   &timer->sys_timer_id);
        if (status != UCS_OK) {
            goto err;
        }

    }

    status = ucs_timerq_add(&timer->timerq, interval, timer_id_p);
    if (status != UCS_OK) {
        goto err;
    }

    sys_interval = ucs_timerq_min_interval(&timer->timerq);
    status = ucs_async_signal_sys_timer_set_interval(timer->sys_timer_id,
                                                     sys_interval);
    if (status != UCS_OK) {
        goto err_remove;
    }

    return UCS_OK;

err_remove:
    ucs_timerq_remove(&timer->timerq, *timer_id_p);
err:
    ucs_timer_reset_if_empty(timer);
    return status;
}

/* Remove a timer, possible resetting the timerq */
static ucs_status_t
ucs_async_signal_timerq_remove_timer(ucs_async_signal_timer_t *timer,
                                     int timer_id)
{
    ucs_status_t status;

    status = ucs_timerq_remove(&timer->timerq, timer_id);
    if (status != UCS_OK) {
        return status;
    }

    ucs_timer_reset_if_empty(timer);
    return UCS_OK;
}

static ucs_async_signal_timer_t *ucs_async_signal_find_timer(pid_t tid)
{
    ucs_async_signal_timer_t *timer;

    for (timer = ucs_async_signal_global_context.timers;
         timer < &ucs_async_signal_global_context.timers[UCS_SIGNAL_MAX_TIMERQS];
         ++timer)
    {
        if (timer->tid == tid) {
            return timer;
        }
    }

    return NULL;
}

static ucs_status_t ucs_async_signal_add_timer(ucs_async_context_t *async,
                                               ucs_time_t interval, int *timer_id_p)
{
    ucs_async_signal_timer_t *timer;
    ucs_status_t status;
    pid_t tid;

    ucs_trace_func("async=%p interval=%.2fus",
                   async, ucs_time_to_usec(interval));
    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    /* Must install signal handler before arming the timer */
    status = ucs_async_signal_install_handler();
    if (status != UCS_OK) {
        goto err;
    }

    ucs_async_signal_allow(0);
    pthread_mutex_lock(&ucs_async_signal_global_context.timers_lock);

    /* Find existing or available timer queue for the current thread */
    tid    = ucs_async_signal_context_tid(async);
    timer  = ucs_async_signal_find_timer(tid);
    if (timer == NULL) {
        timer = ucs_async_signal_find_timer(0); /* Search for free slot */
    }

    if (timer == NULL) {
        status = UCS_ERR_EXCEEDS_LIMIT;
    } else {
        status = ucs_async_signal_timerq_add_timer(timer, tid, interval, timer_id_p);
    }

    pthread_mutex_unlock(&ucs_async_signal_global_context.timers_lock);
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
    ucs_async_signal_timer_t *timer;
    ucs_status_t status;

    ucs_trace_func("async=%p timer_id=%d", async, timer_id);

    UCS_ASYNC_SIGNAL_CHECK_THREAD(async);

    ucs_async_signal_allow(0);
    pthread_mutex_lock(&ucs_async_signal_global_context.timers_lock);

    timer = ucs_async_signal_find_timer(ucs_async_signal_context_tid(async));
    if (timer == NULL) {
        status = UCS_ERR_NO_ELEM;
    } else {
        status = ucs_async_signal_timerq_remove_timer(timer, timer_id);
    }

    pthread_mutex_unlock(&ucs_async_signal_global_context.timers_lock);
    ucs_async_signal_allow(1);

    if (status == UCS_OK) {
        ucs_async_signal_uninstall_handler();
    }
    return status;
}

static void ucs_async_signal_global_init()
{
    pthread_mutex_init(&ucs_async_signal_global_context.timers_lock, NULL);
}

static void ucs_async_signal_global_cleanup()
{
    if (ucs_async_signal_global_context.event_count != 0) {
        ucs_warn("signal handler not removed (%d events remaining)",
                 ucs_async_signal_global_context.event_count);
    }
    pthread_mutex_destroy(&ucs_async_signal_global_context.timers_lock);
}

ucs_async_ops_t ucs_async_signal_ops = {
    .init               = ucs_async_signal_global_init,
    .cleanup            = ucs_async_signal_global_cleanup,
    .block              = ucs_async_signal_block_all,
    .unblock            = ucs_async_signal_unblock_all,
    .context_init       = ucs_async_signal_init,
    .context_cleanup    = ucs_async_signal_cleanup,
    .context_try_block  = ucs_async_signal_try_block,
    .context_unblock    = ucs_async_signal_unblock,
    .add_event_fd       = ucs_async_signal_add_event_fd,
    .remove_event_fd    = ucs_async_signal_remove_event_fd,
    .modify_event_fd    = ucs_async_signal_modify_event_fd,
    .add_timer          = ucs_async_signal_add_timer,
    .remove_timer       = ucs_async_signal_remove_timer,
};

