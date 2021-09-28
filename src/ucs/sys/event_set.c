/**
 * Copyright (C) Hiroyuki Sato. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "event_set.h"

#include <ucs/debug/memtrack_int.h>
#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>
#include <ucs/sys/compiler.h>

#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/epoll.h>


enum {
    UCS_SYS_EVENT_SET_EXTERNAL_EVENT_FD = UCS_BIT(0),
};

struct ucs_sys_event_set {
    int      event_fd;
    unsigned flags;
};

const unsigned ucs_sys_event_set_max_wait_events =
    UCS_ALLOCA_MAX_SIZE / sizeof(struct epoll_event);


static inline int ucs_event_set_map_to_raw_events(ucs_event_set_types_t events)
{
    int raw_events = 0;

    if (events & UCS_EVENT_SET_EVREAD) {
         raw_events |= EPOLLIN;
    }
    if (events & UCS_EVENT_SET_EVWRITE) {
         raw_events |= EPOLLOUT;
    }
    if (events & UCS_EVENT_SET_EVERR) {
         raw_events |= EPOLLERR;
    }
    if (events & UCS_EVENT_SET_EDGE_TRIGGERED) {
        raw_events  |= EPOLLET;
    }
    return raw_events;
}

static inline int ucs_event_set_map_to_events(int raw_events)
{
    ucs_event_set_types_t events = 0;

    if (raw_events & EPOLLIN) {
         events |= UCS_EVENT_SET_EVREAD;
    }
    if (raw_events & EPOLLOUT) {
         events |= UCS_EVENT_SET_EVWRITE;
    }
    if (raw_events & EPOLLERR) {
         events |= UCS_EVENT_SET_EVERR;
    }
    if (raw_events & EPOLLET) {
        events  |= UCS_EVENT_SET_EDGE_TRIGGERED;
    }
    return events;
}

static ucs_sys_event_set_t *ucs_event_set_alloc(int event_fd, unsigned flags)
{
    ucs_sys_event_set_t *event_set;

    event_set = ucs_malloc(sizeof(ucs_sys_event_set_t), "ucs_sys_event_set");
    if (event_set == NULL) {
        ucs_error("unable to allocate memory ucs_sys_event_set_t object");
        return NULL;
    }

    event_set->flags    = flags;
    event_set->event_fd = event_fd;
    return event_set;
}

ucs_status_t ucs_event_set_create_from_fd(ucs_sys_event_set_t **event_set_p,
                                          int event_fd)
{
    *event_set_p = ucs_event_set_alloc(event_fd,
                                       UCS_SYS_EVENT_SET_EXTERNAL_EVENT_FD);
    if (*event_set_p == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_create(ucs_sys_event_set_t **event_set_p)
{
    ucs_status_t status;
    int event_fd;

    /* Create epoll set the thread will wait on */
    event_fd = epoll_create(1);
    if (event_fd < 0) {
        ucs_error("epoll_create() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    *event_set_p = ucs_event_set_alloc(event_fd, 0);
    if (*event_set_p == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto err_close_event_fd;
    }

    return UCS_OK;

err_close_event_fd:
    close(event_fd);
    return status;
}

ucs_status_t ucs_event_set_add(ucs_sys_event_set_t *event_set, int fd,
                               ucs_event_set_types_t events,
                               void *callback_data)
{
    struct epoll_event raw_event;
    int ret;

    memset(&raw_event, 0, sizeof(raw_event));
    raw_event.events   = ucs_event_set_map_to_raw_events(events);
    raw_event.data.ptr = callback_data;

    ret = epoll_ctl(event_set->event_fd, EPOLL_CTL_ADD, fd, &raw_event);
    if (ret < 0) {
        ucs_error("epoll_ctl(event_fd=%d, ADD, fd=%d) failed: %m",
                  event_set->event_fd, fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_mod(ucs_sys_event_set_t *event_set, int fd,
                               ucs_event_set_types_t events,
                               void *callback_data)
{
    struct epoll_event raw_event;
    int ret;

    memset(&raw_event, 0, sizeof(raw_event));
    raw_event.events   = ucs_event_set_map_to_raw_events(events);
    raw_event.data.ptr = callback_data;

    ret = epoll_ctl(event_set->event_fd, EPOLL_CTL_MOD, fd, &raw_event);
    if (ret < 0) {
        ucs_error("epoll_ctl(event_fd=%d, MOD, fd=%d) failed: %m",
                  event_set->event_fd, fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_del(ucs_sys_event_set_t *event_set, int fd)
{
    int ret;

    ret = epoll_ctl(event_set->event_fd, EPOLL_CTL_DEL, fd, NULL);
    if (ret < 0) {
        ucs_error("epoll_ctl(event_fd=%d, DEL, fd=%d) failed: %m",
                  event_set->event_fd, fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_wait(ucs_sys_event_set_t *event_set,
                                unsigned *num_events, int timeout_ms,
                                ucs_event_set_handler_t event_set_handler,
                                void *arg)
{
    struct epoll_event *events;
    int nready, i, io_events;

    ucs_assert(event_set_handler != NULL);
    ucs_assert(num_events != NULL);
    ucs_assert(*num_events <= ucs_sys_event_set_max_wait_events);

    events = ucs_alloca(sizeof(*events) * *num_events);

    nready = epoll_wait(event_set->event_fd, events, *num_events, timeout_ms);
    if (ucs_unlikely(nready < 0)) {
        *num_events = 0;
        if (errno == EINTR) {
            return UCS_INPROGRESS;
        }
        ucs_error("epoll_wait() failed: %m");
        return UCS_ERR_IO_ERROR;
    }

    ucs_assert(nready <= *num_events);
    ucs_trace_poll("epoll_wait(event_fd=%d, num_events=%u, timeout=%d) "
                   "returned %u",
                   event_set->event_fd, *num_events, timeout_ms, nready);

    for (i = 0; i < nready; i++) {
        io_events = ucs_event_set_map_to_events(events[i].events);
        event_set_handler(events[i].data.ptr, io_events, arg);
    }

    *num_events = nready;
    return UCS_OK;
}

void ucs_event_set_cleanup(ucs_sys_event_set_t *event_set)
{
    if (!(event_set->flags & UCS_SYS_EVENT_SET_EXTERNAL_EVENT_FD)) {
        close(event_set->event_fd);
    }
    ucs_free(event_set);
}

ucs_status_t ucs_event_set_fd_get(ucs_sys_event_set_t *event_set,
                                  int *event_fd_p)
{
    ucs_assert(event_set != NULL);
    *event_fd_p = event_set->event_fd;
    return UCS_OK;
}
