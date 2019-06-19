/**
 * Copyright (C) Hiroyuki Sato. 2019. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "event_set.h"

#include <ucs/debug/memtrack.h>
#include <ucs/debug/log.h>
#include <ucs/debug/assert.h>
#include <ucs/sys/math.h>

#include <sys/epoll.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#define UCS_EVENT_EPOLL_MAX_EVENTS 16

struct ucs_sys_event_set {
    int epfd;
};


static inline int ucs_event_set_map_to_raw_events(int events)
{
    int raw_events = 0;

    if (events & UCS_EVENT_SET_EVREAD) {
         raw_events |= EPOLLIN;
    }
    if (events & UCS_EVENT_SET_EVWRITE) {
         raw_events |= EPOLLOUT;
    }
    return raw_events;
}

static inline int ucs_event_set_map_to_events(int raw_events)
{
    int events = 0;

    if (raw_events & EPOLLIN) {
         events |= UCS_EVENT_SET_EVREAD;
    }
    if (raw_events & EPOLLOUT) {
         events |= UCS_EVENT_SET_EVWRITE;
    }
    return events;
}

ucs_status_t ucs_event_set_create(ucs_sys_event_set_t **event_set_p)
{
    ucs_sys_event_set_t *event_set;
    ucs_status_t status;

    event_set = ucs_malloc(sizeof(ucs_sys_event_set_t), "ucs_sys_event_set");
    if (event_set == NULL) {
        ucs_error("unable to allocate memory ucs_sys_event_set_t object");
        status = UCS_ERR_NO_MEMORY;
        goto out_create;
    }

    /* Create epoll set the thread will wait on */
    event_set->epfd = epoll_create(1);
    if (event_set->epfd < 0) {
        ucs_error("epoll_create() failed: %m");
        status = UCS_ERR_IO_ERROR;
        goto err_free;
    }

    *event_set_p = event_set;
    return UCS_OK;

err_free:
    ucs_free(event_set);
out_create:
    return status;
}

ucs_status_t ucs_event_set_add(ucs_sys_event_set_t *event_set, int event_fd,
                               ucs_event_set_type_t events, void *callback_data)
{
    struct epoll_event raw_event;
    int ret;

    memset(&raw_event, 0, sizeof(raw_event));
    raw_event.events   = ucs_event_set_map_to_raw_events(events);
    raw_event.data.ptr = callback_data;

    ret = epoll_ctl(event_set->epfd, EPOLL_CTL_ADD, event_fd, &raw_event);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, ADD, fd=%d) failed: %m", event_set->epfd,
                  event_fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_mod(ucs_sys_event_set_t *event_set, int event_fd,
                               ucs_event_set_type_t events, void *callback_data)
{
    struct epoll_event raw_event;
    int ret;

    memset(&raw_event, 0, sizeof(raw_event));
    raw_event.events   = ucs_event_set_map_to_raw_events(events);
    raw_event.data.ptr = callback_data;

    ret = epoll_ctl(event_set->epfd, EPOLL_CTL_MOD, event_fd, &raw_event);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, MOD, fd=%d) failed: %m", event_set->epfd,
                  event_fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_del(ucs_sys_event_set_t *event_set, int event_fd)
{
    int ret;

    ret = epoll_ctl(event_set->epfd, EPOLL_CTL_DEL, event_fd, NULL);
    if (ret < 0) {
        ucs_error("epoll_ctl(epfd=%d, DEL, fd=%d) failed: %m", event_set->epfd,
                  event_fd);
        return UCS_ERR_IO_ERROR;
    }

    return UCS_OK;
}

ucs_status_t ucs_event_set_wait(ucs_sys_event_set_t *event_set,
                                unsigned max_events, int timeout_ms,
                                ucs_event_set_handler_t event_set_handler,
                                void *arg, unsigned *read_events)
{
    struct epoll_event ep_events[UCS_EVENT_EPOLL_MAX_EVENTS];
    unsigned max_wait_events = max_events;
    int nready, events, i;
    int timeout_wait_ms = timeout_ms;

    ucs_assert(event_set_handler != NULL);
    ucs_assert(read_events != NULL);

    *read_events = 0;
    do {
	nready = epoll_wait(event_set->epfd, ep_events, max_wait_events,
			    timeout_wait_ms);
        ucs_assert(nready <= max_wait_events);
	if (nready < 0 && errno != EINTR ) {
	    ucs_error("epoll_wait() failed: %m");
	    return UCS_ERR_IO_ERROR;
	}

	ucs_trace_data("epoll_wait(epfd=%d, timeout=%d) returned %d",
		       event_set->epfd, timeout_wait_ms, nready);

	for (i = 0; i < nready; i++) {
	    events = ucs_event_set_map_to_events(ep_events[i].events);
	    event_set_handler(ep_events[i].data.ptr, events, arg);
	}

	*read_events    += nready;
        max_wait_events -= nready;
        /* After 1st iteration, epoll_wait always returns immediately */
        timeout_wait_ms = 0;
    } while (max_wait_events > 0);

    return UCS_OK;
}

void ucs_event_set_cleanup(ucs_sys_event_set_t *event_set)
{
    close(event_set->epfd);
    ucs_free(event_set);
}

ucs_status_t ucs_event_set_fd_get(ucs_sys_event_set_t *event_set, int *fd_p)
{
    if (event_set == NULL) {
         return UCS_ERR_INVALID_PARAM;
    }

    *fd_p = event_set->epfd;
    return UCS_OK;
}
