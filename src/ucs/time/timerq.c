/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#include "timerq.h"

#include <ucs/debug/log.h>
#include <ucs/sys/math.h>

static void ucs_timer_reschedule(ucs_timer_queue_t *timerq, ucs_timer_t *timer)
{
    timerq->expiration = ucs_min(timerq->expiration, timer->expiration);
}

ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq)
{
    ucs_trace_func("timerq=%p", timerq);

    ucs_list_head_init(&timerq->timers);
    timerq->expiration = UCS_TIME_INFINITY;
    return UCS_OK;
}

void ucs_timerq_cleanup(ucs_timer_queue_t *timerq)
{
    ucs_timer_t *timer;

    ucs_trace_func("timerq=%p", timerq);

    while (!ucs_list_is_empty(&timerq->timers)) {
        timer = ucs_list_extract_head(&timerq->timers, ucs_timer_t, list);
        ucs_warn("removing timer cb=%p", timer->cb);
        ucs_free(timer);
    }
}

void ucs_timerq_sweep_internal(ucs_timer_queue_t *timerq, ucs_time_t current_time)
{
    ucs_timer_t *timer, *tmp;

    timerq->expiration = UCS_TIME_INFINITY;
    ucs_list_for_each_safe(timer, tmp, &timerq->timers, list) {
        if (current_time >= timer->expiration) {
            ucs_invoke_callback(timer->cb);
            timer->expiration = current_time + timer->interval;
            ucs_timer_reschedule(timerq, timer);
        }
        timerq->expiration = ucs_min(timerq->expiration, timer->expiration);
    }
}

ucs_status_t ucs_timer_add(ucs_timer_queue_t *timerq, ucs_callback_t *callback,
                          ucs_time_t interval)
{
    ucs_timer_t *timer;

    timer = ucs_malloc(sizeof *timer, "timer");
    if (timer == NULL) {
        ucs_error("failed to allocate timer");
        return UCS_ERR_NO_MEMORY;
    }

    timer->cb = callback;
    timer->interval = interval;
    timer->expiration = 0; /* will fire the next time sweep is called */
    ucs_list_add_tail(&timerq->timers, &timer->list);
    ucs_timer_reschedule(timerq, timer);
    return UCS_OK;
}

void ucs_timer_remove(ucs_timer_queue_t *timerq, ucs_callback_t *callback)
{
    ucs_timer_t *timer, *tmp;

    ucs_list_for_each_safe(timer, tmp, &timerq->timers, list) {
        if (timer->cb == callback) {
            ucs_list_del(&timer->list);
            ucs_free(timer);
        }
    }
}
