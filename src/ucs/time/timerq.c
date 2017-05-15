/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "timerq.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>


ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq)
{
    ucs_trace_func("timerq=%p", timerq);

    pthread_spin_init(&timerq->lock, 0);
    timerq->timers       = NULL;
    timerq->num_timers   = 0;
    /* coverity[missing_lock] */
    timerq->min_interval = UCS_TIME_INFINITY;
    return UCS_OK;
}

void ucs_timerq_cleanup(ucs_timer_queue_t *timerq)
{
    ucs_trace_func("timerq=%p", timerq);

    if (timerq->num_timers > 0) {
        ucs_warn("timer queue with %d timers being destroyed", timerq->num_timers);
    }
    ucs_free(timerq->timers);
}

ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, int timer_id,
                            ucs_time_t interval)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p interval=%.2fus timer_id=%d", timerq,
                   ucs_time_to_usec(interval), timer_id);

    pthread_spin_lock(&timerq->lock);

    /* Make sure ID is unique */
    for (ptr = timerq->timers; ptr < timerq->timers + timerq->num_timers; ++ptr) {
        if (ptr->id == timer_id) {
            status = UCS_ERR_ALREADY_EXISTS;
            goto out_unlock;
        }
    }

    /* Resize timer array */
    ptr = ucs_realloc(timerq->timers, (timerq->num_timers + 1) * sizeof(ucs_timer_t),
                      "timerq");
    if (ptr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }
    timerq->timers = ptr;
    ++timerq->num_timers;
    timerq->min_interval = ucs_min(interval, timerq->min_interval);
    ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);

    /* Initialize the new timer */
    ptr = &timerq->timers[timerq->num_timers - 1];
    ptr->expiration = 0; /* will fire the next time sweep is called */
    ptr->interval   = interval;
    ptr->id         = timer_id;

    status = UCS_OK;

out_unlock:
    pthread_spin_unlock(&timerq->lock);
    return status;
}

ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, int timer_id)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p timer_id=%d", timerq, timer_id);

    status = UCS_ERR_NO_ELEM;

    pthread_spin_lock(&timerq->lock);
    timerq->min_interval = UCS_TIME_INFINITY;
    ptr = timerq->timers;
    while (ptr < timerq->timers + timerq->num_timers) {
        if (ptr->id == timer_id) {
            *ptr = timerq->timers[--timerq->num_timers];
            status = UCS_OK;
        } else {
            timerq->min_interval = ucs_min(timerq->min_interval, ptr->interval);
            ++ptr;
        }
    }

    /* TODO realloc - shrink */
    if (timerq->num_timers == 0) {
        ucs_assert(timerq->min_interval == UCS_TIME_INFINITY);
        free(timerq->timers);
        timerq->timers = NULL;
    } else {
        ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);
    }

    pthread_spin_unlock(&timerq->lock);
    return status;
}
