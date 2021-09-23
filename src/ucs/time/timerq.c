/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "timerq.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>
#include <ucs/sys/math.h>
#include <stdlib.h>


ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq)
{
    ucs_trace_func("timerq=%p", timerq);

    ucs_recursive_spinlock_init(&timerq->lock, 0);
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
    ucs_recursive_spinlock_destroy(&timerq->lock);
}

ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, int timer_id,
                            ucs_time_t interval)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p interval=%.2fus timer_id=%d", timerq,
                   ucs_time_to_usec(interval), timer_id);

    ucs_recursive_spin_lock(&timerq->lock);

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
    ucs_recursive_spin_unlock(&timerq->lock);
    return status;
}

ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, int timer_id)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p timer_id=%d", timerq, timer_id);

    status = UCS_ERR_NO_ELEM;

    ucs_recursive_spin_lock(&timerq->lock);
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
        ucs_free(timerq->timers);
        timerq->timers = NULL;
    } else {
        ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);
    }

    ucs_recursive_spin_unlock(&timerq->lock);
    return status;
}
