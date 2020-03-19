/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "timerq.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
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
    ucs_status_t status;

    ucs_trace_func("timerq=%p", timerq);

    if (timerq->num_timers > 0) {
        ucs_warn("timer queue with %d timers being destroyed", timerq->num_timers);
    }
    ucs_free(timerq->timers);

    status = ucs_recursive_spinlock_destroy(&timerq->lock);
    if (status != UCS_OK) {
        ucs_warn("ucs_recursive_spinlock_destroy() failed (%d)", status);
    }
}

ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, ucs_time_t interval,
                            int *timer_id_p)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p interval=%.2fus", timerq,
                   ucs_time_to_usec(interval));

    ucs_recursive_spin_lock(&timerq->lock);

    /* Resize timer array */
    ptr = ucs_realloc(timerq->timers, (timerq->num_timers + 1) * sizeof(ucs_timer_t),
                      "timerq");
    if (ptr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }
    timerq->timers = ptr;
    *timer_id_p = timerq->num_timers++;
    timerq->min_interval = ucs_min(interval, timerq->min_interval);
    ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);

    /* Initialize the new timer */
    ptr = &timerq->timers[timerq->num_timers - 1];
    ptr->expiration = 0; /* will fire the next time sweep is called */
    ptr->interval   = interval;
    ptr->id         = *timer_id_p;

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
        free(timerq->timers);
        timerq->timers = NULL;
    } else {
        ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);
    }

    ucs_recursive_spin_unlock(&timerq->lock);
    return status;
}
