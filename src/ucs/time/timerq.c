/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "timerq.h"

#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>
#include <stdlib.h>


ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq, int flags)
{
    ucs_trace_func("timerq=%p", timerq);

    ucs_spinlock_init(&timerq->lock);
    timerq->timers       = NULL;
    timerq->used_cnt     = 0;
    timerq->alloc_cnt    = 0;
    /* coverity[missing_lock] */
    timerq->min_interval = UCS_TIME_INFINITY;
    timerq->flags        = flags;
    return UCS_OK;
}

void ucs_timerq_cleanup(ucs_timer_queue_t *timerq)
{
    ucs_status_t status;

    ucs_trace_func("timerq=%p", timerq);

    if (timerq->used_cnt > 0) {
        ucs_warn("timer queue with %d timers being destroyed", timerq->used_cnt);
    }
    ucs_free(timerq->timers);

    status = ucs_spinlock_destroy(&timerq->lock);
    if (status != UCS_OK) {
        ucs_warn("ucs_spinlock_destroy() failed (%d)", status);
    }
}

ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, uint64_t timer_id,
                            ucs_time_t interval, unsigned *timer_index)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p interval=%.2fus timer_id=%lu", timerq,
                   ucs_time_to_usec(interval), timer_id);

    ucs_spin_lock(&timerq->lock);

#ifdef ENABLE_ASSERT
    /* Enforce ID uniqueness */
    if (ucs_unlikely(timerq->flags & UCS_TIMERQ_FLAG_UNIQUE_IDS)) {
        for (ptr = timerq->timers; ptr < timerq->timers + timerq->used_cnt; ++ptr) {
            ucs_assert(ptr->id != timer_id);
        }
    }
#endif

    /* Resize timer array */
    if (ucs_unlikely(timerq->used_cnt == timerq->alloc_cnt)) {
        timerq->alloc_cnt = (timerq->alloc_cnt + 1) * 2;
        ptr = ucs_realloc(timerq->timers, timerq->alloc_cnt * sizeof(ucs_timer_t),
                "timerq");
        if (ptr == NULL) {
            status = UCS_ERR_NO_MEMORY;
            goto out_unlock;
        }
        timerq->timers = ptr;
    }

    timerq->min_interval = ucs_min(interval, timerq->min_interval);
    ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);

    /* Initialize the new timer */
    ptr = &timerq->timers[timerq->used_cnt];
    ptr->expiration = 0; /* will fire the next time sweep is called */
    ptr->interval   = interval;
    ptr->id         = timer_id;
    if (timer_index) {
        *timer_index = timerq->used_cnt;
    }
    timerq->used_cnt++;
    status = UCS_OK;

out_unlock:
    ucs_spin_unlock(&timerq->lock);
    return status;
}

ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, uint64_t timer_id,
                               unsigned timer_index_hint)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p timer_id=%lu", timerq, timer_id);

    status = UCS_ERR_NO_ELEM;

    ucs_spin_lock(&timerq->lock);

    if (timer_index_hint < timerq->used_cnt) {
        ptr = &timerq->timers[timer_index_hint];
        if (ptr->id == timer_id) {
            *ptr = timerq->timers[--timerq->used_cnt];
            status = UCS_OK;
            goto removal_done;
        }
    }

    timerq->min_interval = UCS_TIME_INFINITY;
    ptr = timerq->timers;
    while (ptr < timerq->timers + timerq->used_cnt) {
        if (ptr->id == timer_id) {
            *ptr = timerq->timers[--timerq->used_cnt];
            status = UCS_OK;
        } else {
            timerq->min_interval = ucs_min(timerq->min_interval, ptr->interval);
            ++ptr;
        }
    }

    if (ucs_unlikely(timerq->used_cnt == 0)) {
        free(timerq->timers);
        timerq->timers = NULL;
        timerq->alloc_cnt = 0;
        timerq->min_interval = UCS_TIME_INFINITY;
    } else {
        ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);
    }

removal_done:
    ucs_spin_unlock(&timerq->lock);
    return status;
}

ucs_status_t ucs_timerq_modify(ucs_timer_queue_t *timerq, uint64_t timer_id,
                               ucs_time_t new_interval, unsigned timer_index_hint)
{
    ucs_status_t status;
    ucs_timer_t *ptr;

    ucs_trace_func("timerq=%p interval=%.2fus timer_id=%lu", timerq,
                   ucs_time_to_usec(new_interval), timer_id);

    status = UCS_ERR_NO_ELEM;

    ucs_spin_lock(&timerq->lock);

    if (timer_index_hint < timerq->used_cnt) {
        ptr = &timerq->timers[timer_index_hint];
        if (ptr->id == timer_id) {
            ptr->expiration += new_interval - ptr->interval;
            ptr->interval    = new_interval;
            status           = UCS_OK;
            if (ucs_likely(timerq->flags & UCS_TIMERQ_FLAG_UNIQUE_IDS)) {
                goto modification_done;
            }
        }
    }

    ptr = timerq->timers;
    while (ptr < timerq->timers + timerq->used_cnt) {
        if (ptr->id == timer_id) {
            ptr->expiration += new_interval - ptr->interval;
            ptr->interval    = new_interval;
            status           = UCS_OK;
            if (timerq->flags & UCS_TIMERQ_FLAG_UNIQUE_IDS) {
                goto modification_done;
            }
        }
        ptr++;
    }

modification_done:
    ucs_spin_unlock(&timerq->lock);
    return status;
}
