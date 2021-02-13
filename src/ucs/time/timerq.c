/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
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
#include <ucs/sys/sys.h>
#include <stdlib.h>

ucs_mpool_ops_t uct_timerq_mpool_ops = {
    .chunk_alloc   = ucs_mpool_chunk_malloc,
    .chunk_release = ucs_mpool_chunk_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL
};

ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq, const char *name)
{
    ucs_mpool_params_t mp_params;
    ucs_trace_func("timerq=%p", timerq);

    ucs_mpool_params_reset(&mp_params);

    mp_params.elem_size       = sizeof(ucs_timer_t);
    mp_params.elems_per_chunk = ucs_get_page_size() / sizeof(ucs_timer_t);
    mp_params.ops             = &uct_timerq_mpool_ops;
    mp_params.name            = "timerq";

    ucs_ptr_array_locked_init(&timerq->timers, name);
    /* coverity[missing_lock] */
    timerq->min_interval = UCS_TIME_INFINITY;
    return ucs_mpool_init(&mp_params, &timerq->timers_mp);
}

void ucs_timerq_cleanup(ucs_timer_queue_t *timerq)
{
    ucs_trace_func("timerq=%p", timerq);

    ucs_ptr_array_locked_cleanup(&timerq->timers, 0);
    timerq->min_interval = UCS_TIME_INFINITY;
    ucs_mpool_cleanup(&timerq->timers_mp, 0);
}

ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, ucs_time_t interval,
                            unsigned *timer_id_p)
{
    ucs_status_t status;
    ucs_timer_t *ptr;
    unsigned index;

    ucs_trace_func("timerq=%p interval=%.2fus", timerq,
                   ucs_time_to_usec(interval));

    /* Initialize the new timer */
    ucs_ptr_array_locked_acquire_lock(&timerq->timers);
    ptr = (ucs_timer_t *)ucs_mpool_get(&timerq->timers_mp);
    ucs_ptr_array_locked_release_lock(&timerq->timers);
    if (ptr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }

    ptr->expiration = 0; /* will fire the next time sweep is called */
    ptr->interval   = interval;
    index           = ucs_ptr_array_locked_insert(&timerq->timers, (void *)ptr);
    /*
     * TODO: At the moment ucs_ptr_array_locked_insert can't fail, since
     * it returns the index of the new element and the index is unsigned.
     * Once this is fixed, need to handle case when insert failed.
     */
    *timer_id_p = index;

    timerq->min_interval = ucs_min(interval, timerq->min_interval);
    ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);

    status = UCS_OK;
out_unlock:
    return status;
}

ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, unsigned timer_id)
{
    unsigned index;
    ucs_timer_t *timer;
    ucs_time_t interval;

    ucs_trace_func("timerq=%p timer_id=%u", timerq, timer_id);

    if (!ucs_ptr_array_locked_lookup(&timerq->timers, timer_id, (void**)&timer)) {
        return UCS_ERR_NO_ELEM;
    }
    interval = timer->interval;

    ucs_ptr_array_locked_remove(&timerq->timers, timer_id);
    ucs_mpool_put(timer);

    if (ucs_unlikely(ucs_ptr_array_locked_is_empty(&timerq->timers))) {
        timerq->min_interval = UCS_TIME_INFINITY;
        return UCS_OK;
    }

    if (ucs_unlikely(interval > timerq->min_interval)) {
        return UCS_OK;
    }

    /* Update min_interval */
    timerq->min_interval = UCS_TIME_INFINITY;
    ucs_ptr_array_locked_for_each(timer, index, &timerq->timers) {
        timerq->min_interval = ucs_min(timerq->min_interval, timer->interval);
    }

    if (ucs_ptr_array_locked_is_empty(&timerq->timers)) {
        ucs_assert(timerq->min_interval == UCS_TIME_INFINITY);
        ucs_ptr_array_locked_cleanup(&timerq->timers, 1);
    } else {
        ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);
    }

    return UCS_OK;
}
