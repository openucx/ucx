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
#include <ucs/debug/memtrack.h>
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
    ucs_status_t status;
    ucs_trace_func("timerq=%p", timerq);

    ucs_ptr_array_locked_init(&timerq->timers, name);
    /* coverity[missing_lock] */
    timerq->min_interval = UCS_TIME_INFINITY;
    status = ucs_mpool_init(&timerq->timers_mp, 0, 
                            sizeof(ucs_timer_t), 0, UCS_SYS_CACHE_LINE_SIZE,
                            ucs_get_page_size() / sizeof(ucs_timer_t), UINT_MAX,
                            &uct_timerq_mpool_ops, "timerq");
    return status;
}

void ucs_timerq_cleanup(ucs_timer_queue_t *timerq)
{
    ucs_trace_func("timerq=%p", timerq);

    ucs_ptr_array_locked_cleanup(&timerq->timers);
    timerq->min_interval = UCS_TIME_INFINITY;
    ucs_mpool_cleanup(&timerq->timers_mp, 1);
}

ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, ucs_time_t interval,
                            int *timer_id_p)
{
    ucs_status_t status;
    ucs_timer_t *ptr;
    unsigned index;

    ucs_trace_func("timerq=%p interval=%.2fus", timerq,
                   ucs_time_to_usec(interval));

    /* Initialize the new timer */
    ptr = (ucs_timer_t *)ucs_mpool_get(&timerq->timers_mp);
    if (ptr == NULL) {
        status = UCS_ERR_NO_MEMORY;
        goto out_unlock;
    }

    ptr->expiration = 0; /* will fire the next time sweep is called */
    ptr->interval   = interval;
    index = ucs_ptr_array_locked_insert(&timerq->timers, (void *)ptr);
    /*
     * TODO: At the moment ucs_ptr_array_locked_insert can't fail, since
     * it returns the index of the new element and the index is unsigned.
     * Once this is fixed, need to handle case when insert failed.
     */
    *timer_id_p = index;
    ptr->id     = index;

    timerq->min_interval = ucs_min(interval, timerq->min_interval);
    ucs_assert(timerq->min_interval != UCS_TIME_INFINITY);

    status = UCS_OK;
out_unlock:
    return status;
}

ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, int timer_id)
{
    ucs_status_t status;
    ucs_timer_t *timer = NULL;
    void *ptr;
    int _index = 0;

    ucs_trace_func("timerq=%p timer_id=%d", timerq, timer_id);

    if (!ucs_ptr_array_locked_lookup(&timerq->timers, timer_id, &ptr)) {
        status = UCS_ERR_NO_ELEM;
        goto out_no_elem;
    }
    /* Update min_interval */
    timerq->min_interval = UCS_TIME_INFINITY;
    ucs_ptr_array_locked_for_each(timer, _index, &timerq->timers) {
        if (timer->id != timer_id) {
            timerq->min_interval = ucs_min(timerq->min_interval, timer->interval);
        } else {
            ucs_mpool_put((void *)timer);
        }
    }
    ucs_ptr_array_locked_remove(&timerq->timers, timer_id);

    status = UCS_OK;
out_no_elem:
    return status;
}
