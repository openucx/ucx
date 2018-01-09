/**
* Copyright (C) Mellanox Technologies Ltd. 2012-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include <ucs/time/timer_wheel.h>

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/math.h>


ucs_status_t ucs_twheel_init(ucs_twheel_t *twheel, ucs_time_t resolution,
                             ucs_time_t current_time)
{
    unsigned i;

    twheel->res         = ucs_roundup_pow2(resolution);
    twheel->res_order   = (unsigned) ucs_log2(twheel->res);
    twheel->num_slots   = 1024;
    twheel->current     = 0;
    twheel->now         = current_time;
    twheel->wheel       = ucs_malloc(sizeof(*twheel->wheel) * twheel->num_slots,
                                     "twheel");

    for (i = 0; i < twheel->num_slots; i++) {
        ucs_list_head_init(&twheel->wheel[i]);
    }

    ucs_debug("high res timer created log=%d resolution=%lf usec wanted: %lf usec",
              twheel->res_order, ucs_time_to_usec(twheel->res), ucs_time_to_usec(resolution));
    return UCS_OK;
}

void ucs_twheel_cleanup(ucs_twheel_t *twheel)
{
    ucs_free(twheel->wheel);
}

ucs_status_t ucs_wtimer_init(ucs_wtimer_t *t, ucs_twheel_callback_t cb)
{
    t->cb        = cb;
    t->is_active = 0;
    return UCS_OK;
}

void __ucs_wtimer_add(ucs_twheel_t *t, ucs_wtimer_t *timer, ucs_time_t delta)
{
    uint64_t slot;

    timer->is_active = 1;
    slot = delta>>t->res_order;
    if (ucs_unlikely(slot == 0)) {
        /* nothing really wrong with adding timer to the current slot. However
         * we want to guard against the case we spend to much time in hi res
         * timer processing */
        ucs_fatal("Timer resolution is too low. Min resolution %lf usec, wanted %lf usec",
                ucs_time_to_usec(t->res), ucs_time_to_usec(delta));
    }
    ucs_assert(slot > 0);

    if (ucs_unlikely(slot >= t->num_slots)) {
        slot = t->num_slots - 1;
    }

    slot = (t->current + slot) % t->num_slots;
    ucs_assert(slot != t->current);

    ucs_list_add_tail(&t->wheel[slot], &timer->list);
}

void __ucs_twheel_sweep(ucs_twheel_t *t, ucs_time_t current_time)
{
    ucs_wtimer_t *timer;
    uint64_t slot;

    slot   = (current_time - t->now) >> t->res_order;
    t->now = current_time;

    if (ucs_unlikely(slot >= t->num_slots)) {
        slot = t->num_slots - 1;
    }

    slot = (t->current + slot) % t->num_slots;

    for (; t->current != slot; t->current = (t->current+1) % t->num_slots) {
        while (!ucs_list_is_empty(&t->wheel[t->current])) {
            timer = ucs_list_extract_head(&t->wheel[t->current], ucs_wtimer_t, list);
            timer->is_active = 0;
            timer->cb(timer);
        }
    }
}
