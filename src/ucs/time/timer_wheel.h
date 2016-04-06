/**
* Copyright (C) Mellanox Technologies Ltd. 2012-2013.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_WHEEL_H
#define UCS_WHEEL_H

#include <ucs/datastruct/list.h>
#include <ucs/time/time.h>
#include <ucs/debug/log.h>


/* Forward declarations */
typedef struct ucs_wtimer       ucs_wtimer_t;
typedef struct ucs_timer_wheel  ucs_twheel_t;


/**
 * Timer wheel callback
 */
typedef void (*ucs_twheel_callback_t)(ucs_wtimer_t *self);


/**
 * UCS high resolution timer.
 */
struct ucs_wtimer {
    ucs_twheel_callback_t  cb;         /* User callback */
    ucs_list_link_t        list;       /* Link in the list of timers */
    int                    is_active;
};


struct ucs_timer_wheel {
    ucs_time_t             res;
    ucs_time_t             now;        /* when wheel was last updated */
    uint64_t               current;
    ucs_list_link_t        *wheel;
    unsigned               res_order;
    unsigned               num_slots;
};


/**
 * Initialize the timer queue.
 *
 * @param twheel        Timer queue to initialize.
 * @param resolution    Timer resolution. Timer wheel range is from now to now + UCS_TWHEEL_NSLOTS * res
 * @param current_time  Current time to initialize the timer with.
 */
ucs_status_t ucs_twheel_init(ucs_twheel_t *twheel, ucs_time_t resolution,
                             ucs_time_t current_time);


/**
 * Cleanup the timer queue.
 *
 * @param twheel    Timer queue to clean up.
 */
void ucs_twheel_cleanup(ucs_twheel_t *twheel);


/**
 * Initialize wheel timer
 *
 * @param cb          Callback to call
 */
ucs_status_t ucs_wtimer_init(ucs_wtimer_t *t, ucs_twheel_callback_t cb);


/**
 * Go through the timers in the timer queue, dispatch expired timers.
 *
 * @param twheel        Timer wheel to dispatch timers on.
 * @param current_time  Current time to dispatch the timers for.
 *
 * @note Timers which expired between calls to this function will also be dispatched.
 * @note There is no guarantee on the order of dispatching.
 */
void __ucs_twheel_sweep(ucs_twheel_t *t, ucs_time_t current_time);
static inline void ucs_twheel_sweep(ucs_twheel_t *t, ucs_time_t current_time)
{
    if (ucs_unlikely(current_time - t->now >= t->res)) {
        __ucs_twheel_sweep(t, current_time);
    }
}

/**
 * Get current time
 */
static inline ucs_time_t ucs_twheel_get_time(ucs_twheel_t *t)
{
    return t->now;
}

/**
 * Add a one shot timer.
 *
 * @param twheel     Timer queue to schedule on.
 * @param timer      Timer callback to invoke every time.
 * @param delta      Invocation time
 *
 * NOTE: adding timer already in queue will do nothing
 */
void __ucs_wtimer_add(ucs_twheel_t *t, ucs_wtimer_t *timer, ucs_time_t delta);
static inline ucs_status_t ucs_wtimer_add(ucs_twheel_t *t, ucs_wtimer_t *timer,
                                          ucs_time_t delta)
{
    if (ucs_likely(timer->is_active)) {
        /* most of the times we try to schedule already active timer */
        return UCS_ERR_BUSY;
    }

    __ucs_wtimer_add(t, timer, delta);
    return UCS_OK;
}


/**
 * Remove a timer.
 *
 * @param timer      timer to remove.
 */
static inline void ucs_wtimer_remove(ucs_wtimer_t *timer)
{
    if (ucs_likely(timer->is_active)) {
        ucs_list_del(&timer->list);
        timer->is_active = 0;
    }
}

#endif
