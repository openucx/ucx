/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_TIMERQ_H
#define UCS_TIMERQ_H

#include <ucs/datastruct/list.h>
#include <ucs/type/callback.h>
#include <ucs/time/time.h>


/**
 * UCS timer.
 */
typedef struct ucs_timer {
    ucs_callback_t *cb;              /* User callback */
    ucs_time_t     interval;         /* Re-scheduling interval */
    ucs_time_t     expiration;       /* Absolute timer expiration time */
    ucs_list_link_t    list;             /* Link in the list of timers */
} ucs_timer_t;


typedef struct ucs_timer_queue {
    ucs_time_t     expiration;       /* Expiration of next timer */
    ucs_list_link_t    timers;           /* List of timers */
} ucs_timer_queue_t;


/**
 * Initialize the timer queue.
 *
 * @param timerq        Timer queue to initialize.
 */
ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq);

/**
 * Cleanup the timer queue.
 *
 * @param timerq    Timer queue to clean up.
 */
void ucs_timerq_cleanup(ucs_timer_queue_t *timerq);

void ucs_timerq_sweep_internal(ucs_timer_queue_t *timerq, ucs_time_t current_time);

/**
 * Go through the timers in the timer queue, dispatch expired timers, and reschedule
 * periodic timers.
 *
 * @param timerq        Timer queue to dispatch timers on.
 * @param current_time  Current time to dispatch the timers for.
 *
 * @note Timers which expired between calls to this function will also be dispatched.
 * @note There is no guarantee on the order of dispatching.
 */
static inline void ucs_timerq_sweep(ucs_timer_queue_t *timerq, ucs_time_t current_time)
{
    if (current_time >= timerq->expiration) {
        ucs_timerq_sweep_internal(timerq, current_time);
    }
}

/**
 * Add a periodic timer.
 *
 * @param timerq     Timer queue to schedule on.
 * @param callback   Callback to invoke every time.
 * @param interval   Timer interval.
 *
 * NOTE: The timer callback is allowed to remove itself from the queue by
 * calling ucs_timer_remove().
 */
ucs_status_t ucs_timer_add(ucs_timer_queue_t *timerq, ucs_callback_t *callback,
                          ucs_time_t interval);

/**
 * Remove a timer.
 *
 * @param timerq     Time queue this timer was scheduled on.
 * @param callback   Callback to remove.
 */
void ucs_timer_remove(ucs_timer_queue_t *timerq, ucs_callback_t *callback);

#endif
