/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TIMERQ_H
#define UCS_TIMERQ_H

#include <ucs/datastruct/queue.h>
#include <ucs/time/time.h>
#include <ucs/type/status.h>
#include <ucs/sys/preprocessor.h>
#include <ucm/util/sys.h>
#include <ucs/type/spinlock.h>
#include <ucs/datastruct/ptr_array.h>
#include <ucs/datastruct/mpool.h>


typedef struct ucs_timer {
    ucs_time_t                 expiration;/* Absolute timer expiration time */
    ucs_time_t                 interval;  /* Re-scheduling interval */
    int                        id;
} ucs_timer_t;


typedef struct ucs_timer_queue {
    ucs_time_t                 min_interval; /* Expiration of next timer */
    ucs_ptr_array_locked_t     *timers;      /* Array of timers */
    ucs_mpool_t                timers_mp;   /* Mem pool for the timers */
} ucs_timer_queue_t;


/**
 * Initialize the timer queue.
 *
 * @param timerq        Timer queue to initialize.
 * @param name		Name for the timer queue
 */
ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq, const char *name);


/**
 * Cleanup the timer queue.
 *
 * @param timerq    Timer queue to clean up.
 */
void ucs_timerq_cleanup(ucs_timer_queue_t *timerq);


/**
 * Add a periodic timer.
 *
 * @param timerq     Timer queue to schedule on.
 * @param interval   Timer interval.
 * @param timer_id_p Filled with the ID of the new timer in the queue.
 */
ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq,
                        ucs_time_t interval, int *timer_id_p);

/**
 * Remove a timer.
 *
 * @param timerq     Time queue this timer was scheduled on.
 * @param timer_id   Timer ID to remove.
 */
ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, int timer_id);


/**
 * @return Minimal timer interval.
 */
static inline ucs_time_t ucs_timerq_min_interval(ucs_timer_queue_t *timerq) {
    return timerq->min_interval;
}


/**
 * @return Number of timers in the queue.
 */
static inline int ucs_timerq_size(ucs_timer_queue_t *timerq) {
    return ucs_ptr_array_locked_get_size(timerq->timers);
}


/**
 * @return Whether there are no timers.
 */
static inline int ucs_timerq_is_empty(ucs_timer_queue_t *timerq) {
    return ucs_timerq_size(timerq) == 0;
}


/**
 * Go through the expired timers in the timer queue.
 *
 * @param _timer        Variable to be assigned with a pointer to the timer.
 * @param _timerq       Timer queue to dispatch timers on.
 * @param _current_time Current time to dispatch the timers for.
 *
 * @note Timers which expired between calls to this function will also be dispatched.
 * @note There is no guarantee on the order of dispatching.
 */
#define ucs_timerq_for_each_expired(_timer, _timerq, _current_time, _code) \
    { \
        ucs_time_t __current_time = _current_time; \
        unsigned _index;	\
        void *_ptr;	\
        ucs_ptr_array_locked_for_each(_ptr, _index, (_timerq)->timers)	\
        { \
            timer = (ucs_timer_t *)_ptr;	\
            if (__current_time >= (_timer)->expiration) { \
                /* Update expiration time */ \
                (_timer)->expiration = __current_time + (_timer)->interval; \
                _code; \
            } \
        } \
    }

#endif
