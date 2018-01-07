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
#include <pthread.h>


typedef struct ucs_timer {
    ucs_time_t                 expiration;/* Absolute timer expiration time */
    ucs_time_t                 interval;  /* Re-scheduling interval */
    int                        id;
} ucs_timer_t;


typedef struct ucs_timer_queue {
    pthread_spinlock_t         lock;
    ucs_time_t                 min_interval; /* Expiration of next timer */
    ucs_timer_t                *timers;      /* Array of timers */
    unsigned                   num_timers;   /* Number of timers */
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


/**
 * Add a periodic timer.
 *
 * @param timerq     Timer queue to schedule on.
 * @param timer_id   Timer ID to add.
 * @param interval   Timer interval.
 */
ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, int timer_id,
                            ucs_time_t interval);


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
    return timerq->num_timers;
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
        pthread_spin_lock(&(_timerq)->lock); /* Grab lock */ \
        for (_timer = (_timerq)->timers; \
             _timer != (_timerq)->timers + (_timerq)->num_timers; \
             ++_timer) \
        { \
            if (__current_time >= (_timer)->expiration) { \
                /* Update expiration time */ \
                (_timer)->expiration = __current_time + (_timer)->interval; \
                _code; \
            } \
        } \
        pthread_spin_unlock(&(_timerq)->lock); /* Release lock  */ \
    }

#endif
