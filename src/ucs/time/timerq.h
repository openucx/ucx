/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
* Copyright (C) Huawei Technologies Co., Ltd. 2020. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_TIMERQ_H
#define UCS_TIMERQ_H

#include <ucs/datastruct/queue.h>
#include <ucs/time/time.h>
#include <ucs/type/status.h>
#include <ucs/sys/preprocessor.h>
#include <ucs/type/spinlock.h>

enum {
    UCS_TIMERQ_FLAG_UNIQUE_IDS = UCS_BIT(0)
};

typedef struct ucs_timer {
    ucs_time_t                 expiration;/* Absolute timer expiration time */
    ucs_time_t                 interval;  /* Re-scheduling interval */
    uint64_t                   id;
} ucs_timer_t;


typedef struct ucs_timer_queue {
    ucs_spinlock_t             lock;
    ucs_time_t                 min_interval; /* Expiration of next timer */
    ucs_timer_t                *timers;      /* Array of timers */
    unsigned                   alloc_cnt;    /* Number of timers allocated */
    unsigned                   used_cnt;     /* Number of timers used */
    int                        flags;        /* Various flags */
} ucs_timer_queue_t;


/**
 * Initialize the timer queue.
 *
 * @param timerq        Timer queue to initialize.
 * @param flags         Modifiers for timer queue creation.
 */
ucs_status_t ucs_timerq_init(ucs_timer_queue_t *timerq, int flags);


/**
 * Cleanup the timer queue.
 *
 * @param timerq    Timer queue to clean up.
 */
void ucs_timerq_cleanup(ucs_timer_queue_t *timerq);


/**
 * Add a periodic timer.
 *
 * @param timerq       Timer queue to schedule on.
 * @param timer_id     Timer ID to add.
 * @param interval     Timer interval.
 * @param timer_index  Selected location inside the queue (optional).
 */
ucs_status_t ucs_timerq_add(ucs_timer_queue_t *timerq, uint64_t timer_id,
                            ucs_time_t interval, unsigned *timer_index);


/**
 * Remove a timer.
 *
 * @param timerq           Time queue this timer was scheduled on.
 * @param timer_id         Timer ID to remove.
 * @param timer_index_hint Possible timer location inside the queue (optional)
 */
ucs_status_t ucs_timerq_remove(ucs_timer_queue_t *timerq, uint64_t timer_id,
                               unsigned timer_index_hint);


/**
 * Modify the interval of a timer.
 *
 * @param timerq           Time queue this timer was scheduled on.
 * @param timer_id         Timer ID to modify.
 * @param new_interval     New timer interval.
 * @param timer_index_hint Possible timer location inside the queue (optional)
 */
ucs_status_t ucs_timerq_modify(ucs_timer_queue_t *timerq, uint64_t timer_id,
                               ucs_time_t new_interval, unsigned timer_index_hint);


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
    return timerq->used_cnt;
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
        ucs_spin_lock(&(_timerq)->lock); /* Grab lock */ \
        for (_timer = (_timerq)->timers; \
             _timer != (_timerq)->timers + (_timerq)->used_cnt; \
             ++_timer) \
        { \
            if (__current_time >= (_timer)->expiration) { \
                /* Update expiration time */ \
                (_timer)->expiration = __current_time + (_timer)->interval; \
                _code; \
            } \
        } \
        ucs_spin_unlock(&(_timerq)->lock); /* Release lock  */ \
    }

#endif
