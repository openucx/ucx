/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_MPMC_H
#define UCS_MPMC_H

#include <ucs/type/status.h>
#include <ucs/sys/math.h>

#define UCS_MPMC_VALID_SHIFT        31
#define UCS_MPMC_VALUE_MAX          UCS_BIT(UCS_MPMC_VALID_SHIFT)

/**
 * A Multi-producer-multi-consumer thread-safe queue.
 * Every push/pull is a single atomic operation in "good" scenario.
 * The queue can contain small integers up to UCS_MPMC_VALUE_MAX.
 *
 * TODO make the queue resizeable.
 */
typedef struct ucs_mpmc_queue {
    uint32_t           length;      /* Array size. Rounded to power of 2. */
    int                shift;
    volatile uint32_t  producer;    /* Producer index */
    volatile uint32_t  consumer;    /* Consumer index */
    uint32_t           *queue;      /* Array of data */
} ucs_mpmc_queue_t;


/**
 * Initialize MPMC queue.
 *
 * @param length   Queue length.
 */
ucs_status_t ucs_mpmc_queue_init(ucs_mpmc_queue_t *mpmc, uint32_t length);


/**
 * Destroy MPMC queue.
 */
void ucs_mpmc_queue_cleanup(ucs_mpmc_queue_t *mpmc);


/**
 * Atomically push a value to the queue.
 *
 * @param value Value to push.
 * @return UCS_ERR_EXCEEDS_LIMIT if the queue is full.
 */
ucs_status_t ucs_mpmc_queue_push(ucs_mpmc_queue_t *mpmc, uint32_t value);


/**
 * Atomically pull a value from the queue.
 *
 * @param value_p Filled with the value, if successful.
 * @param UCS_ERR_NO_PROGRESS if there is currently no available item to retrieve,
 *                            or another thread removed the current item.
 */
ucs_status_t ucs_mpmc_queue_pull(ucs_mpmc_queue_t *mpmc, uint32_t *value_p);


/**
 * @retrurn nonzero if queue is empty, 0 if queue *may* be non-empty.
 */
static inline int ucs_mpmc_queue_is_empty(ucs_mpmc_queue_t *mpmc)
{
    return mpmc->producer == mpmc->consumer;
}

#endif
