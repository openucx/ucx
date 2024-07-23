/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_MPMC_H
#define UCS_MPMC_H

#include "queue.h"

#include <ucs/type/status.h>
#include <ucs/type/spinlock.h>

/**
 * A Multi-producer-multi-consumer thread-safe queue.
 * Every push/pull is a single atomic operation in "good" scenario.
 */
typedef struct ucs_mpmc_queue {
    ucs_spinlock_t     lock;        /* Protects 'queue' */
    ucs_queue_head_t   queue;       /* Queue of data */
} ucs_mpmc_queue_t;


/**
 * MPMC queue element type.
 */
typedef struct ucs_mpmc_elem {
    ucs_queue_elem_t super;
    uint64_t         value;
} ucs_mpmc_elem_t;


/**
 * MPMC queue element value predicate.
 *
 * @param [in] value MPMC queue element value to check.
 * @param [in] arg   User-defined argument.
 *
 * @return Predicate result value - nonzero means "true", zero means "false".
 */
typedef int (*ucs_mpmc_queue_predicate_t)(uint64_t value, void *arg);


/**
 * Initialize MPMC queue.
 *
 * @param length   Queue length.
 */
ucs_status_t ucs_mpmc_queue_init(ucs_mpmc_queue_t *mpmc);


/**
 * Destroy MPMC queue.
 */
void ucs_mpmc_queue_cleanup(ucs_mpmc_queue_t *mpmc);


/**
 * Atomically push a value to the queue.
 *
 * @param value Value to push.
 * @return UCS_ERR_NO_MEMORY if it fails to allocate the MPMC queue element.
 */
ucs_status_t ucs_mpmc_queue_push(ucs_mpmc_queue_t *mpmc, uint64_t value);


/**
 * Atomically pull a value from the queue.
 *
 * @param value_p Filled with the value, if successful.
 * @param UCS_ERR_NO_PROGRESS if there is currently no available item to retrieve,
 *                            or another thread removed the current item.
 */
ucs_status_t ucs_mpmc_queue_pull(ucs_mpmc_queue_t *mpmc, uint64_t *value_p);


/**
 * Remove all elements from the MPMC queue with the given value for which the
 * given predicate returns "true" (nonzero) value.
 * This can be used from any context and any thread.
 *
 * @param  [in] mpmc      MPMC queue.
 * @param  [in] predicate Predicate to check candidates for removal.
 * @param  [in] arg       User-defined argument for the predicate.
 */
void ucs_mpmc_queue_remove_if(ucs_mpmc_queue_t *mpmc,
                              ucs_mpmc_queue_predicate_t predicate, void *arg);


/**
 * @retrurn nonzero if queue is empty, 0 if queue *may* be non-empty.
 */
static inline int ucs_mpmc_queue_is_empty(ucs_mpmc_queue_t *mpmc)
{
    return ucs_queue_is_empty(&mpmc->queue);
}

#endif
