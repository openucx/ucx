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
 * Iterate over MPMC queue elements. The queue must not be modified during the
 * iteration.
 *
 * @param elem Variable which will hold point to the MPMC queue element.
 * @param mpmc_queue MPMC Queue to iterate on.
 */
#define ucs_mpmc_queue_for_each(elem, mpmc_queue) \
    ucs_queue_for_each(elem, &(mpmc_queue)->queue, super)


/**
 * Lock MPMC queue.
 */
void ucs_mpmc_queue_block(ucs_mpmc_queue_t *mpmc);


/**
 * Unlock MPMC queue.
 */
void ucs_mpmc_queue_unblock(ucs_mpmc_queue_t *mpmc);


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
 * @retrurn nonzero if queue is empty, 0 if queue *may* be non-empty.
 */
static inline int ucs_mpmc_queue_is_empty(ucs_mpmc_queue_t *mpmc)
{
    return ucs_queue_is_empty(&mpmc->queue);
}

#endif
