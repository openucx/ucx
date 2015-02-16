/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_CALLBACKQ_H_
#define UCS_CALLBACKQ_H_

#include <ucs/datastruct/queue.h>
#include <ucs/type/callback.h>


/**
 * A queue of callbacks with sequence numbers. Supports executing all
 * callbacks with a lesser sequence number.
 */
typedef struct ucs_callbackq {
    ucs_queue_head_t queue;
} ucs_callbackq_t;


/**
 * Callback queue element.
 */
typedef struct ucs_callbackq_elem ucs_callbackq_elem_t;
struct ucs_callbackq_elem {
    ucs_callback_t   super;
    ucs_queue_elem_t queue;
    uint16_t         sn;
} UCS_S_PACKED;


/**
 * Initialize the callback queue.
 *
 * @param cbq    Callback queue to initialize.
 */
static inline void ucs_callbackq_init(ucs_callbackq_t *cbq)
{
    ucs_queue_head_init(&cbq->queue);
}


/**
 * Cleanup the callback queue.
 *
 * @param cbq    Callback queue to cleanup.
 */
static inline void ucs_callbackq_cleanup(ucs_callbackq_t *cbq)
{
    ucs_assert(ucs_queue_is_empty(&cbq->queue));
}


/**
 * Add element to the callback queue
 *
 * @param cbq    Callback queue to add element to.
 * @param elem   Element to add.
 */
static inline void ucs_callbackq_push(ucs_callbackq_t *cbq, ucs_callbackq_elem_t *elem)
{
    ucs_assert(ucs_queue_is_empty(&cbq->queue) ||
               UCS_CIRCULAR_COMPARE16(elem->sn, >=,
                                      ucs_queue_tail_elem_non_empty(&cbq->queue,
                                                                    ucs_callbackq_elem_t,
                                                                    queue)->sn));
    ucs_queue_push(&cbq->queue, &elem->queue);
}


/**
 * Remove and execute all callbacks with sequence number less-than-equal the one provided.
 *
 * @param cbq    Callback queue to execute callback from.
 * @param sn     Provided sequence number.
 */
static inline unsigned ucs_callbackq_pull(ucs_callbackq_t *cbq, uint16_t sn)
{
    ucs_callbackq_elem_t *elem;
    unsigned count;

    count = 0;
    while (!ucs_queue_is_empty(&cbq->queue)) {
        elem = ucs_queue_head_elem_non_empty(&cbq->queue, ucs_callbackq_elem_t,
                                             queue);
        if (!UCS_CIRCULAR_COMPARE16(elem->sn, <=, sn)) {
            break;
        }

        /* TODO prefetch */
        ucs_queue_pull_non_empty(&cbq->queue);
        elem->super.func(&elem->super);
        ++count;
    }

    return count;
}


#endif
