/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_QUEUE_H_
#define UCS_QUEUE_H_

#include "queue_types.h"

#include <ucs/debug/assert.h>
#include <stddef.h>


/**
 * Initialize a queue.
 *
 * @param queue  Queue to initialize.
 */
static inline void ucs_queue_head_init(ucs_queue_head_t *queue)
{
    queue->ptail = &queue->head;
}

/**
 * @return Queue length.
 */
static inline size_t ucs_queue_length(ucs_queue_head_t *queue)
{
    ucs_queue_elem_t **pelem;
    size_t length;

    length = 0;
    for (pelem = &queue->head; pelem != queue->ptail; pelem = &(*pelem)->next) {
        ++length;
    }
    return length;
}

/**
 * @return Whether the queue is empty.
 */
static inline int ucs_queue_is_empty(ucs_queue_head_t *queue)
{
    return queue->ptail == &queue->head;
}

/**
 * Enqueue an element to the tail of the queue.
 *
 * @param queue  Queue to add to.
 * @param elem   Element to add.
 */
static inline void ucs_queue_push(ucs_queue_head_t *queue, ucs_queue_elem_t *elem)
{
    *queue->ptail = elem;
    queue->ptail = &elem->next;
#if ENABLE_ASSERT
    elem->next = NULL; /* For sanity check below */
#endif
}

/**
 * Add an element to the head of the queue.
 *
 * @param queue  Queue to add to.
 * @param elem   Element to add.
 */
static inline void ucs_queue_push_head(ucs_queue_head_t *queue,
                                       ucs_queue_elem_t *elem)
{
    elem->next = queue->head;
    queue->head = elem;
    if (queue->ptail == &queue->head) {
        queue->ptail = &elem->next;
    }
}

/**
 * Dequeue an element from the head of the queue, assuming the queue is not empty.
 *
 * @param queue  Non-empty queue to pull from.
 * @return  Element from the head of the queue.
 */
static inline ucs_queue_elem_t *ucs_queue_pull_non_empty(ucs_queue_head_t *queue)
{
    ucs_queue_elem_t *elem;

    elem = queue->head;
    queue->head = elem->next;
    if (queue->ptail == &elem->next) {
        queue->ptail = &queue->head;
    }
    return elem;
}

/**
 * Delete an element.
 * The element must be valid when deleting it.
 * After the call, iter points to the next element, and the element may be released.
 */
static inline void ucs_queue_del_iter(ucs_queue_head_t *queue, ucs_queue_iter_t iter)
{
    if (queue->ptail == &(*iter)->next) {
        queue->ptail = iter; /* deleting the last element */
        *iter = NULL;        /* make *ptail point to NULL */
    } else {
        *iter = (*iter)->next;
    }

    /* Sanity check */
    ucs_assertv((queue->head != NULL) || (queue->ptail == &queue->head),
               "head=%p ptail=%p &head=%p iter=%p", queue->head, queue->ptail,
               &queue->head, iter);

    /* If the queue is empty, head must point to null */
    ucs_assertv((queue->ptail != &queue->head) || (queue->head == NULL),
               "head=%p ptail=%p &head=%p iter=%p", queue->head, queue->ptail,
               &queue->head, iter);
}

/**
 * Dequeue an element from the head of the queue.
 *
 * @param queue  Queue to pull from.
 * @return  Element from the head of the queue, or NULL if the queue is empty.
 */
static inline ucs_queue_elem_t *ucs_queue_pull(ucs_queue_head_t *queue)
{
    if (ucs_queue_is_empty(queue))
        return NULL;
    return ucs_queue_pull_non_empty(queue);
}

/**
 * Insert all elements from one queue to another queue, leaving the first queue
 * empty.
 *
 * @param queue     Queue to push elements to.
 * @param new_elems Queue of elements to add.
 */
static inline void ucs_queue_splice(ucs_queue_head_t *queue,
                                    ucs_queue_head_t *new_elems)
{
    if (!ucs_queue_is_empty(new_elems)) {
        *queue->ptail = new_elems->head;
        queue->ptail = new_elems->ptail;
        new_elems->ptail = &new_elems->head;
    }
}

/**
 * Convenience macro to pull from a non-empty queue and return the containing element.
 *
 * @param queue   Non-empty queue to pull from.
 * @param type    Container element type.
 * @param member  Queue element member inside the container.
 *
 * @return Pulled element.
 */
#define ucs_queue_pull_elem_non_empty(queue, type, member) \
    ucs_container_of(ucs_queue_pull_non_empty(queue), type, member)

/**
 * Convenience macro to get the head element of a non-empty queue.
 *
 * @param queue   Non-empty queue whose head element to get.
 * @param type    Container element type.
 * @param member  Queue element member inside the container.
 *
 * @return Head element.
 */
#define ucs_queue_head_elem_non_empty(queue, type, member) \
    ucs_container_of((queue)->head, type, member)

/**
 * Convenience macro to get the tail element of a non-empty queue.
 *
 * @param queue   Non-empty queue whose head element to get.
 * @param type    Container element type.
 * @param member  Queue element member inside the container.
 *
 * @return Head element.
 */
#define ucs_queue_tail_elem_non_empty(queue, type, member) \
    ucs_container_of((queue)->ptail, type, member)

/**
 * Iterate over queue elements. The queue must not be modified during the iteration.
 *
 * @param elem    Variable which will hold point to the element in the queue.
 * @param queue   Queue to iterate on.
 * @param member  Member inside 'elem' which is the queue link.
 */
#define ucs_queue_for_each(elem, queue, member) \
    for (*(queue)->ptail = NULL, \
             elem = ucs_container_of((queue)->head, typeof(*elem), member); \
         &elem->member != NULL; \
         elem = ucs_container_of(elem->member.next, typeof(*elem), member))

/**
 * Iterate over queue elements. The current element may be safely removed from
 * the queue using ucs_queue_del_iter().
 *
 * @param elem    Variable which will hold point to the element in the queue.
 * @param iter    Iterator variable. May be passed to ucs_queue_del_iter().
 * @param queue   Queue to iterate on.
 * @param member  Member inside 'elem' which is the queue link.
 */
#define ucs_queue_for_each_safe(elem, iter, queue, member) \
    for (iter = &(queue)->head, \
         elem = ucs_container_of(*iter, typeof(*elem), member); \
          iter != (queue)->ptail; \
          iter = (*iter == &elem->member) ? &(*iter)->next : iter, \
            elem = ucs_container_of(*iter, typeof(*elem), member))

/**
 * Iterate and extract elements from the queue while a condition is true.
 *
 * @param elem    Variable which will hold point to the element in the queue.
 * @param queue   Queue to iterate on.
 * @param member  Member inside 'elem' which is the queue link.
 * @param cond    Condition to continue iterating.
 *
 * TODO optimize
 */
#define ucs_queue_for_each_extract(elem, queue, member, cond) \
    for (elem = ucs_container_of((queue)->head, typeof(*elem), member); \
         \
         !ucs_queue_is_empty(queue) && (cond) && ucs_queue_pull_non_empty(queue); \
         \
         elem = ucs_container_of((queue)->head, typeof(*elem), member))


/*
 * Queue iteration
 */

static inline ucs_queue_iter_t ucs_queue_iter_begin(ucs_queue_head_t *q)
{
    return &q->head;
}

static inline ucs_queue_iter_t ucs_queue_iter_next(ucs_queue_iter_t i)
{
    return  &(*i)->next;
}

static inline int ucs_queue_iter_end(ucs_queue_head_t *q, ucs_queue_iter_t i)
{
    return i == q->ptail;
}

static inline void ucs_queue_remove(ucs_queue_head_t *queue, ucs_queue_elem_t *elem)
{
    ucs_queue_iter_t iter = ucs_queue_iter_begin(queue);

    while (!ucs_queue_iter_end(queue, iter)) {
        if (*iter == elem) {
            ucs_queue_del_iter(queue, iter);
            return;
        }
        iter = ucs_queue_iter_next(iter);
    }
}

#define ucs_queue_iter_elem(elem, iter, member) \
    ucs_container_of(*iter, typeof(*elem), member)

#endif
