/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "arbiter.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


void ucs_arbiter_init(ucs_arbiter_t *arbiter)
{
    ucs_list_head_init(&arbiter->list);
}

void ucs_arbiter_group_init(ucs_arbiter_group_t *group)
{
    group->tail = NULL;
    UCS_ARBITER_GROUP_GUARD_INIT(group);
}

void ucs_arbiter_cleanup(ucs_arbiter_t *arbiter)
{
    ucs_assert_always(ucs_arbiter_is_empty(arbiter));
}

void ucs_arbiter_group_cleanup(ucs_arbiter_group_t *group)
{
    UCS_ARBITER_GROUP_GUARD_CHECK(group);
    ucs_assert_always(ucs_arbiter_group_is_empty(group));
}

static inline int ucs_arbiter_group_head_is_scheduled(ucs_arbiter_elem_t *head)
{
    return head->list.next != NULL;
}

static inline void ucs_arbiter_group_head_reset(ucs_arbiter_elem_t *head)
{
    head->list.next = NULL; /* Not scheduled yet */
}

void ucs_arbiter_group_push_elem_always(ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem)
{
    ucs_arbiter_elem_t *tail = group->tail;

    UCS_ARBITER_GROUP_GUARD_CHECK(group);

    if (tail == NULL) {
        ucs_arbiter_group_head_reset(elem);
        elem->next = elem;        /* Connect to itself */
    } else {
        elem->next = tail->next;  /* Point to first element */
        tail->next = elem;        /* Point previous element to new one */
    }

    elem->group = group;  /* Always point to group */
    group->tail = elem;   /* Update group tail */
}

void ucs_arbiter_group_push_head_elem_always(ucs_arbiter_t *arbiter,
                                             ucs_arbiter_group_t *group,
                                             ucs_arbiter_elem_t *elem)
{
    ucs_arbiter_elem_t *tail = group->tail;
    ucs_arbiter_elem_t *head;

    UCS_ARBITER_GROUP_GUARD_CHECK(group);

    elem->group = group;      /* Always point to group */
    ucs_arbiter_group_head_reset(elem);

    if (tail == NULL) {
        elem->next  = elem;   /* Connect to itself */
        group->tail = elem;   /* Update group tail */
        return;
    }

    head       = tail->next;
    elem->next = head;        /* Point to first element */
    tail->next = elem;        /* Point previous element to new one */

    if (!ucs_arbiter_group_head_is_scheduled(head)) {
        return;
    }

    ucs_list_replace(&head->list, &elem->list);
}

void ucs_arbiter_group_head_desched(ucs_arbiter_t *arbiter,
                                    ucs_arbiter_elem_t *head)
{
    if (ucs_arbiter_group_head_is_scheduled(head)) {
        ucs_list_del(&head->list);
    }
}

void ucs_arbiter_group_purge(ucs_arbiter_t *arbiter,
                             ucs_arbiter_group_t *group,
                             ucs_arbiter_callback_t cb, void *cb_arg)
{
    ucs_arbiter_elem_t *tail            = group->tail;
    ucs_arbiter_elem_t dummy_group_head = {};
    ucs_arbiter_elem_t *ptr, *next, *prev;
    ucs_arbiter_cb_result_t result;
    ucs_arbiter_elem_t *head;
    int sched_group;

    if (tail == NULL) {
        return; /* Empty group */
    }

    UCS_ARBITER_GROUP_GUARD_CHECK(group);

    head = tail->next;
    next = head;
    prev = tail;

    sched_group = ucs_arbiter_group_head_is_scheduled(head);
    if (sched_group) {
        /* put a placeholder on the arbiter queue */
        ucs_list_replace(&head->list, &dummy_group_head.list);
    }

    do {
        ptr       = next;
        next      = ptr->next;
        /* Can't touch the element after cb is called if it gets removed. But it
         * can be reused later as well, so it's next should be NULL. */
        ptr->next = NULL;
        result    = cb(arbiter, group, ptr, cb_arg);

        if (result == UCS_ARBITER_CB_RESULT_REMOVE_ELEM) {
            if (ptr == head) {
                head = next;
                if (ptr == tail) {
                    /* Last element is being removed - mark group as empty */
                    group->tail = NULL;
                    if (sched_group) {
                        ucs_list_del(&dummy_group_head.list);
                    }
                    /* Break here to avoid further processing of the group */
                    return;
                }
            } else if (ptr == tail) {
                group->tail = prev;
                /* tail->next should point to head, make sure next is head
                 * (it is assigned 2 lines below) */
                ucs_assert_always(next == head);
            }
            prev->next = next;
        } else {
            /* keep the element */
            ptr->next = next; /* Restore next pointer */
            prev      = ptr;
        }
    } while (ptr != tail);

    ucs_assert(group->tail != NULL);

    if (sched_group) {
        /* restore group head (could be old or new) instead of the dummy element */
        ucs_list_replace(&dummy_group_head.list, &head->list);
    } else {
        /* mark the group head (could be old or new) as unscheduled */
        ucs_arbiter_group_head_reset(head);
    }
}

int ucs_arbiter_group_is_scheduled(ucs_arbiter_group_t *group)
{
    ucs_arbiter_elem_t *head;

    if (ucs_arbiter_group_is_empty(group)) {
        return 0;
    }

    head = group->tail->next;
    return ucs_arbiter_group_head_is_scheduled(head);
}

static void
ucs_arbiter_schedule_head_if_not_scheduled(ucs_arbiter_t *arbiter,
                                           ucs_arbiter_elem_t *head)
{
    if (!ucs_arbiter_group_head_is_scheduled(head)) {
        ucs_list_add_tail(&arbiter->list, &head->list);
    }
}

void ucs_arbiter_group_schedule_nonempty(ucs_arbiter_t *arbiter,
                                         ucs_arbiter_group_t *group)
{
    ucs_arbiter_elem_t *tail = group->tail;
    ucs_arbiter_elem_t *head;

    ucs_assert(tail != NULL);
    head = tail->next;

    if (head == NULL) {
        /* It means that 1 element group is scheduled during dispatch.
         * Restore next pointer.
         */
        head = tail;
    }

    ucs_arbiter_schedule_head_if_not_scheduled(arbiter, head);
    UCS_ARBITER_GROUP_ARBITER_SET(group, arbiter);
}

void ucs_arbiter_group_desched_nonempty(ucs_arbiter_t *arbiter,
                                        ucs_arbiter_group_t *group)
{
    ucs_arbiter_elem_t *head = group->tail->next;

    if (!ucs_arbiter_group_head_is_scheduled(head)) {
        return;
    }

    UCS_ARBITER_GROUP_ARBITER_CHECK(group, arbiter);
    UCS_ARBITER_GROUP_ARBITER_SET(group, NULL);
    ucs_list_del(&head->list);
    ucs_arbiter_group_head_reset(head);
}

void ucs_arbiter_dispatch_nonempty(ucs_arbiter_t *arbiter, unsigned per_group,
                                   ucs_arbiter_callback_t cb, void *cb_arg)
{
    ucs_arbiter_elem_t *group_head, *group_tail, *next_elem;
    ucs_arbiter_cb_result_t result;
    unsigned group_dispatch_count;
    ucs_arbiter_group_t *group;
    UCS_LIST_HEAD(resched_list);
    int sched_group;

    ucs_assert(!ucs_list_is_empty(&arbiter->list));

    for (;;) {
        group_head = ucs_list_extract_head(&arbiter->list, ucs_arbiter_elem_t,
                                           list);
        ucs_assert(group_head != NULL);

        /* Reset group head to allow the group to be moved to another arbiter by
         * the dispatch callback. For example, when a DC endpoint is moved from
         * waiting-for-DCI arbiter to waiting-for-TX-resources arbiter.
         */
        ucs_arbiter_group_head_reset(group_head);

        group_dispatch_count = 0;
        sched_group          = 1;
        group                = group_head->group;
        UCS_ARBITER_GROUP_GUARD_CHECK(group);

        do {
            /* zero pointer to next elem here because:
             * 1. if the element is removed from the arbiter it must be kept in
             *    initialized state otherwise push will fail
             * 2. we can't zero the pointer after calling the callback because
             *    the callback could release the element.
             */
            next_elem        = group_head->next;
            group_head->next = NULL;
            ucs_assert(group_head->group == group);

            ucs_trace_poll("dispatching arbiter element %p", group_head);
            UCS_ARBITER_GROUP_GUARD_ENTER(group);
            result = cb(arbiter, group, group_head, cb_arg);
            UCS_ARBITER_GROUP_GUARD_EXIT(group);
            ucs_trace_poll("dispatch result: %d", result);
            ++group_dispatch_count;

            if (result == UCS_ARBITER_CB_RESULT_REMOVE_ELEM) {
                group_tail = group->tail;
                if (group_head == group_tail) {
                    /* Last element */
                    group->tail = NULL; /* Group is empty now */
                    sched_group = 0;
                    group_head  = NULL; /* for debugging */
                    UCS_ARBITER_GROUP_ARBITER_SET(group, NULL);
                    break;
                } else {
                    /* Not last element */
                    ucs_assert(group_head == group_tail->next);
                    ucs_assert(group_head != next_elem);
                    group_head       = next_elem;  /* Update group head */
                    group_tail->next = group_head; /* Tail points to new head */
                    ucs_arbiter_group_head_reset(group_head);
                }
            } else {
                /* element is not removed, restore next pointer */
                group_head->next = next_elem;

                /* group must still be active */
                ucs_assert(sched_group == 1);

                if (result == UCS_ARBITER_CB_RESULT_STOP) {
                    /* exit the outmost loop and make sure that next dispatch()
                     * will continue from the current group */
                    ucs_list_add_head(&arbiter->list, &group_head->list);
                    goto out;
                } else if (result != UCS_ARBITER_CB_RESULT_NEXT_GROUP) {
                    /* resched/desched must avoid adding the group to the arbiter */
                    sched_group = 0;
                    if (result == UCS_ARBITER_CB_RESULT_DESCHED_GROUP) {
                        UCS_ARBITER_GROUP_ARBITER_SET(group, NULL);
                    } else if (result == UCS_ARBITER_CB_RESULT_RESCHED_GROUP) {
                        ucs_list_add_tail(&resched_list, &group_head->list);
                    } else {
                        ucs_bug("unexpected return value from arbiter callback");
                    }
                    break;
                }
            }
        } while (group_dispatch_count < per_group);

        if (sched_group) {
            /* the group could be scheduled again from dispatch callback */
            ucs_arbiter_schedule_head_if_not_scheduled(arbiter, group_head);
            ucs_assert(!ucs_list_is_empty(&arbiter->list));
        } else if (ucs_list_is_empty(&arbiter->list)) {
            break;
        }
    }

out:
    ucs_list_splice_tail(&arbiter->list, &resched_list);
}

void ucs_arbiter_dump(ucs_arbiter_t *arbiter, FILE *stream)
{
    ucs_arbiter_elem_t *group_head, *elem;
    int first;

    fprintf(stream, "-------\n");
    if (ucs_list_is_empty(&arbiter->list)) {
        fprintf(stream, "(empty)\n");
        goto out;
    }

    first = 1;
    ucs_list_for_each(group_head, &arbiter->list, list) {
        elem = group_head;
        if (first) {
            fprintf(stream, "=> ");
            first = 0;
        } else {
            fprintf(stream, " * ");
        }
        do {
            fprintf(stream, "[%p", elem);
            if (elem == group_head) {
                fprintf(stream, " prev_g:%p", elem->list.prev);
                fprintf(stream, " next_g:%p", elem->list.next);
            }
            fprintf(stream, " next_e:%p grp:%p]", elem->next, elem->group);
            if (elem->next != group_head) {
                fprintf(stream, "->");
            }
            elem = elem->next;
        } while (elem != group_head);
        fprintf(stream, "\n");
    }

out:
    fprintf(stream, "-------\n");
}
