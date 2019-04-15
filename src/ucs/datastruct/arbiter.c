/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "arbiter.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>


#define SENTINEL ((ucs_arbiter_elem_t*)0x1)

void ucs_arbiter_init(ucs_arbiter_t *arbiter)
{
    arbiter->current = NULL;
    UCS_ARBITER_GUARD_INIT(arbiter);
}

void ucs_arbiter_group_init(ucs_arbiter_group_t *group)
{
    group->tail = NULL;
}

void ucs_arbiter_cleanup(ucs_arbiter_t *arbiter)
{
    ucs_assert(arbiter->current == NULL);
}

void ucs_arbiter_group_cleanup(ucs_arbiter_group_t *group)
{
    ucs_assert(group->tail == NULL);
}

void ucs_arbiter_group_push_elem_always(ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem)
{
    ucs_arbiter_elem_t *tail = group->tail;

    if (tail == NULL) {
        elem->list.next = NULL;   /* Not scheduled yet */
        elem->next      = elem;   /* Connect to itself */
    } else {
        elem->next = tail->next;  /* Point to first element */
        tail->next = elem;        /* Point previous element to new one */
    }

    elem->group = group;  /* Always point to group */
    group->tail = elem;   /* Update group tail */
}

static int ucs_arbiter_group_is_scheduled(ucs_arbiter_elem_t *head)
{
    return head->list.next != NULL;
}

void ucs_arbiter_group_push_head_elem_always(ucs_arbiter_t *arbiter,
                                             ucs_arbiter_group_t *group,
                                             ucs_arbiter_elem_t *elem)
{
    ucs_arbiter_elem_t *tail = group->tail;
    ucs_arbiter_elem_t *head;

    elem->group     = group;  /* Always point to group */
    elem->list.next = NULL;   /* Not scheduled yet */

    if (tail == NULL) {
        elem->next  = elem;   /* Connect to itself */
        group->tail = elem;   /* Update group tail */
        return;
    }

    head       = tail->next;
    elem->next = head;        /* Point to first element */
    tail->next = elem;        /* Point previous element to new one */

    if (!ucs_arbiter_group_is_scheduled(head)) {
        return;
    }

    ucs_assert(arbiter != NULL);

    if (head->list.next == &head->list) {
        /* single group which was scheduled */
        ucs_assert(arbiter->current == head);
        ucs_list_head_init(&elem->list);
        arbiter->current = elem;
    } else {
        ucs_list_insert_replace(head->list.prev, head->list.next, &elem->list);
        if (arbiter->current == head) {
            arbiter->current = elem;
        }
    }
}

void ucs_arbiter_group_head_desched(ucs_arbiter_t *arbiter,
                                    ucs_arbiter_elem_t *head)
{
    ucs_arbiter_elem_t *next;

    if (!ucs_arbiter_group_is_scheduled(head)) {
        return; /* Not scheduled */
    }

    /* If this group is the next to be scheduled, skip it */
    if (arbiter->current == head) {
        next = ucs_list_next(&head->list, ucs_arbiter_elem_t, list);
        arbiter->current = (next == head) ? NULL : next;
    }

    ucs_list_del(&head->list);
}

void ucs_arbiter_group_purge(ucs_arbiter_t *arbiter,
                             ucs_arbiter_group_t *group,
                             ucs_arbiter_callback_t cb, void *cb_arg)
{
    ucs_arbiter_elem_t *tail       = group->tail;
    ucs_arbiter_elem_t *next_group = NULL;
    ucs_arbiter_elem_t *prev_group = NULL;
    ucs_arbiter_elem_t *ptr, *next, *prev;
    ucs_arbiter_elem_t *head, *orig_head;
    ucs_arbiter_cb_result_t result;
    int is_scheduled;

    if (tail == NULL) {
        return; /* Empty group */
    }

    orig_head    = head = tail->next;
    is_scheduled = ucs_arbiter_group_is_scheduled(head);
    next         = head;
    prev         = tail;

    if (is_scheduled) {
        prev_group = ucs_list_prev(&head->list, ucs_arbiter_elem_t, list);
        next_group = ucs_list_next(&head->list, ucs_arbiter_elem_t, list);
    }

    do {
        ptr       = next;
        next      = ptr->next;
        /* Can't touch the element if it gets removed. But it can be reused
         * later as well, so it's next should be NULL. */
        ptr->next = NULL;
        result    = cb(arbiter, ptr, cb_arg);

        if (result == UCS_ARBITER_CB_RESULT_REMOVE_ELEM) {
            if (ptr == head) {
                head = next;
                if (ptr == tail) {
                    /* Last element is being removed - mark group as empty */
                    group->tail = NULL;
                    /* Break here to keep ptr->next = NULL, otherwise ptr->next
                       will be set to itself below */
                    break;
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

    if (is_scheduled) {
        if (orig_head == prev_group) {
            /* this is the only group which was scheduled */
            if (group->tail == NULL) {
                /* group became empty - no more groups scheduled */
                arbiter->current = NULL;
            } else if (orig_head != head) {
                /* keep the group scheduled, but with new head element */
                arbiter->current = head;
                ucs_list_head_init(&head->list);
            }
        } else {
            if (group->tail == NULL) {
                /* group became empty - deschedule it */
                prev_group->list.next = &next_group->list;
                next_group->list.prev = &prev_group->list;
                if (arbiter->current == orig_head) {
                    arbiter->current = next_group;
                }
            } else if (orig_head != head) {
                /* keep the group scheduled, but with new head element */
                ucs_list_insert_replace(&prev_group->list, &next_group->list,
                                        &head->list);
                if (arbiter->current == orig_head) {
                    arbiter->current = head;
                }
            }
        }
    } else if ((orig_head != head) && (group->tail != NULL)) {
        /* Mark new head as unscheduled */
        head->list.next = NULL;
    }
}

void ucs_arbiter_group_schedule_nonempty(ucs_arbiter_t *arbiter,
                                         ucs_arbiter_group_t *group)
{
    ucs_arbiter_elem_t *tail = group->tail;
    ucs_arbiter_elem_t *current, *head;

    UCS_ARBITER_GUARD_CHECK(arbiter);

    ucs_assert(tail != NULL);
    head = tail->next;

    if (head == NULL) {
        /* it means that 1 element group is
         * scheduled during dispatch.
         * Restore next pointer.
         */
        head = tail->next = tail;
    }

    if (ucs_arbiter_group_is_scheduled(head)) {
        return; /* Already scheduled */
    }

    current = arbiter->current;
    if (current == NULL) {
        ucs_list_head_init(&head->list);
        arbiter->current = head;
    } else {
        ucs_list_insert_before(&current->list, &head->list);
    }
}

void ucs_arbiter_dispatch_nonempty(ucs_arbiter_t *arbiter, unsigned per_group,
                                   ucs_arbiter_callback_t cb, void *cb_arg)
{
    ucs_arbiter_elem_t *group_head, *last_elem, *elem, *next_elem;
    ucs_list_link_t *elem_list_next;
    ucs_arbiter_elem_t *next_group, *prev_group;
    ucs_arbiter_group_t *group;
    ucs_arbiter_cb_result_t result;
    unsigned group_dispatch_count;
    int is_single_group;
    UCS_LIST_HEAD(resched_groups);

    next_group = arbiter->current;
    ucs_assert(next_group != NULL);

    do {
        group_head    = next_group;
        ucs_assert(group_head != NULL);
        prev_group    = ucs_list_prev(&group_head->list, ucs_arbiter_elem_t, list);
        next_group    = ucs_list_next(&group_head->list, ucs_arbiter_elem_t, list);
        ucs_assert(prev_group != NULL);
        ucs_assert(next_group != NULL);
        ucs_assert(prev_group->list.next == &group_head->list);
        ucs_assert(next_group->list.prev == &group_head->list);

        group_dispatch_count = 0;
        group                = group_head->group;
        last_elem            = group->tail;
        next_elem            = group_head;
        is_single_group      = group_head == prev_group;

        do {
            elem            = next_elem;
            next_elem       = elem->next;
            /* zero pointer to next elem here because:
             * - user callback may free() the element
             * - push_elem() will fail if next is not NULL
             *   and elem is reused later. For example in
             *   rc/ud transports control.
             */
            elem->next      = NULL;
            elem_list_next  = elem->list.next;
            elem->list.next = NULL;

            ucs_assert(elem->group == group);
            ucs_trace_poll("dispatching arbiter element %p", elem);
            UCS_ARBITER_GUARD_ENTER(arbiter);
            result = cb(arbiter, elem, cb_arg);
            UCS_ARBITER_GUARD_EXIT(arbiter);
            ucs_trace_poll("dispatch result %d", result);
            ++group_dispatch_count;

            if (result == UCS_ARBITER_CB_RESULT_REMOVE_ELEM) {
                 if (elem == last_elem) {
                    /* Only element */
                    group->tail = NULL; /* Group is empty now */
                    if (is_single_group) {
                        next_group = NULL; /* No more groups */
                    } else {
                        /* Remove the group */
                        prev_group->list.next = &next_group->list;
                        next_group->list.prev = &prev_group->list;
                    }
                } else {
                    /* Not only element */
                    ucs_assert(elem == last_elem->next); /* first element should be removed */
                    if (is_single_group) {
                        next_group = next_elem; /* No more groups, point arbiter
                                                   to next element in this group */
                        ucs_list_head_init(&next_elem->list);
                    } else {
                        ucs_list_insert_replace(&prev_group->list,
                                                &next_group->list,
                                                &next_elem->list);
                    }
                    last_elem->next = next_elem; /* Tail points to new head */
                }
            } else if (result == UCS_ARBITER_CB_RESULT_NEXT_GROUP) {
                elem->next = next_elem;
                /* avoid infinite loop */
                elem->list.next = elem_list_next;
                break;
            } else if ((result == UCS_ARBITER_CB_RESULT_DESCHED_GROUP) ||
                       (result == UCS_ARBITER_CB_RESULT_RESCHED_GROUP)) {
                elem->next = next_elem;
                if (is_single_group) {
                    next_group = NULL; /* No more groups */
                } else {
                    prev_group->list.next = &next_group->list;
                    next_group->list.prev = &prev_group->list;
                }
                if (result == UCS_ARBITER_CB_RESULT_RESCHED_GROUP) {
                    ucs_list_add_tail(&resched_groups, &elem->list);
                }
                break;
            } else if (result == UCS_ARBITER_CB_RESULT_STOP) {
                elem->next = next_elem;
                elem->list.next = elem_list_next;
                /* make sure that next dispatch() will continue
                 * from the current group */
                arbiter->current = elem;
                goto out;
            } else {
                elem->next = next_elem;
                elem->list.next = elem_list_next;
                ucs_bug("unexpected return value from arbiter callback");
            }
        } while ((elem != last_elem) && (group_dispatch_count < per_group));
    } while (next_group != NULL);
    arbiter->current = NULL;
out:
    ucs_list_for_each_safe(elem, next_elem, &resched_groups, list) {
        ucs_list_del(&elem->list);
        elem->list.next = NULL;
        ucs_trace_poll("reschedule group %p", elem->group);
        ucs_arbiter_group_schedule_nonempty(arbiter, elem->group);
    }
}

void ucs_arbiter_dump(ucs_arbiter_t *arbiter, FILE *stream)
{
    ucs_arbiter_elem_t *first_group, *group_head, *elem;

    fprintf(stream, "-------\n");
    if (arbiter->current == NULL) {
        fprintf(stream, "(empty)\n");
        goto out;
    }

    first_group = arbiter->current;
    group_head = first_group;
    do {
        elem = group_head;
        if (group_head == first_group) {
            fprintf(stream, "=> ");
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
        group_head = ucs_list_next(&group_head->list, ucs_arbiter_elem_t, list);
    } while (group_head != first_group);

out:
    fprintf(stream, "-------\n");
}
