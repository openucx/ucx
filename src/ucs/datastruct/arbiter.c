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

static inline void ucs_arbiter_elem_set_scheduled(ucs_arbiter_elem_t *elem,
                                                  ucs_arbiter_group_t *group)
{
    elem->group = group;
}

void ucs_arbiter_group_push_elem_always(ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem)
{
    ucs_arbiter_elem_t *tail = group->tail;

    if (tail == NULL) {
        /* group is empty */
        ucs_arbiter_group_head_reset(elem);
        elem->next = elem;        /* Connect to itself */
    } else {
        elem->next = tail->next;  /* Point to first element */
        tail->next = elem;        /* Point previous element to new one */
    }

    group->tail = elem;   /* Update group tail */
    ucs_arbiter_elem_set_scheduled(elem, group);
}

void ucs_arbiter_group_push_head_elem_always(ucs_arbiter_group_t *group,
                                             ucs_arbiter_elem_t *elem)
{
    ucs_arbiter_elem_t *tail = group->tail;
    ucs_arbiter_elem_t *head;

    UCS_ARBITER_GROUP_GUARD_CHECK(group);
    ucs_assert(!ucs_arbiter_elem_is_scheduled(elem));

    ucs_arbiter_group_head_reset(elem);
    ucs_arbiter_elem_set_scheduled(elem, group);

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
         * can be reused later as well, so it's group should be NULL. */
        ucs_arbiter_elem_init(ptr);
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
            ucs_arbiter_elem_set_scheduled(ptr, group);
            prev       = ptr;
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

size_t ucs_arbiter_group_num_elems(ucs_arbiter_group_t *group)
{
    ucs_arbiter_elem_t *elem = group->tail;
    size_t num_elems;

    if (elem == NULL) {
        return 0;
    }

    num_elems = 0;
    do {
        ++num_elems;
        elem = elem->next;
    } while (elem != group->tail);

    return num_elems;
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

    ucs_assert(!ucs_arbiter_group_is_empty(group));
    head = tail->next;

    ucs_assert(head != NULL);
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

static inline void
ucs_arbiter_remove_and_reset_if_scheduled(ucs_arbiter_elem_t *elem)
{
    if (ucs_unlikely(ucs_arbiter_group_head_is_scheduled(elem))) {
         ucs_list_del(&elem->list);
         ucs_arbiter_group_head_reset(elem);
     }
}

static inline void
ucs_arbiter_group_head_replace(ucs_arbiter_group_t *group,
                               ucs_arbiter_elem_t *group_head,
                               ucs_arbiter_elem_t *new_group_head)
{
    /* check if this is really the group head */
    ucs_assert(!ucs_arbiter_group_is_empty(group));
    ucs_assert(group->tail->next == group_head);

    if (group_head->next == group_head) {
        group->tail          = new_group_head;
    } else {
        new_group_head->next = group_head->next;
    }
    group->tail->next = new_group_head;
}

void ucs_arbiter_dispatch_nonempty(ucs_arbiter_t *arbiter, unsigned per_group,
                                   ucs_arbiter_callback_t cb, void *cb_arg)
{
    ucs_arbiter_elem_t *group_head;
    ucs_arbiter_cb_result_t result;
    unsigned group_dispatch_count;
    ucs_arbiter_group_t *group;
    UCS_LIST_HEAD(resched_list);
    ucs_arbiter_elem_t dummy;

    ucs_assert(!ucs_list_is_empty(&arbiter->list));

    ucs_arbiter_group_head_reset(&dummy);

    do {
        group_head = ucs_list_extract_head(&arbiter->list, ucs_arbiter_elem_t,
                                           list);
        ucs_assert(group_head != NULL);

        /* Reset group head to allow the group to be moved to another arbiter by
         * the dispatch callback. For example, when a DC endpoint is moved from
         * waiting-for-DCI arbiter to waiting-for-TX-resources arbiter.
         */
        ucs_arbiter_group_head_reset(group_head);

        group_dispatch_count = 0;
        group                = group_head->group;
        dummy.group          = group;
        UCS_ARBITER_GROUP_GUARD_CHECK(group);

        for (;;) {
            ucs_assert(group_head->group   == group);
            ucs_assert(dummy.group         == group);
            ucs_assert(group_dispatch_count < per_group);

            /* reset the dispatched element here because:
             * 1. if the element is removed from the arbiter it must be kept in
             *    initialized state otherwise push will fail
             * 2. we can't reset the element after calling the callback because
             *    the callback could release the element.
             */
            ucs_arbiter_elem_init(group_head);
            ucs_assert(!ucs_arbiter_group_head_is_scheduled(group_head));

            /* replace group head by a dummy element, to allow scheduling more
             * elements on this group from the dispatch callback.
             */
            ucs_arbiter_group_head_replace(group, group_head, &dummy);

            /* dispatch the element */
            ucs_trace_poll("dispatching arbiter element %p", group_head);
            UCS_ARBITER_GROUP_GUARD_ENTER(group);
            result = cb(arbiter, group, group_head, cb_arg);
            UCS_ARBITER_GROUP_GUARD_EXIT(group);
            ucs_trace_poll("dispatch result: %d", result);
            ++group_dispatch_count;

            /* recursive push to head (during dispatch) is not allowed */
            ucs_assert(group->tail->next == &dummy);

            /* element is not removed */
            if (ucs_unlikely(result != UCS_ARBITER_CB_RESULT_REMOVE_ELEM)) {
                /* restore group pointer */
                ucs_arbiter_elem_set_scheduled(group_head, group);

                /* the head should not move, since dummy replaces it */
                ucs_assert(!ucs_arbiter_group_head_is_scheduled(group_head));

                /* replace dummy element by group_head */
                ucs_arbiter_group_head_replace(group, &dummy, group_head);

                if (result == UCS_ARBITER_CB_RESULT_DESCHED_GROUP) {
                    /* take over a recursively scheduled group */
                    if (ucs_unlikely(ucs_arbiter_group_head_is_scheduled(&dummy))) {
                        ucs_list_replace(&dummy.list, &group_head->list);
                        UCS_ARBITER_GROUP_ARBITER_SET(group, dummy.group->arbiter);
                        ucs_arbiter_group_head_reset(&dummy);
                    } else {
                        UCS_ARBITER_GROUP_ARBITER_SET(group, NULL);
                    }
                } else {
                    /* remove a recursively scheduled group, give priority
                     * to the original order */
                    ucs_arbiter_remove_and_reset_if_scheduled(&dummy);

                    if (result == UCS_ARBITER_CB_RESULT_NEXT_GROUP) {
                        /* add to arbiter tail */
                        ucs_list_add_tail(&arbiter->list, &group_head->list);
                    } else if (result == UCS_ARBITER_CB_RESULT_RESCHED_GROUP) {
                        /* add to resched list */
                        ucs_list_add_tail(&resched_list, &group_head->list);
                    } else if (result == UCS_ARBITER_CB_RESULT_STOP) {
                        /* exit the outmost loop and make sure that next dispatch()
                         * will continue from the current group */
                        ucs_list_add_head(&arbiter->list, &group_head->list);
                        goto out;
                    } else {
                        ucs_bug("unexpected return value from arbiter callback");
                    }
                }

                break;
            }

            /* last element removed */
            if (dummy.next == &dummy) {
                group->tail = NULL; /* group is empty now */
                group_head  = NULL; /* for debugging */
                ucs_arbiter_remove_and_reset_if_scheduled(&dummy);
                UCS_ARBITER_GROUP_ARBITER_SET(group, NULL);
                break;
            }

            /* non-last element removed */
            group_head        = dummy.next;  /* Update group head */
            group->tail->next = group_head;  /* Tail points to new head */

            if (ucs_unlikely(ucs_arbiter_group_head_is_scheduled(&dummy))) {
                /* take over a recursively scheduled group */
                ucs_list_replace(&dummy.list, &group_head->list);
                ucs_arbiter_group_head_reset(&dummy);
                /* the group is already scheduled, continue to next group */
                break;
            } else if (group_dispatch_count >= per_group) {
                /* add to arbiter tail and continue to next group */
                ucs_list_add_tail(&arbiter->list, &group_head->list);
                break;
            }

            /* continue with new group head */
            ucs_arbiter_group_head_reset(group_head);
        }
    } while (!ucs_list_is_empty(&arbiter->list));

out:
    ucs_list_splice_tail(&arbiter->list, &resched_list);
}

void ucs_arbiter_dump(ucs_arbiter_t *arbiter, FILE *stream)
{
    static const int max_groups = 100;
    ucs_arbiter_elem_t *group_head, *elem;
    int count;

    fprintf(stream, "-------\n");
    if (ucs_list_is_empty(&arbiter->list)) {
        fprintf(stream, "(empty)\n");
        goto out;
    }

    count = 0;
    ucs_list_for_each(group_head, &arbiter->list, list) {
        elem = group_head;
        if (ucs_list_head(&arbiter->list, ucs_arbiter_elem_t, list) == group_head) {
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
        ++count;
        if (count > max_groups) {
            fprintf(stream, "more than %d groups - not printing any more\n",
                    max_groups);
            break;
        }
    }

out:
    fprintf(stream, "-------\n");
}
