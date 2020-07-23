/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARBITER_H_
#define UCS_ARBITER_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <stdio.h>
#include <ucs/debug/assert.h>

/*
 *  A mechanism to arbitrate among groups of queued work elements, which attempts
 * to be "fair" with respect to the groups.
 *
 * - "Arbiter" - the top-level entity.
 * - "Element" - a single work element.
 * - "Group"   - queue of work elements which would be dispatched in-order
 *
 * The arbiter contains a double-linked list of the group head elements. The
 * next group head to dispatch is the first entry in the list. Whenever a group
 * is rescheduled it's moved to the tail of the list. At any point a group head
 * can be removed from the "middle" of the list.
 *
 * The groups and elements are arranged like this:
 *  - every arbitrated element points to the group (head).
 *  - first element in the group points to previous and next group (list)
 *  - first element in the group points to the first element of next group (next_group).
 *  - all except last element point to the next element in same group, and the
 *    last one points to the first (next).
 *
 * Note:
 *  Every elements holds 4 pointers. It could be done with 3 pointers, so that
 *  the pointer to the previous group is put instead of "next" pointer in the last
 *  element in the group, when it is put on the arbiter queue. However in makes
 *  the code much more complicated.
 *
 *
 * Arbiter:
 *   +=============+
 *   | list        +
 *   +======+======+
 *   | next | prev |
 *   +==|===+===|==+
 *      |       +----------------------------------------------+
 *      |                                                      |
 *      +------+                                               |
 *             |                                               |
 *             |                                               |
 * Elements:   V                                               V
 *       +------------+          +------------+          +------------+
 *       | list       |<-------->| list       |<-------->| list       |
 *       +------------+          +------------+          +------------+
 *    +->| next       +---+   +->| next       +---+      + next       +
 *    |  +------------+   |   |  +------------+   |      +------------+
 *    |  | group      |   |   |  | group      |   |      | group      |
 *    |  +------------+   |   |  +------------+   |      +--------+---+
 *    |                   |   |                   |          ^    |
 *    |                   |   |                   |          |    |
 *    |  +------------+   |   |  +------------+   |          |    |
 *    |  | list       |<--+   |  | list       |<--+          |    |
 *    |  +------------+       |  +------------+              |    |
 *    +--+ next       +       +--+ next       |              |    |
 *       +------------+          +------------+              |    |
 *       | group      |          | group      |              |    |
 *       +---------+--+          +--------+---+              |    |
 *            ^    |                 ^    |                  |    |
 * Groups:    |    |                 |    |                  |    |
 *            |    |                 |    |                  |    |
 *     +------+    |          +------+    |           +------+    |
 *     | tail |<---+          | tail |<---+           | tail |<---+
 *     +------+               +------+                +------+
 *
 */

typedef struct ucs_arbiter        ucs_arbiter_t;
typedef struct ucs_arbiter_group  ucs_arbiter_group_t;
typedef struct ucs_arbiter_elem   ucs_arbiter_elem_t;


/**
 * Arbitration callback result codes.
 */
typedef enum {
    UCS_ARBITER_CB_RESULT_REMOVE_ELEM,  /* Remove the current element, move to
                                           the next element. */
    UCS_ARBITER_CB_RESULT_NEXT_GROUP,   /* Keep current element and move to next
                                           group. Group IS NOT descheduled */
    UCS_ARBITER_CB_RESULT_DESCHED_GROUP,/* Keep current element but remove the
                                           current group and move to next group. */
    UCS_ARBITER_CB_RESULT_RESCHED_GROUP,/* Keep current element, do not process
                                           the group anymore during current
                                           dispatch cycle. After dispatch()
                                           is finished group automatically
                                           scheduled */
    UCS_ARBITER_CB_RESULT_STOP          /* Stop dispatching work altogether. Next dispatch()
                                           will start from the group that returned STOP */
} ucs_arbiter_cb_result_t;

#if UCS_ENABLE_ASSERT
#define UCS_ARBITER_GROUP_GUARD_DEFINE          int guard
#define UCS_ARBITER_GROUP_GUARD_INIT(_group)    (_group)->guard = 0
#define UCS_ARBITER_GROUP_GUARD_ENTER(_group)   (_group)->guard++
#define UCS_ARBITER_GROUP_GUARD_EXIT(_group)    (_group)->guard--
#define UCS_ARBITER_GROUP_GUARD_CHECK(_group) \
    ucs_assertv((_group)->guard == 0, \
                "scheduling arbiter group %p while it's being dispatched", _group)
#define UCS_ARBITER_GROUP_ARBITER_DEFINE        ucs_arbiter_t *arbiter
#define UCS_ARBITER_GROUP_ARBITER_SET(_group, _arbiter) \
    (_group)->arbiter = (_arbiter)
#else
#define UCS_ARBITER_GROUP_GUARD_DEFINE
#define UCS_ARBITER_GROUP_GUARD_INIT(_group)
#define UCS_ARBITER_GROUP_GUARD_ENTER(_group)
#define UCS_ARBITER_GROUP_GUARD_EXIT(_group)
#define UCS_ARBITER_GROUP_GUARD_CHECK(_group)
#define UCS_ARBITER_GROUP_ARBITER_DEFINE
#define UCS_ARBITER_GROUP_ARBITER_SET(_group, _arbiter)
#endif

#define UCS_ARBITER_GROUP_ARBITER_CHECK(_group, _arbiter) \
    ucs_assertv((_group)->arbiter == (_arbiter), \
                "%p == %p", (_group)->arbiter, _group)


/**
 * Arbiter callback function.
 *
 * @param [in]  arbiter  The arbiter.
 * @param [in]  elem     Current work element.
 * @param [in]  arg      User-defined argument.
 *
 * @return According to @ref ucs_arbiter_cb_result_t.
 */
typedef ucs_arbiter_cb_result_t (*ucs_arbiter_callback_t)(ucs_arbiter_t *arbiter,
                                                          ucs_arbiter_group_t *group,
                                                          ucs_arbiter_elem_t *elem,
                                                          void *arg);


/**
 * Top-level arbiter.
 */
struct ucs_arbiter {
    ucs_list_link_t         list;
};


/**
 * Arbitration group.
 */
struct ucs_arbiter_group {
    ucs_arbiter_elem_t      *tail;
    UCS_ARBITER_GROUP_GUARD_DEFINE;
    UCS_ARBITER_GROUP_ARBITER_DEFINE;
};


/**
 * Arbitrated work element.
 * In order to keep it small, one of the fields is a union.
 */
struct ucs_arbiter_elem {
    ucs_list_link_t         list;       /* List link in the scheduler queue */
    ucs_arbiter_elem_t      *next;      /* Next element, last points to head */
    ucs_arbiter_group_t     *group;     /* Always points to the group */
};


/**
 * Initialize the arbiter object.
 *
 * @param [in]  arbiter  Arbiter object to initialize.
 */
void ucs_arbiter_init(ucs_arbiter_t *arbiter);
void ucs_arbiter_cleanup(ucs_arbiter_t *arbiter);


/**
 * Initialize a group object.
 *
 * @param [in]  group    Group to initialize.
 */
void ucs_arbiter_group_init(ucs_arbiter_group_t *group);
void ucs_arbiter_group_cleanup(ucs_arbiter_group_t *group);


/**
 * Initialize an element object.
 *
 * @param [in]  elem    Element to initialize.
 */
static inline void ucs_arbiter_elem_init(ucs_arbiter_elem_t *elem)
{
    elem->group = NULL;
}


/**
 * Check if a group is scheduled on an arbiter.
 */
int ucs_arbiter_group_is_scheduled(ucs_arbiter_group_t *group);


/**
 * Add a new work element to a group - internal function
 */
void ucs_arbiter_group_push_elem_always(ucs_arbiter_group_t *group,
                                        ucs_arbiter_elem_t *elem);


/**
 * Add a new work element to the head of a group - internal function
 */
void ucs_arbiter_group_push_head_elem_always(ucs_arbiter_group_t *group,
                                             ucs_arbiter_elem_t *elem);


/**
 * Call the callback for each element from a group. If the callback returns
 * UCS_ARBITER_CB_RESULT_REMOVE_ELEM, remove it from the group.
 *
 * @param [in]  arbiter  Arbiter object to remove the group from.
 * @param [in]  group    Group to clean up.
 * @param [in]  cb       Callback to be called for each element.
 * @param [in]  cb_arg   Last argument for the callback.
 */
void ucs_arbiter_group_purge(ucs_arbiter_t *arbiter, ucs_arbiter_group_t *group,
                             ucs_arbiter_callback_t cb, void *cb_arg);


/**
 * @return Number of elements in the group
 */
size_t ucs_arbiter_group_num_elems(ucs_arbiter_group_t *group);


void ucs_arbiter_dump(ucs_arbiter_t *arbiter, FILE *stream);


/* Internal function */
void ucs_arbiter_group_schedule_nonempty(ucs_arbiter_t *arbiter,
                                         ucs_arbiter_group_t *group);


/* Internal function */
void ucs_arbiter_group_desched_nonempty(ucs_arbiter_t *arbiter,
                                        ucs_arbiter_group_t *group);


/* Internal function */
void ucs_arbiter_dispatch_nonempty(ucs_arbiter_t *arbiter, unsigned per_group,
                                   ucs_arbiter_callback_t cb, void *cb_arg);


/**
 * Return true if arbiter has no groups scheduled
 *
 * @param [in]  arbiter    Arbiter object to dispatch work on.
 */
static inline int ucs_arbiter_is_empty(ucs_arbiter_t *arbiter)
{
    return ucs_list_is_empty(&arbiter->list);
}


/**
 * @return the last group element.
 */
static inline ucs_arbiter_elem_t*
ucs_arbiter_group_tail(ucs_arbiter_group_t *group)
{
    return group->tail;
}


/**
 * @return whether the group does not have any queued elements.
 */
static inline int ucs_arbiter_group_is_empty(ucs_arbiter_group_t *group)
{
    return group->tail == NULL;
}


/**
 * Schedule a group for arbitration. If the group is already there, the operation
 * will have no effect.
 *
 * @param [in]  arbiter  Arbiter object to schedule the group on.
 * @param [in]  group    Group to schedule.
 */
static inline void ucs_arbiter_group_schedule(ucs_arbiter_t *arbiter,
                                              ucs_arbiter_group_t *group)
{
    if (ucs_unlikely(!ucs_arbiter_group_is_empty(group))) {
        ucs_arbiter_group_schedule_nonempty(arbiter, group);
    }
}


/**
 * Deschedule already scheduled group. If the group is not scheduled, the operation
 * will have no effect
 *
 * @param [in]  arbiter  Arbiter object that  group on.
 * @param [in]  group    Group to deschedule.
 */
static inline void ucs_arbiter_group_desched(ucs_arbiter_t *arbiter,
                                             ucs_arbiter_group_t *group)
{
    if (ucs_unlikely(!ucs_arbiter_group_is_empty(group))) {
        ucs_arbiter_group_desched_nonempty(arbiter, group);
    }
}


/**
 * @return Whether the element is queued in an arbiter group.
 *         (an element can't be queued more than once)
 */
static inline int ucs_arbiter_elem_is_scheduled(ucs_arbiter_elem_t *elem)
{
    return elem->group != NULL;
}


/**
 * Add a new work element to a group if it is not already there
 *
 * @param [in]  group    Group to add the element to.
 * @param [in]  elem     Work element to add.
 */
static inline void
ucs_arbiter_group_push_elem(ucs_arbiter_group_t *group,
                            ucs_arbiter_elem_t *elem)
{
    if (ucs_arbiter_elem_is_scheduled(elem)) {
        return;
    }

    ucs_arbiter_group_push_elem_always(group, elem);
}


/**
 * Add a new work element to the head of a group if it is not already there
 *
 * @param [in]  group    Group to add the element to.
 * @param [in]  elem     Work element to add.
 */
static inline void
ucs_arbiter_group_push_head_elem(ucs_arbiter_group_t *group,
                                 ucs_arbiter_elem_t *elem)
{
    if (ucs_arbiter_elem_is_scheduled(elem)) {
        return;
    }

    ucs_arbiter_group_push_head_elem_always(group, elem);
}


/**
 * Dispatch work elements in the arbiter. For every group, up to per_group work
 * elements are dispatched, as long as the callback returns REMOVE_ELEM or
 * NEXT_GROUP. Then, the same is done for the next group, until either the
 * arbiter becomes empty or the callback returns STOP. If a group is either out
 * of elements, or its callback returns REMOVE_GROUP, it will be removed until
 * ucs_arbiter_group_schedule() is used to put it back on the arbiter.
 *
 * @param [in]  arbiter    Arbiter object to dispatch work on.
 * @param [in]  per_group  How many elements to dispatch from each group.
 * @param [in]  cb         User-defined callback to be called for each element.
 * @param [in]  cb_arg     Last argument for the callback.
 */
static inline void
ucs_arbiter_dispatch(ucs_arbiter_t *arbiter, unsigned per_group,
                     ucs_arbiter_callback_t cb, void *cb_arg)
{
    if (ucs_unlikely(!ucs_arbiter_is_empty(arbiter))) {
        ucs_arbiter_dispatch_nonempty(arbiter, per_group, cb, cb_arg);
    }
}


/**
 * @return true if element is the only one in the group
 */
static inline int
ucs_arbiter_elem_is_only(ucs_arbiter_elem_t *elem)
{
    return elem->next == elem;
}

#endif
