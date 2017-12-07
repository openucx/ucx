/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_LIST_H_
#define UCS_LIST_H_

#include <ucs/sys/compiler_def.h>
#include <ucs/datastruct/list_types.h>


#define UCS_LIST_INITIALIZER(_prev, _next) \
    { (_prev), (_next) }


/**
 * Declare an empty list
 */
#define UCS_LIST_HEAD(name) \
    ucs_list_link_t name = UCS_LIST_INITIALIZER(&(name), &(name))


/**
 * Initialize list head.
 *
 * @param head  List head struct to initialize.
 */
static inline void ucs_list_head_init(ucs_list_link_t *head)
{
    head->prev = head->next = head;
}

/**
 * Insert an element in-between to list elements. Any elements which were in this
 * section will be discarded.
 *
 * @param prev Element to insert after
 * @param next Element to insert before.
 */
static inline void ucs_list_insert_replace(ucs_list_link_t *prev,
                                           ucs_list_link_t *next,
                                           ucs_list_link_t *elem)
{
    elem->prev = prev;
    elem->next = next;
    prev->next = elem;
    next->prev = elem;
}

/**
 * Insert an item to a list after another item.
 *
 * @param pos         Item after which to insert.
 * @param new_link    Item to insert.
 */
static inline void ucs_list_insert_after(ucs_list_link_t *pos,
                                         ucs_list_link_t *new_link)
{
    ucs_list_insert_replace(pos, pos->next, new_link);
}

/**
 * Insert an item to a list before another item.
 *
 * @param pos         Item before which to insert.
 * @param new_link    Item to insert.
 */
static inline void ucs_list_insert_before(ucs_list_link_t *pos,
                                          ucs_list_link_t *new_link)
{
    ucs_list_insert_replace(pos->prev, pos, new_link);
}

/**
 * Remove an item from its list.
 *
 * @param link  Item to remove.
 */
static inline void ucs_list_del(ucs_list_link_t *link)
{
    link->prev->next = link->next;
    link->next->prev = link->prev;
}

/**
 * @return Whether the list is empty.
 */
static inline int ucs_list_is_empty(ucs_list_link_t *head)
{
    return head->next == head;
}

/**
 * Move the items from 'newlist' to the tail of the list pointed by 'head'
 *
 * @param head       List to whose tail to add the items.
 * @param newlist    List of items to add.
 *
 * @note The contents of 'newlist' is left unmodified.
 */
static inline void ucs_list_splice_tail(ucs_list_link_t *head,
                                        ucs_list_link_t *newlist)
{
    ucs_list_link_t *first, *last, *tail;

    if (ucs_list_is_empty(newlist)) {
        return;
    }

    first = newlist->next; /* First element in the new list */
    last  = newlist->prev; /* Last element in the new list */
    tail  = head->prev;    /* Last element in the original list */

    first->prev = tail;
    tail->next = first;

    last->next = head;
    head->prev = last;
}

/**
 * Count the members of the list
 */
static inline unsigned long ucs_list_length(ucs_list_link_t *head)
{
    unsigned long length;
    ucs_list_link_t *ptr;

    for (ptr = head->next, length = 0; ptr != head; ptr = ptr->next, ++length);
    return length;
}

/*
 * Convenience macros
 */
#define ucs_list_add_head(_head, _item) \
    ucs_list_insert_after(_head, _item)
#define ucs_list_add_tail(_head, _item) \
    ucs_list_insert_before(_head, _item)

/**
 * Get the previous element in the list
 */
#define ucs_list_prev(_elem, _type, _member) \
    (ucs_container_of((_elem)->prev, _type, _member))

/**
 * Get the next element in the list
 */
#define ucs_list_next(_elem, _type, _member) \
    (ucs_container_of((_elem)->next, _type, _member))

/**
 * Get the first element in the list
 */
#define ucs_list_head   ucs_list_next

/**
 * Get the last element in the list
 */
#define ucs_list_tail   ucs_list_prev

/**
 * Iterate over members of the list.
 */
#define ucs_list_for_each(_elem, _head, _member) \
    for (_elem = ucs_container_of((_head)->next, typeof(*_elem), _member); \
        &(_elem)->_member != (_head); \
        _elem = ucs_container_of((_elem)->_member.next, typeof(*_elem), _member))

/**
 * Iterate over members of the list, the user may invalidate the current entry.
 */
#define ucs_list_for_each_safe(_elem, _telem, _head, _member) \
    for (_elem = ucs_container_of((_head)->next, typeof(*_elem), _member), \
        _telem = ucs_container_of(_elem->_member.next, typeof(*_elem), _member); \
        &_elem->_member != (_head); \
        _elem = _telem, \
        _telem = ucs_container_of(_telem->_member.next, typeof(*_telem), _member))

/**
 * Extract list head
 */
#define ucs_list_extract_head(_head, _type, _member) \
    ({ \
        ucs_list_link_t *tmp = (_head)->next; \
        ucs_list_del(tmp); \
        ucs_container_of(tmp, _type, _member); \
    })

#endif
