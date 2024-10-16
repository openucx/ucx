/*
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2014. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_LIST_H_
#define UCS_LIST_H_

#include <ucs/debug/debug.h>
#include <ucs/sys/compiler_def.h>
#include <ucs/sys/math.h>
#include <ucs/debug/memtrack_int.h>


BEGIN_C_DECLS

#define UCS_LIST_INITIALIZER(_prev, _next) \
    { (_prev), (_next) }

#define UCS_LIST_CHECK_ELEM_IS_VALID(_elem) \
    UCS_ASAN_ADDRESS_IS_VALID(_elem, sizeof(ucs_list_link_t));

/**
 * Declare an empty list
 */
#define UCS_LIST_HEAD(name) \
    ucs_list_link_t name = UCS_LIST_INITIALIZER(&(name), &(name))


/**
 * A link in a circular list.
 */
typedef struct ucs_list_link {
    struct ucs_list_link  *prev;
    struct ucs_list_link  *next;
} ucs_list_link_t;


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
 * Replace an element in a list with another element.
 *
 * @param elem         Element in the list to replace.
 * @param replacement  New element to insert in place of 'elem'.
 */
static inline void ucs_list_replace(ucs_list_link_t *elem,
                                    ucs_list_link_t *replacement)
{
    ucs_list_insert_replace(elem->prev, elem->next, replacement);
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
static inline void ucs_list_del(ucs_list_link_t *elem)
{
    elem->prev->next = elem->next;
    elem->next->prev = elem->prev;
}

/**
 * @return Whether the list is empty.
 */
static inline int ucs_list_is_empty(const ucs_list_link_t *head)
{
    UCS_LIST_CHECK_ELEM_IS_VALID(head->next);
    return head->next == head;
}

/**
 * @return Whether @a elem is the first element in the list @a head.
 */
static inline int
ucs_list_is_first(const ucs_list_link_t *head, const ucs_list_link_t *elem)
{
    UCS_LIST_CHECK_ELEM_IS_VALID(head);
    UCS_LIST_CHECK_ELEM_IS_VALID(elem->prev);
    return elem->prev == head;
}

/**
 * @return Whether @a elem is the last element in the list @a head.
 */
static inline int
ucs_list_is_last(const ucs_list_link_t *head, const ucs_list_link_t *elem)
{
    UCS_LIST_CHECK_ELEM_IS_VALID(head);
    UCS_LIST_CHECK_ELEM_IS_VALID(elem->next);
    return elem->next == head;
}

/**
 * @return Whether the list @a head contains only then element @a elem.
 */
static inline int
ucs_list_is_only(const ucs_list_link_t *head, const ucs_list_link_t *elem)
{
    return ucs_list_is_first(head, elem) && ucs_list_is_last(head, elem);
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

    UCS_LIST_CHECK_ELEM_IS_VALID(head->next);
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
    UCS_LIST_CHECK_ELEM_IS_VALID(_head); \
    for (_elem = ucs_container_of((_head)->next, ucs_typeof(*_elem), _member); \
        &(_elem)->_member != (_head); \
        _elem = ucs_container_of((_elem)->_member.next, ucs_typeof(*_elem), _member))

/**
 * Iterate over members of the list, the user may invalidate the current entry.
 */
#define ucs_list_for_each_safe(_elem, _telem, _head, _member) \
    UCS_LIST_CHECK_ELEM_IS_VALID(_head); \
    for (_elem = ucs_container_of((_head)->next, ucs_typeof(*_elem), _member), \
        _telem = ucs_container_of(_elem->_member.next, ucs_typeof(*_elem), _member); \
        &_elem->_member != (_head); \
        _elem = _telem, \
        _telem = ucs_container_of(_telem->_member.next, ucs_typeof(*_telem), _member))

/**
 * Extract list head
 */
#define ucs_list_extract_head(_head, _type, _member) \
    ({ \
        ucs_list_link_t *tmp = (_head)->next; \
        ucs_list_del(tmp); \
        ucs_container_of(tmp, _type, _member); \
    })

/**
 * Print the list links
 *
 * @param head       List head to print
 * @param num_nodes  Number of elements in the list
 */
static inline void ucs_list_print_links(ucs_list_link_t *head, unsigned int num_nodes)
{
    ucs_list_link_t *node;
    unsigned int i;

    node = head;
    for (i = 0; i < num_nodes; i++) {
        printf("Node #%u: %p prev=%p next=%p", i, (void *)node, (void *)node->prev, (void *)node->next);
        node = node->next;
    }
}

/**
 * Mix the order of the list elements
 *
 * @param head       List head to shuffle
 * @param num_nodes  Number of elements in the list
 */
static inline void ucs_list_mix_order(ucs_list_link_t *head,
                                      unsigned int num_nodes)
{
    ucs_list_link_t **nodes_array;
    ucs_list_link_t *temp_node;
    int i, j;

    if (num_nodes <= 1) {
        printf("Nothing to shuffle, num_nodes=%u", num_nodes);
        return;
    }

    nodes_array = (ucs_list_link_t **)ucs_malloc(num_nodes * sizeof(*nodes_array), "nodes_array");
    if (nodes_array == NULL) {
        printf("Failed to allocate memory for nodes array");
        return;
    }

    temp_node = head;
    for (i = 0; i < num_nodes; i++) {
        nodes_array[i] = temp_node;
        temp_node = temp_node->next;
    }

    /* Fisher-Yates shuffle algorithm */
    for (i = num_nodes - 1; i > 0; i--) {
        j = ucs_rand() % (i + 1);
        temp_node = nodes_array[i];
        nodes_array[i] = nodes_array[j];
        nodes_array[j] = temp_node;

        nodes_array[i]->next = (i < num_nodes - 1)? nodes_array[i + 1] : NULL;
        nodes_array[i]->prev = (i > 0)? nodes_array[i - 1] : NULL;
    }

    nodes_array[0]->prev = nodes_array[num_nodes - 1];
    nodes_array[num_nodes - 1]->next = nodes_array[0];

    ucs_free(nodes_array);
}

END_C_DECLS

#endif
