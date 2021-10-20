/*
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_HLIST_H_
#define UCS_HLIST_H_

#include "list.h"

#include <stddef.h>


BEGIN_C_DECLS

/**
 * Detached-head circular list: unlike the basic double-linked list, the head
 * element is separate from the list, and it points to first element, or NULL if
 * the list is empty.
 * It reduces the size of list head from 2 pointers to 1 pointer, and allows
 * storing the head element inside a reallocating hash table, but adds some
 * overhead to basic list operations.
 */


/**
 * List element of a detached-head list.
 */
typedef struct ucs_hlist_link {
    ucs_list_link_t list;
} ucs_hlist_link_t;


/**
 * Head of a circular detached-head list.
 */
typedef struct ucs_hlist_head {
    ucs_hlist_link_t *ptr;
} ucs_hlist_head_t;


/**
 * Initialize a detached-head list.
 *
 * @param [in] head   List head to initialize.
 */
static UCS_F_ALWAYS_INLINE void
ucs_hlist_head_init(ucs_hlist_head_t *head)
{
    head->ptr = NULL;
}


/**
 * Check if a detached-head list is empty.
 *
 * @param [in] head   List head to check.
 *
 * @return Whether the list is empty.
 */
static UCS_F_ALWAYS_INLINE int
ucs_hlist_is_empty(const ucs_hlist_head_t *head)
{
    return head->ptr == NULL;
}


/**
 * Common function to add elements to the list head or tail.
 *
 * @param [in] head              List head to add to.
 * @param [in] elem              Element to add.
 * @param [in] set_head_to_elem  Whether to set the list head to the newly added
 *                               element.
 */
static UCS_F_ALWAYS_INLINE void
ucs_hlist_add_common(ucs_hlist_head_t *head, ucs_hlist_link_t *elem,
                     int set_head_to_elem)
{
    if (head->ptr == NULL) {
        head->ptr = elem;
        ucs_list_head_init(&elem->list);
    } else {
        ucs_list_insert_before(&head->ptr->list, &elem->list);
        if (set_head_to_elem) {
            head->ptr = elem;
        }
    }
}


/**
 * Add element to the beginning of a detached-head list.
 *
 * @param [in] head  List head to add to.
 * @param [in] elem  Element to add.
 */
static UCS_F_ALWAYS_INLINE void
ucs_hlist_add_head(ucs_hlist_head_t *head, ucs_hlist_link_t *elem)
{
    ucs_hlist_add_common(head, elem, 1);
}


/**
 * Add element to the end of a detached-head list.
 *
 * @param [in] head  List head to add to.
 * @param [in] elem  Element to add.
 */
static UCS_F_ALWAYS_INLINE void
ucs_hlist_add_tail(ucs_hlist_head_t *head, ucs_hlist_link_t *elem)
{
    ucs_hlist_add_common(head, elem, 0);
}


/**
 * Remove an element from a detached-head list.
 *
 * @param [in] head    List head to remove from.
 * @param [in] elem    Element to remove.
 * @param [in] is_head Flag that shows that the element is head
 *
 * @note If the element is not present in the list, this function has undefined
 *       behavior.
 */
static UCS_F_ALWAYS_INLINE void
ucs_hlist_del_common(ucs_hlist_head_t *head, ucs_hlist_link_t *elem,
                     int is_head)
{
    if (ucs_list_is_empty(&elem->list)) {
        /* Remove elem if it's not the only one in the list.
         * We assume here that head->ptr == elem, but cannot assert() to avoid
         * dependency of assert.h */
        head->ptr = NULL;
    } else {
        if (is_head) {
            /* removing head of non-empty list, point to next elem */
            head->ptr = ucs_list_next(&elem->list, ucs_hlist_link_t, list);
        }
        ucs_list_del(&elem->list);
    }
}


/**
 * Remove an element from a detached-head list.
 *
 * @param [in] head  List head to remove from.
 * @param [in] elem  Element to remove.
 *
 * @note If the element is not present in the list, this function has undefined
 *       behavior.
 */
static UCS_F_ALWAYS_INLINE void
ucs_hlist_del(ucs_hlist_head_t *head, ucs_hlist_link_t *elem)
{
    ucs_hlist_del_common(head, elem, elem == head->ptr);
}


/**
 * Remove the first element from a detached-head list, and return it.
 *
 * @param [in] head  List head to remove from.
 *
 * @return The former list head element, or NULL if the list is empty.
 */
static UCS_F_ALWAYS_INLINE ucs_hlist_link_t*
ucs_hlist_extract_head(ucs_hlist_head_t *head)
{
    ucs_hlist_link_t *elem;

    if (head->ptr == NULL) {
        return NULL;
    }

    elem = head->ptr;
    ucs_hlist_del_common(head, elem, 1);

    return elem;
}


/**
 * Get list head element as the containing type, assuming the list is not empty.
 *
 * @param _head    List head.
 * @param _type    Containing structure type.
 * @param _member  List element inside the containing structure.
 *
 * @note If the list is empty this macro has undefined behavior.
 */
#define ucs_hlist_head_elem(_head, _type, _member) \
    ucs_container_of((_head)->ptr, _type, _member)


/**
 * Get list next element as the containing type.
 *
 * @param _elem    List element.
 * @param _type    Containing structure type.
 * @param _member  List element inside the containing structure.
 */
#define ucs_hlist_next_elem(_elem, _member) \
    ucs_container_of(ucs_list_next(&(_elem)->_member.list, ucs_hlist_link_t, \
                                   list), \
                     ucs_typeof(*(_elem)), _member)


/**
 * Iterate over detached-head list.
 *
 * @param _elem     Variable to hold the current list element
 * @param _head     Pointer to list head.
 * @param _member   List element inside the containing structure.
 *
 * @note The iteration is implemented by first setting the element to NULL, then
 * inside 'for' loop condition (which is done before each iteration), we advance
 * the element pointer and check for end condition: in the first iteration,
 * when elem is NULL, we check that the list is not empty. For subsequent
 * iterations, we check that elem has not reached the list head yet.
 */
#define ucs_hlist_for_each(_elem, _head, _member) \
    for (_elem = NULL; \
         (_elem == NULL) ? \
             /* First iteration: check _head->ptr != NULL. 2nd && condition is \
              * actually _elem != NULL which is always expected to be true. \
              * We can't check (&_elem->_member != NULL) because some compilers \
              * assume pointer-to-member is never NULL */ \
             (!ucs_hlist_is_empty(_head) && \
              ((_elem = ucs_hlist_head_elem(_head, ucs_typeof(*(_elem)), _member)) \
                     != NULL)) : \
             /* rest of iterations: check _elem != _head->ptr */ \
             ((_elem = ucs_hlist_next_elem(_elem, _member)) != \
                     ucs_hlist_head_elem(_head, ucs_typeof(*(_elem)), _member)); \
         )


/**
 * Remove the first element from a detached-head list, and return its containing
 * type. The function is intended for internal use only in hlist. It has to be
 * used for non-empty hlist, otherwise, the result of the function is undefined.
 *
 * @param _head    List head to remove from.
 * @param _type    Type of the structure containing list element.
 * @param _member  List element inside the containing structure.
 */
#define ucs_hlist_extract_head_elem(_head, _type, _member) \
    ucs_container_of(ucs_hlist_extract_head(_head), _type, _member)


/**
 * Iterate over detached-head list, while removing the head element, until the
 * list becomes empty.
 *
 * @param _elem     Variable to hold the current list element
 * @param _head     Pointer to list head.
 * @param _member   List element inside the containing structure.
 */
#define ucs_hlist_for_each_extract(_elem, _head, _member) \
    for (_elem = ucs_hlist_extract_head_elem(_head, ucs_typeof(*(_elem)), _member); \
         _elem != UCS_PTR_BYTE_OFFSET(NULL, -ucs_offsetof(ucs_typeof(*(_elem)), _member)); \
         _elem = ucs_hlist_extract_head_elem(_head, ucs_typeof(*(_elem)), _member))


/**
 * Iterate over detached-head list, while removing the head element passes
 * @a _cond or the list becomes empty.
 *
 * @param _elem     Variable to hold the current list element
 * @param _head     Pointer to list head.
 * @param _member   List element inside the containing structure.
 * @param _cond     Condition to test for @a _head element before extract it.
 */
#define ucs_hlist_for_each_extract_if(_elem, _head, _member, _cond) \
    for (_elem = ucs_hlist_head_elem(_head, ucs_typeof(*(_elem)), _member); \
         (_elem != UCS_PTR_BYTE_OFFSET(NULL, -ucs_offsetof(ucs_typeof(*(_elem)), \
                                                           _member))) && \
         (_cond) && ucs_hlist_extract_head(_head); \
         _elem = ucs_hlist_head_elem(_head, ucs_typeof(*(_elem)), _member))


END_C_DECLS

#endif
