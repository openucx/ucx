/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_LIST_TYPES_H_
#define UCS_LIST_TYPES_H_


/**
 * A link in a circular list.
 */
typedef struct ucs_list_link {
    struct ucs_list_link  *prev;
    struct ucs_list_link  *next;
} ucs_list_link_t;


#endif /* UCS_LIST_TYPES_H_ */
