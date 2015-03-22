/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_TYPE_THREAD_MODE_H
#define UCS_TYPE_THREAD_MODE_H


/**
 * Specifies thread sharing mode of an object.
 */
typedef enum {
    UCS_THREAD_MODE_SINGLE,   /**< Only one thread can access */
    UCS_THREAD_MODE_FUNNELED, /**< Multiple threads can access, but only one at a time */
    UCS_THREAD_MODE_MULTI,    /**< Multiple threads can access concurrently */
    UCS_THREAD_MODE_LAST
} ucs_thread_mode_t;


#endif
