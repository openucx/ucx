/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
* Copyright (C) The University of Tennessee and The University 
*               of Tennessee Research Foundation. 2015. ALL RIGHTS RESERVED.
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
    UCS_THREAD_MODE_SINGLE,     /**< Only the main thread can access (multiple threads may exist and never access) */
    UCS_THREAD_MODE_SERIALIZED, /**< Multiple threads can access, but only one at a time */
    UCS_THREAD_MODE_MULTI,      /**< Multiple threads can access concurrently */
    UCS_THREAD_MODE_LAST
} ucs_thread_mode_t;


#endif
