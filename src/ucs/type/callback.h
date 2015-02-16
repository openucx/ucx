/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_TYPE_CALLBACK_H
#define UCS_TYPE_CALLBACK_H


typedef struct ucs_callback  ucs_callback_t;
typedef void                 (*ucs_callback_func_t)(ucs_callback_t *self);

/**
 * A generic callback which can be embedded into structures.
 */
struct ucs_callback {
    ucs_callback_func_t func;
};

#endif
