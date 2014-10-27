/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_TYPES_STATUS_H_
#define UCS_TYPES_STATUS_H_


/**
 * Status codes
 */
typedef enum {
    UCS_SUCCESS              =  0,
    UCS_ERR_INPROGRESS       =  1,
    UCS_ERR_INVALID_PARAM    = -1
} ucs_status_t;


#endif
