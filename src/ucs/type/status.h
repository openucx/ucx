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
    /* Operation completed successfully */
    UCS_OK                         =   0,

    /* Operation is queued and still in progress */
    UCS_INPROGRESS                 =   1,

    /* Failure codes */
    UCS_ERR_NO_MESSAGE             =  -1,
    UCS_ERR_WOULD_BLOCK            =  -2,
    UCS_ERR_IO_ERROR               =  -3,
    UCS_ERR_NO_MEMORY              =  -4,
    UCS_ERR_INVALID_PARAM          =  -5,
    UCS_ERR_UNREACHABLE            =  -6,
    UCS_ERR_INVALID_ADDR           =  -7,
    UCS_ERR_NOT_IMPLEMENTED        =  -8,
    UCS_ERR_MESSAGE_TRUNCATED      =  -9,
    UCS_ERR_NO_PROGRESS            = -10,
    UCS_ERR_BUFFER_TOO_SMALL       = -11,
    UCS_ERR_NO_ELEM                = -12,
    UCS_ERR_SOME_CONNECTS_FAILED   = -13,
    UCS_ERR_NO_DEVICE              = -14,
    UCS_ERR_BUSY                   = -15,
    UCS_ERR_CANCELED               = -16,
    UCS_ERR_SHMEM_SEGMENT          = -17,
    UCS_ERR_ALREADY_EXISTS         = -18,
    UCS_ERR_OUT_OF_RANGE           = -19,
    UCS_ERR_TIMED_OUT              = -20,
    UCS_ERR_EXCEEDS_LIMIT          = -21,
    UCS_ERR_UNSUPPORTED            = -22,
    UCS_ERR_LAST
} ucs_status_t;


/**
 * @param  status UCS status code.
 *
 * @return Verbose status message.
 */
const char *ucs_status_string(ucs_status_t status);


#endif
