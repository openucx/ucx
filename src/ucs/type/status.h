/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
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
    UCS_ERR_NO_RESOURCE            =  -2,
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
    UCS_ERR_LAST                   = -100
} ucs_status_t;


/**
 * A pointer can represent one of these values:
 * - NULL / UCS_OK
 * - Error code pointer (UCS_ERR_xx)
 * - Valid pointer
 */
typedef void *ucs_status_ptr_t;

#define UCS_PTR_STATUS(_ptr)    ((ucs_status_t)(intptr_t)(_ptr))
#define UCS_PTR_IS_ERR(_ptr)    (((uintptr_t)(_ptr)) >= ((uintptr_t)UCS_ERR_LAST))
#define UCS_PTR_IS_PTR(_ptr)    (((uintptr_t)(_ptr) - 1) < ((uintptr_t)UCS_ERR_LAST - 1))
#define UCS_STATUS_PTR(_status) ((void*)(intptr_t)(_status))


/**
 * @param  status UCS status code.
 *
 * @return Verbose status message.
 */
const char *ucs_status_string(ucs_status_t status);


#endif
