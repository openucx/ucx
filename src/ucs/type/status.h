/**
 * @file        uct.h
 * @date        2014-2016
 * @copyright   Mellanox Technologies Ltd. All rights reserved.
 * @copyright   The University of Tennessee and the University of Tennessee research foundation. All rights reserved.
 * @brief       Unified Communication Services
 */

#ifndef UCS_TYPES_STATUS_H_
#define UCS_TYPES_STATUS_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/**
 * @defgroup UCS_API Unified Communication Services (UCS) API
 * @{
 * This section describes UCS API.
 * @}
 */

/**
* @defgroup UCS_RESOURCE   UCS Communication Resource
* @ingroup UCS_API
* @{
* This section describes a concept of the Communication Resource and routines
* associated with the concept.
* @}
*/

/**
 * @ingroup UCS_RESOURCE
 * @brief Status codes
 *
 * @note In order to evaluate the necessary steps to recover from a certain
 * error, all error codes which can be returned by the external API are grouped
 * by the largest entity permanently effected by the error. Each group ranges
 * between its UCS_ERR_FIRST_<name> and UCS_ERR_LAST_<name> enum values.
 * For example, if a link fails it may be sufficient to destroy (and possibly
 * replace) it, in contrast to an endpoint-level error.
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
    UCS_ERR_REMOTE_DISCONNECT      = -24,

    UCS_ERR_FIRST_LINK_FAILURE     = -40,
    UCS_ERR_LAST_LINK_FAILURE      = -59,
    UCS_ERR_FIRST_ENDPOINT_FAILURE = -60,
    UCS_ERR_LAST_ENDPOINT_FAILURE  = -79,
    UCS_ERR_ENDPOINT_TIMEOUT       = -80,

    UCS_ERR_LAST                   = -100
} UCS_S_PACKED ucs_status_t ;


#define UCS_IS_LINK_ERROR(_code) \
    (((_code) <= UCS_ERR_FIRST_LINK_FAILURE) && \
     ((_code) >= UCS_ERR_LAST_LINK_FAILURE)

#define UCS_IS_ENDPOINT_ERROR(_code) \
    (((_code) <= UCS_ERR_FIRST_ENDPOINT_FAILURE) && \
     ((_code) >= UCS_ERR_LAST_ENDPOINT_FAILURE)

/**
 * @ingroup UCS_RESOURCE
 * @brief Status pointer
 *
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
#define UCS_STATUS_IS_ERR(_status)  (_status < 0)


/**
 * @param  status UCS status code.
 *
 * @return Verbose status message.
 */
const char *ucs_status_string(ucs_status_t status);

END_C_DECLS

#endif
