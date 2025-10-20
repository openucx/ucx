/**
 * @file        status.h
 * @date        2014-2019
 * @copyright   NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
 * @copyright   The University of Tennessee and the University of Tennessee research foundation. All rights reserved.
 * @brief       Unified Communication Services
 */

#ifndef UCS_TYPES_STATUS_H_
#define UCS_TYPES_STATUS_H_

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

/** @file status.h */

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
 * @brief X-macro for defining status codes and their string representations
 *
 * This macro allows defining status codes and their associated messages in one
 * place, avoiding duplication between enum definitions and string conversions.
 *
 * Usage: UCS_FOREACH_STATUS(_macro) where _macro(ID, VALUE, MSG) is expanded for each status.
 */
#define UCS_FOREACH_STATUS(_macro) \
    _macro(UCS_OK,                           0, "Success") \
    _macro(UCS_INPROGRESS,                   1, "Operation in progress") \
    _macro(UCS_ERR_NO_MESSAGE,              -1, "No pending message") \
    _macro(UCS_ERR_NO_RESOURCE,             -2, "No resources are available to initiate the operation") \
    _macro(UCS_ERR_IO_ERROR,                -3, "Input/output error") \
    _macro(UCS_ERR_NO_MEMORY,               -4, "Out of memory") \
    _macro(UCS_ERR_INVALID_PARAM,           -5, "Invalid parameter") \
    _macro(UCS_ERR_UNREACHABLE,             -6, "Destination is unreachable") \
    _macro(UCS_ERR_INVALID_ADDR,            -7, "Address not valid") \
    _macro(UCS_ERR_NOT_IMPLEMENTED,         -8, "Function not implemented") \
    _macro(UCS_ERR_MESSAGE_TRUNCATED,       -9, "Message truncated") \
    _macro(UCS_ERR_NO_PROGRESS,            -10, "No progress") \
    _macro(UCS_ERR_BUFFER_TOO_SMALL,       -11, "Provided buffer is too small") \
    _macro(UCS_ERR_NO_ELEM,                -12, "No such element") \
    _macro(UCS_ERR_SOME_CONNECTS_FAILED,   -13, "Failed to connect some of the requested endpoints") \
    _macro(UCS_ERR_NO_DEVICE,              -14, "No such device") \
    _macro(UCS_ERR_BUSY,                   -15, "Device is busy") \
    _macro(UCS_ERR_CANCELED,               -16, "Request canceled") \
    _macro(UCS_ERR_SHMEM_SEGMENT,          -17, "Shared memory error") \
    _macro(UCS_ERR_ALREADY_EXISTS,         -18, "Element already exists") \
    _macro(UCS_ERR_OUT_OF_RANGE,           -19, "Index out of range") \
    _macro(UCS_ERR_TIMED_OUT,              -20, "Operation timed out") \
    _macro(UCS_ERR_EXCEEDS_LIMIT,          -21, "User-defined limit was reached") \
    _macro(UCS_ERR_UNSUPPORTED,            -22, "Unsupported operation") \
    _macro(UCS_ERR_REJECTED,               -23, "Operation rejected by remote peer") \
    _macro(UCS_ERR_NOT_CONNECTED,          -24, "Endpoint is not connected") \
    _macro(UCS_ERR_CONNECTION_RESET,       -25, "Connection reset by remote peer") \
    _macro(UCS_ERR_FIRST_LINK_FAILURE,     -40, "First link failure") \
    _macro(UCS_ERR_LAST_LINK_FAILURE,      -59, "Last link failure") \
    _macro(UCS_ERR_FIRST_ENDPOINT_FAILURE, -60, "First endpoint failure") \
    _macro(UCS_ERR_ENDPOINT_TIMEOUT,       -80, "Endpoint timeout") \
    _macro(UCS_ERR_LAST_ENDPOINT_FAILURE,  -89, "Last endpoint failure") \
    _macro(UCS_ERR_LAST,                  -100, "Last error code")

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
#define UCS_STATUS_ENUMIFY(ID, VALUE, _) ID = VALUE,

typedef enum {
    UCS_FOREACH_STATUS(UCS_STATUS_ENUMIFY)
} UCS_S_PACKED ucs_status_t;


#define UCS_IS_LINK_ERROR(_code) \
    (((_code) <= UCS_ERR_FIRST_LINK_FAILURE) && \
     ((_code) >= UCS_ERR_LAST_LINK_FAILURE))

#define UCS_IS_ENDPOINT_ERROR(_code) \
    (((_code) <= UCS_ERR_FIRST_ENDPOINT_FAILURE) && \
     ((_code) >= UCS_ERR_LAST_ENDPOINT_FAILURE))

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

#define UCS_PTR_IS_ERR(_ptr)       (((uintptr_t)(_ptr)) >= ((uintptr_t)UCS_ERR_LAST))
#define UCS_PTR_IS_PTR(_ptr)       (((uintptr_t)(_ptr) - 1) < ((uintptr_t)UCS_ERR_LAST - 1))
#define UCS_PTR_RAW_STATUS(_ptr)   ((ucs_status_t)(intptr_t)(_ptr))
#define UCS_PTR_STATUS(_ptr)       (UCS_PTR_IS_PTR(_ptr) ? UCS_INPROGRESS : UCS_PTR_RAW_STATUS(_ptr))
#define UCS_STATUS_PTR(_status)    ((void*)(intptr_t)(_status))
#define UCS_STATUS_IS_ERR(_status) ((_status) < 0)

/**
 * @brief Helper macro to generate switch case for status to string conversion
 */
#define UCS_STATUS_STRINGIFY(ID, _, MSG) case ID: return MSG;

/**
 * @brief Common status code to string cases
 *
 * This macro defines the common switch cases for converting status codes to
 * strings. It's used by both the host and device implementations to avoid
 * code duplication.
 */
#define UCS_STATUS_STRING_CASES UCS_FOREACH_STATUS(UCS_STATUS_STRINGIFY)

/**
 * @param  status UCS status code.
 *
 * @return Verbose status message.
 */
const char *ucs_status_string(ucs_status_t status);

END_C_DECLS

#endif
