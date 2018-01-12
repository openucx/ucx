/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "status.h"

#include <stdio.h>


const char *ucs_status_string(ucs_status_t status)
{
    static char error_str[128] = {0};

    switch (status) {
    case UCS_OK:
        return "Success";
    case UCS_INPROGRESS:
        return "Operation in progress";
    case UCS_ERR_NO_MESSAGE:
        return "No pending message";
    case UCS_ERR_NO_RESOURCE:
        return "No resources are available to initiate the operation";
    case UCS_ERR_IO_ERROR:
        return "Input/output error";
    case UCS_ERR_NO_MEMORY:
        return "Out of memory";
    case UCS_ERR_INVALID_PARAM:
        return "Invalid parameter";
    case UCS_ERR_UNREACHABLE:
        return "Destination is unreachable";
    case UCS_ERR_INVALID_ADDR:
        return "Address not valid";
    case UCS_ERR_NOT_IMPLEMENTED:
        return "Function not implemented";
    case UCS_ERR_MESSAGE_TRUNCATED:
        return "Message truncated";
    case UCS_ERR_NO_PROGRESS:
        return "No progress";
    case UCS_ERR_BUFFER_TOO_SMALL:
        return "Provided buffer is too small";
    case UCS_ERR_NO_ELEM:
        return "No such element";
    case UCS_ERR_SOME_CONNECTS_FAILED:
        return "Failed to connect some of the requested endpoints";
    case UCS_ERR_NO_DEVICE:
        return "No such device";
    case UCS_ERR_BUSY:
        return "Device is busy";
    case UCS_ERR_CANCELED:
        return "Request canceled";
    case UCS_ERR_SHMEM_SEGMENT:
        return "Shared memory error";
    case UCS_ERR_ALREADY_EXISTS:
        return "Element already exists";
    case UCS_ERR_OUT_OF_RANGE:
        return "Index out of range";
    case UCS_ERR_TIMED_OUT:
        return "Operation timed out";
    case UCS_ERR_EXCEEDS_LIMIT:
        return "User-defined limit was reached";
    case UCS_ERR_UNSUPPORTED:
        return "Unsupported operation";
    case UCS_ERR_ENDPOINT_TIMEOUT:
        return "Endpoint timeout";
    default:
        snprintf(error_str, sizeof(error_str) - 1, "Unknown error %d", status);
        return error_str;
    };
}
