/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCT_TL_LOG_H_
#define UCT_TL_LOG_H_

#include <uct/api/uct.h>
#include <ucs/debug/log.h>


/**
 * In release mode - do nothing.
 *
 * In debug mode, if _condition is not true, return an error. This could be less
 * optimal because of additional checks, and that compiler needs to generate code
 * for error flow as well.
 */
#define UCT_CHECK_PARAM(_condition, _err_message, ...) \
    if (ENABLE_PARAMS_CHECK && !(_condition)) { \
        ucs_error(_err_message, ## __VA_ARGS__); \
        return UCS_ERR_INVALID_PARAM; \
    }


/**
 * In debug mode, if _condition is not true, generate 'Invalid length' error.
 */
#define UCT_CHECK_LENGTH(_condition, _name) \
    UCT_CHECK_PARAM(_condition, "Invalid %s length", _name)


/**
 * In debug mode, check that active message ID is valid.
 */
#define UCT_CHECK_AM_ID(_am_id) \
    UCT_CHECK_PARAM((_am_id) < UCT_AM_ID_MAX, \
                    "Invalid active message id (valid range: 0..%d)", (int)UCT_AM_ID_MAX - 1)


/**
 * In debug mode, print packet description to the log.
 */
#define uct_log_data(_file, _line, _function, _info) \
    __ucs_log(_file, _line, _function, UCS_LOG_LEVEL_TRACE_DATA, "%s", buf);


/**
 * Log callback which prints information about transport headers.
 */
typedef void (*uct_log_data_dump_func_t)(void *data, size_t length, size_t valid_length,
                                         char *bufer, size_t max);


#endif
