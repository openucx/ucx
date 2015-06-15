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
