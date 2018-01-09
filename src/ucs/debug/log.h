/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/


#ifndef UCS_LOG_H_
#define UCS_LOG_H_

#ifdef HAVE_CONFIG_H
#  include "config.h" /* Defines UCS_MAX_LOG_LEVEL */
#endif

#include <ucs/sys/compiler_def.h>
#include <ucs/config/global_opts.h>
#include <stdarg.h>
#include <stdint.h>


BEGIN_C_DECLS


#define ucs_log_is_enabled(_level) \
    ucs_unlikely(((_level) <= UCS_MAX_LOG_LEVEL) && ((_level) <= (ucs_global_opts.log_level)))


#define ucs_log(_level, _fmt, ...) \
    do { \
        if (ucs_log_is_enabled(_level)) { \
            ucs_log_dispatch(__FILE__, __LINE__, __FUNCTION__, (_level), \
                             _fmt, ## __VA_ARGS__); \
        } \
    } while (0)


#define ucs_error(_fmt, ...)        ucs_log(UCS_LOG_LEVEL_ERROR, _fmt, ## __VA_ARGS__)
#define ucs_warn(_fmt, ...)         ucs_log(UCS_LOG_LEVEL_WARN, _fmt,  ## __VA_ARGS__)
#define ucs_info(_fmt, ...)         ucs_log(UCS_LOG_LEVEL_INFO, _fmt, ## __VA_ARGS__)
#define ucs_debug(_fmt, ...)        ucs_log(UCS_LOG_LEVEL_DEBUG, _fmt, ##  __VA_ARGS__)
#define ucs_trace(_fmt, ...)        ucs_log(UCS_LOG_LEVEL_TRACE, _fmt, ## __VA_ARGS__)
#define ucs_trace_req(_fmt, ...)    ucs_log(UCS_LOG_LEVEL_TRACE_REQ, _fmt, ## __VA_ARGS__)
#define ucs_trace_data(_fmt, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_DATA, _fmt, ## __VA_ARGS__)
#define ucs_trace_async(_fmt, ...)  ucs_log(UCS_LOG_LEVEL_TRACE_ASYNC, _fmt, ## __VA_ARGS__)
#define ucs_trace_func(_fmt, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _fmt ")", __FUNCTION__, ## __VA_ARGS__)
#define ucs_trace_poll(_fmt, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_POLL, _fmt, ## __VA_ARGS__)


/**
 * Print a message regardless of current log level. Output can be
 * enabled/disabled via environment variable/configuration settings.
 *
 * During debugging it can be useful to add a few prints to the code
 * without changing a current log level. Also it is useful to be able
 * to see messages only from specific processes. For example, one may
 * want to see prints only from rank 0 when debugging MPI.
 *
 * The function is intended for debugging only. It should not be used
 * in the real code.
 */

#define ucs_print(_fmt, ...) \
    do { \
        if (ucs_global_opts.log_print_enable) { \
            ucs_log_dispatch(__FILE__, __LINE__, __FUNCTION__, \
                             UCS_LOG_LEVEL_PRINT, _fmt, ## __VA_ARGS__); \
        } \
    } while(0)


typedef enum {
    UCS_LOG_FUNC_RC_STOP,
    UCS_LOG_FUNC_RC_CONTINUE
} ucs_log_func_rc_t;


/**
 * Function type for handling log messages.
 *
 * @param file     Source file name.
 * @param line     Source line number.
 * @param function Function name.
 * @param message  Log message - format string
 * @param ap       Log message format parameters.
 *
 * @return UCS_LOG_FUNC_RC_CONTINUE - continue to next log handler
 *         UCS_LOG_FUNC_RC_STOP     - don't continue
 */
typedef ucs_log_func_rc_t (*ucs_log_func_t)(const char *file, unsigned line,
                                            const char *function, ucs_log_level_t level,
                                            const char *message, va_list ap);


extern const char *ucs_log_level_names[];
extern const char *ucs_log_category_names[];


/**
 * Dispatch a logging message.
 *
 * @param [in] file       Source file name.
 * @param [in] line       Source line number.
 * @param [in] function   Function name which generated the log.
 * @param [in] level      Log level of the message,
 * @param [in] message    Log format
 */
void ucs_log_dispatch(const char *file, unsigned line, const char *function,
                      ucs_log_level_t level, const char *format, ...)
    UCS_F_PRINTF(5, 6);


/**
 * Flush logging output.
 */
void ucs_log_flush();


/**
 * Default log handler, which prints the message to the output configured in
 * UCS global options. See @ref ucs_log_func_t.
 */
ucs_log_func_rc_t
ucs_log_default_handler(const char *file, unsigned line, const char *function,
                        ucs_log_level_t level, const char *format, va_list ap);


/**
 * Show a fatal error
 */
void ucs_log_fatal_error(const char *format, ...);


/**
 * Initialize/cleanup logging subsystem.
 */
void ucs_log_early_init();
void ucs_log_init();
void ucs_log_cleanup();


const char *ucs_log_bitmap_to_str(unsigned n, uint8_t *bitmap, size_t length);

const char *ucs_log_dump_hex(const void* data, size_t length, char *buf,
                             size_t max);

/**
 * Add/remove logging handlers
 */
void ucs_log_push_handler(ucs_log_func_t handler);
void ucs_log_pop_handler();

END_C_DECLS

#endif
