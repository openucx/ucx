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

#include <ucs/sys/compiler.h>
#include <ucs/config/global_opts.h>
#include <stdint.h>

BEGIN_C_DECLS

#define ucs_log_enabled(_level) \
    ucs_unlikely(((_level) <= UCS_MAX_LOG_LEVEL) && ((_level) <= (ucs_global_opts.log_level)))

#define ucs_log(_level, _message, ...) \
    do { \
        if (ucs_log_enabled(_level)) { \
            __ucs_log(__FILE__, __LINE__, __FUNCTION__, (_level), \
                      _message, ## __VA_ARGS__); \
        } \
    } while (0)


#define ucs_assert_always(_expression) \
    do { \
        if (!ucs_likely(_expression)) { \
            __ucs_abort("assertion failure", __FILE__, __LINE__, __FUNCTION__, \
                        "Assertion `%s' failed", #_expression); \
        } \
    } while (0)

#define ucs_assertv_always(_expression, _fmt, ...) \
    do { \
        if (!ucs_likely(_expression)) { \
            __ucs_abort("assertion failure", __FILE__, __LINE__, __FUNCTION__, \
                        "Assertion `%s' failed: " _fmt, #_expression, \
                        ## __VA_ARGS__); \
        } \
    } while (0)

#if ENABLE_ASSERT

#define ucs_bug(_message, ...)                  __ucs_abort("bug", __FILE__, \
                                                            __LINE__, __FUNCTION__, \
                                                            "Bug: " _message, ## __VA_ARGS__)
#define ucs_assert(_expression)                 ucs_assert_always(_expression)
#define ucs_assertv(_expression, _fmt, ...)     ucs_assertv_always(_expression, _fmt, ## __VA_ARGS__)

#else

#define ucs_bug(_message, ...)
#define ucs_assert(_expression)
#define ucs_assertv(_expression, _fmt, ...)

#endif


#define ucs_fatal(_message, ...) \
    __ucs_abort("fatal error", __FILE__, __LINE__, __FUNCTION__, \
                "Fatal: " _message, ## __VA_ARGS__)

#define ucs_error(_message, ...)        ucs_log(UCS_LOG_LEVEL_ERROR, _message, ## __VA_ARGS__)
#define ucs_warn(_message, ...)         ucs_log(UCS_LOG_LEVEL_WARN, _message,  ## __VA_ARGS__)
#define ucs_info(_message, ...)         ucs_log(UCS_LOG_LEVEL_INFO, _message, ## __VA_ARGS__)
#define ucs_debug(_message, ...)        ucs_log(UCS_LOG_LEVEL_DEBUG, _message, ##  __VA_ARGS__)
#define ucs_trace(_message, ...)        ucs_log(UCS_LOG_LEVEL_TRACE, _message, ## __VA_ARGS__)
#define ucs_trace_req(_message, ...)    ucs_log(UCS_LOG_LEVEL_TRACE_REQ, _message, ## __VA_ARGS__)
#define ucs_trace_data(_message, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_DATA, _message, ## __VA_ARGS__)
#define ucs_trace_async(_message, ...)  ucs_log(UCS_LOG_LEVEL_TRACE_ASYNC, _message, ## __VA_ARGS__)
#define ucs_trace_func(_message, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_FUNC, "%s(" _message ")", __FUNCTION__, ## __VA_ARGS__)
#define ucs_trace_poll(_message, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_POLL, _message, ## __VA_ARGS__)

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

#define ucs_print(_message, ...) \
    do { \
        if (ucs_global_opts.log_print_enable) { \
            __ucs_log(__FILE__, __LINE__, __FUNCTION__, UCS_LOG_LEVEL_PRINT, \
                      _message, ## __VA_ARGS__); \
        } \
    } while(0)


typedef enum {
    UCS_LOG_FUNC_RC_STOP,
    UCS_LOG_FUNC_RC_CONTINUE
} ucs_log_func_rc_t;


/**
 * Function for printing log messages.
 *
 * @param file     Source file name.
 * @param line     Source line number.
 * @param function Function name.
 * @param prefix   Log message prefix.
 * @param message  Log message - format string
 * @param ap       Log message format parameters.
 *
 * @return UCS_LOG_FUNC_RC_CONTINUE - continue to next log handler
 *         UCS_LOG_FUNC_RC_STOP     - don't continue
 */
typedef ucs_log_func_rc_t (*ucs_log_func_t)(const char *file, unsigned line,
                                            const char *function, ucs_log_level_t level,
                                            const char *prefix, const char *message,
                                            va_list ap);


extern const char *ucs_log_level_names[];
extern const char *ucs_log_category_names[];
struct ibv_sge;
struct ibv_wc;
struct ibv_send_wr;

/**
 * Initialize logging subsystem.
 */
void ucs_log_early_init();
void ucs_log_init();
void ucs_log_cleanup();

/**
 * Flush out logs.
 */
void ucs_log_flush();

void __ucs_log(const char *file, unsigned line, const char *function,
               ucs_log_level_t level, const char *message, ...)
    UCS_F_PRINTF(5, 6);

ucs_log_func_rc_t
ucs_log_default_handler(const char *file, unsigned line, const char *function,
                        ucs_log_level_t level, const char *prefix, const char *message,
                        va_list ap);

void __ucs_abort(const char *error_type, const char *file, unsigned line,
                 const char *function, const char *message, ...)
    UCS_F_NORETURN UCS_F_PRINTF(5, 6);

void ucs_log_fatal_error(const char *fmt, ...);

const char *ucs_log_bitmap_to_str(unsigned n, uint8_t *bitmap, size_t length);

const char *ucs_log_dump_hex(const void* data, size_t length, char *buf, size_t max);

void ucs_log_push_handler(ucs_log_func_t handler);
void ucs_log_pop_handler();

END_C_DECLS

#endif

