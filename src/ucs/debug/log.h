/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/


#ifndef UCS_LOG_H_
#define UCS_LOG_H_

#ifdef HAVE_CONFIG_H
#  include "config.h" /* Defines UCS_MAX_LOG_LEVEL */
#endif

#include <ucs/sys/compiler.h>
#include <ucs/config/global_opts.h>
#include <stdint.h>


#define ucs_log_level  (ucs_global_opts.log_level)

#define ucs_log(_level, _message, ...) \
    do { \
        if (ucs_unlikely(((_level) <= UCS_MAX_LOG_LEVEL) && ((_level) <= ucs_log_level))) { \
            __ucs_log(__FILE__, __LINE__, __FUNCTION__, (_level), \
                      _message, ## __VA_ARGS__); \
        } \
    } while (0)


#define ucs_assert_always(_expression) \
    do { \
        if (!ucs_likely(_expression)) { \
            __ucs_abort(__FILE__, __LINE__, __FUNCTION__, "Assertion `%s' failed", \
                        #_expression); \
        } \
    } while (0)

#define ucs_assertv_always(_expression, _fmt, ...) \
    do { \
        if (!ucs_likely(_expression)) { \
            __ucs_abort(__FILE__, __LINE__, __FUNCTION__, "Assertion `%s' failed: "_fmt, \
                        #_expression, ## __VA_ARGS__); \
        } \
    } while (0)

#if ENABLE_ASSERT

#define ucs_assert(_expression)                 ucs_assert_always(_expression)
#define ucs_assertv(_expression, _fmt, ...)     ucs_assertv_always(_expression, _fmt, ## __VA_ARGS__)

#else

#define ucs_assert(_expression)
#define ucs_assertv(_expression, _fmt, ...)

#endif


#define ucs_fatal(_message, ...) \
    __ucs_abort(__FILE__, __LINE__, __FUNCTION__, "Fatal: " _message, ## __VA_ARGS__)

#define ucs_error(_message, ...)        ucs_log(UCS_LOG_LEVEL_ERROR, _message, ## __VA_ARGS__)
#define ucs_warn(_message, ...)         ucs_log(UCS_LOG_LEVEL_WARN, _message,  ## __VA_ARGS__)
#define ucs_info(_message, ...)         ucs_log(UCS_LOG_LEVEL_INFO, _message, ## __VA_ARGS__)
#define ucs_debug(_message, ...)        ucs_log(UCS_LOG_LEVEL_DEBUG, _message, ##  __VA_ARGS__)
#define ucs_trace(_message, ...)        ucs_log(UCS_LOG_LEVEL_TRACE, _message, ## __VA_ARGS__)
#define ucs_trace_req(_message, ...)    ucs_log(UCS_LOG_LEVEL_TRACE_REQ, _message, ## __VA_ARGS__)
#define ucs_trace_data(_message, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_DATA, _message, ## __VA_ARGS__)
#define ucs_trace_async(_message, ...)  ucs_log(UCS_LOG_LEVEL_TRACE_ASYNC, _message, ## __VA_ARGS__)
#define ucs_trace_func(_message, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_FUNC, "%s("_message")", __FUNCTION__, ## __VA_ARGS__)
#define ucs_trace_poll(_message, ...)   ucs_log(UCS_LOG_LEVEL_TRACE_POLL, "%s("_message")", __FUNCTION__, ## __VA_ARGS__)


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
               unsigned level, const char *message, ...)
    UCS_F_PRINTF(5, 6);

void __ucs_vlog(const char *file, unsigned line, const char *function,
                unsigned level, const char *prefix, const char *message,
                va_list ap);

void __ucs_abort(const char *file, unsigned line, const char *function,
                 const char *message, ...)
    UCS_F_NORETURN UCS_F_PRINTF(4, 5);

void ucs_log_fatal_error(const char *fmt, ...);

const char *ucs_log_bitmap_to_str(unsigned n, uint8_t *bitmap, size_t length);

void ucs_log_dump_hex(const void* data, size_t length, char *buf, size_t max);


#endif

