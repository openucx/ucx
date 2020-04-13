/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef _UCS_ASSERT_H
#define _UCS_ASSERT_H

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/sys/compiler_def.h>


BEGIN_C_DECLS

/** @file assert.h */

/**
 * Fail if _expression evaluates to 0
 */
#define ucs_assert_always(_expression) \
    do { \
        if (!ucs_likely(_expression)) { \
            ucs_fatal_error_format(__FILE__, __LINE__, __FUNCTION__, \
                                   "Assertion `%s' failed", #_expression); \
        } \
    } while (0)


/**
 * Fail if _expression evaluates to 0 and print a formatted error message
 */
#define ucs_assertv_always(_expression, _fmt, ...) \
    do { \
        if (!ucs_likely(_expression)) { \
            ucs_fatal_error_format(__FILE__, __LINE__, __FUNCTION__, \
                                   "Assertion `%s' failed: " _fmt, \
                                   #_expression, ## __VA_ARGS__); \
        } \
    } while (0)


/**
 * Generate a fatal error
 */
#define ucs_fatal(_fmt, ...) \
    ucs_fatal_error_format(__FILE__, __LINE__, __FUNCTION__, \
                           "Fatal: " _fmt, ## __VA_ARGS__)


#if defined (ENABLE_ASSERT) || defined(__COVERITY__) || defined(__clang_analyzer__)

#define UCS_ENABLE_ASSERT 1

/**
 * Generate a program bug report if assertions are enabled
 */
#define ucs_bug(_fmt, ...) \
    ucs_fatal_error_format(__FILE__, __LINE__, __FUNCTION__, \
                           "Bug: " _fmt, ## __VA_ARGS__)

#define ucs_assert(...)       ucs_assert_always(__VA_ARGS__)
#define ucs_assertv(...)      ucs_assertv_always(__VA_ARGS__)

#else

#define UCS_ENABLE_ASSERT 0

#define ucs_bug(...)
#define ucs_assert(...)
#define ucs_assertv(...)

#endif


/**
 * Generate a fatal error and stop the program.
 *
 * @param [in] file        Source file name
 * @param [in] line        Source line number
 * @param [in] function    Calling function name
 * @param [in] format      Error message format string. Multi-line message is
 *                         supported.
 */
void ucs_fatal_error_format(const char *file, unsigned line,
                            const char *function, const char *format, ...)
    UCS_F_NORETURN UCS_F_PRINTF(4, 5);


/**
 * Generate a fatal error and stop the program.
 *
 * @param [in] file        Source file name
 * @param [in] line        Source line number
 * @param [in] function    Calling function name
 * @param [in] message_buf Error message buffer. Multi-line message is
 *                         supported.
 *
 * IMPORTANT NOTE: message_buf could be overridden by this function
 */
void ucs_fatal_error_message(const char *file, unsigned line,
                             const char *function, char *message_buf)
    UCS_F_NORETURN;


END_C_DECLS

#endif
