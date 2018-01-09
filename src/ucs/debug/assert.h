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


/**
 * Fail if _expression evaluates to 0
 */
#define ucs_assert_always(_expression) \
    do { \
        if (!ucs_likely(_expression)) { \
            ucs_fatal_error("assertion failure", __FILE__, __LINE__, \
                            __FUNCTION__, "Assertion `%s' failed", #_expression); \
        } \
    } while (0)


/**
 * Fail if _expression evaluates to 0 and print a formatted error message
 */
#define ucs_assertv_always(_expression, _fmt, ...) \
    do { \
        if (!ucs_likely(_expression)) { \
            ucs_fatal_error("assertion failure", __FILE__, __LINE__, __FUNCTION__, \
                            "Assertion `%s' failed: " _fmt, #_expression, \
                            ## __VA_ARGS__); \
        } \
    } while (0)


/**
 * Generate a fatal error
 */
#define ucs_fatal(_fmt, ...) \
    ucs_fatal_error("fatal error", __FILE__, __LINE__, __FUNCTION__, \
                    "Fatal: " _fmt, ## __VA_ARGS__)


#if ENABLE_ASSERT

/**
 * Generate a program bug report if assertions are enabled
 */
#define ucs_bug(_fmt, ...) \
    ucs_fatal_error("bug", __FILE__, __LINE__, __FUNCTION__, \
                    "Bug: " _fmt, ## __VA_ARGS__)

#define ucs_assert(...)       ucs_assert_always(__VA_ARGS__)
#define ucs_assertv(...)      ucs_assertv_always(__VA_ARGS__)

#else

#define ucs_bug(...)
#define ucs_assert(...)
#define ucs_assertv(...)

#endif


/**
 * Generate a fatal error and stop the program.
 *
 * @param [in] error_type  Fatal error type
 * @param [in] file        Source file name
 * @param [in] line        Source line number
 * @param [in] function    Calling function name
 * @param [in] format      Error message format string
 */
void ucs_fatal_error(const char *error_type, const char *file, unsigned line,
                     const char *function, const char *format, ...)
    UCS_F_NORETURN UCS_F_PRINTF(5, 6);


END_C_DECLS

#endif
