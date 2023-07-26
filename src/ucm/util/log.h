/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2015. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCM_LOG_H_
#define UCM_LOG_H_

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucm/api/ucm.h>
#include <ucs/config/types.h>


#define ucm_log(_level, _message, ...) \
    if (((_level) <= UCS_MAX_LOG_LEVEL) && \
        ((_level) <= (int)ucm_global_opts.log_level)) { \
        __ucm_log(__FILE__, __LINE__, __FUNCTION__, (_level), _message, \
                  ## __VA_ARGS__); \
    }


#define ucm_fatal(_message, ...) ucm_log(UCS_LOG_LEVEL_FATAL, _message, ## __VA_ARGS__)
#define ucm_error(_message, ...) ucm_log(UCS_LOG_LEVEL_ERROR, _message, ## __VA_ARGS__)
#define ucm_warn(_message, ...)  ucm_log(UCS_LOG_LEVEL_WARN,  _message, ## __VA_ARGS__)
#define ucm_diag(_message, ...)  ucm_log(UCS_LOG_LEVEL_DIAG,  _message, ## __VA_ARGS__)
#define ucm_info(_message, ...)  ucm_log(UCS_LOG_LEVEL_INFO,  _message, ## __VA_ARGS__)
#define ucm_debug(_message, ...) ucm_log(UCS_LOG_LEVEL_DEBUG, _message, ## __VA_ARGS__)
#define ucm_trace(_message, ...) ucm_log(UCS_LOG_LEVEL_TRACE, _message, ## __VA_ARGS__)


#define ucm_assert_always(_expression) \
    do { \
        if (!ucs_likely(_expression)) { \
            ucm_fatal("Assertion `%s' failed", #_expression); \
        } \
    } while (0)


#define ucm_assertv_always(_expression, _fmt, ...) \
    do { \
        if (!ucs_likely(_expression)) { \
            ucm_fatal("Assertion `%s' failed: " _fmt, \
                      #_expression, ## __VA_ARGS__); \
        } \
    } while (0)


#if defined (ENABLE_ASSERT) || defined(__COVERITY__) || defined(__clang_analyzer__)
#  define ucm_assert(...)    ucm_assert_always(__VA_ARGS__)
#else
#  define ucm_assert(...)    {}
#endif


extern const char *ucm_log_level_names[];


void ucm_init_log();

void __ucm_log(const char *file, unsigned line, const char *function,
               ucs_log_level_t level, const char *message, ...)
    UCS_F_PRINTF(5, 6);

#endif
