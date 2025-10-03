/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2025. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_DEVICE_LOG_H
#define UCS_DEVICE_LOG_H

#include "device_common.h"

#include <ucs/sys/compiler_def.h>
#include <ucs/config/types.h>
#include <stddef.h>

/* Maximal log level for which the logging code is compiled in.
   Application kernels can override this value by defining
   UCS_DEVICE_MAX_LOG_LEVEL before including this file. */
#ifndef UCS_DEVICE_MAX_LOG_LEVEL
#define UCS_DEVICE_MAX_LOG_LEVEL UCS_LOG_LEVEL_DEBUG
#endif


/* Helper macro to print a message from a device function including the
 * thread and block indices */
#define ucs_device_log(_level, _log_config, _fmt, ...) \
    do { \
        if ((UCS_LOG_LEVEL_##_level <= UCS_DEVICE_MAX_LOG_LEVEL) && \
            (UCS_LOG_LEVEL_##_level <= (_log_config)->level)) { \
            const uint64_t _ts = ucs_device_get_time_ns(); \
            printf("[%06lu.%06lu] (%4d:%-3d) %10s:%-4d %-6s " _fmt "\n", \
                   _ts / 1000000000ul, (_ts % 1000000000ul) / 1000ul, \
                   threadIdx.x, blockIdx.x, \
                   ucs_device_log_source_file(__FILE__), __LINE__, \
                   UCS_LOG_LEVEL_NAME_##_level, ##__VA_ARGS__); \
        } \
    } while (0)


/* Log level names */
#define UCS_LOG_LEVEL_NAME_ERROR      "ERROR"
#define UCS_LOG_LEVEL_NAME_WARN       "WARN"
#define UCS_LOG_LEVEL_NAME_DIAG       "DIAG"
#define UCS_LOG_LEVEL_NAME_INFO       "INFO"
#define UCS_LOG_LEVEL_NAME_DEBUG      "DEBUG"
#define UCS_LOG_LEVEL_NAME_TRACE      "TRACE"
#define UCS_LOG_LEVEL_NAME_TRACE_DATA "DATA"
#define UCS_LOG_LEVEL_NAME_TRACE_POLL "POLL"


static UCS_F_DEVICE_LIB const char *ucs_device_basename(const char *path)
{
    const char *basename = path;
    const char *p;

    for (p = path; *p != '\0'; p++) {
        if (*p == '/') {
            basename = p + 1;
        }
    }

    return basename;
}


UCS_F_DEVICE const char *ucs_device_log_source_file(const char *file)
{
    static const char *cached_source_file = NULL;

    if (cached_source_file == NULL) {
        cached_source_file = ucs_device_basename(file);
    }

    return cached_source_file;
}

#endif /* UCS_DEVICE_LOG_H */
