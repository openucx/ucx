/**
* Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "assert.h"

#include <ucs/config/global_opts.h>
#include <ucs/debug/debug_int.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>


void ucs_fatal_error_message(const char *file, unsigned line,
                             const char *function, const char *message)
{
    const char *line_start;
    const char *line_end;

    ucs_log_flush();

    if (message != NULL) {
        line_start = message;
        while ((line_end = strchr(line_start, '\n')) != NULL) {
            ucs_log_fatal_error("%13s:%-4u %.*s", ucs_basename(file), line,
                                (int)(line_end - line_start), line_start);
            line_start = line_end + 1;
        }

        /* Handle last line (or only line if no newlines) */
        if (*line_start != '\0') {
            ucs_log_fatal_error("%13s:%-4u %s", ucs_basename(file), line,
                                line_start);
        }
    }

    ucs_handle_error(message);
    abort();
}

void ucs_fatal_error_format(const char *file, unsigned line,
                            const char *function, const char *format, ...)
{
    const size_t buffer_size = ucs_log_get_buffer_size();
    char *buffer;
    va_list ap;

    buffer = ucs_alloca(buffer_size);
    va_start(ap, format);
    ucs_vsnprintf_safe(buffer, buffer_size, format, ap);
    va_end(ap);

    ucs_fatal_error_message(file, line, function, buffer);
}
