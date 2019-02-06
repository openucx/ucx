/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "assert.h"

#include <ucs/config/global_opts.h>
#include <ucs/debug/debug.h>
#include <ucs/debug/log.h>
#include <ucs/sys/compiler.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>


void ucs_fatal_error_message(const char *file, unsigned line,
                             const char *function, char *message_buf)
{
    char *message_line, *save_ptr = NULL;
    const char *short_file;

    ucs_log_flush();

    short_file = strrchr(file, '/');
    short_file = (short_file == NULL) ? file : short_file + 1;

    message_line = (message_buf == NULL) ? NULL :
                   strtok_r(message_buf, "\n", &save_ptr);
    while (message_line != NULL) {
        ucs_log_fatal_error("%13s:%-4u %s", short_file, line, message_line);
        message_line = strtok_r(NULL, "\n", &save_ptr);
    }

    ucs_handle_error(message_buf);
    abort();
}

void ucs_fatal_error_format(const char *file, unsigned line,
                            const char *function, const char *format, ...)
{
    size_t buffer_size = ucs_log_get_buffer_size();
    char *buffer;
    va_list ap;

    buffer = ucs_alloca(buffer_size + 1);
    va_start(ap, format);
    vsnprintf(buffer, buffer_size, format, ap);
    va_end(ap);

    ucs_fatal_error_message(file, line, function, buffer);
}
