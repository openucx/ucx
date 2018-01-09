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


void ucs_fatal_error(const char *error_type, const char *file, unsigned line,
                     const char *function, const char *format, ...)
{
    size_t buffer_size = ucs_global_opts.log_buffer_size;
    const char *short_file;
    char *buffer;
    va_list ap;

    buffer = ucs_alloca(buffer_size + 1);
    va_start(ap, format);
    vsnprintf(buffer, buffer_size, format, ap);
    va_end(ap);

    ucs_debug_cleanup();
    ucs_log_flush();

    short_file = strrchr(file, '/');
    short_file = (short_file == NULL) ? file : short_file + 1;
    ucs_handle_error(error_type, "%13s:%-4u %s", short_file, line, buffer);

    abort();
}

