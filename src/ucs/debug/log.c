/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#include "log.h"

#include <ucs/debug/debug.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/config/parser.h>

#define UCS_MAX_LOG_HANDLERS    32


const char *ucs_log_level_names[] = {
    [UCS_LOG_LEVEL_FATAL]        = "FATAL",
    [UCS_LOG_LEVEL_ERROR]        = "ERROR",
    [UCS_LOG_LEVEL_WARN]         = "WARN",
    [UCS_LOG_LEVEL_INFO]         = "INFO",
    [UCS_LOG_LEVEL_DEBUG]        = "DEBUG",
    [UCS_LOG_LEVEL_TRACE]        = "TRACE",
    [UCS_LOG_LEVEL_TRACE_REQ]    = "REQ",
    [UCS_LOG_LEVEL_TRACE_DATA]   = "DATA",
    [UCS_LOG_LEVEL_TRACE_ASYNC]  = "ASYNC",
    [UCS_LOG_LEVEL_TRACE_FUNC]   = "FUNC",
    [UCS_LOG_LEVEL_TRACE_POLL]   = "POLL",
    [UCS_LOG_LEVEL_LAST]         = NULL,
    [UCS_LOG_LEVEL_PRINT]        = "PRINT"
};

static unsigned ucs_log_num_handlers   = 0;
static ucs_log_func_t ucs_log_handlers[UCS_MAX_LOG_HANDLERS];
static int ucs_log_initialized         = 0;
static char ucs_log_hostname[256]      = {0};
static int  ucs_log_pid                = 0;
static FILE *ucs_log_file              = NULL;
static int ucs_log_file_close          = 0;
static unsigned threads_count          = 0;
static pthread_spinlock_t threads_lock = 0;
static pthread_t threads[128]          = {0};


static int ucs_log_get_thread_num(void)
{
    pthread_t self = pthread_self();
    unsigned i;

    for (i = 0; i < threads_count; ++i) {
        if (threads[i] == self) {
            return i;
        }
    }

    pthread_spin_lock(&threads_lock);

    for (i = 0; i < threads_count; ++i) {
        if (threads[i] == self) {
            goto unlock_and_return_i;
        }
    }

    if (threads_count >= sizeof(threads) / sizeof(threads[0])) {
        i = -1;
        goto unlock_and_return_i;
    }

    i = threads_count;
    ++threads_count;
    threads[i] = self;

unlock_and_return_i:
    pthread_spin_unlock(&threads_lock);
    return i;
}

void ucs_log_flush()
{
    if (ucs_log_file != NULL) {
        fflush(ucs_log_file);
        fsync(fileno(ucs_log_file));
    }
}

ucs_log_func_rc_t
ucs_log_default_handler(const char *file, unsigned line, const char *function,
                        ucs_log_level_t level, const char *format, va_list ap)
{
    size_t buffer_size = ucs_config_memunits_get(ucs_global_opts.log_buffer_size,
                                                 256, 2048);
    const char *short_file;
    struct timeval tv;
    char *valg_buf;
    char *buf;

    if (!ucs_log_is_enabled(level) && (level != UCS_LOG_LEVEL_PRINT)) {
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

   buf = ucs_alloca(buffer_size + 1);
    buf[buffer_size] = 0;

    vsnprintf(buf, buffer_size, format, ap);

    short_file = strrchr(file, '/');
    short_file = (short_file == NULL) ? file : short_file + 1;
    gettimeofday(&tv, NULL);

    if (level <= ucs_global_opts.log_level_trigger) {
        ucs_handle_error(ucs_log_level_names[level], "%13s:%-4u %s: %s",
                         short_file, line, ucs_log_level_names[level], buf);
    } else if (RUNNING_ON_VALGRIND) {
        valg_buf = ucs_alloca(buffer_size + 1);
        snprintf(valg_buf, buffer_size,
                 "[%lu.%06lu] %16s:%-4u %-4s %-5s %s\n", tv.tv_sec, tv.tv_usec,
                 short_file, line, "UCX", ucs_log_level_names[level], buf);
        VALGRIND_PRINTF("%s", valg_buf);
    } else if (ucs_log_initialized) {
        fprintf(ucs_log_file,
                "[%lu.%06lu] [%s:%-5d:%d] %16s:%-4u %-4s %-5s %s\n",
                tv.tv_sec, tv.tv_usec, ucs_log_hostname, ucs_log_pid,
                ucs_log_get_thread_num(), short_file, line, "UCX",
                ucs_log_level_names[level], buf);
    } else {
        fprintf(stdout,
                "[%lu.%06lu] %16s:%-4u %-4s %-5s %s\n",
                tv.tv_sec, tv.tv_usec, short_file, line,
                "UCX", ucs_log_level_names[level], buf);
    }

    /* flush the log file if the log_level of this message is fatal or error */
    if (level <= UCS_LOG_LEVEL_ERROR) {
        ucs_log_flush();
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

void ucs_log_push_handler(ucs_log_func_t handler)
{
    if (ucs_log_num_handlers < UCS_MAX_LOG_HANDLERS) {
        ucs_log_handlers[ucs_log_num_handlers++] = handler;
    }
}

void ucs_log_pop_handler()
{
    if (ucs_log_num_handlers > 0) {
        --ucs_log_num_handlers;
    }
}

void ucs_log_dispatch(const char *file, unsigned line, const char *function,
                      ucs_log_level_t level, const char *format, ...)
{
    ucs_log_func_rc_t rc;
    unsigned index;
    va_list ap;

    /* Call handlers in reverse order */
    rc    = UCS_LOG_FUNC_RC_CONTINUE;
    index = ucs_log_num_handlers;
    while ((index > 0) && (rc == UCS_LOG_FUNC_RC_CONTINUE)) {
        --index;
        va_start(ap, format);
        rc = ucs_log_handlers[index](file, line, function, level, format, ap);
        va_end(ap);
    }
}

void ucs_log_fatal_error(const char *format, ...)
{
    size_t buffer_size = ucs_global_opts.log_buffer_size;
    FILE *stream = stderr;
    char *buffer, *p;
    va_list ap;
    int ret;

    buffer = ucs_alloca(buffer_size + 1);
    p = buffer;

    /* Print hostname:pid */
    snprintf(p, buffer_size, "[%s:%-5d:%d:%d] ", ucs_log_hostname, ucs_log_pid,
             ucs_log_get_thread_num(), ucs_get_tid());
    buffer_size -= strlen(p);
    p           += strlen(p);

    /* Print rest of the message */
    va_start(ap, format);
    vsnprintf(p, buffer_size, format, ap);
    va_end(ap);
    buffer_size -= strlen(p);
    p           += strlen(p);

    /* Newline */
    snprintf(p, buffer_size, "\n");

    /* Flush stderr, and write the message directly to the pipe */
    fflush(stream);
    ret = write(fileno(stream), buffer, strlen(buffer));
    (void)ret;
}

/**
 * Print a bitmap as a list of ranges.
 *
 * @param n        Number equivalent to the first bit in the bitmap.
 * @param bitmap   Compressed array of bits.
 * @param length   Number of bits in the bitmap.
 */
const char *ucs_log_bitmap_to_str(unsigned n, uint8_t *bitmap, size_t length)
{
    static char buf[512] = {0};
    int first, in_range;
    unsigned prev = 0, end = 0;
    char *p, *endp;
    size_t i;

    p = buf;
    endp = buf + sizeof(buf) - 4;

    first = 1;
    in_range = 0;
    for (i = 0; i < length; ++i) {
        if (bitmap[i / 8] & UCS_BIT(i % 8)) {
            if (first) {
                p += snprintf(p, endp - p, "%d", n);
                if (p > endp) {
                    goto overflow;
                }
            } else if (n == prev + 1) {
                in_range = 1;
                end = n;
            } else {
                if (in_range) {
                    p += snprintf(p, endp - p, "-%d", end);
                    if (p > endp) {
                        goto overflow;
                    }
                }
                in_range = 0;
                p += snprintf(p, endp - p, ",%d", n);
                if (p > endp) {
                    goto overflow;
                }
            }
            first = 0;
            prev = n;
        }
        ++n;
    }
    if (in_range) {
        p += snprintf(p, endp - p, "-%d", end);
        if (p > endp) {
            goto overflow;
        }
    }
    return buf;

overflow:
    strcpy(p, "...");
    return buf;
}


const char * ucs_log_dump_hex(const void* data, size_t length, char *buf,
                              size_t max)
{
    static const char hexchars[] = "0123456789abcdef";
    char *p, *endp;
    uint8_t value;
    size_t i;

    p    = buf;
    endp = buf + max - 2;

    i = 0;
    while ((p < endp) && (i < length)) {
        if (((i % 4) == 0) && (i > 0)) {
            *(p++) = ':';
        }
        value = *(uint8_t*)(data + i);
        p[0] = hexchars[value / 16];
        p[1] = hexchars[value % 16];
        p += 2;
        ++i;
    }
    *p = 0;
    return buf;
}

void ucs_log_early_init()
{
    ucs_log_initialized      = 0;
    ucs_log_hostname[0]      = 0;
    ucs_log_pid              = getpid();
    ucs_log_file             = NULL;
    ucs_log_file_close       = 0;
    threads_count            = 0;
    pthread_spin_init(&threads_lock, 0);
}

void ucs_log_init()
{
    const char *next_token;

    if (ucs_log_initialized) {
        return;
    }

    ucs_log_initialized = 1; /* Set this to 1 immediately to avoid infinite recursion */

    strcpy(ucs_log_hostname, ucs_get_host_name());
    ucs_log_file       = stdout;
    ucs_log_file_close = 0;

    ucs_log_push_handler(ucs_log_default_handler);

    if (strlen(ucs_global_opts.log_file) != 0) {
         ucs_open_output_stream(ucs_global_opts.log_file, UCS_LOG_LEVEL_FATAL,
                                &ucs_log_file, &ucs_log_file_close, &next_token);
    }
}

void ucs_log_cleanup()
{
    ucs_log_flush();
    if (ucs_log_file_close) {
        fclose(ucs_log_file);
    }
    ucs_log_file = NULL;
    ucs_log_initialized  = 0;
    ucs_log_num_handlers = 0;
}
