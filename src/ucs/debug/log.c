/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "log.h"

#include <ucs/arch/atomic.h>
#include <ucs/debug/debug_int.h>
#include <ucs/datastruct/khash.h>
#include <ucs/sys/compiler.h>
#include <ucs/sys/checker.h>
#include <ucs/sys/string.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/math.h>
#include <ucs/type/spinlock.h>
#include <ucs/config/parser.h>
#include <fnmatch.h>


#define UCS_MAX_LOG_HANDLERS    32

#define UCS_LOG_TIME_FMT        "[%lu.%06lu]"
#define UCS_LOG_METADATA_FMT    "%17s:%-4u %-4s %-5s %*s"
#define UCS_LOG_PROC_DATA_FMT   "[%s:%-5d:%s]"

#define UCS_LOG_COMPACT_FMT     UCS_LOG_TIME_FMT " " UCS_LOG_PROC_DATA_FMT "  "
#define UCS_LOG_SHORT_FMT       UCS_LOG_TIME_FMT " [%s] " UCS_LOG_METADATA_FMT "%s\n"
#define UCS_LOG_FMT             UCS_LOG_TIME_FMT " " UCS_LOG_PROC_DATA_FMT " " \
                                UCS_LOG_METADATA_FMT "%s\n"

#define UCS_LOG_TIME_ARG(_tv)  (_tv)->tv_sec, (_tv)->tv_usec

#define UCS_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf) \
    (_short_file), (_line), (_comp_conf)->name, \
    ucs_log_level_names[_level], (ucs_log_current_indent * 2), ""

#define UCS_LOG_PROC_DATA_ARG() \
    ucs_log_hostname, ucs_log_pid, ucs_log_get_thread_name()

#define UCS_LOG_COMPACT_ARG(_tv)\
    UCS_LOG_TIME_ARG(_tv), UCS_LOG_PROC_DATA_ARG()

#define UCS_LOG_SHORT_ARG(_short_file, _line, _level, _comp_conf, _tv, \
                          _message) \
    UCS_LOG_TIME_ARG(_tv), ucs_log_get_thread_name(), \
            UCS_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf), \
            (_message)

#define UCS_LOG_ARG(_short_file, _line, _level, _comp_conf, _tv, _message) \
    UCS_LOG_TIME_ARG(_tv), UCS_LOG_PROC_DATA_ARG(), \
    UCS_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf), (_message)

KHASH_MAP_INIT_STR(ucs_log_filter, char);

const char *ucs_log_level_names[] = {
    [UCS_LOG_LEVEL_FATAL]        = "FATAL",
    [UCS_LOG_LEVEL_ERROR]        = "ERROR",
    [UCS_LOG_LEVEL_WARN]         = "WARN",
    [UCS_LOG_LEVEL_DIAG]         = "DIAG",
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

static unsigned ucs_log_handlers_count       = 0;
static int ucs_log_initialized               = 0;
static int __thread ucs_log_current_indent   = 0;
static char ucs_log_hostname[HOST_NAME_MAX]  = {0};
static int ucs_log_pid                       = 0;
static FILE *ucs_log_file                    = NULL;
static char *ucs_log_file_base_name          = NULL;
static int ucs_log_file_close                = 0;
static int ucs_log_file_last_idx             = 0;
static uint32_t ucs_log_thread_count         = 0;
static char __thread ucs_log_thread_name[32] = {0};
static ucs_log_func_t ucs_log_handlers[UCS_MAX_LOG_HANDLERS];
static ucs_spinlock_t ucs_log_global_filter_lock;
static khash_t(ucs_log_filter) ucs_log_global_filter;


static const char *ucs_log_get_thread_name()
{
    char *name = ucs_log_thread_name;
    uint32_t thread_num;

    if (ucs_unlikely(name[0] == '\0')) {
        thread_num = ucs_atomic_fadd32(&ucs_log_thread_count, 1);
        ucs_snprintf_safe(ucs_log_thread_name, sizeof(ucs_log_thread_name),
                          "%u", thread_num);
    }

    return name;
}

void ucs_log_flush()
{
    if (ucs_log_file != NULL) {
        fflush(ucs_log_file);

        if (ucs_log_file_close) { /* non-stdout/stderr */
            fsync(fileno(ucs_log_file));
        }
    }
}

size_t ucs_log_get_buffer_size()
{
    return ucs_config_memunits_get(ucs_global_opts.log_buffer_size,
                                   256, UCS_ALLOCA_MAX_SIZE);
}

static void ucs_log_get_file_name(char *log_file_name, size_t max, int idx)
{
    ucs_assert(idx <= ucs_global_opts.log_file_rotate);

    if (idx == 0) {
        ucs_strncpy_zero(log_file_name, ucs_log_file_base_name, max);
        return;
    }

    ucs_snprintf_zero(log_file_name, max, "%s.%d",
                      ucs_log_file_base_name, idx);
}

static void ucs_log_file_rotate()
{
    char old_log_file_name[PATH_MAX];
    char new_log_file_name[PATH_MAX];
    int idx, ret;

    if (ucs_log_file_last_idx == ucs_global_opts.log_file_rotate) {
        /* remove the last file and log rotation from the
         * `log_file_rotate - 1` file */
        ucs_log_get_file_name(old_log_file_name,
                              sizeof(old_log_file_name),
                              ucs_log_file_last_idx);
        unlink(old_log_file_name);
    } else {
        ucs_log_file_last_idx++;
    }

    ucs_assert(ucs_log_file_last_idx <= ucs_global_opts.log_file_rotate);

    for (idx = ucs_log_file_last_idx - 1; idx >= 0; --idx) {
        ucs_log_get_file_name(old_log_file_name,
                              sizeof(old_log_file_name), idx);
        ucs_log_get_file_name(new_log_file_name,
                              sizeof(new_log_file_name), idx + 1);

        if (access(old_log_file_name, W_OK) != 0) {
            ucs_fatal("unable to write to %s", old_log_file_name);
        }

        /* coverity[toctou] */
        ret = rename(old_log_file_name, new_log_file_name);
        if (ret) {
            ucs_fatal("failed to rename %s to %s: %m",
                      old_log_file_name, new_log_file_name);
        }


        if (access(old_log_file_name, F_OK) != -1) {
            ucs_fatal("%s must not exist on the filesystem", old_log_file_name);
        }

        if (access(new_log_file_name, W_OK) != 0) {
            ucs_fatal("unable to write to %s", new_log_file_name);
        }
    }
}

static void ucs_log_handle_file_max_size(int log_entry_len)
{
    const char *next_token;

    /* check if it is necessary to find a new storage for logs */
    if ((log_entry_len + ftell(ucs_log_file)) < ucs_global_opts.log_file_size) {
        return;
    }

    fclose(ucs_log_file);

    if (ucs_global_opts.log_file_rotate != 0) {
        ucs_log_file_rotate();
    } else {
        unlink(ucs_log_file_base_name);
    }

    ucs_open_output_stream(ucs_log_file_base_name, UCS_LOG_LEVEL_FATAL,
                           &ucs_log_file, &ucs_log_file_close,
                           &next_token, NULL);
}

void ucs_log_print_compact(const char *str)
{
    struct timeval tv;

    gettimeofday(&tv, NULL);

    if (RUNNING_ON_VALGRIND) {
        VALGRIND_PRINTF(UCS_LOG_TIME_FMT " %s\n", UCS_LOG_TIME_ARG(&tv), str);
    } else if (ucs_log_initialized) {
        if (ucs_log_file_close) { /* non-stdout/stderr */
            ucs_log_handle_file_max_size(strlen(str) + 1);
        }

        fprintf(ucs_log_file, UCS_LOG_COMPACT_FMT " %s\n",
                UCS_LOG_COMPACT_ARG(&tv), str);
    } else {
        fprintf(stdout, UCS_LOG_COMPACT_FMT " %s\n", UCS_LOG_COMPACT_ARG(&tv),
                str);
    }
}

static void ucs_log_print(const char *short_file, int line,
                          ucs_log_level_t level,
                          const ucs_log_component_config_t *comp_conf,
                          const struct timeval *tv, const char *message)
{
    size_t buffer_size;
    int log_entry_len;
    char *log_buf;

    if (RUNNING_ON_VALGRIND) {
        buffer_size = ucs_log_get_buffer_size();
        log_buf     = ucs_alloca(buffer_size + 1);
        snprintf(log_buf, buffer_size, UCS_LOG_SHORT_FMT,
                UCS_LOG_SHORT_ARG(short_file, line, level,
                                  comp_conf, tv, message));
        VALGRIND_PRINTF("%s", log_buf);
    } else if (ucs_log_initialized) {
        if (ucs_log_file_close) { /* non-stdout/stderr */
            /* get log entry size */
            log_entry_len = snprintf(NULL, 0, UCS_LOG_FMT,
                                     UCS_LOG_ARG(short_file, line, level,
                                                 comp_conf, tv, message));
            ucs_log_handle_file_max_size(log_entry_len);
        }

        fprintf(ucs_log_file, UCS_LOG_FMT,
                UCS_LOG_ARG(short_file, line, level,
                            comp_conf, tv, message));
    } else {
        fprintf(stdout, UCS_LOG_SHORT_FMT,
                UCS_LOG_SHORT_ARG(short_file, line, level,
                                  comp_conf, tv, message));
    }
}

ucs_log_func_rc_t
ucs_log_default_handler(const char *file, unsigned line, const char *function,
                        ucs_log_level_t level,
                        const ucs_log_component_config_t *comp_conf,
                        const char *format, va_list ap)
{
    size_t buffer_size = ucs_log_get_buffer_size();
    char *saveptr      = "";
    const char *short_file;
    struct timeval tv;
    khiter_t khiter;
    char *log_line;
    char match;
    int khret;
    char *buf;

    if (!ucs_log_component_is_enabled(level, comp_conf) &&
        (level != UCS_LOG_LEVEL_PRINT)) {
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    ucs_spin_lock(&ucs_log_global_filter_lock);
    khiter = kh_get(ucs_log_filter, &ucs_log_global_filter, file);
    if (ucs_unlikely(khiter == kh_end(&ucs_log_global_filter))) {
        /* Add source file name to the hash */
        match  = fnmatch(ucs_global_opts.log_component.file_filter, file, 0) !=
                 FNM_NOMATCH;
        khiter = kh_put(ucs_log_filter, &ucs_log_global_filter, file, &khret);
        ucs_assert((khret == UCS_KH_PUT_BUCKET_EMPTY) ||
                   (khret == UCS_KH_PUT_BUCKET_CLEAR));
        kh_val(&ucs_log_global_filter, khiter) = match;
    } else {
        match = kh_val(&ucs_log_global_filter, khiter);
    }
    ucs_spin_unlock(&ucs_log_global_filter_lock);

    if (!match) {
        return UCS_LOG_FUNC_RC_CONTINUE;
    }

    buf = ucs_alloca(buffer_size + 1);
    buf[buffer_size] = 0;
    vsnprintf(buf, buffer_size, format, ap);

    if (level <= ucs_global_opts.log_level_trigger) {
        ucs_fatal_error_message(file, line, function, buf);
    } else {
        short_file = ucs_basename(file);
        gettimeofday(&tv, NULL);

        log_line = strtok_r(buf, "\n", &saveptr);
        while (log_line != NULL) {
            ucs_log_print(short_file, line, level, comp_conf, &tv, log_line);
            log_line = strtok_r(NULL, "\n", &saveptr);
        }
    }

    /* flush the log file if the log_level of this message is fatal or error */
    if (level <= UCS_LOG_LEVEL_ERROR) {
        ucs_log_flush();
    }

    return UCS_LOG_FUNC_RC_CONTINUE;
}

void ucs_log_push_handler(ucs_log_func_t handler)
{
    if (ucs_log_handlers_count < UCS_MAX_LOG_HANDLERS) {
        ucs_log_handlers[ucs_log_handlers_count++] = handler;
    }
}

void ucs_log_pop_handler()
{
    if (ucs_log_handlers_count > 0) {
        --ucs_log_handlers_count;
    }
}

void ucs_log_indent(int delta)
{
    ucs_log_current_indent += delta;
    ucs_assert(ucs_log_current_indent >= 0);
}

int ucs_log_get_current_indent()
{
    return ucs_log_current_indent;
}

unsigned ucs_log_num_handlers()
{
    return ucs_log_handlers_count;
}

void ucs_log_dispatch(const char *file, unsigned line, const char *function,
                      ucs_log_level_t level, ucs_log_component_config_t *comp_conf,
                      const char *format, ...)
{
    ucs_log_func_rc_t rc;
    unsigned idx;
    va_list ap;

    /* Call handlers in reverse order */
    rc    = UCS_LOG_FUNC_RC_CONTINUE;
    idx = ucs_log_handlers_count;
    while ((idx > 0) && (rc == UCS_LOG_FUNC_RC_CONTINUE)) {
        --idx;
        va_start(ap, format);
        rc = ucs_log_handlers[idx](file, line, function,
                                   level, comp_conf, format, ap);
        va_end(ap);
    }
}

void ucs_log_fatal_error(const char *format, ...)
{
    size_t buffer_size = ucs_log_get_buffer_size();
    FILE *stream = stderr;
    char *buffer, *p;
    va_list ap;
    int ret;

    buffer = ucs_alloca(buffer_size + 1);
    p = buffer;

    /* Print hostname:pid */
    snprintf(p, buffer_size, "[%s:%-5d:%s:%d] ", ucs_log_hostname, ucs_log_pid,
             ucs_log_get_thread_name(), ucs_get_tid());
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

void ucs_log_early_init()
{
    ucs_log_initialized   = 0;
    ucs_log_hostname[0]   = 0;
    ucs_log_pid           = getpid();
    ucs_log_file          = NULL;
    ucs_log_file_last_idx = 0;
    ucs_log_file_close    = 0;
    ucs_log_thread_count  = 0;
}

static void ucs_log_atfork_child()
{
    ucs_log_pid = getpid();
}

void ucs_log_init()
{
    const char *next_token;

    if (ucs_log_initialized) {
        return;
    }

    ucs_log_initialized = 1; /* Set this to 1 immediately to avoid infinite recursion */

    if (ucs_global_opts.log_file_size < ucs_log_get_buffer_size()) {
        ucs_fatal("the maximal log file size (%zu) has to be >= %zu",
                  ucs_global_opts.log_file_size,
                  ucs_log_get_buffer_size());
    }

    if (ucs_global_opts.log_file_rotate > INT_MAX) {
        ucs_fatal("the log file rotate (%u) has to be <= %d",
                  ucs_global_opts.log_file_rotate, INT_MAX);
    }

    ucs_spinlock_init(&ucs_log_global_filter_lock, 0);
    kh_init_inplace(ucs_log_filter, &ucs_log_global_filter);

    strcpy(ucs_log_hostname, ucs_get_host_name());
    ucs_log_file           = stdout;
    ucs_log_file_base_name = NULL;
    ucs_log_file_close     = 0;
    ucs_log_file_last_idx  = 0;

    ucs_log_push_handler(ucs_log_default_handler);

    if (strlen(ucs_global_opts.log_file) != 0) {
        ucs_open_output_stream(ucs_global_opts.log_file, UCS_LOG_LEVEL_FATAL,
                               &ucs_log_file, &ucs_log_file_close,
                               &next_token, &ucs_log_file_base_name);
    }

    pthread_atfork(NULL, NULL, ucs_log_atfork_child);
}

void ucs_log_cleanup()
{
    ucs_assert(ucs_log_initialized);

    ucs_log_flush();
    if (ucs_log_file_close) {
        fclose(ucs_log_file);
    }

    ucs_spinlock_destroy(&ucs_log_global_filter_lock);
    kh_destroy_inplace(ucs_log_filter, &ucs_log_global_filter);
    ucs_free(ucs_log_file_base_name);
    ucs_log_file_base_name = NULL;
    ucs_log_file           = NULL;
    ucs_log_file_last_idx  = 0;
    ucs_log_initialized    = 0;
    ucs_log_handlers_count = 0;
}

void ucs_log_print_backtrace(ucs_log_level_t level)
{
    backtrace_h bckt;
    backtrace_line_h bckt_line;
    int i;
    char buf[1024];
    ucs_status_t status;

    status = ucs_debug_backtrace_create(&bckt, 1);
    if (status != UCS_OK) {
        return;
    }

    ucs_log(level, "==== backtrace (tid:%7d) ====\n", ucs_get_tid());
    for (i = 0; ucs_debug_backtrace_next(bckt, &bckt_line); ++i) {
        ucs_debug_print_backtrace_line(buf, sizeof(buf), i, bckt_line);
        ucs_log(level, "%s", buf);
    }
    ucs_log(level, "=================================\n");

    ucs_debug_backtrace_destroy(bckt);
}

void ucs_log_set_thread_name(const char *format, ...)
{
    va_list ap;

    va_start(ap, format);
    memset(ucs_log_thread_name, 0, sizeof(ucs_log_thread_name));
    vsnprintf(ucs_log_thread_name, sizeof(ucs_log_thread_name) - 1, format, ap);
    va_end(ap);
}
