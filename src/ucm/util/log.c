/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "log.h"

#include <ucs/type/component.h>
#include <sys/time.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <stdio.h>
#include <ctype.h>
#include <errno.h>


#define UCM_LOG_BUG_SIZE   256

static int  ucm_log_fileno = 1; /* stdout */
static char ucm_log_hostname[40] = {0};

const char *ucm_log_level_names[] = {
    [UCS_LOG_LEVEL_FATAL] = "FATAL",
    [UCS_LOG_LEVEL_ERROR] = "ERROR",
    [UCS_LOG_LEVEL_WARN]  = "WARN",
    [UCS_LOG_LEVEL_INFO]  = "INFO",
    [UCS_LOG_LEVEL_DEBUG] = "DEBUG",
    [UCS_LOG_LEVEL_TRACE] = "TRACE",
    NULL
};

/* Flags for ucm_log_ltoa */
#define UCM_LOG_LTOA_FLAG_SIGN   UCS_BIT(0)  /* print sign */
#define UCM_LOG_LTOA_FLAG_UNSIGN UCS_BIT(1)  /* unsigned number */
#define UCM_LOG_LTOA_FLAG_LONG   UCS_BIT(2)  /* long number */
#define UCM_LOG_LTOA_FLAG_PAD0   UCS_BIT(3)  /* pad with zeroes */
#define UCM_LOG_LTOA_PAD_LEFT    UCS_BIT(4)  /* pad to left */


static char *ucm_log_add_padding(char *p, char *end, int pad, char fill)
{
    while ((pad > 0) && (p < end)) {
        *(p++) = fill;
        --pad;
    }
    return p;
}

/*
 * Convert a long integer to a string.
 * @return Pointer to the end of the string (after last character written).
 */
static char *ucm_log_ltoa(char *p, char *end, long n, int base, int flags,
                          int pad)
{
    static const char digits[] = "0123456789abcdef";
    long div;

    if (((n < 0) || (flags & UCM_LOG_LTOA_FLAG_SIGN)) && (p < end)) {
        *(p++) = (n < 0 ) ? '-' : '+';
    }

    if (n == 0) {
        if (p < end) {
            *(p++) = '0';
        }
        goto out;
    }

    n = labs(n);

    div = 1;
    while ((n / div) != 0) {
        div *= base;
        --pad;
    }

    if (!(flags & UCM_LOG_LTOA_PAD_LEFT)) {
        p = ucm_log_add_padding(p, end, pad,
                                (flags & UCM_LOG_LTOA_FLAG_PAD0) ? '0' : ' ');
    }

    div /= base;
    while ((p < end) && (div > 0)) {
        *(p++) = digits[(n / div + base) % base];
        div /= base;
    }

    if (flags & UCM_LOG_LTOA_PAD_LEFT) {
        p = ucm_log_add_padding(p, end, pad, ' ');
    }

out:
    return p;
}

/*
 * Implement basic formatted print.
 * We can't use snprintf() because it may potentially call malloc().
 *
 * Supported format characters:
 *  %[-]?[0-9]?s
 *  %m
 *  %%
 *  %[+]?[0-9]?[l]?[dxup]
 */
static void ucm_log_vsnprintf(char *buf, size_t max, const char *fmt, va_list ap)
{
    const char *pf;
    char *pb, *endb, *ps;
    union {
        char          *s;
        long          d;
        unsigned long u;
        uintptr_t     p;
    } value;
    int flags;
    int pad;
    int base;
    int eno;

    pf   = fmt;
    pb   = buf;
    endb = buf + max - 1;
    eno  = errno;

    while ((pb < endb) && (*pf != '\0')) {
        if (*pf != '%') {
            *(pb++) = *(pf++);
            continue;
        }

        /* Data field */
        pad   = 0;
        flags = 0;
        base  = 10;
        while (pb < endb) {
            ++pf;
            switch (*pf) {
            /* The '%' character */
            case '%':
                *(pb++) = '%';
                goto done;

            /* Error message */
            case 'm':
                ps = strerror_r(eno, pb, endb - pb);
                if (ps != pb) {
                    strncpy(pb, ps, endb - pb);
                }
                pb += strlen(pb);
                goto done;

            /* String */
            case 's':
                value.s = va_arg(ap, char *);
                pad -= strlen(value.s);
                if (!(flags & UCM_LOG_LTOA_PAD_LEFT)) {
                    pb = ucm_log_add_padding(pb, endb, pad, ' ');
                }
                while ((pb < endb) && (*value.s != '\0')) {
                    *(pb++) = *(value.s++);
                }
                if (flags & UCM_LOG_LTOA_PAD_LEFT) {
                    pb = ucm_log_add_padding(pb, endb, pad, ' ');
                }
                goto done;

            /* Signed number */
            case 'd':
                if (flags & UCM_LOG_LTOA_FLAG_LONG) {
                    value.d = va_arg(ap, long);
                } else {
                    value.d = va_arg(ap, int);
                }
                pb = ucm_log_ltoa(pb, endb, value.d, base, flags, pad);
                goto done;

            /* Hex number */
            case 'x':
                base = 16;
                /* Fall thru */

            /* Unsigned number */
            case 'u':
                if (flags & UCM_LOG_LTOA_FLAG_LONG) {
                    value.u = va_arg(ap, unsigned long);
                } else {
                    value.u = va_arg(ap, unsigned);
                }
                flags |= UCM_LOG_LTOA_FLAG_UNSIGN;
                pb = ucm_log_ltoa(pb, endb, value.u, base, flags, pad);
                goto done;

            /* Pointer */
            case 'p':
                value.p = va_arg(ap, uintptr_t);
                if (pb < endb) {
                    *(pb++) = '0';
                }
                if (pb < endb) {
                    *(pb++) = 'x';
                }
                pb = ucm_log_ltoa(pb, endb, value.p, 16, flags, pad);
                goto done;

            /* Flags and modifiers */
            case '+':
                flags |= UCM_LOG_LTOA_FLAG_SIGN;
                break;
            case '-':
                flags |= UCM_LOG_LTOA_PAD_LEFT;
                break;
            case 'l':
                flags |= UCM_LOG_LTOA_FLAG_LONG;
                break;
            case '0':
                if (pad == 0) {
                    flags |= UCM_LOG_LTOA_FLAG_PAD0;
                }
                /* Fall thru */
            default:
                if (isdigit(*pf)) {
                    pad = (pad * 10) + (*pf - '0');
                }
                break;
            }
        }
done:
        ++pf;
    }
    *pb = '\0';
}

static void ucm_log_snprintf(char *buf, size_t max, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    ucm_log_vsnprintf(buf, max, fmt, ap);
    va_end(ap);
}

void __ucm_log(const char *file, unsigned line, const char *function,
               ucs_log_level_t level, const char *message, ...)
{
    char buf[UCM_LOG_BUG_SIZE];
    size_t length;
    va_list ap;
    struct timeval tv;
    ssize_t nwrite;

    gettimeofday(&tv, NULL);
    ucm_log_snprintf(buf, UCM_LOG_BUG_SIZE - 1, "[%lu.%06lu] [%s:%d] %18s:%-4d UCX  %s ",
                     tv.tv_sec, tv.tv_usec, ucm_log_hostname, getpid(),
                     basename(file), line, ucm_log_level_names[level]);
    buf[UCM_LOG_BUG_SIZE - 1] = '\0';

    length = strlen(buf);
    va_start(ap, message);
    ucm_log_vsnprintf(buf + length, UCM_LOG_BUG_SIZE - length, message, ap);
    va_end(ap);
    strncat(buf, "\n", UCM_LOG_BUG_SIZE - 1);

    /* Use writev to avoid potential calls to malloc() in buffered IO functions */
    nwrite = write(ucm_log_fileno, buf, strlen(buf));
    (void)nwrite;

    if (level <= UCS_LOG_LEVEL_FATAL) {
        abort();
    }
}

UCS_STATIC_INIT {
    gethostname(ucm_log_hostname, sizeof(ucm_log_hostname));
}
