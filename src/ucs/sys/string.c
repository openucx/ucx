/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2017.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "string.h"
#include "math.h"
#include "sys.h"
#include <ucs/config/parser.h>
#include <ucs/arch/bitops.h>

#include <string.h>
#include <ctype.h>
#include <stdio.h>
#include <time.h>


void ucs_fill_filename_template(const char *tmpl, char *buf, size_t max)
{
    char *p, *end;
    const char *pf, *pp;
    size_t length;
    time_t t;

    p = buf;
    end = buf + max - 1;
    *end = 0;
    pf = tmpl;
    while (*pf != 0 && p < end) {
        pp = strchr(pf, '%');
        if (pp == NULL) {
            strncpy(p, pf, end - p);
            p = end;
            break;
        }

        length = ucs_min(pp - pf, end - p);
        strncpy(p, pf, length);
        p += length;

        switch (*(pp + 1)) {
        case 'p':
            snprintf(p, end - p, "%d", getpid());
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'h':
            snprintf(p, end - p, "%s", ucs_get_host_name());
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'c':
            snprintf(p, end - p, "%02d", ucs_get_first_cpu());
            pf = pp + 2;
            p += strlen(p);
            break;
        case 't':
            t = time(NULL);
            strftime(p, end - p, "%Y-%m-%d-%H:%M:%S", localtime(&t));
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'u':
            snprintf(p, end - p, "%s", basename(ucs_get_user_name()));
            pf = pp + 2;
            p += strlen(p);
            break;
        case 'e':
            snprintf(p, end - p, "%s", basename(ucs_get_exe()));
            pf = pp + 2;
            p += strlen(p);
            break;
        default:
            *(p++) = *pp;
            pf = pp + 1;
            break;
        }

        p += strlen(p);
    }
    *p = 0;
}

void ucs_snprintf_zero(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;

    memset(buf, 0, size);
    va_start(ap, fmt);
    vsnprintf(buf, size, fmt, ap);
    va_end(ap);
}

void ucs_strncpy_zero(char *dest, const char *src, size_t max)
{
    if (max) {
        strncpy(dest, src, max - 1);
        dest[max - 1] = '\0';
    }
}

uint64_t ucs_string_to_id(const char* str)
{
    uint64_t id = 0;
    strncpy((char*)&id, str, sizeof(id) - 1); /* Last character will be \0 */
    return id;
}

size_t ucs_string_quantity_prefix_value(char prefix)
{
    switch (prefix) {
    case 'B':
        return 1;
    case 'K':
        return UCS_KBYTE;
    case 'M':
        return UCS_MBYTE;
    case 'G':
        return UCS_GBYTE;
    case 'T':
        return UCS_TBYTE;
    default:
        return 0;
    }
}

void ucs_memunits_to_str(size_t value, char *buf, size_t max)
{
    static const char * suffixes[] = {"", "K", "M", "G", "T", NULL};

    const char **suffix;

    if (value == SIZE_MAX) {
        strncpy(buf, "(inf)", max);
    } else {
        suffix = &suffixes[0];
        while ((value >= 1024) && ((value % 1024) == 0) && *(suffix + 1)) {
            value /= 1024;
            ++suffix;
        }
        snprintf(buf, max, "%zu%s", value, *suffix);
    }
}

ucs_status_t ucs_str_to_memunits(const char *buf, void *dest)
{
    char units[3];
    int num_fields;
    size_t value;
    size_t bytes;

    /* Special value: infinity */
    if (!strcasecmp(buf, UCS_NUMERIC_INF_STR)) {
        *(size_t*)dest = UCS_MEMUNITS_INF;
        return UCS_OK;
    }

    /* Special value: auto */
    if (!strcasecmp(buf, "auto")) {
        *(size_t*)dest = UCS_MEMUNITS_AUTO;
        return UCS_OK;
    }

    memset(units, 0, sizeof(units));
    num_fields = sscanf(buf, "%ld%c%c", &value, &units[0], &units[1]);
    if (num_fields == 1) {
        bytes = 1;
    } else if ((num_fields == 2) || (num_fields == 3)) {
        bytes = ucs_string_quantity_prefix_value(toupper(units[0]));
        if (!bytes || ((num_fields == 3) && tolower(units[1]) != 'b')) {
            return UCS_ERR_INVALID_PARAM;
        }
    } else {
        return UCS_ERR_INVALID_PARAM;
    }

    *(size_t*)dest = value * bytes;
    return UCS_OK;
}

void ucs_snprintf_safe(char *buf, size_t size, const char *fmt, ...)
{
    va_list ap;

    if (size == 0) {
        return;
    }

    va_start(ap, fmt);
    vsnprintf(buf, size - 1, fmt, ap);
    buf[size - 1] = '\0';
    va_end(ap);
}

char* ucs_strncpy_safe(char *dst, const char *src, size_t len)
{
    size_t length;

    if (!len) {
        return dst;
    }

    /* copy string into dst including null terminator */
    length = ucs_min(len, strnlen(src, len) + 1);

    memcpy(dst, src, length);
    dst[length - 1] = '\0';
    return dst;
}

char *ucs_strtrim(char *str)
{
    char *start, *end;

    /* point 'p' at first non-space character */
    start = str;
    while (isspace(*start)) {
        ++start;
    }

    if (*start) {
        /* write '\0' after the last non-space character */
        end = start + strlen(start) - 1;
        while (isspace(*end)) {
            --end;
        }
        *(end + 1) = '\0';
    }

    return start;
}

const char * ucs_str_dump_hex(const void* data, size_t length, char *buf,
                              size_t max, size_t per_line)
{
    static const char hexchars[] = "0123456789abcdef";
    char *p, *endp;
    uint8_t value;
    size_t i;

    p     = buf;
    endp  = buf + max - 2;
    i     = 0;
    while ((p < endp) && (i < length)) {
        if (i > 0) {
            if ((i % per_line) == 0) {
                *(p++) = '\n';
            } else if ((i % 4) == 0) {
                *(p++) = ':';
            }

            if (p == endp) {
                break;
            }
        }

        value = *(const uint8_t*)(UCS_PTR_BYTE_OFFSET(data, i));
        p[0]  = hexchars[value / 16];
        p[1]  = hexchars[value % 16];
        p    += 2;
        ++i;
    }
    *p = 0;
    return buf;
}

const char* ucs_flags_str(char *buf, size_t max,
                          uint64_t flags, const char **str_table)
{
    size_t i, len = 0;

    for (i = 0; *str_table; ++str_table, ++i) {
        if (flags & UCS_BIT(i)) { /* not using ucs_for_each_bit to silence coverity */
            snprintf(buf + len, max - len, "%s,", *str_table);
            len = strlen(buf);
        }
    }

    if (len > 0) {
        buf[len - 1] = '\0'; /* remove last ',' */
    } else {
        buf[0] = '\0';
    }

    return buf;
}
