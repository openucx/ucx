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

#include <string.h>
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
    strncpy(dest, src, max - 1);
    dest[max - 1] = '\0';
}

uint64_t ucs_string_to_id(const char* str)
{
    uint64_t id = 0;
    strncpy((char*)&id, str, sizeof(id) - 1); /* Last character will be \0 */
    return id;
}

void ucs_memunits_to_str(size_t value, char *buf, size_t max)
{
    static const char * suffixes[] = {"", "k", "m", "g", "t"};

    const char **suffix;

    if (value == SIZE_MAX) {
        strncpy(buf, "(inf)", max);
    } else {
        suffix = &suffixes[0];
        while ((value >= 1024) && ((value % 1024) == 0)) {
            value /= 1024;
            ++suffix;
        }
        snprintf(buf, max, "%zu%s", value, *suffix);
    }
}

const char* ucs_sockaddr_str(const struct sockaddr *sock_addr, char *str, size_t max_size)
{
    struct sockaddr_in6 *addr_in6;
    struct sockaddr_in *addr_in;

    switch (sock_addr->sa_family) {
    case AF_INET:
        addr_in = (struct sockaddr_in *) sock_addr;
        inet_ntop(AF_INET, &addr_in->sin_addr, str, max_size);
        max_size -= strlen(str);
        snprintf(str + strlen(str), max_size, ":%d", ntohs(addr_in->sin_port));
        return str;
    case AF_INET6:
        addr_in6 = (struct sockaddr_in6 *)sock_addr;
        inet_ntop(AF_INET6, &addr_in6->sin6_addr, str, max_size);
        max_size -= strlen(str);
        snprintf(str + strlen(str), max_size, ":%d", ntohs(addr_in6->sin6_port));
        return str;
    default:
        return "Invalid string";
    }
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
