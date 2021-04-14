/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "string_buffer.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack.h>
#include <ucs/sys/string.h>
#include <ucs/sys/math.h>
#include <string.h>
#include <ctype.h>

#include <ucs/datastruct/array.inl>


/* Minimal reserve size when appending new data */
#define UCS_STRING_BUFFER_RESERVE  32

UCS_ARRAY_IMPL(string_buffer, size_t, char, static UCS_F_ALWAYS_INLINE)

void ucs_string_buffer_init(ucs_string_buffer_t *strb)
{
    ucs_array_init_dynamic(&strb->str);
}

void ucs_string_buffer_init_fixed(ucs_string_buffer_t *strb, char *buffer,
                                  size_t capacity)
{
    ucs_array_init_fixed(&strb->str, buffer, capacity);
}

void ucs_string_buffer_cleanup(ucs_string_buffer_t *strb)
{
    ucs_array_cleanup_dynamic(&strb->str);
}

size_t ucs_string_buffer_length(ucs_string_buffer_t *strb)
{
    return ucs_array_length(&strb->str);
}

void ucs_string_buffer_appendf(ucs_string_buffer_t *strb, const char *fmt, ...)
{
    ucs_status_t status;
    size_t max_print;
    va_list ap;
    int ret;

    ucs_array_reserve(string_buffer, &strb->str,
                      ucs_array_length(&strb->str) + UCS_STRING_BUFFER_RESERVE);

    /* try to write to existing buffer */
    va_start(ap, fmt);
    max_print = ucs_array_available_length(&strb->str);
    ret       = vsnprintf(ucs_array_end(&strb->str), max_print, fmt, ap);
    va_end(ap);

    /* if failed, grow the buffer accommodate for the expected extra length */
    if (ret >= max_print) {
        status = ucs_array_reserve(string_buffer, &strb->str,
                                   ucs_array_length(&strb->str) + ret + 1);
        if (status != UCS_OK) {
            /* cannot grow the buffer, just set null terminator at the end, and
             * the string will contain only what could fit in.
             */
            ucs_array_length(&strb->str) = ucs_array_capacity(&strb->str) - 1;
            *ucs_array_end(&strb->str)   = '\0';
            goto out;
        }

        va_start(ap, fmt);
        max_print = ucs_array_available_length(&strb->str);
        ret       = vsnprintf(ucs_array_end(&strb->str), max_print, fmt, ap);
        va_end(ap);

        /* since we've grown the buffer, it should be sufficient now */
        ucs_assertv(ret < max_print, "ret=%d max_print=%zu", ret, max_print);
    }

    /* string length grows by the amount of characters written by vsnprintf */
    ucs_array_set_length(&strb->str, ucs_array_length(&strb->str) + ret);

    /* \0 should be written by vsnprintf */
out:
    ucs_assert(ucs_array_available_length(&strb->str) >= 1);
    ucs_assert(*ucs_array_end(&strb->str) == '\0');
}

void ucs_string_buffer_append_hex(ucs_string_buffer_t *strb, const void *data,
                                  size_t size, size_t per_line)
{
    size_t prev_length    = ucs_array_length(&strb->str);
    size_t hexdump_length = (size * 2) + (size / 4) + (size / per_line);
    size_t new_length;

    ucs_array_reserve(string_buffer, &strb->str, prev_length + hexdump_length);
    ucs_str_dump_hex(data, size, ucs_array_end(&strb->str),
                     ucs_array_available_length(&strb->str), per_line);

    new_length = prev_length + strlen(ucs_array_end(&strb->str));
    ucs_array_set_length(&strb->str, new_length);
    ucs_assert(*ucs_array_end(&strb->str) == '\0');
}

void ucs_string_buffer_rtrim(ucs_string_buffer_t *strb, const char *charset)
{
    char *ptr = ucs_array_end(&strb->str);

    while (ucs_array_length(&strb->str) > 0) {
        --ptr;
        if (((charset == NULL) && !isspace(*ptr)) ||
            ((charset != NULL) && (strchr(charset, *ptr) == NULL))) {
            /* if the last character should NOT be removed - stop */
            break;
        }

        ucs_array_set_length(&strb->str, ucs_array_length(&strb->str) - 1);
    }

    /* mark the new end of string */
    *(ptr + 1) = '\0';
}

const char *ucs_string_buffer_cstr(const ucs_string_buffer_t *strb)
{
    char *c_str;

    if (ucs_array_is_empty(&strb->str)) {
        return "";
    }

    c_str = ucs_array_begin(&strb->str);
    ucs_assert(c_str != NULL);
    return c_str;
}

void ucs_string_buffer_dump(const ucs_string_buffer_t *strb,
                            const char *line_prefix, FILE *stream)
{
    const char *next_tok, *tok;
    size_t size, remaining;

    if (ucs_array_is_empty(&strb->str)) {
        return;
    }

    tok      = ucs_array_begin(&strb->str);
    next_tok = strchr(tok, '\n');
    while (next_tok != NULL) {
        fputs(line_prefix, stream);

        /* Write the line, handle partial writes */
        remaining = UCS_PTR_BYTE_DIFF(tok, next_tok + 1);
        while (remaining > 0) {
            size       = fwrite(tok, sizeof(*tok), remaining, stream);
            tok        = UCS_PTR_BYTE_OFFSET(tok, size);
            remaining -= size;
        }

        next_tok = strchr(tok, '\n');
    }

    /* Write last line */
    if (*tok != '\0') {
        fputs(line_prefix, stream);
        fputs(tok, stream);
    }
}

char *ucs_string_buffer_extract_mem(ucs_string_buffer_t *strb)
{
    char *c_str;

    if (ucs_array_is_fixed(&strb->str)) {
        c_str = ucs_strdup(ucs_array_begin(&strb->str), "ucs_string_buffer");
    } else {
        c_str = ucs_array_begin(&strb->str);
        ucs_array_init_dynamic(&strb->str);
    }

    return c_str;
}
