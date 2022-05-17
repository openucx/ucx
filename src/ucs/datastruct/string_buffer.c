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
#include <ucs/debug/memtrack_int.h>
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

void ucs_string_buffer_reset(ucs_string_buffer_t *strb)
{
    ucs_array_length(&strb->str) = 0;
}

size_t ucs_string_buffer_length(ucs_string_buffer_t *strb)
{
    return ucs_array_length(&strb->str);
}

static void ucs_string_buffer_add_null_terminator(ucs_string_buffer_t *strb)
{
    ucs_assert(ucs_array_available_length(&strb->str) >= 1);
    *ucs_array_end(&strb->str) = '\0';
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
            ucs_string_buffer_add_null_terminator(strb);
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

void ucs_string_buffer_append_flags(ucs_string_buffer_t *strb, uint64_t mask,
                                    const char **flag_names)
{
    unsigned flag;

    ucs_for_each_bit(flag, mask) {
        if (flag_names == NULL) {
            ucs_string_buffer_appendf(strb, "%u,", flag);
        } else {
            ucs_string_buffer_appendf(strb, "%s|", flag_names[flag]);
        }
    }
    ucs_string_buffer_rtrim(strb, ",|");
}

void ucs_string_buffer_append_iovec(ucs_string_buffer_t *strb,
                                    const struct iovec *iov, size_t iovcnt)
{
    size_t iov_index;

    for (iov_index = 0; iov_index < iovcnt; ++iov_index) {
        ucs_string_buffer_appendf(strb, "%p,%zu|", iov[iov_index].iov_base,
                                  iov[iov_index].iov_len);
    }
    ucs_string_buffer_rtrim(strb, "|");
}

void ucs_string_buffer_rtrim(ucs_string_buffer_t *strb, const char *charset)
{
    char *ptr = ucs_array_end(&strb->str);

    if (ucs_array_is_empty(&strb->str)) {
        /* If the string is empty, do not write '\0' terminator */
        return;
    }

    do {
        --ptr;
        if (((charset == NULL) && !isspace(*ptr)) ||
            ((charset != NULL) && (strchr(charset, *ptr) == NULL))) {
            /* if the last character should NOT be removed - stop */
            break;
        }

        ucs_array_set_length(&strb->str, ucs_array_length(&strb->str) - 1);
    } while (!ucs_array_is_empty(&strb->str));

    ucs_string_buffer_add_null_terminator(strb);
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

char *ucs_string_buffer_next_token(ucs_string_buffer_t *strb, char *token,
                                   const char *delimiters)
{
    char *next_token;

    /* The token must be either NULL or inside the string buffer array */
    ucs_assert((token == NULL) || ((token >= ucs_array_begin(&strb->str)) &&
                                   (token < ucs_array_end(&strb->str))));

    next_token = (token == NULL) ? ucs_array_begin(&strb->str) :
                                   (token + strlen(token) + 1);
    if (next_token >= ucs_array_end(&strb->str)) {
        /* No more tokens */
        return NULL;
    }

    return strsep(&next_token, delimiters);
}

void ucs_string_buffer_appendc(ucs_string_buffer_t *strb, int c, size_t count)
{
    size_t length = ucs_array_length(&strb->str);
    size_t append_length;

    (void)ucs_array_reserve(string_buffer, &strb->str, length + count + 1);

    if (ucs_array_available_length(&strb->str) < 1) {
        /* No room to add anything */
        return;
    }

    append_length = ucs_min(count, ucs_array_available_length(&strb->str) - 1);
    memset(ucs_array_end(&strb->str), c, append_length);
    ucs_array_set_length(&strb->str, length + append_length);

    ucs_string_buffer_add_null_terminator(strb);
}
