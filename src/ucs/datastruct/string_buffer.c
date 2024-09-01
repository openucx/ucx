/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
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
#include <ucs/sys/sock.h>
#include <string.h>
#include <ctype.h>


/* Minimal reserve size when appending new data */
#define UCS_STRING_BUFFER_RESERVE  32

void ucs_string_buffer_init(ucs_string_buffer_t *strb)
{
    ucs_array_init_dynamic(strb);
}

void ucs_string_buffer_init_fixed(ucs_string_buffer_t *strb, char *buffer,
                                  size_t capacity)
{
    ucs_array_init_fixed(strb, buffer, capacity);
    if (capacity > 0) {
        ucs_array_elem(strb, 0) = '\0';
    }
}

void ucs_string_buffer_cleanup(ucs_string_buffer_t *strb)
{
    ucs_array_cleanup_dynamic(strb);
}

void ucs_string_buffer_reset(ucs_string_buffer_t *strb)
{
    ucs_array_clear(strb);
}

size_t ucs_string_buffer_length(ucs_string_buffer_t *strb)
{
    return ucs_array_length(strb);
}

static void ucs_string_buffer_add_null_terminator(ucs_string_buffer_t *strb)
{
    ucs_assert(ucs_array_available_length(strb) >= 1);
    *ucs_array_end(strb) = '\0';
}

void ucs_string_buffer_appendf(ucs_string_buffer_t *strb, const char *fmt, ...)
{
    ucs_status_t status;
    size_t max_print;
    va_list ap;
    int ret;

    ucs_array_reserve(strb, ucs_array_length(strb) + UCS_STRING_BUFFER_RESERVE);
    ucs_assert(ucs_array_begin(strb) != NULL); /* For coverity */

    /* try to write to existing buffer */
    va_start(ap, fmt);
    max_print = ucs_array_available_length(strb);
    ret       = vsnprintf(ucs_array_end(strb), max_print, fmt, ap);
    va_end(ap);

    /* if failed, grow the buffer accommodate for the expected extra length */
    if (ret >= max_print) {
        status = ucs_array_reserve(strb, ucs_array_length(strb) + ret + 1);
        if (status != UCS_OK) {
            /* cannot grow the buffer, just set null terminator at the end, and
             * the string will contain only what could fit in.
             */
            ucs_array_length(strb) = ucs_array_capacity(strb) - 1;
            ucs_string_buffer_add_null_terminator(strb);
            goto out;
        }

        va_start(ap, fmt);
        max_print = ucs_array_available_length(strb);
        ret       = vsnprintf(ucs_array_end(strb), max_print, fmt, ap);
        va_end(ap);

        /* since we've grown the buffer, it should be sufficient now */
        ucs_assertv(ret < max_print, "ret=%d max_print=%zu", ret, max_print);
    }

    /* string length grows by the amount of characters written by vsnprintf */
    ucs_array_set_length(strb, ucs_array_length(strb) + ret);

    /* \0 should be written by vsnprintf */
out:
    ucs_assert(ucs_array_available_length(strb) >= 1);
    ucs_assert(*ucs_array_end(strb) == '\0');
}

void ucs_string_buffer_append_hex(ucs_string_buffer_t *strb, const void *data,
                                  size_t size, size_t per_line)
{
    size_t prev_length    = ucs_array_length(strb);
    size_t hexdump_length = (size * 2) + (size / 4) + (size / per_line);
    size_t new_length;

    ucs_array_reserve(strb, prev_length + hexdump_length);
    ucs_str_dump_hex(data, size, ucs_array_end(strb),
                     ucs_array_available_length(strb), per_line);

    new_length = prev_length + strlen(ucs_array_end(strb));
    ucs_array_set_length(strb, new_length);
    ucs_assert(*ucs_array_end(strb) == '\0');
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

void ucs_string_buffer_append_saddr(ucs_string_buffer_t *strb,
                                    const struct sockaddr *sa)
{
    char sockstr[UCS_SOCKADDR_STRING_LEN];

    ucs_sockaddr_str(sa, sockstr, UCS_SOCKADDR_STRING_LEN);
    ucs_string_buffer_appendf(strb, "%s", sockstr);
}

static int ucs_string_buffer_match_charset(char ch, const char *charset)
{
    return (charset == NULL) ? isspace(ch) : (strchr(charset, ch) != NULL);
}

void ucs_string_buffer_rtrim(ucs_string_buffer_t *strb, const char *charset)
{
    char *ptr = ucs_array_end(strb);

    if (ucs_array_is_empty(strb)) {
        /* If the string is empty, do not write '\0' terminator */
        return;
    }

    do {
        --ptr;
        /* if the last character should NOT be removed - stop */
        if (!ucs_string_buffer_match_charset(*ptr, charset)) {
            break;
        }

        ucs_array_set_length(strb, ucs_array_length(strb) - 1);
    } while (!ucs_array_is_empty(strb));

    ucs_string_buffer_add_null_terminator(strb);
}

void ucs_string_buffer_rbrk(ucs_string_buffer_t *strb, const char *delim)
{
    char *begin = ucs_array_begin(strb);
    char *ptr;

    if (ucs_array_is_empty(strb)) {
        return;
    }

    for (ptr = ucs_array_last(strb); ptr >= begin; --ptr) {
        if (ucs_string_buffer_match_charset(*ptr, delim)) {
            ucs_array_set_length(strb, UCS_PTR_BYTE_DIFF(begin, ptr));
            ucs_string_buffer_add_null_terminator(strb);
            break;
        }
    }
}

const char *ucs_string_buffer_cstr(const ucs_string_buffer_t *strb)
{
    char *c_str;

    if (ucs_array_is_empty(strb)) {
        return "";
    }

    c_str = ucs_array_begin(strb);
    ucs_assert(c_str != NULL);
    return c_str;
}

void ucs_string_buffer_dump(const ucs_string_buffer_t *strb,
                            const char *line_prefix, FILE *stream)
{
    const char *next_tok, *tok;
    size_t size, remaining;

    if (ucs_array_is_empty(strb)) {
        return;
    }

    tok      = ucs_array_begin(strb);
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

    if (ucs_array_is_fixed(strb)) {
        c_str = ucs_strdup(ucs_array_begin(strb), "ucs_string_buffer");
    } else {
        c_str = ucs_array_begin(strb);
        ucs_array_init_dynamic(strb);
    }

    return c_str;
}

char *ucs_string_buffer_next_token(ucs_string_buffer_t *strb, char *token,
                                   const char *delimiters)
{
    char *next_token;

    /* The token must be either NULL or inside the string buffer array */
    ucs_assert((token == NULL) || ((token >= ucs_array_begin(strb)) &&
                                   (token < ucs_array_end(strb))));

    next_token = (token == NULL) ? ucs_array_begin(strb) :
                                   (token + strlen(token) + 1);
    if (next_token >= ucs_array_end(strb)) {
        /* No more tokens */
        return NULL;
    }

    return strsep(&next_token, delimiters);
}

void ucs_string_buffer_appendc(ucs_string_buffer_t *strb, int c, size_t count)
{
    size_t length = ucs_array_length(strb);
    size_t append_length;

    ucs_array_reserve(strb, length + count + 1);

    if (ucs_array_available_length(strb) < 1) {
        /* No room to add anything */
        return;
    }

    ucs_assert(ucs_array_begin(strb) != NULL); /* For coverity */
    append_length = ucs_min(count, ucs_array_available_length(strb) - 1);
    memset(ucs_array_end(strb), c, append_length);
    ucs_array_set_length(strb, length + append_length);

    ucs_string_buffer_add_null_terminator(strb);
}

void ucs_string_buffer_translate(ucs_string_buffer_t *strb,
                                 ucs_string_buffer_translate_cb_t cb)
{
    char *src_ptr, *dst_ptr;
    char new_char;

    if (ucs_array_is_empty(strb)) {
        return;
    }

    src_ptr = dst_ptr = ucs_array_begin(strb);
    while (src_ptr < ucs_array_end(strb)) {
        new_char = cb(*src_ptr);
        if (new_char != '\0') {
            *dst_ptr++ = new_char;
        }
        ++src_ptr;
    }

    *dst_ptr = '\0';
    ucs_array_set_length(strb, dst_ptr - ucs_array_begin(strb));
}
