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
#include <ucs/sys/math.h>
#include <string.h>
#include <ctype.h>


#define UCS_STRING_BUFFER_GROW                32
#define UCS_STRING_BUFFER_ALLOC_NAME          "string_buffer"


static void ucs_string_buffer_reset(ucs_string_buffer_t *strb)
{
    strb->buffer   = NULL;
    strb->length   = 0;
    strb->capacity = 0;
}

void ucs_string_buffer_init(ucs_string_buffer_t *strb)
{
    ucs_string_buffer_reset(strb);
}

void ucs_string_buffer_cleanup(ucs_string_buffer_t *strb)
{
    ucs_free(strb->buffer);
    ucs_string_buffer_reset(strb);
}

/*
 * Grow the buffer to at least the required size and at least double the
 * previous size (to reduce the amortized cost of realloc)
 */
static ucs_status_t ucs_string_buffer_grow(ucs_string_buffer_t *strb,
                                           size_t min_capacity)
{
    size_t new_capacity;
    char *new_buffer;

    new_capacity = ucs_max(strb->capacity * 2, min_capacity);
    new_buffer   = ucs_realloc(strb->buffer, new_capacity,
                               UCS_STRING_BUFFER_ALLOC_NAME);
    if (new_buffer == NULL) {
        ucs_error("failed to grow string from %zu to %zu characters",
                  strb->capacity, new_capacity);
        return UCS_ERR_NO_MEMORY;
    }

    strb->buffer   = new_buffer;
    strb->capacity = new_capacity;
    /* length stays the same */
    return UCS_OK;
}

ucs_status_t ucs_string_buffer_appendf(ucs_string_buffer_t *strb,
                                       const char *fmt, ...)
{
    ucs_status_t status;
    size_t max_print;
    va_list ap;
    int ret;

    /* set minimal initial size */
    if ((strb->capacity - strb->length) <= 1) {
        status = ucs_string_buffer_grow(strb,
                                        strb->capacity + UCS_STRING_BUFFER_GROW);
        if (status != UCS_OK) {
            return status;
        }
    }

    /* try to write to existing buffer */
    va_start(ap, fmt);
    max_print = strb->capacity - strb->length - 1;
    ret       = vsnprintf(strb->buffer + strb->length, max_print, fmt, ap);
    va_end(ap);

    /* if failed, grow the buffer accommodate for the expected extra length */
    if (ret >= max_print) {
        status = ucs_string_buffer_grow(strb, strb->length + ret + 1);
        if (status != UCS_OK) {
            return status;
        }

        va_start(ap, fmt);
        max_print = strb->capacity - strb->length;
        ret       = vsnprintf(strb->buffer + strb->length, max_print, fmt, ap);
        va_end(ap);

        /* since we've grown the buffer, it should be sufficient now */
        ucs_assertv(ret < max_print, "ret=%d max_print=%zu", ret, max_print);
    }

    /* string length grows by the amount of characters written by vsnprintf */
    strb->length += ret;

    ucs_assert(strb->length < strb->capacity);
    ucs_assert(strb->buffer[strb->length] == '\0'); /* \0 is written by vsnprintf */

    return UCS_OK;
}

void ucs_string_buffer_rtrim(ucs_string_buffer_t *strb, const char *charset)
{
    char *ptr;

    ptr = &strb->buffer[strb->length];
    while (strb->length > 0) {
        --ptr;
        if (((charset == NULL) && !isspace(*ptr)) ||
            ((charset != NULL) && (strchr(charset, *ptr) == NULL))) {
            /* if the last character should NOT be removed - stop */
            break;
        }

        --strb->length;
    }

    /* mark the new end of string */
    *(ptr + 1) = '\0';
}

const char *ucs_string_buffer_cstr(const ucs_string_buffer_t *strb)
{
    if (strb->length == 0) {
        return "";
    }

    ucs_assert(strb->buffer  != NULL);
    ucs_assert(strb->capacity > 0);
    return strb->buffer;
}
