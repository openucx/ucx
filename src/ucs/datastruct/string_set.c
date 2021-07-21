/**
 * Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "string_set.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/memtrack_int.h>


#define UCS_STRING_SET_ALLOC_NAME          "string_set"


void ucs_string_set_init(ucs_string_set_t *sset)
{
    kh_init_inplace(ucs_string_set, sset);
}

void ucs_string_set_cleanup(ucs_string_set_t *sset)
{
    char *str;

    kh_foreach_key(sset, str, {
        ucs_free(str);
    });
    kh_destroy_inplace(ucs_string_set, sset);
}

/* Adds string by pointer, and releases the string if add fails or the string
 * already exists in the set
 */
static ucs_status_t ucs_string_set_add_ptr(ucs_string_set_t *sset, char *str)
{
    int ret;

    kh_put(ucs_string_set, sset, str, &ret);

    switch (ret) {
    case -1:
        ucs_free(str);
        return UCS_ERR_NO_MEMORY;
    case 0:
        /* key already present */
        ucs_free(str);
        return UCS_OK;
    case 1:
    case 2:
        /* key inserted */
        return UCS_OK;
    default:
        ucs_error("unexpected return value from kh_put(ucs_string_set): %d", ret);
        return UCS_ERR_INVALID_PARAM;
    }
}

ucs_status_t ucs_string_set_add(ucs_string_set_t *sset, const char *str)
{
    char *str_copy;

    str_copy = ucs_strdup(str, UCS_STRING_SET_ALLOC_NAME);
    if (str_copy == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    return ucs_string_set_add_ptr(sset, str_copy);
}

ucs_status_t ucs_string_set_addf(ucs_string_set_t *sset, const char *fmt, ...)
{
    int length;
    va_list ap;
    char *str;

    va_start(ap, fmt);
    length = vsnprintf(NULL, 0, fmt, ap);
    va_end(ap);

    str = ucs_malloc(length + 1, UCS_STRING_SET_ALLOC_NAME);
    if (str == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    va_start(ap, fmt);
    vsnprintf(str, length + 1, fmt, ap);
    va_end(ap);

    return ucs_string_set_add_ptr(sset, str);
}

int ucs_string_set_contains(const ucs_string_set_t *sset, const char *str)
{
    return kh_get(ucs_string_set, sset, (char*)str) != kh_end(sset);
}

static int ucs_string_set_compare_func(const void *a, const void *b)
{
    return strcmp(*(const char**)a, *(const char**)b);
}

ucs_status_t ucs_string_set_print_sorted(const ucs_string_set_t *sset,
                                         ucs_string_buffer_t *strb,
                                         const char *sep)
{
    const char **sorted_strings;
    size_t idx, count;
    char *str;

    /* allocate a temporary array to hold the sorted strings */
    count          = kh_size(sset);
    sorted_strings = ucs_calloc(count, sizeof(*sorted_strings), "string_set");
    if (sorted_strings == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    /* collect and sort the strings */
    idx = 0;
    kh_foreach_key(sset, str, {
        sorted_strings[idx++] = str;
    })
    ucs_assert(idx == count);
    qsort(sorted_strings, count, sizeof(*sorted_strings),
          ucs_string_set_compare_func);

    /* append the sorted strings to the string buffer */
    for (idx = 0; idx < count; ++idx) {
        ucs_string_buffer_appendf(strb, "%s%s", (idx > 0) ? sep : "",
                                  sorted_strings[idx]);
    }

    ucs_free(sorted_strings);

    return UCS_OK;
}
