/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#include "parser.h"

#include <ucs/arch/atomic.h>
#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/datastruct/list.h>
#include <ucs/datastruct/khash.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug.h>
#include <ucs/time/time.h>
#include <fnmatch.h>
#include <ctype.h>


/* width of titles in docstring */
#define UCS_CONFIG_PARSER_DOCSTR_WIDTH         10

/* String literal for allow-list */
#define UCS_CONFIG_PARSER_ALL "all"

/* list of prefixes for a configuration variable, used to dump all possible
 * aliases.
 */
typedef struct ucs_config_parser_prefix_list {
    const char                  *prefix;
    ucs_list_link_t             list;
} ucs_config_parser_prefix_t;


typedef UCS_CONFIG_ARRAY_FIELD(void, data) ucs_config_array_field_t;

KHASH_SET_INIT_STR(ucs_config_env_vars)


/* Process environment variables */
extern char **environ;


UCS_LIST_HEAD(ucs_config_global_list);
static khash_t(ucs_config_env_vars) ucs_config_parser_env_vars = {0};
static pthread_mutex_t ucs_config_parser_env_vars_hash_lock    = PTHREAD_MUTEX_INITIALIZER;
static char ucs_config_parser_negate                           = '^';


const char *ucs_async_mode_names[] = {
    [UCS_ASYNC_MODE_SIGNAL]          = "signal",
    [UCS_ASYNC_MODE_THREAD_SPINLOCK] = "thread_spinlock",
    [UCS_ASYNC_MODE_THREAD_MUTEX]    = "thread_mutex",
    [UCS_ASYNC_MODE_POLL]            = "poll",
    [UCS_ASYNC_MODE_LAST]            = NULL
};

UCS_CONFIG_DEFINE_ARRAY(string, sizeof(char*), UCS_CONFIG_TYPE_STRING);

/* Fwd */
static ucs_status_t
ucs_config_parser_set_value_internal(void *opts, ucs_config_field_t *fields,
                                     const char *name, const char *value,
                                     const char *table_prefix, int recurse);


static int __find_string_in_list(const char *str, const char **list)
{
    int i;

    for (i = 0; *list; ++list, ++i) {
        if (strcasecmp(*list, str) == 0) {
            return i;
        }
    }
    return -1;
}

int ucs_config_sscanf_string(const char *buf, void *dest, const void *arg)
{
    *((char**)dest) = strdup(buf);
    return 1;
}

int ucs_config_sprintf_string(char *buf, size_t max,
                              const void *src, const void *arg)
{
    strncpy(buf, *((char**)src), max);
    return 1;
}

ucs_status_t ucs_config_clone_string(const void *src, void *dest, const void *arg)
{
    char *new_str = strdup(*(char**)src);
    if (new_str == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    *((char**)dest) = new_str;
    return UCS_OK;
}

void ucs_config_release_string(void *ptr, const void *arg)
{
    free(*(char**)ptr);
}

int ucs_config_sscanf_int(const char *buf, void *dest, const void *arg)
{
    return sscanf(buf, "%i", (unsigned*)dest);
}

ucs_status_t ucs_config_clone_int(const void *src, void *dest, const void *arg)
{
    *(int*)dest = *(int*)src;
    return UCS_OK;
}

int ucs_config_sprintf_int(char *buf, size_t max,
                           const void *src, const void *arg)
{
    return snprintf(buf, max, "%i", *(unsigned*)src);
}

int ucs_config_sscanf_uint(const char *buf, void *dest, const void *arg)
{
    if (!strcasecmp(buf, UCS_NUMERIC_INF_STR)) {
        *(unsigned*)dest = UINT_MAX;
        return 1;
    } else {
        return sscanf(buf, "%u", (unsigned*)dest);
    }
}

ucs_status_t ucs_config_clone_uint(const void *src, void *dest, const void *arg)
{
    *(unsigned*)dest = *(unsigned*)src;
    return UCS_OK;
}

int ucs_config_sprintf_uint(char *buf, size_t max,
                            const void *src, const void *arg)
{
    unsigned value = *(unsigned*)src;
    if (value == UINT_MAX) {
        snprintf(buf, max, UCS_NUMERIC_INF_STR);
        return 1;
    } else {
        return snprintf(buf, max, "%u", value);
    }
}

int ucs_config_sscanf_ulong(const char *buf, void *dest, const void *arg)
{
    return sscanf(buf, "%lu", (unsigned long*)dest);
}

int ucs_config_sprintf_ulong(char *buf, size_t max,
                             const void *src, const void *arg)
{
    return snprintf(buf, max, "%lu", *(unsigned long*)src);
}

ucs_status_t ucs_config_clone_ulong(const void *src, void *dest, const void *arg)
{
    *(unsigned long*)dest = *(unsigned long*)src;
    return UCS_OK;
}

int ucs_config_sscanf_double(const char *buf, void *dest, const void *arg)
{
    return sscanf(buf, "%lf", (double*)dest);
}

int ucs_config_sprintf_double(char *buf, size_t max,
                              const void *src, const void *arg)
{
    return snprintf(buf, max, "%.3f", *(double*)src);
}

ucs_status_t ucs_config_clone_double(const void *src, void *dest, const void *arg)
{
    *(double*)dest = *(double*)src;
    return UCS_OK;
}

int ucs_config_sscanf_hex(const char *buf, void *dest, const void *arg)
{
    /* Special value: auto */
    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        *(size_t*)dest = UCS_HEXUNITS_AUTO;
        return 1;
    } else if (strncasecmp(buf, "0x", 2) == 0) {
        return (sscanf(buf + 2, "%x", (unsigned int*)dest));
    } else {
        return 0;
    }
}

int ucs_config_sprintf_hex(char *buf, size_t max,
                           const void *src, const void *arg)
{
    uint16_t val = *(uint16_t*)src;

    if (val == UCS_HEXUNITS_AUTO) {
        return snprintf(buf, max, UCS_VALUE_AUTO_STR);
    }

    return snprintf(buf, max, "0x%x", *(unsigned int*)src);
}

int ucs_config_sscanf_bool(const char *buf, void *dest, const void *arg)
{
    if (!strcasecmp(buf, "y") || !strcasecmp(buf, "yes") || !strcmp(buf, "1")) {
        *(int*)dest = 1;
        return 1;
    } else if (!strcasecmp(buf, "n") || !strcasecmp(buf, "no") || !strcmp(buf, "0")) {
        *(int*)dest = 0;
        return 1;
    } else {
        return 0;
    }
}

int ucs_config_sprintf_bool(char *buf, size_t max, const void *src, const void *arg)
{
    return snprintf(buf, max, "%c", *(int*)src ? 'y' : 'n');
}

int ucs_config_sscanf_ternary(const char *buf, void *dest, const void *arg)
{
    UCS_STATIC_ASSERT(UCS_NO  == 0);
    UCS_STATIC_ASSERT(UCS_YES == 1);
    if (!strcasecmp(buf, "try") || !strcasecmp(buf, "maybe")) {
        *(int*)dest = UCS_TRY;
        return 1;
    }

    return ucs_config_sscanf_bool(buf, dest, arg);
}

int ucs_config_sscanf_ternary_auto(const char *buf, void *dest, const void *arg)
{
    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        *(int*)dest = UCS_AUTO;
        return 1;
    }

    return ucs_config_sscanf_ternary(buf, dest, arg);
}

int ucs_config_sprintf_ternary_auto(char *buf, size_t max,
                                    const void *src, const void *arg)
{
    if (*(int*)src == UCS_AUTO) {
        return snprintf(buf, max, UCS_VALUE_AUTO_STR);
    } else if (*(int*)src == UCS_TRY) {
        return snprintf(buf, max, "try");
    }

    return ucs_config_sprintf_bool(buf, max, src, arg);
}

int ucs_config_sscanf_on_off(const char *buf, void *dest, const void *arg)
{
    if (!strcasecmp(buf, "on") || !strcmp(buf, "1")) {
        *(int*)dest = UCS_CONFIG_ON;
        return 1;
    } else if (!strcasecmp(buf, "off") || !strcmp(buf, "0")) {
        *(int*)dest = UCS_CONFIG_OFF;
        return 1;
    } else {
        return 0;
    }
}

int ucs_config_sscanf_on_off_auto(const char *buf, void *dest, const void *arg)
{
    if (!strcasecmp(buf, "try")   ||
        !strcasecmp(buf, "maybe") ||
        !strcasecmp(buf, "auto")) {
        *(int*)dest = UCS_CONFIG_AUTO;
        return 1;
    } else {
        return ucs_config_sscanf_on_off(buf, dest, arg);
    }
}

int ucs_config_sprintf_on_off_auto(char *buf, size_t max,
                                   const void *src, const void *arg)
{
    switch (*(int*)src) {
    case UCS_CONFIG_AUTO:
        return snprintf(buf, max, "auto");
    case UCS_CONFIG_ON:
        return snprintf(buf, max, "on");
    case UCS_CONFIG_OFF:
        return snprintf(buf, max, "off");
    default:
        return snprintf(buf, max, "%d", *(int*)src);
    }
}

int ucs_config_sscanf_enum(const char *buf, void *dest, const void *arg)
{
    int i;

    i = __find_string_in_list(buf, (const char**)arg);
    if (i < 0) {
        return 0;
    }

    *(unsigned*)dest = i;
    return 1;
}

int ucs_config_sprintf_enum(char *buf, size_t max,
                            const void *src, const void *arg)
{
    char * const *table = arg;
    strncpy(buf, table[*(unsigned*)src], max);
    return 1;
}

static void __print_table_values(char * const *table, char *buf, size_t max)
{
    char *ptr = buf, *end = buf + max;

    for (; *table; ++table) {
        snprintf(ptr, end - ptr, "|%s", *table);
        ptr += strlen(ptr);
    }

    snprintf(ptr, end - ptr, "]");

    *buf = '[';
}

void ucs_config_help_enum(char *buf, size_t max, const void *arg)
{
    __print_table_values(arg, buf, max);
}

ucs_status_t ucs_config_clone_log_comp(const void *src, void *dst, const void *arg)
{
    const ucs_log_component_config_t *src_comp = src;
    ucs_log_component_config_t       *dst_comp = dst;

    dst_comp->log_level = src_comp->log_level;
    ucs_strncpy_safe(dst_comp->name, src_comp->name, sizeof(dst_comp->name));

    return UCS_OK;
}

int ucs_config_sscanf_bitmap(const char *buf, void *dest, const void *arg)
{
    char *str = strdup(buf);
    char *p, *saveptr;
    int ret, i;

    if (str == NULL) {
        return 0;
    }

    ret = 1;
    *((unsigned*)dest) = 0;
    p = strtok_r(str, ",", &saveptr);
    while (p != NULL) {
        i = __find_string_in_list(p, (const char**)arg);
        if (i < 0) {
            ret = 0;
            break;
        }
        *((unsigned*)dest) |= UCS_BIT(i);
        p = strtok_r(NULL, ",", &saveptr);
    }

    free(str);
    return ret;
}

int ucs_config_sprintf_bitmap(char *buf, size_t max,
                              const void *src, const void *arg)
{
    ucs_flags_str(buf, max, *((unsigned*)src), (const char**)arg);
    return 1;
}

void ucs_config_help_bitmap(char *buf, size_t max, const void *arg)
{
    snprintf(buf, max, "comma-separated list of: ");
    __print_table_values(arg, buf + strlen(buf), max - strlen(buf));
}

int ucs_config_sscanf_bitmask(const char *buf, void *dest, const void *arg)
{
    int ret = sscanf(buf, "%u", (unsigned*)dest);
    if (*(unsigned*)dest != 0) {
        *(unsigned*)dest = UCS_BIT(*(unsigned*)dest) - 1;
    }
    return ret;
}

int ucs_config_sprintf_bitmask(char *buf, size_t max,
                               const void *src, const void *arg)
{
    return snprintf(buf, max, "%u", __builtin_popcount(*(unsigned*)src));
}

int ucs_config_sscanf_time(const char *buf, void *dest, const void *arg)
{
    char units[3];
    int num_fields;
    double value;
    double per_sec;

    memset(units, 0, sizeof(units));
    num_fields = sscanf(buf, "%lf%c%c", &value, &units[0], &units[1]);
    if (num_fields == 1) {
        per_sec = 1;
    } else if (num_fields == 2 || num_fields == 3) {
        if (!strcmp(units, "m")) {
            per_sec = 1.0 / 60.0;
        } else if (!strcmp(units, "s")) {
            per_sec = 1;
        } else if (!strcmp(units, "ms")) {
            per_sec = UCS_MSEC_PER_SEC;
        } else if (!strcmp(units, "us")) {
            per_sec = UCS_USEC_PER_SEC;
        } else if (!strcmp(units, "ns")) {
            per_sec = UCS_NSEC_PER_SEC;
        } else {
            return 0;
        }
    } else {
        return 0;
    }

    *(double*)dest = value / per_sec;
    return 1;
}

int ucs_config_sprintf_time(char *buf, size_t max,
                            const void *src, const void *arg)
{
    return snprintf(buf, max, "%.2fus", *(double*)src * UCS_USEC_PER_SEC);
}

int ucs_config_sscanf_time_units(const char *buf, void *dest, const void *arg)
{
    double value;
    int ret;

    if (!strcmp(buf, "inf")) {
        *(ucs_time_t*)dest = UCS_TIME_INFINITY;
        return 1;
    } else if (!strcmp(buf, "auto")) {
        *(ucs_time_t*)dest = UCS_TIME_AUTO;
        return 1;
    }

    ret = ucs_config_sscanf_time(buf, &value, arg);
    if (ret == 0) {
        return 0;
    }

    *(ucs_time_t*)dest = ucs_time_from_sec(value);
    return 1;
}

int ucs_config_sprintf_time_units(char *buf, size_t max,
                                  const void *src, const void *arg)
{
    double value;

    if (*(ucs_time_t*)src == UCS_TIME_INFINITY) {
        return snprintf(buf, max, "inf");
    } else if (*(ucs_time_t*)src == UCS_TIME_AUTO) {
        return snprintf(buf, max, "auto");
    }

    value = ucs_time_to_sec(*(ucs_time_t*)src);
    return ucs_config_sprintf_time(buf, max, &value, arg);
}

int ucs_config_sscanf_bw(const char *buf, void *dest, const void *arg)
{
    double *dst     = (double*)dest;
    char    str[16] = {0};
    int     offset  = 0;
    size_t  divider;
    size_t  units;
    double  value;
    int     num_fields;

    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        *dst = UCS_CONFIG_BW_AUTO;
        return 1;
    }

    num_fields = sscanf(buf, "%lf%15s", &value, str);
    if (num_fields < 2) {
        return 0;
    }

    ucs_assert(num_fields == 2);

    units = (str[0] == 'b') ? 1 : ucs_string_quantity_prefix_value(str[0]);
    if (!units) {
        return 0;
    }

    offset = (units == 1) ? 0 : 1;

    switch (str[offset]) {
    case 'B':
        divider = 1;
        break;
    case 'b':
        divider = 8;
        break;
    default:
        return 0;
    }

    offset++;
    if (strcmp(str + offset, "ps") &&
        strcmp(str + offset, "/s") &&
        strcmp(str + offset, "s")) {
        return 0;
    }

    ucs_assert((divider == 1) || (divider == 8)); /* bytes or bits */
    *dst = value * units / divider;
    return 1;
}

int ucs_config_sprintf_bw(char *buf, size_t max, const void *src,
                          const void *arg)
{
    static const double max_value = 50000.0;
    double value                  = *(double*)src;
    const char **suffix;

    if (UCS_CONFIG_BW_IS_AUTO(value)) {
        ucs_strncpy_safe(buf, UCS_VALUE_AUTO_STR, max);
        return 1;
    }

    suffix = &ucs_memunits_suffixes[0];
    while ((value > max_value) && (*(suffix + 1) != NULL)) {
        value /= 1024;
        ++suffix;
    }

    ucs_snprintf_safe(buf, max, "%.2f%sBps", value, *suffix);
    return 1;
}

int ucs_config_sscanf_bw_spec(const char *buf, void *dest, const void *arg)
{
    ucs_config_bw_spec_t *dst = (ucs_config_bw_spec_t*)dest;
    char                 *delim;

    delim = strchr(buf, ':');
    if (!delim) {
        return 0;
    }

    if (!ucs_config_sscanf_bw(delim + 1, &dst->bw, arg)) {
        return 0;
    }

    dst->name = ucs_strndup(buf, delim - buf, __func__);
    return dst->name != NULL;
}

int ucs_config_sprintf_bw_spec(char *buf, size_t max,
                               const void *src, const void *arg)
{
    ucs_config_bw_spec_t *bw  = (ucs_config_bw_spec_t*)src;
    int                   len;

    if (max) {
        snprintf(buf, max, "%s:", bw->name);
        len = strlen(buf);
        ucs_config_sprintf_bw(buf + len, max - len, &bw->bw, arg);
    }

    return 1;
}

ucs_status_t ucs_config_clone_bw_spec(const void *src, void *dest, const void *arg)
{
    ucs_config_bw_spec_t *s = (ucs_config_bw_spec_t*)src;
    ucs_config_bw_spec_t *d = (ucs_config_bw_spec_t*)dest;

    d->bw   = s->bw;
    d->name = ucs_strdup(s->name, __func__);

    return d->name ? UCS_OK : UCS_ERR_NO_MEMORY;
}

void ucs_config_release_bw_spec(void *ptr, const void *arg)
{
    ucs_free(((ucs_config_bw_spec_t*)ptr)->name);
}

int ucs_config_sscanf_signo(const char *buf, void *dest, const void *arg)
{
    char *endptr;
    int signo;

    signo = strtol(buf, &endptr, 10);
    if (*endptr == '\0') {
        *(int*)dest = signo;
        return 1;
    }

    if (!strncmp(buf, "SIG", 3)) {
        buf += 3;
    }

    return ucs_config_sscanf_enum(buf, dest, ucs_signal_names);
}

int ucs_config_sprintf_signo(char *buf, size_t max,
                             const void *src, const void *arg)
{
    return ucs_config_sprintf_enum(buf, max, src, ucs_signal_names);
}

int ucs_config_sscanf_memunits(const char *buf, void *dest, const void *arg)
{
    if (ucs_str_to_memunits(buf, dest) != UCS_OK) {
        return 0;
    }
    return 1;
}

int ucs_config_sprintf_memunits(char *buf, size_t max,
                                const void *src, const void *arg)
{
    ucs_memunits_to_str(*(size_t*)src, buf, max);
    return 1;
}

int ucs_config_sscanf_ulunits(const char *buf, void *dest, const void *arg)
{
    /* Special value: auto */
    if (!strcasecmp(buf, UCS_VALUE_AUTO_STR)) {
        *(unsigned long*)dest = UCS_ULUNITS_AUTO;
        return 1;
    } else if (!strcasecmp(buf, UCS_NUMERIC_INF_STR)) {
        *(unsigned long*)dest = UCS_ULUNITS_INF;
        return 1;
    }

    return ucs_config_sscanf_ulong(buf, dest, arg);
}

int ucs_config_sprintf_ulunits(char *buf, size_t max,
                               const void *src, const void *arg)
{
    unsigned long val = *(unsigned long*)src;

    if (val == UCS_ULUNITS_AUTO) {
        return snprintf(buf, max, UCS_VALUE_AUTO_STR);
    } else if (val == UCS_ULUNITS_INF) {
        return snprintf(buf, max, UCS_NUMERIC_INF_STR);
    }

    return ucs_config_sprintf_ulong(buf, max, src, arg);
}

int ucs_config_sscanf_range_spec(const char *buf, void *dest, const void *arg)
{
    ucs_range_spec_t *range_spec = dest;
    unsigned first, last;
    char *p, *str;
    int ret = 1;

    str = strdup(buf);
    if (str == NULL) {
        return 0;
    }

    /* Check if got a range or a single number */
    p = strchr(str, '-');
    if (p == NULL) {
        /* got only one value (not a range) */
        if (1 != sscanf(buf, "%u", &first)) {
            ret = 0;
            goto out;
        }
        last = first;
    } else {
        /* got a range of numbers */
        *p = 0;      /* split str */

        if ((1 != sscanf(str, "%u", &first))
            || (1 != sscanf(p + 1, "%u", &last))) {
            ret = 0;
            goto out;
        }
    }

    range_spec->first = first;
    range_spec->last = last;

out:
    free (str);
    return ret;
}

int ucs_config_sprintf_range_spec(char *buf, size_t max,
                                  const void *src, const void *arg)
{
    const ucs_range_spec_t *range_spec = src;

    if (range_spec->first == range_spec->last) {
        snprintf(buf, max, "%d", range_spec->first);
    } else {
        snprintf(buf, max, "%d-%d", range_spec->first, range_spec->last);
    }

    return 1;
}

ucs_status_t ucs_config_clone_range_spec(const void *src, void *dest, const void *arg)
{
    const ucs_range_spec_t *src_range_spec = src;
    ucs_range_spec_t *dest_ragne_spec      = dest;

    dest_ragne_spec->first = src_range_spec->first;
    dest_ragne_spec->last = src_range_spec->last;

    return UCS_OK;
}

int ucs_config_sscanf_array(const char *buf, void *dest, const void *arg)
{
    ucs_config_array_field_t *field = dest;
    void *temp_field;
    const ucs_config_array_t *array = arg;
    char *str_dup, *token, *saveptr;
    int ret;
    unsigned i;

    str_dup = strdup(buf);
    if (str_dup == NULL) {
        return 0;
    }

    saveptr = NULL;
    token = strtok_r(str_dup, ",", &saveptr);
    temp_field = ucs_calloc(UCS_CONFIG_ARRAY_MAX, array->elem_size, "config array");
    i = 0;
    while (token != NULL) {
        ret = array->parser.read(token, (char*)temp_field + i * array->elem_size,
                                 array->parser.arg);
        if (!ret) {
            ucs_free(temp_field);
            free(str_dup);
            return 0;
        }

        ++i;
        if (i >= UCS_CONFIG_ARRAY_MAX) {
            break;
        }
        token = strtok_r(NULL, ",", &saveptr);
    }

    field->data = temp_field;
    field->count = i;
    free(str_dup);
    return 1;
}

int ucs_config_sprintf_array(char *buf, size_t max,
                             const void *src, const void *arg)
{
    const ucs_config_array_field_t *field = src;
    const ucs_config_array_t *array       = arg;
    size_t offset;
    unsigned i;
    int ret;

    offset = 0;
    for (i = 0; i < field->count; ++i) {
        if (i > 0 && offset < max) {
            buf[offset++] = ',';
        }
        ret = array->parser.write(buf + offset, max - offset,
                                  (char*)field->data + i * array->elem_size,
                                  array->parser.arg);
        if (!ret) {
            return 0;
        }

        offset += strlen(buf + offset);
    }
    return 1;
}

ucs_status_t ucs_config_clone_array(const void *src, void *dest, const void *arg)
{
    const ucs_config_array_field_t *src_array = src;
    const ucs_config_array_t *array           = arg;
    ucs_config_array_field_t *dest_array      = dest;
    ucs_status_t status;
    unsigned i;

    if (src_array->count > 0) {
        dest_array->data = ucs_calloc(src_array->count, array->elem_size,
                                      "config array");
        if (dest_array->data == NULL) {
            return UCS_ERR_NO_MEMORY;
        }
    } else {
        dest_array->data = NULL;
    }

    dest_array->count = src_array->count;
    for (i = 0; i < src_array->count; ++i) {
        status = array->parser.clone((const char*)src_array->data  + i * array->elem_size,
                                    (char*)dest_array->data + i * array->elem_size,
                                    array->parser.arg);
        if (status != UCS_OK) {
            ucs_free(dest_array->data);
            return status;
        }
    }

    return UCS_OK;
}

void ucs_config_release_array(void *ptr, const void *arg)
{
    ucs_config_array_field_t *array_field = ptr;
    const ucs_config_array_t *array = arg;
    unsigned i;

    for (i = 0; i < array_field->count; ++i) {
        array->parser.release((char*)array_field->data  + i * array->elem_size,
                              array->parser.arg);
    }
    ucs_free(array_field->data);
}

void ucs_config_help_array(char *buf, size_t max, const void *arg)
{
    const ucs_config_array_t *array = arg;

    snprintf(buf, max, "comma-separated list of: ");
    array->parser.help(buf + strlen(buf), max - strlen(buf), array->parser.arg);
}

int ucs_config_sscanf_allow_list(const char *buf, void *dest, const void *arg)
{
    ucs_config_allow_list_t *field  = dest;
    unsigned offset                 = 0;

    if (buf[0] == ucs_config_parser_negate) {
        field->mode = UCS_CONFIG_ALLOW_LIST_NEGATE;
        offset++;
    } else {
        field->mode = UCS_CONFIG_ALLOW_LIST_ALLOW;
    }

    if (!ucs_config_sscanf_array(&buf[offset], &field->array, arg)) {
        return 0;
    }

    if ((field->array.count >= 1) &&
        !strcmp(field->array.names[0], UCS_CONFIG_PARSER_ALL)) {
        field->mode = UCS_CONFIG_ALLOW_LIST_ALLOW_ALL;
        ucs_config_release_array(&field->array, arg);
        if (field->array.count != 1) {
            return 0;
        }

        field->array.count = 0;
    }

    return 1;
}

int ucs_config_sprintf_allow_list(char *buf, size_t max, const void *src,
                                  const void *arg)
{
    const ucs_config_allow_list_t *allow_list = src;
    size_t offset                             = 0;

    if (allow_list->mode == UCS_CONFIG_ALLOW_LIST_ALLOW_ALL) {
        snprintf(buf, max, UCS_CONFIG_PARSER_ALL);
        return 1;
    }
    
    if (allow_list->mode == UCS_CONFIG_ALLOW_LIST_NEGATE) {
        buf[offset++] = ucs_config_parser_negate;
        max--;
    }

    return ucs_config_sprintf_array(&buf[offset], max, &allow_list->array, arg);
}

ucs_status_t ucs_config_clone_allow_list(const void *src, void *dest, const void *arg)
{
    const ucs_config_allow_list_t *src_list = src;
    ucs_config_allow_list_t *dest_list      = dest;

    dest_list->mode = src_list->mode;
    return ucs_config_clone_array(&src_list->array, &dest_list->array, arg);
}

void ucs_config_release_allow_list(void *ptr, const void *arg)
{
    ucs_config_allow_list_t *allow_list = ptr;

    if (allow_list->mode == UCS_CONFIG_ALLOW_LIST_ALLOW_ALL) {
        return;
    }

    ucs_config_release_array(&allow_list->array, arg);
}

void ucs_config_help_allow_list(char *buf, size_t max, const void *arg)
{
    const ucs_config_array_t *array = arg;

    snprintf(
        buf,
        max, "comma-separated list (use \"all\" for including \
              all items or \'^\' for negation) of: ");
    array->parser.help(buf + strlen(buf), max - strlen(buf), array->parser.arg);
}

int ucs_config_sscanf_table(const char *buf, void *dest, const void *arg)
{
    char *tokens;
    char *token, *saveptr1;
    char *name, *value, *saveptr2;
    ucs_status_t status;

    tokens = strdup(buf);
    if (tokens == NULL) {
        return 0;
    }

    saveptr1 = NULL;
    saveptr2 = NULL;
    token = strtok_r(tokens, ";", &saveptr1);
    while (token != NULL) {
        name  = strtok_r(token, "=", &saveptr2);
        value = strtok_r(NULL,  "=", &saveptr2);
        if (name == NULL || value == NULL) {
            free(tokens);
            ucs_error("Could not parse list of values in '%s' (token: '%s')", buf, token);
            return 0;
        }

        status = ucs_config_parser_set_value_internal(dest, (ucs_config_field_t*)arg,
                                                     name, value, NULL, 1);
        if (status != UCS_OK) {
            if (status == UCS_ERR_NO_ELEM) {
                ucs_error("Field '%s' does not exist", name);
            } else {
                ucs_debug("Failed to set %s to '%s': %s", name, value,
                          ucs_status_string(status));
            }
            free(tokens);
            return 0;
        }

        token = strtok_r(NULL, ";", &saveptr1);
    }

    free(tokens);
    return 1;
}

ucs_status_t ucs_config_clone_table(const void *src, void *dst, const void *arg)
{
    return ucs_config_parser_clone_opts(src, dst, (ucs_config_field_t*)arg);
}

void ucs_config_release_table(void *ptr, const void *arg)
{
    ucs_config_parser_release_opts(ptr, (ucs_config_field_t*)arg);
}

void ucs_config_help_table(char *buf, size_t max, const void *arg)
{
    snprintf(buf, max, "Table");
}

void ucs_config_release_nop(void *ptr, const void *arg)
{
}

void ucs_config_help_generic(char *buf, size_t max, const void *arg)
{
    strncpy(buf, (char*)arg, max);
}

static inline int ucs_config_is_deprecated_field(const ucs_config_field_t *field)
{
    return (field->offset == UCS_CONFIG_DEPRECATED_FIELD_OFFSET);
}

static inline int ucs_config_is_alias_field(const ucs_config_field_t *field)
{
    return (field->dfl_value == NULL);
}

static inline int ucs_config_is_table_field(const ucs_config_field_t *field)
{
    return (field->parser.read == ucs_config_sscanf_table);
}

static void ucs_config_print_doc_line_by_line(const ucs_config_field_t *field,
                                              void (*cb)(int num, const char *line, void *arg),
                                              void *arg)
{
    char *doc, *line, *p;
    int num;

    line = doc = strdup(field->doc);
    p = strchr(line, '\n');
    num = 0;
    while (p != NULL) {
        *p = '\0';
        cb(num, line, arg);
        line = p + 1;
        p = strchr(line, '\n');
        ++num;
    }
    cb(num, line, arg);
    free(doc);
}

static ucs_status_t
ucs_config_parser_parse_field(ucs_config_field_t *field, const char *value, void *var)
{
    char syntax_buf[256];
    int ret;

    ret = field->parser.read(value, var, field->parser.arg);
    if (ret != 1) {
        if (ucs_config_is_table_field(field)) {
            ucs_error("Could not set table value for %s: '%s'", field->name, value);

        } else {
            field->parser.help(syntax_buf, sizeof(syntax_buf) - 1, field->parser.arg);
            ucs_error("Invalid value for %s: '%s'. Expected: %s", field->name,
                      value, syntax_buf);
        }
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void ucs_config_parser_release_field(ucs_config_field_t *field, void *var)
{
    field->parser.release(var, field->parser.arg);
}

static int ucs_config_field_is_last(const ucs_config_field_t *field)
{
    return field->name == NULL;
}

ucs_status_t
ucs_config_parser_set_default_values(void *opts, ucs_config_field_t *fields)
{
    ucs_config_field_t *field, *sub_fields;
    ucs_status_t status;
    void *var;

    for (field = fields; !ucs_config_field_is_last(field); ++field) {
        if (ucs_config_is_alias_field(field) ||
            ucs_config_is_deprecated_field(field)) {
            continue;
        }

        var = (char*)opts + field->offset;

        /* If this field is a sub-table, recursively set the values for it.
         * Defaults can be subsequently set by parser.read(). */
        if (ucs_config_is_table_field(field)) {
            sub_fields = (ucs_config_field_t*)field->parser.arg;
            status = ucs_config_parser_set_default_values(var, sub_fields);
            if (status != UCS_OK) {
                return status;
            }
        }

        status = ucs_config_parser_parse_field(field, field->dfl_value, var);
        if (status != UCS_OK) {
            return status;
        }
    }

    return UCS_OK;
}

/**
 * table_prefix == NULL  -> unused
 */
static ucs_status_t
ucs_config_parser_set_value_internal(void *opts, ucs_config_field_t *fields,
                                     const char *name, const char *value,
                                     const char *table_prefix, int recurse)
{
    char value_buf[256] = "";
    ucs_config_field_t *field, *sub_fields;
    size_t prefix_len;
    ucs_status_t status;
    ucs_status_t UCS_V_UNUSED status_restore;
    int UCS_V_UNUSED ret;
    unsigned count;
    void *var;

    prefix_len = (table_prefix == NULL) ? 0 : strlen(table_prefix);

    count = 0;
    for (field = fields; !ucs_config_field_is_last(field); ++field) {

        var = (char*)opts + field->offset;

        if (ucs_config_is_table_field(field)) {
            sub_fields = (ucs_config_field_t*)field->parser.arg;

            /* Check with sub-table prefix */
            if (recurse) {
                status = ucs_config_parser_set_value_internal(var, sub_fields,
                                                             name, value,
                                                             field->name, 1);
                if (status == UCS_OK) {
                    ++count;
                } else if (status != UCS_ERR_NO_ELEM) {
                    return status;
                }
            }

            /* Possible override with my prefix */
            if (table_prefix != NULL) {
                status = ucs_config_parser_set_value_internal(var, sub_fields,
                                                             name, value,
                                                             table_prefix, 0);
                if (status == UCS_OK) {
                    ++count;
                } else if (status != UCS_ERR_NO_ELEM) {
                    return status;
                }
            }
        } else if (((table_prefix == NULL) || !strncmp(name, table_prefix, prefix_len)) &&
                   !strcmp(name + prefix_len, field->name))
        {
            if (ucs_config_is_deprecated_field(field)) {
                return UCS_ERR_NO_ELEM;
            }

            /* backup current value to restore it in case if new value
             * is not accepted */
            ret = field->parser.write(value_buf, sizeof(value_buf) - 1, var,
                                      field->parser.arg);
            ucs_assert(ret != 0); /* write success */
            ucs_config_parser_release_field(field, var);
            status = ucs_config_parser_parse_field(field, value, var);
            if (status != UCS_OK) {
                status_restore = ucs_config_parser_parse_field(field, value_buf, var);
                /* current value must be valid */
                ucs_assert(status_restore == UCS_OK);
                return status;
            }
            ++count;
        }
    }

    return (count == 0) ? UCS_ERR_NO_ELEM : UCS_OK;
}

static void ucs_config_parser_mark_env_var_used(const char *name, int *added)
{
    khiter_t iter;
    char *key;
    int ret;

    *added = 0;

    if (!ucs_global_opts.warn_unused_env_vars) {
        return;
    }

    pthread_mutex_lock(&ucs_config_parser_env_vars_hash_lock);

    iter = kh_get(ucs_config_env_vars, &ucs_config_parser_env_vars, name);
    if (iter != kh_end(&ucs_config_parser_env_vars)) {
        goto out; /* already exists */
    }

    key = ucs_strdup(name, "config_parser_env_var");
    if (key == NULL) {
        ucs_error("strdup(%s) failed", name);
        goto out;
    }

#ifndef __clang_analyzer__
    /* Exclude this code from Clang examination as it generates
     * false-postive warning about potential leak of memory
     * pointed to by 'key' variable */
    iter = kh_put(ucs_config_env_vars, &ucs_config_parser_env_vars, key, &ret);
    if ((ret <= 0) || (iter == kh_end(&ucs_config_parser_env_vars))) {
        ucs_warn("kh_put(key=%s) failed", key);
        ucs_free(key);
        goto out;
    }
#else
    ucs_free(key);
#endif

    *added = 1;

out:
    pthread_mutex_unlock(&ucs_config_parser_env_vars_hash_lock);
}

static ucs_status_t ucs_config_apply_env_vars(void *opts, ucs_config_field_t *fields,
                                             const char *prefix, const char *table_prefix,
                                             int recurse, int ignore_errors)
{
    ucs_config_field_t *field, *sub_fields;
    ucs_status_t status;
    size_t prefix_len;
    const char *env_value;
    void *var;
    char buf[256];
    int added;

    /* Put prefix in the buffer. Later we replace only the variable name part */
    snprintf(buf, sizeof(buf) - 1, "%s%s", prefix, table_prefix ? table_prefix : "");
    prefix_len = strlen(buf);

    /* Parse environment variables */
    for (field = fields; !ucs_config_field_is_last(field); ++field) {

        var = (char*)opts + field->offset;

        if (ucs_config_is_table_field(field)) {
            sub_fields = (ucs_config_field_t*)field->parser.arg;

            /* Parse with sub-table prefix */
            if (recurse) {
                status = ucs_config_apply_env_vars(var, sub_fields, prefix,
                                                   field->name, 1, ignore_errors);
                if (status != UCS_OK) {
                    return status;
                }
            }

            /* Possible override with my prefix */
            if (table_prefix) {
                status = ucs_config_apply_env_vars(var, sub_fields, prefix,
                                                   table_prefix, 0, ignore_errors);
                if (status != UCS_OK) {
                    return status;
                }
            }
        } else {
            /* Read and parse environment variable */
            strncpy(buf + prefix_len, field->name, sizeof(buf) - prefix_len - 1);
            env_value = getenv(buf);
            if (env_value == NULL) {
                continue;
            }

            ucs_config_parser_mark_env_var_used(buf, &added);

            if (ucs_config_is_deprecated_field(field)) {
                if (added && !ignore_errors) {
                    ucs_warn("%s is deprecated (set %s%s=n to suppress this warning)",
                             buf, UCS_DEFAULT_ENV_PREFIX,
                             UCS_GLOBAL_OPTS_WARN_UNUSED_CONFIG);
                }
            } else {
                ucs_config_parser_release_field(field, var);
                status = ucs_config_parser_parse_field(field, env_value, var);
                if (status != UCS_OK) {
                    /* If set to ignore errors, restore the default value */
                    ucs_status_t tmp_status =
                        ucs_config_parser_parse_field(field, field->dfl_value,
                                                      var);
                    if (ignore_errors) {
                        status = tmp_status;
                    }
                }
                if (status != UCS_OK) {
                    return status;
                }
            }
        }
    }

    return UCS_OK;
}

/* Find if env_prefix consists of multiple prefixes and returns pointer
 * to rightmost in this case, otherwise returns NULL
 */ 
static ucs_status_t ucs_config_parser_get_sub_prefix(const char *env_prefix,
                                                     const char **sub_prefix_p)
{
    size_t len;

    /* env_prefix always has "_" at the end and we want to find the last but one
     * "_" in the env_prefix */
    len = strlen(env_prefix);
    if (len < 2) {
        ucs_error("Invalid value of env_prefix: '%s'", env_prefix);
        return UCS_ERR_INVALID_PARAM;
    }

    len -= 2;
    while ((len > 0) && (env_prefix[len - 1] != '_')) {
        len -= 1;
    }
    *sub_prefix_p = (len > 0) ? (env_prefix + len): NULL;

    return UCS_OK;
}

ucs_status_t ucs_config_parser_fill_opts(void *opts, ucs_config_field_t *fields,
                                         const char *env_prefix,
                                         const char *table_prefix,
                                         int ignore_errors)
{
    const char   *sub_prefix = NULL;
    ucs_status_t status;

    /* Set default values */
    status = ucs_config_parser_set_default_values(opts, fields);
    if (status != UCS_OK) {
        goto err;
    }

    ucs_assert(env_prefix != NULL);
    status = ucs_config_parser_get_sub_prefix(env_prefix, &sub_prefix);
    if (status != UCS_OK) {
        goto err;
    }

    /* Apply environment variables */
    if (sub_prefix != NULL) {
        status = ucs_config_apply_env_vars(opts, fields, sub_prefix, table_prefix,
                                           1, ignore_errors);
        if (status != UCS_OK) {
            goto err_free;
        }
    }

    /* Apply environment variables with custom prefix */
    status = ucs_config_apply_env_vars(opts, fields, env_prefix, table_prefix,
                                        1, ignore_errors);
    if (status != UCS_OK) {
        goto err_free;
    }

    return UCS_OK;

err_free:
    ucs_config_parser_release_opts(opts, fields); /* Release default values */
err:
    return status;
}

ucs_status_t ucs_config_parser_set_value(void *opts, ucs_config_field_t *fields,
                                        const char *name, const char *value)
{
    return ucs_config_parser_set_value_internal(opts, fields, name, value, NULL, 1);
}

ucs_status_t ucs_config_parser_get_value(void *opts, ucs_config_field_t *fields,
                                         const char *name, char *value,
                                         size_t max)
{
    ucs_config_field_t  *field;
    ucs_config_field_t  *sub_fields;
    void                *sub_opts;
    void                *value_ptr;
    size_t              name_len;
    ucs_status_t        status;

    if (!opts || !fields || !name || (!value && (max > 0))) {
        return UCS_ERR_INVALID_PARAM;
    }

    for (field = fields, status = UCS_ERR_NO_ELEM;
         !ucs_config_field_is_last(field) && (status == UCS_ERR_NO_ELEM); ++field) {

        name_len = strlen(field->name);

        ucs_trace("compare name \"%s\" with field \"%s\" which is %s subtable",
                  name, field->name,
                  ucs_config_is_table_field(field) ? "a" : "NOT a");

        if (ucs_config_is_table_field(field) &&
            !strncmp(field->name, name, name_len)) {

            sub_fields = (ucs_config_field_t*)field->parser.arg;
            sub_opts   = (char*)opts + field->offset;
            status     = ucs_config_parser_get_value(sub_opts, sub_fields,
                                                     name + name_len,
                                                     value, max);
        } else if (!strncmp(field->name, name, strlen(name))) {
            if (value) {
                value_ptr = (char *)opts + field->offset;
                field->parser.write(value, max, value_ptr, field->parser.arg);
            }
            status = UCS_OK;
        }
    }

    return status;
}

ucs_status_t ucs_config_parser_clone_opts(const void *src, void *dst,
                                         ucs_config_field_t *fields)
{
    ucs_status_t status;

    ucs_config_field_t *field;
    for (field = fields; !ucs_config_field_is_last(field); ++field) {
        if (ucs_config_is_alias_field(field) ||
            ucs_config_is_deprecated_field(field)) {
            continue;
        }

        status = field->parser.clone((const char*)src + field->offset,
                                    (char*)dst + field->offset,
                                    field->parser.arg);
        if (status != UCS_OK) {
            ucs_error("Failed to clone the filed '%s': %s", field->name,
                      ucs_status_string(status));
            return status;
        }
    }

    return UCS_OK;
}

void ucs_config_parser_release_opts(void *opts, ucs_config_field_t *fields)
{
    ucs_config_field_t *field;

    for (field = fields; !ucs_config_field_is_last(field); ++field) {
        if (ucs_config_is_alias_field(field) ||
            ucs_config_is_deprecated_field(field)) {
            continue;
        }

        ucs_config_parser_release_field(field, (char*)opts + field->offset);
    }
}

/*
 * Finds the "real" field, which the given field is alias of.
 * *p_alias_table_offset is filled with the offset of the sub-table containing
 * the field, it may be non-0 if the alias is found in a sub-table.
 */
static const ucs_config_field_t *
ucs_config_find_aliased_field(const ucs_config_field_t *fields,
                              const ucs_config_field_t *alias,
                              size_t *p_alias_table_offset)
{
    const ucs_config_field_t *field, *result;
    size_t offset;

    for (field = fields; !ucs_config_field_is_last(field); ++field) {
        if (field == alias) {
            /* skip */
            continue;
        } else if (ucs_config_is_table_field(field)) {
            result = ucs_config_find_aliased_field(field->parser.arg, alias,
                                                   &offset);
            if (result != NULL) {
                *p_alias_table_offset = offset + field->offset;
                return result;
            }
        } else if (field->offset == alias->offset) {
            *p_alias_table_offset = 0;
            return field;
        }
    }

    return NULL;
}

static void __print_stream_cb(int num, const char *line, void *arg)
{
    FILE *stream = arg;
    fprintf(stream, "# %s\n", line);
}

static void
ucs_config_parser_print_field(FILE *stream, const void *opts, const char *env_prefix,
                              ucs_list_link_t *prefix_list, const char *name,
                              const ucs_config_field_t *field, unsigned long flags,
                              const char *docstr, ...)
{
    ucs_config_parser_prefix_t *prefix, *head;
    char value_buf[128]  = {0};
    char syntax_buf[256] = {0};
    va_list ap;

    ucs_assert(!ucs_list_is_empty(prefix_list));
    head = ucs_list_head(prefix_list, ucs_config_parser_prefix_t, list);

    if (ucs_config_is_deprecated_field(field)) {
        snprintf(value_buf, sizeof(value_buf), " (deprecated)");
        snprintf(syntax_buf, sizeof(syntax_buf), "N/A");
    } else {
        snprintf(value_buf, sizeof(value_buf), "=");
        field->parser.write(value_buf + 1, sizeof(value_buf) - 2,
                            (char*)opts + field->offset,
                            field->parser.arg);
        field->parser.help(syntax_buf, sizeof(syntax_buf) - 1, field->parser.arg);
    }

    if (flags & UCS_CONFIG_PRINT_DOC) {
        fprintf(stream, "#\n");
        ucs_config_print_doc_line_by_line(field, __print_stream_cb, stream);
        fprintf(stream, "#\n");
        fprintf(stream, "# %-*s %s\n", UCS_CONFIG_PARSER_DOCSTR_WIDTH, "syntax:",
                syntax_buf);

        /* Extra docstring */
        if (docstr != NULL) {
            fprintf(stream, "# ");
            va_start(ap, docstr);
            vfprintf(stream, docstr, ap);
            va_end(ap);
            fprintf(stream, "\n");
        }

        /* Parents in configuration hierarchy */
        if (prefix_list->next != prefix_list->prev) {
            fprintf(stream, "# %-*s", UCS_CONFIG_PARSER_DOCSTR_WIDTH, "inherits:");
            ucs_list_for_each(prefix, prefix_list, list) {
                if (prefix == head) {
                    continue;
                }

                fprintf(stream, " %s%s%s", env_prefix, prefix->prefix, name);
                if (prefix != ucs_list_tail(prefix_list, ucs_config_parser_prefix_t, list)) {
                    fprintf(stream, ",");
                }
            }
            fprintf(stream, "\n");
        }

        fprintf(stream, "#\n");
    }

    fprintf(stream, "%s%s%s%s\n", env_prefix, head->prefix, name, value_buf);

    if (flags & UCS_CONFIG_PRINT_DOC) {
        fprintf(stream, "\n");
    }
}

static void
ucs_config_parser_print_opts_recurs(FILE *stream, const void *opts,
                                    const ucs_config_field_t *fields,
                                    unsigned flags, const char *prefix,
                                    ucs_list_link_t *prefix_list)
{
    const ucs_config_field_t *field, *aliased_field;
    ucs_config_parser_prefix_t *head;
    ucs_config_parser_prefix_t inner_prefix;
    size_t alias_table_offset;

    for (field = fields; !ucs_config_field_is_last(field); ++field) {
        if (ucs_config_is_table_field(field)) {
            /* Parse with sub-table prefix.
             * We start the leaf prefix and continue up the hierarchy.
             */
            /* Do not add the same prefix several times in a sequence. It can
             * happen when similiar prefix names were used during config
             * table inheritance, e.g. "IB_" -> "RC_" -> "RC_". We check the
             * previous entry only, since it is currently impossible if
             * something like "RC_" -> "IB_" -> "RC_" will be used. */
            if (ucs_list_is_empty(prefix_list) ||
                strcmp(ucs_list_tail(prefix_list,
                                     ucs_config_parser_prefix_t,
                                     list)->prefix, field->name)) {
                inner_prefix.prefix = field->name;
                ucs_list_add_tail(prefix_list, &inner_prefix.list);
            } else {
                inner_prefix.prefix = NULL;
            }

            ucs_config_parser_print_opts_recurs(stream,
                                                UCS_PTR_BYTE_OFFSET(opts, field->offset),
                                                field->parser.arg, flags,
                                                prefix, prefix_list);

            if (inner_prefix.prefix != NULL) {
                ucs_list_del(&inner_prefix.list);
            }
        } else if (ucs_config_is_alias_field(field)) {
            if (flags & UCS_CONFIG_PRINT_HIDDEN) {
                aliased_field =
                    ucs_config_find_aliased_field(fields, field,
                                                  &alias_table_offset);
                if (aliased_field == NULL) {
                    ucs_fatal("could not find aliased field of %s", field->name);
                }

                head = ucs_list_head(prefix_list, ucs_config_parser_prefix_t, list);

                ucs_config_parser_print_field(stream,
                                              UCS_PTR_BYTE_OFFSET(opts, alias_table_offset),
                                              prefix, prefix_list,
                                              field->name, aliased_field,
                                              flags, "%-*s %s%s%s",
                                              UCS_CONFIG_PARSER_DOCSTR_WIDTH,
                                              "alias of:", prefix,
                                              head->prefix,
                                              aliased_field->name);
            }
        } else {
            if (ucs_config_is_deprecated_field(field) &&
                !(flags & UCS_CONFIG_PRINT_HIDDEN)) {
                continue;
            }
            ucs_config_parser_print_field(stream, opts, prefix, prefix_list,
                                          field->name, field, flags, NULL);
        }
    }
}

void ucs_config_parser_print_opts(FILE *stream, const char *title, const void *opts,
                                  ucs_config_field_t *fields, const char *table_prefix,
                                  const char *prefix, ucs_config_print_flags_t flags)
{
    ucs_config_parser_prefix_t table_prefix_elem;
    UCS_LIST_HEAD(prefix_list);

    if (flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "\n");
        fprintf(stream, "#\n");
        fprintf(stream, "# %s\n", title);
        fprintf(stream, "#\n");
        fprintf(stream, "\n");
    }

    if (flags & UCS_CONFIG_PRINT_CONFIG) {
        table_prefix_elem.prefix = table_prefix ? table_prefix : "";
        ucs_list_add_tail(&prefix_list, &table_prefix_elem.list);
        ucs_config_parser_print_opts_recurs(stream, opts, fields, flags,
                                            prefix, &prefix_list);
    }

    if (flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "\n");
    }
}

void ucs_config_parser_print_all_opts(FILE *stream, const char *prefix,
                                      ucs_config_print_flags_t flags,
                                      ucs_list_link_t *config_list)
{
    const ucs_config_global_list_entry_t *entry;
    ucs_status_t status;
    char title[64];
    void *opts;

    ucs_list_for_each(entry, config_list, list) {
        if ((entry->table == NULL) ||
            (ucs_config_field_is_last(&entry->table[0]))) {
            /* don't print title for an empty configuration table */
            continue;
        }

        opts = ucs_malloc(entry->size, "tmp_opts");
        if (opts == NULL) {
            ucs_error("could not allocate configuration of size %zu", entry->size);
            continue;
        }

        status = ucs_config_parser_fill_opts(opts, entry->table, prefix,
                                             entry->prefix, 0);
        if (status != UCS_OK) {
            ucs_free(opts);
            continue;
        }

        snprintf(title, sizeof(title), "%s configuration", entry->name);
        ucs_config_parser_print_opts(stream, title, opts, entry->table,
                                     entry->prefix, prefix, flags);

        ucs_config_parser_release_opts(opts, entry->table);
        ucs_free(opts);
    }
}

static void ucs_config_parser_warn_unused_env_vars(const char *prefix)
{
    char unused_env_vars_names[40];
    int num_unused_vars;
    char **envp, *envstr;
    size_t prefix_len;
    char *var_name;
    char *p, *endp;
    khiter_t iter;
    char *saveptr;
    int truncated;
    int ret;

    if (!ucs_global_opts.warn_unused_env_vars) {
        return;
    }

    pthread_mutex_lock(&ucs_config_parser_env_vars_hash_lock);

    prefix_len      = strlen(prefix);
    p               = unused_env_vars_names;
    endp            = p + sizeof(unused_env_vars_names) - 1;
    *endp           = '\0';
    truncated       = 0;
    num_unused_vars = 0;

    for (envp = environ; !truncated && (*envp != NULL); ++envp) {
        envstr = ucs_strdup(*envp, "env_str");
        if (envstr == NULL) {
            continue;
        }

        var_name = strtok_r(envstr, "=", &saveptr);
        if (!var_name || strncmp(var_name, prefix, prefix_len)) {
            ucs_free(envstr);
            continue; /* Not UCX */
        }

        iter = kh_get(ucs_config_env_vars, &ucs_config_parser_env_vars, var_name);
        if (iter == kh_end(&ucs_config_parser_env_vars)) {
            ret = snprintf(p, endp - p, " %s,", var_name);
            if (ret > endp - p) {
                truncated = 1;
                *p = '\0';
            } else {
                p += strlen(p);
                ++num_unused_vars;
            }
        }

        ucs_free(envstr);
    }

    if (num_unused_vars > 0) {
        if (!truncated) {
            p[-1] = '\0'; /* remove trailing comma */
        }
        ucs_warn("unused env variable%s:%s%s (set %s%s=n to suppress this warning)",
                 (num_unused_vars > 1) ? "s" : "", unused_env_vars_names,
                 truncated ? "..." : "", UCS_DEFAULT_ENV_PREFIX,
                 UCS_GLOBAL_OPTS_WARN_UNUSED_CONFIG);
    }

    pthread_mutex_unlock(&ucs_config_parser_env_vars_hash_lock);
}

void ucs_config_parser_warn_unused_env_vars_once(const char *env_prefix)
{
    const char   *sub_prefix = NULL;
    int          added;
    ucs_status_t status;

    /* Although env_prefix is not real environment variable put it
     * into table anyway to save prefixes which was already checked.
     * Need to save both env_prefix and base_prefix */
    ucs_config_parser_mark_env_var_used(env_prefix, &added);
    if (!added) {
        return;
    }

    ucs_config_parser_warn_unused_env_vars(env_prefix);
 
    status = ucs_config_parser_get_sub_prefix(env_prefix, &sub_prefix);
    if (status != UCS_OK) {
        return;
    }

    if (sub_prefix == NULL) {
        return;
    }

    ucs_config_parser_mark_env_var_used(sub_prefix, &added);
    if (!added) {
        return;
    }

    ucs_config_parser_warn_unused_env_vars(sub_prefix);
}

size_t ucs_config_memunits_get(size_t config_size, size_t auto_size,
                               size_t max_size)
{
    if (config_size == UCS_MEMUNITS_AUTO) {
        return auto_size;
    } else {
        return ucs_min(config_size, max_size);
    }
}

int ucs_config_names_search(ucs_config_names_array_t config_names,
                            const char *str)
{
    unsigned i;

    for (i = 0; i < config_names.count; ++i) {
        if (!fnmatch(config_names.names[i], str, 0)) {
           return i;
        }
    }

    return -1;
}

UCS_STATIC_CLEANUP {
    const char *key;

    kh_foreach_key(&ucs_config_parser_env_vars, key, {
        ucs_free((void*)key);
    })
    kh_destroy_inplace(ucs_config_env_vars, &ucs_config_parser_env_vars);
}
