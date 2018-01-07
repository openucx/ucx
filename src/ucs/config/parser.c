/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
#include "parser.h"

#include <ucs/sys/sys.h>
#include <ucs/sys/string.h>
#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/debug/debug.h>
#include <ucs/time/time.h>
#include <fnmatch.h>


typedef UCS_CONFIG_ARRAY_FIELD(void, data) ucs_config_array_field_t;


/* Process environment variables */
extern char **environ;


const char *ucs_async_mode_names[] = {
    [UCS_ASYNC_MODE_SIGNAL] = "signal",
    [UCS_ASYNC_MODE_THREAD] = "thread",
    [UCS_ASYNC_MODE_POLL]   = "poll",
    [UCS_ASYNC_MODE_LAST]   = NULL
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

int ucs_config_sprintf_string(char *buf, size_t max, void *src, const void *arg)
{
    strncpy(buf, *((char**)src), max);
    return 1;
}

ucs_status_t ucs_config_clone_string(void *src, void *dest, const void *arg)
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

ucs_status_t ucs_config_clone_int(void *src, void *dest, const void *arg)
{
    *(int*)dest = *(int*)src;
    return UCS_OK;
}

int ucs_config_sprintf_int(char *buf, size_t max, void *src, const void *arg)
{
    return snprintf(buf, max, "%i", *(unsigned*)src);
}

int ucs_config_sscanf_uint(const char *buf, void *dest, const void *arg)
{
    return sscanf(buf, "%u", (unsigned*)dest);
}

ucs_status_t ucs_config_clone_uint(void *src, void *dest, const void *arg)
{
    *(unsigned*)dest = *(unsigned*)src;
    return UCS_OK;
}

int ucs_config_sprintf_uint(char *buf, size_t max, void *src, const void *arg)
{
    return snprintf(buf, max, "%u", *(unsigned*)src);
}

int ucs_config_sscanf_ulong(const char *buf, void *dest, const void *arg)
{
    return sscanf(buf, "%lu", (unsigned long*)dest);
}

int ucs_config_sprintf_ulong(char *buf, size_t max, void *src, const void *arg)
{
    return snprintf(buf, max, "%lu", *(unsigned long*)src);
}

ucs_status_t ucs_config_clone_ulong(void *src, void *dest, const void *arg)
{
    *(unsigned long*)dest = *(unsigned long*)src;
    return UCS_OK;
}

int ucs_config_sscanf_double(const char *buf, void *dest, const void *arg)
{
    return sscanf(buf, "%lf", (double*)dest);
}

int ucs_config_sprintf_double(char *buf, size_t max, void *src, const void *arg)
{
    return snprintf(buf, max, "%.3f", *(double*)src);
}

ucs_status_t ucs_config_clone_double(void *src, void *dest, const void *arg)
{
    *(double*)dest = *(double*)src;
    return UCS_OK;
}

int ucs_config_sscanf_hex(const char *buf, void *dest, const void *arg)
{
    if (strncasecmp(buf, "0x", 2) == 0) {
        return (sscanf(buf + 2, "%x", (unsigned int*)dest));
    } else {
        return 0;
    }
}

int ucs_config_sprintf_hex(char *buf, size_t max, void *src, const void *arg)
{
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

int ucs_config_sprintf_bool(char *buf, size_t max, void *src, const void *arg)
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
    } else {
        return ucs_config_sscanf_bool(buf, dest, arg);
    }
}

int ucs_config_sprintf_ternary(char *buf, size_t max, void *src, const void *arg)
{
    if (*(int*)src == UCS_TRY) {
        return snprintf(buf, max, "try");
    } else {
        return ucs_config_sprintf_bool(buf, max, src, arg);
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

int ucs_config_sprintf_enum(char *buf, size_t max, void *src, const void *arg)
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
    ptr += strlen(ptr);

    *buf = '[';
}

void ucs_config_help_enum(char *buf, size_t max, const void *arg)
{
    __print_table_values(arg, buf, max);
}

int ucs_config_sscanf_bitmap(const char *buf, void *dest, const void *arg)
{
    char *str = strdup(buf);
    char *p;
    int ret, i;

    ret = 1;
    *((unsigned*)dest) = 0;
    p = strtok(str, ",");
    while (p != NULL) {
        i = __find_string_in_list(p, (const char**)arg);
        if (i < 0) {
            ret = 0;
            break;
        }
        *((unsigned*)dest) |= UCS_BIT(i);
        p = strtok(NULL, ",");
    }

    free(str);
    return ret;
}

int ucs_config_sprintf_bitmap(char *buf, size_t max, void *src, const void *arg)
{
    char * const *table;
    int i, len;

    len = 0;
    for (table = arg, i = 0; *table; ++table, ++i) {
        if (*((unsigned*)src) & UCS_BIT(i)) {
            snprintf(buf + len, max - len, "%s,", *table);
            len = strlen(buf);
        }
    }

    if (len > 0) {
        buf[len - 1] = '\0'; /* remove last ',' */
    } else {
        buf[0] = '\0';
    }
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

int ucs_config_sprintf_bitmask(char *buf, size_t max, void *src, const void *arg)
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

int ucs_config_sprintf_time(char *buf, size_t max, void *src, const void *arg)
{
    snprintf(buf, max, "%.2fus", *(double*)src * UCS_USEC_PER_SEC);
    return 1;
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

int ucs_config_sprintf_signo(char *buf, size_t max, void *src, const void *arg)
{
    return ucs_config_sprintf_enum(buf, max, src, ucs_signal_names);
}

int ucs_config_sscanf_memunits(const char *buf, void *dest, const void *arg)
{
    char units[3];
    int num_fields;
    size_t value;
    size_t bytes;

    /* Special value: infinity */
    if (!strcasecmp(buf, "inf")) {
        *(size_t*)dest = UCS_CONFIG_MEMUNITS_INF;
        return 1;
    }

    /* Special value: auto */
   if (!strcasecmp(buf, "auto")) {
        *(size_t*)dest = UCS_CONFIG_MEMUNITS_AUTO;
        return 1;
    }

    memset(units, 0, sizeof(units));
    num_fields = sscanf(buf, "%ld%c%c", &value, &units[0], &units[1]);
    if (num_fields == 1) {
        bytes = 1;
    } else if (num_fields == 2 || num_fields == 3) {
        if (!strcasecmp(units, "b")) {
            bytes = 1;
        } else if (!strcasecmp(units, "kb") || !strcasecmp(units, "k")) {
            bytes = UCS_KBYTE;
        } else if (!strcasecmp(units, "mb") || !strcasecmp(units, "m")) {
            bytes = UCS_MBYTE;
        } else if (!strcasecmp(units, "gb") || !strcasecmp(units, "g")) {
            bytes = UCS_GBYTE;
        } else if (!strcasecmp(units, "tb") || !strcasecmp(units, "t")) {
            bytes = UCS_TBYTE;
        } else {
            return 0;
        }
    } else {
        return 0;
    }

    *(size_t*)dest = value * bytes;
    return 1;
}

int ucs_config_sprintf_memunits(char *buf, size_t max, void *src, const void *arg)
{
    size_t sz = *(size_t*)src;

    if (sz == UCS_CONFIG_MEMUNITS_INF) {
        snprintf(buf, max, "inf");
    } else if (sz == UCS_CONFIG_MEMUNITS_AUTO) {
        snprintf(buf, max, "auto");
    } else {
        ucs_memunits_to_str(sz, buf, max);
    }
    return 1;
}

int ucs_config_sscanf_ulunits(const char *buf, void *dest, const void *arg)
{
    /* Special value: auto */
    if (!strcasecmp(buf, "auto")) {
        *(size_t*)dest = UCS_CONFIG_ULUNITS_AUTO;
        return 1;
    }

    return ucs_config_sscanf_ulong(buf, dest, arg);
}

int ucs_config_sprintf_ulunits(char *buf, size_t max, void *src, const void *arg)
{
    size_t val = *(size_t*)src;

    if (val == UCS_CONFIG_ULUNITS_AUTO) {
        return snprintf(buf, max, "auto");
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

int ucs_config_sprintf_range_spec(char *buf, size_t max, void *src, const void *arg)
{
    ucs_range_spec_t *range_spec = src;

    if (range_spec->first == range_spec->last) {
        snprintf(buf, max, "%d", range_spec->first);
    } else {
        snprintf(buf, max, "%d-%d", range_spec->first, range_spec->last);
    }

    return 1;
}

ucs_status_t ucs_config_clone_range_spec(void *src, void *dest, const void *arg)
{
    ucs_range_spec_t *src_range_spec  = src;
    ucs_range_spec_t *dest_ragne_spec = dest;

    dest_ragne_spec->first = src_range_spec->first;
    dest_ragne_spec->last = src_range_spec->last;

    return UCS_OK;
}

int ucs_config_sscanf_array(const char *buf, void *dest, const void *arg)
{
    ucs_config_array_field_t *field = dest;
    void *temp_field;
    const ucs_config_array_t *array = arg;
    char *dup, *token, *saveptr;
    int ret;
    unsigned i;

    dup = strdup(buf);
    if (dup == NULL) {
        return 0;
    }

    saveptr = NULL;
    token = strtok_r(dup, ",", &saveptr);
    temp_field = ucs_calloc(UCS_CONFIG_ARRAY_MAX, array->elem_size, "config array");
    i = 0;
    while (token != NULL) {
        ret = array->parser.read(token, (char*)temp_field + i * array->elem_size,
                                 array->parser.arg);
        if (!ret) {
            ucs_free(temp_field);
            free(dup);
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
    free(dup);
    return 1;
}

int ucs_config_sprintf_array(char *buf, size_t max, void *src, const void *arg)
{
    ucs_config_array_field_t *field = src;
    const ucs_config_array_t *array = arg;
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

ucs_status_t ucs_config_clone_array(void *src, void *dest, const void *arg)
{
    ucs_config_array_field_t *dest_array = dest, *src_array = src;
    const ucs_config_array_t *array = arg;
    ucs_status_t status;
    unsigned i;

    dest_array->data = ucs_calloc(src_array->count, array->elem_size,
                                  "config array");
    if (dest_array->data == NULL) {
        return UCS_ERR_NO_MEMORY;
    }

    dest_array->count = src_array->count;
    for (i = 0; i < src_array->count; ++i) {
        status = array->parser.clone((char*)src_array->data  + i * array->elem_size,
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

ucs_status_t ucs_config_clone_table(void *src, void *dst, const void *arg)
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

ucs_status_t
ucs_config_parser_set_default_values(void *opts, ucs_config_field_t *fields)
{
    ucs_config_field_t *field, *sub_fields;
    ucs_status_t status;
    void *var;

    for (field = fields; field->name; ++field) {
        if (ucs_config_is_alias_field(field)) {
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
    ucs_config_field_t *field, *sub_fields;
    size_t prefix_len;
    ucs_status_t status;
    unsigned count;
    void *var;

    prefix_len = (table_prefix == NULL) ? 0 : strlen(table_prefix);

    count = 0;
    for (field = fields; field->name; ++field) {

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
            ucs_config_parser_release_field(field, var);
            status = ucs_config_parser_parse_field(field, value, var);
            if (status != UCS_OK) {
                return status;
            }
            ++count;
        }
    }

    return (count == 0) ? UCS_ERR_NO_ELEM : UCS_OK;
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

    /* Put prefix in the buffer. Later we replace only the variable name part */
    snprintf(buf, sizeof(buf) - 1, "%s%s", prefix, table_prefix ? table_prefix : "");
    prefix_len = strlen(buf);

    /* Parse environment variables */
    for (field = fields; field->name; ++field) {

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
            if (env_value != NULL) {
                ucs_config_parser_release_field(field, var);
                status = ucs_config_parser_parse_field(field, env_value, var);
                if (status != UCS_OK) {
                    /* If set to ignore errors, restore the default value */
                    ucs_status_t tmp_status =
                                    ucs_config_parser_parse_field(field, field->dfl_value, var);
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

ucs_status_t ucs_config_parser_fill_opts(void *opts, ucs_config_field_t *fields,
                                         const char *env_prefix,
                                         const char *table_prefix,
                                         int ignore_errors)
{
    ucs_status_t status;
    char prefix[128];

    /* Set default values */
    status = ucs_config_parser_set_default_values(opts, fields);
    if (status != UCS_OK) {
        goto err;
    }

    /* Apply environment variables */
    status = ucs_config_apply_env_vars(opts, fields, UCS_CONFIG_PREFIX,
                                       table_prefix, 1, ignore_errors);
    if (status != UCS_OK) {
        goto err_free;
    }

    /* Apply environment variables with custom prefix */
    if ((env_prefix != NULL) && (strlen(env_prefix) > 0)) {
        snprintf(prefix, sizeof(prefix), "%s%s_", UCS_CONFIG_PREFIX, env_prefix);
        status = ucs_config_apply_env_vars(opts, fields, prefix, table_prefix,
                                           1, ignore_errors);
        if (status != UCS_OK) {
            goto err_free;
        }
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
         field->name && (status == UCS_ERR_NO_ELEM); ++field) {

        name_len = strlen(field->name);

        ucs_trace("compare name \"%s\" with field \"%s\" which is%s subtable",
                  name, field->name,
                  ucs_config_is_table_field(field) ? "" : " NOT");

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
    for (field = fields; field->name; ++field) {
        if (ucs_config_is_alias_field(field)) {
            continue;
        }

        status = field->parser.clone((char*)src + field->offset,
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

    for (field = fields; field->name; ++field) {
        if (ucs_config_is_alias_field(field)) {
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

    for (field = fields; field->name; ++field) {

        if (field == alias) {
            /* skip */
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
                              const char *prefix, const char *name,
                              const ucs_config_field_t *field,
                              unsigned long flags, const char *docstr, ...)
{
    char value_buf[128] = {0};
    char syntax_buf[256] = {0};
    va_list ap;

    field->parser.write(value_buf, sizeof(value_buf) - 1, (char*)opts + field->offset,
                        field->parser.arg);
    field->parser.help(syntax_buf, sizeof(syntax_buf) - 1, field->parser.arg);

    if (flags & UCS_CONFIG_PRINT_DOC) {
        fprintf(stream, "#\n");
        ucs_config_print_doc_line_by_line(field, __print_stream_cb, stream);
        fprintf(stream, "#\n");
        fprintf(stream, "# Syntax: %s\n", syntax_buf);
        fprintf(stream, "#\n");

        /* Extra docstring */
        if (docstr != NULL) {
            fprintf(stream, "# ");
            va_start(ap, docstr);
            vfprintf(stream, docstr, ap);
            va_end(ap);
            fprintf(stream, "\n");
        }
     }

    fprintf(stream, "%s%s%s=%s\n", env_prefix, prefix, name, value_buf);

    if (flags & UCS_CONFIG_PRINT_DOC) {
        fprintf(stream, "\n");
    }
}

static void
ucs_config_parser_print_opts_recurs(FILE *stream, const void *opts,
                                    const ucs_config_field_t *fields,
                                    unsigned flags, const char *env_prefix,
                                    const char *table_prefix)
{
    const ucs_config_field_t *field, *aliased_field;
    size_t alias_table_offset;
    const char *prefix;

    prefix = table_prefix == NULL ? "" : table_prefix;

    for (field = fields; field->name; ++field) {
        if (ucs_config_is_table_field(field)) {
            /* Parse with sub-table prefix */
            if (table_prefix == NULL) {
                ucs_config_parser_print_opts_recurs(stream, opts + field->offset,
                                                    field->parser.arg, flags,
                                                    env_prefix, field->name);
            } else {
                ucs_config_parser_print_opts_recurs(stream, opts + field->offset,
                                                    field->parser.arg, flags,
                                                    env_prefix, table_prefix);
            }
        } else if (ucs_config_is_alias_field(field)) {
            if (flags & UCS_CONFIG_PRINT_HIDDEN) {
                aliased_field = ucs_config_find_aliased_field(fields, field,
                                                              &alias_table_offset);
                if (aliased_field == NULL) {
                    ucs_fatal("could not find aliased field of %s", field->name);
                }
                ucs_config_parser_print_field(stream,
                                              opts + alias_table_offset,
                                              env_prefix, table_prefix,
                                              field->name, aliased_field,
                                              flags, "(alias of %s%s%s)",
                                              env_prefix, table_prefix,
                                              aliased_field->name);
            }
        } else {
            ucs_config_parser_print_field(stream, opts, env_prefix, prefix,
                                          field->name, field, flags, NULL);
        }
     }

}

void ucs_config_parser_print_opts(FILE *stream, const char *title, const void *opts,
                                  ucs_config_field_t *fields, const char *table_prefix,
                                  ucs_config_print_flags_t flags)
{
    if (flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "\n");
        fprintf(stream, "#\n");
        fprintf(stream, "# %s\n", title);
        fprintf(stream, "#\n");
        fprintf(stream, "\n");
    }

    if (flags & UCS_CONFIG_PRINT_CONFIG) {
        ucs_config_parser_print_opts_recurs(stream, opts, fields, flags,
                                            UCS_CONFIG_PREFIX, table_prefix);
    }

    if (flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "\n");
    }
}

size_t ucs_config_memunits_get(size_t config_size, size_t auto_size,
                               size_t max_size)
{
    if (config_size == UCS_CONFIG_MEMUNITS_AUTO) {
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

