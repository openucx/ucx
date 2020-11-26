/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2019. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CONFIG_PARSER_H
#define UCS_CONFIG_PARSER_H

#include "types.h"

#include <ucs/datastruct/list.h>
#include <ucs/type/status.h>
#include <ucs/sys/stubs.h>

#include <stdio.h>


#define UCS_DEFAULT_ENV_PREFIX "UCX_"
#define UCS_CONFIG_ARRAY_MAX   128

BEGIN_C_DECLS

/** @file parser.h */

/*
 * Configuration varaibles syntax:
 *
 * name: <env_prefix><table_prefix><field_name>
 *
 * - env_prefix:     supplied by user to ucs_config_read_XXX() API
 * - table_prefix:   defined in sub-tables. e.g IB_, UD_, ...
 * - field_name:     field_name as defined in the table. e.g ZCOPY_THRESH
 *
 * Examples of full variable names:
 *   - UCS_CIB_RNDV_THRESH
 *   - UCS_IB_TX_MODERATION
 */

typedef struct ucs_config_parser {
    int                      (*read) (const char *buf, void *dest, const void *arg);
    int                      (*write)(char *buf, size_t max,
                                      const void *src, const void *arg);
    ucs_status_t             (*clone)(const void *src, void *dest, const void *arg);
    void                     (*release)(void *ptr, const void *arg);
    void                     (*help)(char *buf, size_t max, const void *arg);
    const void               *arg;
} ucs_config_parser_t;


typedef struct ucs_config_array {
    size_t                   elem_size;
    ucs_config_parser_t      parser;
} ucs_config_array_t;


typedef struct ucs_config_field {
    const char               *name;
    const char               *dfl_value;
    const char               *doc;
    size_t                   offset;
    ucs_config_parser_t      parser;
} ucs_config_field_t;


typedef struct ucs_ib_port_spec {
    char                     *device_name;
    unsigned                 port_num;
} ucs_ib_port_spec_t;


typedef struct ucs_range_spec {
    unsigned                 first;  /* the first value in the range */
    unsigned                 last;   /* the last value in the range */
} ucs_range_spec_t;


typedef struct ucs_config_global_list_entry {
    const char               *name;    /* configuration table name */
    const char               *prefix;  /* configuration prefix */
    ucs_config_field_t       *table;   /* array of configuration fields */
    size_t                   size;     /* size of config structure */
    ucs_list_link_t          list;     /* entry in global list */
} ucs_config_global_list_entry_t;


typedef struct ucs_config_bw_spec {
    char                     *name;
    double                   bw;
} ucs_config_bw_spec_t;


#define UCS_CONFIG_EMPTY_GLOBAL_LIST_ENTRY \
    { \
        .name        = "", \
        .prefix      = "", \
        .table       = NULL, \
        .size        = 0, \
    }


#define UCS_CONFIG_REGISTER_TABLE_ENTRY(_entry, _list) \
    UCS_STATIC_INIT { \
        ucs_list_add_tail(_list, &(_entry)->list); \
    } \
    \
    UCS_STATIC_CLEANUP { \
        ucs_list_del(&(_entry)->list); \
    }

#define UCS_CONFIG_REGISTER_TABLE(_table, _name, _prefix, _type, _list) \
    static ucs_config_global_list_entry_t _table##_config_entry = { \
        .table  = _table, \
        .name   = _name, \
        .prefix = _prefix, \
        .size   = sizeof(_type) \
    }; \
    UCS_CONFIG_REGISTER_TABLE_ENTRY(&_table##_config_entry, _list);

extern ucs_list_link_t ucs_config_global_list;

/*
 * Parsing and printing different data types
 */

int ucs_config_sscanf_string(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_string(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_string(const void *src, void *dest, const void *arg);
void ucs_config_release_string(void *ptr, const void *arg);

int ucs_config_sscanf_int(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_int(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_int(const void *src, void *dest, const void *arg);

int ucs_config_sscanf_uint(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_uint(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_uint(const void *src, void *dest, const void *arg);

int ucs_config_sscanf_ulong(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_ulong(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_ulong(const void *src, void *dest, const void *arg);

int ucs_config_sscanf_double(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_double(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_double(const void *src, void *dest, const void *arg);

int ucs_config_sscanf_hex(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_hex(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_bool(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bool(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_ternary(const char *buf, void *dest, const void *arg);
int ucs_config_sscanf_ternary_auto(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_ternary_auto(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_on_off(const char *buf, void *dest, const void *arg);

int ucs_config_sscanf_on_off_auto(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_on_off_auto(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_enum(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_enum(char *buf, size_t max, const void *src, const void *arg);
void ucs_config_help_enum(char *buf, size_t max, const void *arg);

int ucs_config_sscanf_bitmap(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bitmap(char *buf, size_t max, const void *src, const void *arg);
void ucs_config_help_bitmap(char *buf, size_t max, const void *arg);

int ucs_config_sscanf_bitmask(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bitmask(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_time(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_time(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_time_units(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_time_units(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_bw(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bw(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_bw_spec(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bw_spec(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_bw_spec(const void *src, void *dest, const void *arg);
void ucs_config_release_bw_spec(void *ptr, const void *arg);

int ucs_config_sscanf_signo(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_signo(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_memunits(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_memunits(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_ulunits(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_ulunits(char *buf, size_t max, const void *src, const void *arg);

int ucs_config_sscanf_range_spec(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_range_spec(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_range_spec(const void *src, void *dest, const void *arg);

int ucs_config_sscanf_array(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_array(char *buf, size_t max, const void *src, const void *arg);
ucs_status_t ucs_config_clone_array(const void *src, void *dest, const void *arg);
void ucs_config_release_array(void *ptr, const void *arg);
void ucs_config_help_array(char *buf, size_t max, const void *arg);

int ucs_config_sscanf_table(const char *buf, void *dest, const void *arg);
ucs_status_t ucs_config_clone_table(const void *src, void *dest, const void *arg);
void ucs_config_release_table(void *ptr, const void *arg);
void ucs_config_help_table(char *buf, size_t max, const void *arg);

ucs_status_t ucs_config_clone_log_comp(const void *src, void *dst, const void *arg);

void ucs_config_release_nop(void *ptr, const void *arg);
void ucs_config_help_generic(char *buf, size_t max, const void *arg);

#define UCS_CONFIG_DEPRECATED_FIELD_OFFSET SIZE_MAX

/* Forward declaration of array. Should be in header file. */
#define UCS_CONFIG_DECLARE_ARRAY(_name) \
    extern ucs_config_array_t ucs_config_array_##_name;

/* Definition of array of specific type. Should be in source file. */
#define UCS_CONFIG_DEFINE_ARRAY(_name, _elem_size, ...) \
    ucs_config_array_t ucs_config_array_##_name = {_elem_size, __VA_ARGS__};

#define UCS_CONFIG_TYPE_STRING     {ucs_config_sscanf_string,    ucs_config_sprintf_string, \
                                    ucs_config_clone_string,     ucs_config_release_string, \
                                    ucs_config_help_generic,     "string"}

#define UCS_CONFIG_TYPE_INT        {ucs_config_sscanf_int,       ucs_config_sprintf_int, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "integer"}

#define UCS_CONFIG_TYPE_UINT       {ucs_config_sscanf_uint,      ucs_config_sprintf_uint, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_generic,     "unsigned integer"}

#define UCS_CONFIG_TYPE_ULONG      {ucs_config_sscanf_ulong,     ucs_config_sprintf_ulong, \
                                    ucs_config_clone_ulong,      ucs_config_release_nop, \
                                    ucs_config_help_generic,     "unsigned long"}

#define UCS_CONFIG_TYPE_ULUNITS    {ucs_config_sscanf_ulunits,   ucs_config_sprintf_ulunits, \
                                    ucs_config_clone_ulong,      ucs_config_release_nop, \
                                    ucs_config_help_generic, \
                                    "unsigned long: <number>, \"inf\", or \"auto\""}

#define UCS_CONFIG_TYPE_DOUBLE     {ucs_config_sscanf_double,    ucs_config_sprintf_double, \
                                    ucs_config_clone_double,     ucs_config_release_nop, \
                                    ucs_config_help_generic,     "floating point number"}

#define UCS_CONFIG_TYPE_HEX        {ucs_config_sscanf_hex,       ucs_config_sprintf_hex, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_generic, \
                                    "hex representation of a number or \"auto\""}

#define UCS_CONFIG_TYPE_BOOL       {ucs_config_sscanf_bool,      ucs_config_sprintf_bool, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "<y|n>"}

#define UCS_CONFIG_TYPE_TERNARY    {ucs_config_sscanf_ternary, ucs_config_sprintf_ternary_auto, \
                                    ucs_config_clone_int,      ucs_config_release_nop, \
                                    ucs_config_help_generic,   "<yes|no|try>"}

#define UCS_CONFIG_TYPE_TERNARY_AUTO {ucs_config_sscanf_ternary_auto, ucs_config_sprintf_ternary_auto, \
                                      ucs_config_clone_int,           ucs_config_release_nop, \
                                      ucs_config_help_generic,        "<yes|no|try|auto>"}

#define UCS_CONFIG_TYPE_ON_OFF     {ucs_config_sscanf_on_off,    ucs_config_sprintf_on_off_auto, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "<on|off>"}

#define UCS_CONFIG_TYPE_ON_OFF_AUTO {ucs_config_sscanf_on_off_auto, ucs_config_sprintf_on_off_auto, \
                                     ucs_config_clone_int,          ucs_config_release_nop, \
                                     ucs_config_help_generic,       "<on|off|auto>"}

#define UCS_CONFIG_TYPE_ENUM(t)    {ucs_config_sscanf_enum,      ucs_config_sprintf_enum, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_enum,        t}

#define UCS_CONFIG_TYPE_BITMAP(t)  {ucs_config_sscanf_bitmap,    ucs_config_sprintf_bitmap, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_bitmap,      t}

#define UCS_CONFIG_TYPE_BITMASK    {ucs_config_sscanf_bitmask,   ucs_config_sprintf_bitmask, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_generic,     "bit count"}

#define UCS_CONFIG_TYPE_TIME       {ucs_config_sscanf_time,      ucs_config_sprintf_time, \
                                    ucs_config_clone_double,     ucs_config_release_nop, \
                                    ucs_config_help_generic,     "time value: <number>[s|us|ms|ns]"}

#define UCS_CONFIG_TYPE_TIME_UNITS {ucs_config_sscanf_time_units, ucs_config_sprintf_time_units, \
                                    ucs_config_clone_ulong,       ucs_config_release_nop, \
                                    ucs_config_help_generic, \
                                    "time value: <number>[s|us|ms|ns], \"inf\", or \"auto\""}

#define UCS_CONFIG_TYPE_BW         {ucs_config_sscanf_bw,        ucs_config_sprintf_bw, \
                                    ucs_config_clone_double,     ucs_config_release_nop, \
                                    ucs_config_help_generic,     \
                                    "bandwidth value: <number>[T|G|M|K]B|b[[p|/]s] or \"auto\""}

#define UCS_CONFIG_TYPE_BW_SPEC    {ucs_config_sscanf_bw_spec,   ucs_config_sprintf_bw_spec, \
                                    ucs_config_clone_bw_spec,    ucs_config_release_bw_spec, \
                                    ucs_config_help_generic,     \
                                    "device_name:<number>[T|G|M|K]B|b[[p|/]s] or device_name:auto"}

#define UCS_CONFIG_TYPE_LOG_COMP   {ucs_config_sscanf_enum,      ucs_config_sprintf_enum, \
                                    ucs_config_clone_log_comp,   ucs_config_release_nop, \
                                    ucs_config_help_enum,        ucs_log_level_names}

#define UCS_CONFIG_TYPE_SIGNO      {ucs_config_sscanf_signo,     ucs_config_sprintf_signo, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "system signal (number or SIGxxx)"}

#define UCS_CONFIG_TYPE_MEMUNITS   {ucs_config_sscanf_memunits,  ucs_config_sprintf_memunits, \
                                    ucs_config_clone_ulong,      ucs_config_release_nop, \
                                    ucs_config_help_generic,     \
                                    "memory units: <number>[b|kb|mb|gb], \"inf\", or \"auto\""}

#define UCS_CONFIG_TYPE_ARRAY(a)   {ucs_config_sscanf_array,     ucs_config_sprintf_array, \
                                    ucs_config_clone_array,      ucs_config_release_array, \
                                    ucs_config_help_array,       &ucs_config_array_##a}

#define UCS_CONFIG_TYPE_TABLE(t)   {ucs_config_sscanf_table,     NULL, \
                                    ucs_config_clone_table,      ucs_config_release_table, \
                                    ucs_config_help_table,       t}

#define UCS_CONFIG_TYPE_RANGE_SPEC {ucs_config_sscanf_range_spec,ucs_config_sprintf_range_spec, \
                                    ucs_config_clone_range_spec, ucs_config_release_nop, \
                                    ucs_config_help_generic,     "numbers range: <number>-<number>"}

#define UCS_CONFIG_TYPE_DEPRECATED {(ucs_field_type(ucs_config_parser_t, read))   ucs_empty_function_do_assert, \
                                    (ucs_field_type(ucs_config_parser_t, write))  ucs_empty_function_do_assert, \
                                    (ucs_field_type(ucs_config_parser_t, clone))  ucs_empty_function_do_assert, \
                                    (ucs_field_type(ucs_config_parser_t, release))ucs_empty_function_do_assert, \
                                    (ucs_field_type(ucs_config_parser_t, help))   ucs_empty_function_do_assert, \
                                    ""}

/**
 * Helpers for using an array of strings
 */
#define UCS_CONFIG_TYPE_STRING_ARRAY \
    UCS_CONFIG_TYPE_ARRAY(string)

UCS_CONFIG_DECLARE_ARRAY(string)


/**
 * Helpers for Bandwidth units (see UCS_CONFIG_TYPE_BW)
 */
#define UCS_CONFIG_BW_AUTO            ((double)-2)
#define UCS_CONFIG_BW_IS_AUTO(_value) ((ssize_t)(_value) == UCS_CONFIG_BW_AUTO)


/**
 * Set default values for options.
 *
 * @param opts   User-defined options structure to fill.
 * @param fields Array of fields which define how to parse.
 */
ucs_status_t
ucs_config_parser_set_default_values(void *opts, ucs_config_field_t *fields);


/**
 * Fill existing opts structure.
 *
 * @param opts           User-defined options structure to fill.
 * @param fields         Array of fields which define how to parse.
 * @param env_prefix     Prefix to add to all environment variables,
 *                       env_prefix may consist of multiple sub preifxes
 * @param table_prefix   Optional prefix to add to the variables of top-level table.
 * @param ignore_errors  Whether to ignore parsing errors and continue parsing
 *                       other fields.
 */
ucs_status_t ucs_config_parser_fill_opts(void *opts, ucs_config_field_t *fields,
                                         const char *env_prefix,
                                         const char *table_prefix,
                                         int ignore_errors);

/**
 * Perform deep copy of the options structure.
 *
 * @param src    User-defined options structure to copy from.
 * @param dst    User-defined options structure to copy to.
 * @param table  Array of fields which define the structure of the options.
 */
ucs_status_t ucs_config_parser_clone_opts(const void *src, void *dst,
                                          ucs_config_field_t *fields);

/**
 * Release the options fields.
 * NOTE: Does not release the structure itself.
 *
 * @param opts   User-defined options structure.
 * @param table  Array of fields which define the options.
 */
void ucs_config_parser_release_opts(void *opts, ucs_config_field_t *fields);

/**
 * Print the options - names, values, documentation.
 *
 * @param stream         Output stream to print to.
 * @param opts           User-defined options structure.
 * @param fields         Array of fields which define the options.
 * @param table_prefix   Optional prefix to add to the variables of top-level table.
 * @param prefix         Prefix to add to all environment variables.
 * @param flags          Flags which control the output.
 */
void ucs_config_parser_print_opts(FILE *stream, const char *title, const void *opts,
                                  ucs_config_field_t *fields, const char *table_prefix,
                                  const char *prefix, ucs_config_print_flags_t flags);

/**
 * Print all options defined in the library - names, values, documentation.
 *
 * @param stream         Output stream to print to.
 * @param prefix         Prefix to add to all environment variables.
 * @param flags          Flags which control the output.
 * @param config_list    List of config tables
 */
void ucs_config_parser_print_all_opts(FILE *stream, const char *prefix,
                                      ucs_config_print_flags_t flags,
                                      ucs_list_link_t *config_list);

/**
 * Read a value from options structure.
 *
 * @param opts       User-defined options structure.
 * @param fields     Array of fields which define how to parse.
 * @param name       Option name including subtable prefixes.
 * @param value      Filled with option value (as a string).
 * @param max        Number of bytes reserved in 'value'.
 */
ucs_status_t ucs_config_parser_get_value(void *opts, ucs_config_field_t *fields,
                                        const char *name, char *value, size_t max);

/**
 * Modify existing opts structure with new setting.
 *
 * @param opts       User-defined options structure.
 * @param fields     Array of fields which define how to parse.
 * @param name       Option name to modify.
 * @param value      Value to assign.
 */
ucs_status_t ucs_config_parser_set_value(void *opts, ucs_config_field_t *fields,
                                         const char *name, const char *value);

/**
 * Wrapper for `ucs_config_parser_warn_unused_env_vars`
 * that ensures that this is called once
 *
 * @param env_prefix     Environment variable prefix.
 *                       env_prefix may consist of multiple sub prefixex
 */

void ucs_config_parser_warn_unused_env_vars_once(const char *env_prefix);

/**
 * Translate configuration value of "MEMUNITS" type to actual value.
 *
 * @param config_size  Size specified by configuration.
 * @param auto_size    Default size when configured to 'auto'.
 * @param max_size     Maximal size to trim "inf".
 */
size_t ucs_config_memunits_get(size_t config_size, size_t auto_size,
                               size_t max_size);

/**
 * Look for a string in config names array.
 *
 * @param config_names     lookup array of counters patterns.
 * @param str              string to search.
 */
int ucs_config_names_search(ucs_config_names_array_t config_names,
                            const char *str);

END_C_DECLS

#endif
