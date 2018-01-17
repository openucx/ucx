/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_CONFIG_PARSER_H
#define UCS_CONFIG_PARSER_H

#include "types.h"

#include <ucs/type/status.h>
#include <ucs/sys/compiler_def.h>
#include <stdio.h>


#define UCS_CONFIG_PREFIX      "UCX_"
#define UCS_CONFIG_ARRAY_MAX   128

BEGIN_C_DECLS

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
    int              (*read) (const char *buf, void *dest, const void *arg);
    int              (*write)(char *buf, size_t max, void *src, const void *arg);
    ucs_status_t     (*clone)(void *src, void *dest, const void *arg);
    void             (*release)(void *ptr, const void *arg);
    void             (*help)(char *buf, size_t max, const void *arg);
    const void       *arg;
} ucs_config_parser_t;


typedef struct ucs_config_array {
    size_t              elem_size;
    ucs_config_parser_t parser;
} ucs_config_array_t;


typedef struct ucs_config_field {
    const char           *name;
    const char           *dfl_value;
    const char           *doc;
    size_t               offset;
    ucs_config_parser_t  parser;
} ucs_config_field_t;


typedef struct ucs_ib_port_spec {
    char                     *device_name;
    unsigned                 port_num;
} ucs_ib_port_spec_t;


typedef struct ucs_range_spec {
    unsigned    first;  /* the first value in the range */
    unsigned    last;   /* the last value in the range */
} ucs_range_spec_t;


/*
 * Parsing and printing different data types
 */

int ucs_config_sscanf_string(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_string(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_string(void *src, void *dest, const void *arg);
void ucs_config_release_string(void *ptr, const void *arg);

int ucs_config_sscanf_int(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_int(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_int(void *src, void *dest, const void *arg);

int ucs_config_sscanf_uint(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_uint(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_uint(void *src, void *dest, const void *arg);

int ucs_config_sscanf_ulong(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_ulong(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_ulong(void *src, void *dest, const void *arg);

int ucs_config_sscanf_double(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_double(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_double(void *src, void *dest, const void *arg);

int ucs_config_sscanf_hex(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_hex(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_bool(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bool(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_ternary(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_ternary(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_enum(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_enum(char *buf, size_t max, void *src, const void *arg);
void ucs_config_help_enum(char *buf, size_t max, const void *arg);

int ucs_config_sscanf_bitmap(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bitmap(char *buf, size_t max, void *src, const void *arg);
void ucs_config_help_bitmap(char *buf, size_t max, const void *arg);

int ucs_config_sscanf_bitmask(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_bitmask(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_time(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_time(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_signo(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_signo(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_memunits(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_memunits(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_ulunits(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_ulunits(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_range_spec(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_range_spec(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_range_spec(void *src, void *dest, const void *arg);

int ucs_config_sscanf_array(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_array(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_array(void *src, void *dest, const void *arg);
void ucs_config_release_array(void *ptr, const void *arg);
void ucs_config_help_array(char *buf, size_t max, const void *arg);

int ucs_config_sscanf_table(const char *buf, void *dest, const void *arg);
ucs_status_t ucs_config_clone_table(void *src, void *dest, const void *arg);
void ucs_config_release_table(void *ptr, const void *arg);
void ucs_config_help_table(char *buf, size_t max, const void *arg);

void ucs_config_release_nop(void *ptr, const void *arg);
void ucs_config_help_generic(char *buf, size_t max, const void *arg);


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
                                    "unsigned long: <number> or \"auto\""}

#define UCS_CONFIG_TYPE_DOUBLE     {ucs_config_sscanf_double,    ucs_config_sprintf_double, \
                                    ucs_config_clone_double,     ucs_config_release_nop, \
                                    ucs_config_help_generic,     "floating point number"}

#define UCS_CONFIG_TYPE_HEX        {ucs_config_sscanf_hex,       ucs_config_sprintf_hex, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_generic,     "hex representation of a number"}

#define UCS_CONFIG_TYPE_BOOL       {ucs_config_sscanf_bool,      ucs_config_sprintf_bool, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "<y|n>"}

#define UCS_CONFIG_TYPE_TERNARY    {ucs_config_sscanf_ternary,   ucs_config_sprintf_ternary, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "<yes|no|try>"}

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

/*
 * Helpers for using an array of strings.
 */
#define UCS_CONFIG_TYPE_STRING_ARRAY \
    UCS_CONFIG_TYPE_ARRAY(string)

UCS_CONFIG_DECLARE_ARRAY(string);

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
 * @param env_prefix     Prefix to add to all environment variables.
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
 * @param flags          Flags which control the output.
 */
void ucs_config_parser_print_opts(FILE *stream, const char *title, const void *opts,
                                  ucs_config_field_t *fields, const char *table_prefix,
                                  ucs_config_print_flags_t flags);

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
