/*
* Copyright (C) Mellanox Technologies Ltd. 2001-2014.  ALL RIGHTS RESERVED.
*
* $COPYRIGHT$
* $HEADER$
*/

#ifndef UCS_CONFIG_PARSER_H
#define UCS_CONFIG_PARSER_H

#include "types.h"

#include <ucs/type/status.h>
#include <ucs/sys/math.h>
#include <stdio.h>


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

/* Special values for IB port parser */
#define UCS_IB_CFG_DEVICE_NAME_ANY  ((char *)0xfe)
#define UCS_IB_CFG_DEVICE_NAME_ALL  ((char *)0xff)
#define UCS_IB_CFG_PORT_NUM_ANY     0xfffe
#define UCS_IB_CFG_PORT_NUM_ALL     0xffff


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

int ucs_config_sscanf_port_spec(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_port_spec(char *buf, size_t max, void *src, const void *arg);
ucs_status_t ucs_config_clone_port_spec(void *src, void *dest, const void *arg);
void ucs_config_release_port_spec(void *ptr, const void *arg);

int ucs_config_sscanf_time(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_time(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_signo(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_signo(char *buf, size_t max, void *src, const void *arg);

int ucs_config_sscanf_memunits(const char *buf, void *dest, const void *arg);
int ucs_config_sprintf_memunits(char *buf, size_t max, void *src, const void *arg);

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


#define UCS_CONFIG_DEFINE_ARRAY(_name, _elem_size, ...) \
    ucs_config_array_t ucs_array_##_name = {_elem_size, __VA_ARGS__};

#define UCS_CONFIG_TYPE_STRING     {ucs_config_sscanf_string,    ucs_config_sprintf_string, \
                                    ucs_config_clone_string,     ucs_config_release_string, \
                                    ucs_config_help_generic,     "string"}

#define UCS_CONFIG_TYPE_INT        {ucs_config_sscanf_int,       ucs_config_sprintf_int, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "integer"}

#define UCS_CONFIG_TYPE_UINT       {ucs_config_sscanf_uint,      ucs_config_sprintf_uint, \
                                    ucs_config_clone_uint,       ucs_config_release_nop, \
                                    ucs_config_help_generic,     "unsigned"}

#define UCS_CONFIG_TYPE_ULONG      {ucs_config_sscanf_ulong,     ucs_config_sprintf_ulong, \
                                    ucs_config_clone_ulong,      ucs_config_release_nop, \
                                    ucs_config_help_generic,     "unsigned long"}

#define UCS_CONFIG_TYPE_DOUBLE     {ucs_config_sscanf_double,    ucs_config_sprintf_double, \
                                    ucs_config_clone_double,     ucs_config_release_nop, \
                                    ucs_config_help_generic,     "floating point number"}

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

#define UCS_CONFIG_TYPE_PORT_SPEC  {ucs_config_sscanf_port_spec, ucs_config_sprintf_port_spec, \
                                    ucs_config_clone_port_spec,  ucs_config_release_port_spec, \
                                    ucs_config_help_generic,     "IB port: <device>:<port_num>"}

#define UCS_CONFIG_TYPE_TIME       {ucs_config_sscanf_time,      ucs_config_sprintf_time, \
                                    ucs_config_clone_double,     ucs_config_release_nop, \
                                    ucs_config_help_generic,     "time value: <number>[s|us|ms|ns]"}

#define UCS_CONFIG_TYPE_SIGNO      {ucs_config_sscanf_signo,     ucs_config_sprintf_signo, \
                                    ucs_config_clone_int,        ucs_config_release_nop, \
                                    ucs_config_help_generic,     "system signal (number or SIGxxx)"}

#define UCS_CONFIG_TYPE_MEMUNITS   {ucs_config_sscanf_memunits,  ucs_config_sprintf_memunits, \
                                    ucs_config_clone_ulong,      ucs_config_release_nop, \
                                    ucs_config_help_generic,     "memory units: <number>[b|kb|mb|gb], or \"inf\""}

#define UCS_CONFIG_TYPE_ARRAY(a)   {ucs_config_sscanf_array,     ucs_config_sprintf_array, \
                                    ucs_config_clone_array,      ucs_config_release_array, \
                                    ucs_config_help_array,       &ucs_array_##a}

#define UCS_CONFIG_TYPE_TABLE(t)   {ucs_config_sscanf_table,     NULL, \
                                    ucs_config_clone_table,      ucs_config_release_table, \
                                    ucs_config_help_table,       t}


/**
 * Fill existing opts structure.
 */
ucs_status_t ucs_config_parser_fill_opts(void *opts, ucs_config_field_t *table,
                                         const char *user_prefix,
                                         const char *table_prefix);

/**
 * Perform deep copy of the options structure.
 */
ucs_status_t ucs_config_parser_clone_opts(void *src, void *dst,
                                         ucs_config_field_t *fields);

/**
 * Release the options fields.
 * NOTE: Does not release the structure itself.
 */
void ucs_config_parser_release_opts(void *opts, ucs_config_field_t *fields);

/**
 * Print the options.
 */
void ucs_config_parser_print_opts(FILE *stream, const char *title, void *opts,
                                  ucs_config_field_t *fields, const char *env_prefix,
                                  const char *table_prefix, ucs_config_print_flags_t flags);

/**
 * Read existing opts structure.
 */
ucs_status_t ucs_config_parser_get_value(void *opts, ucs_config_field_t *fields,
                                        const char *name, char *value, size_t max);

/**
 * Modify existing opts structure with new setting.
 */
ucs_status_t ucs_config_parser_set_value(void *opts, ucs_config_field_t *fields,
                                        const char *name, const char *value);

#endif
