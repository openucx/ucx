/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucm_config.h"

#include <ucm/util/log.h>
#include <ucs/config/parser.h>
#include <ucs/type/component.h>
#include <string.h>
#include <stdlib.h>

#define UCM_ENV_PREFIX           UCS_CONFIG_PREFIX "MEM_"
#define UCM_LOG_LEVEL_VAR        "LOG_LEVEL"
#define UCM_ALLOC_ALIGN_VAR      "ALLOC_ALIGN"
#define UCM_EN_EVENTS_VAR        "EVENTS"
#define UCM_EN_MALLOC_HOOKS_VAR  "MALLOC_HOOKS"
#define UCM_EN_RELOC_HOOKS_VAR   "RELOC_HOOKS"


ucm_config_t ucm_global_config = {
    .log_level            = UCS_LOG_LEVEL_WARN,
    .alloc_alignment      = 16,
    .enable_events        = 1,
    .enable_malloc_hooks  = 1,
    .enable_reloc_hooks   = 1
};

static const char *ucm_config_bool_to_string(int value)
{
    return value ? "yes" : "no";
}

static void ucm_config_print_doc(FILE *stream, const char *doc, const char *syntax,
                                      ucs_config_print_flags_t print_flags)
{
    if (!(print_flags & UCS_CONFIG_PRINT_DOC)) {
        return;
    }

    fprintf(stream, "\n");
    fprintf(stream, "#\n");
    fprintf(stream, "# %s\n", doc);
    fprintf(stream, "#\n");
    fprintf(stream, "# Syntax: %s\n", syntax);
    fprintf(stream, "#\n");
}

static void ucm_config_print_bool_doc(FILE *stream, const char *doc,
                                      ucs_config_print_flags_t print_flags)
{
    ucm_config_print_doc(stream, doc, "<yes|no>", print_flags);
}

void ucm_config_print(FILE *stream, ucs_config_print_flags_t print_flags)
{
    if (print_flags & UCS_CONFIG_PRINT_HEADER) {
        fprintf(stream, "#\n");
        fprintf(stream, "# UCM configuration\n");
        fprintf(stream, "#\n");
    }

    if (!(print_flags & UCS_CONFIG_PRINT_CONFIG)) {
        return;
    }

    ucm_config_print_doc(stream, "Logging level", "<fatal|error|warn|info|debug|trace>",
                         print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_LOG_LEVEL_VAR,
            ucm_log_level_names[ucm_global_config.log_level]);

    ucm_config_print_doc(stream, "Minimal alignment of allocated blocks",
                         "long integer", print_flags);
    fprintf(stream, "%s%s=%zu\n", UCM_ENV_PREFIX, UCM_ALLOC_ALIGN_VAR,
            ucm_global_config.alloc_alignment);

    ucm_config_print_bool_doc(stream, "Enable memory events", print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_EVENTS_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_events));

    ucm_config_print_bool_doc(stream, "Enable installing malloc hooks",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_MALLOC_HOOKS_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_malloc_hooks));

    ucm_config_print_bool_doc(stream, "Enable modifying the relocation table",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_RELOC_HOOKS_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_reloc_hooks));
}

static char *ucm_config_getenv(const char *name)
{
    char var_name[64];
    snprintf(var_name, sizeof(var_name), "%s%s", UCM_ENV_PREFIX, name);
    return getenv(var_name);
}

static void ucm_config_set_value_table(const char *name, const char **table,
                                       int *value)
{
    char *str_value;
    int i;

    str_value = ucm_config_getenv(name);
    if (str_value == NULL) {
        return;
    }

    for (i = 0; table[i] != NULL; ++i) {
        if (!strcasecmp(table[i], str_value)) {
            ucm_global_config.log_level = i;
            return;
        }
    }
}

static void ucm_config_set_value_bool(const char *name, int *value)
{
    char *str_value;

    str_value = ucm_config_getenv(name);
    if (str_value == NULL) {
        return;
    }

    if (!strcasecmp(str_value, "1") || !strcasecmp(str_value, "y") || !strcasecmp(str_value, "yes")) {
        *value = 1;
    } else if (!strcasecmp(str_value, "0") || !strcasecmp(str_value, "n") || !strcasecmp(str_value, "no")) {
        *value = 0;
    }
}

static void ucm_config_set_value_size(const char *name, size_t *value)
{
    char *str_value, *endptr;
    size_t n;

    str_value = ucm_config_getenv(name);
    if (str_value == NULL) {
        return;
    }

    n = strtoul(str_value, &endptr, 10);
    if (*endptr == '\0') {
        *value = n;
    }
}

UCS_STATIC_INIT {
    ucm_config_set_value_table(UCM_LOG_LEVEL_VAR, ucm_log_level_names,
                               (int*)&ucm_global_config.log_level);
    ucm_config_set_value_size(UCM_ALLOC_ALIGN_VAR,
                              &ucm_global_config.alloc_alignment);
    ucm_config_set_value_bool(UCM_EN_EVENTS_VAR,
                              &ucm_global_config.enable_events);
    ucm_config_set_value_bool(UCM_EN_MALLOC_HOOKS_VAR,
                              &ucm_global_config.enable_malloc_hooks);
    ucm_config_set_value_bool(UCM_EN_RELOC_HOOKS_VAR,
                              &ucm_global_config.enable_reloc_hooks);
}
