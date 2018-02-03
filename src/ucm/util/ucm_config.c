/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucm_config.h"

#include <ucm/util/log.h>
#include <ucs/config/parser.h>
#include <ucs/type/component.h>
#include <ucs/sys/checker.h>
#include <string.h>
#include <stdlib.h>

#define UCM_ENV_PREFIX           UCS_CONFIG_PREFIX "MEM_"

#define UCM_LOG_LEVEL_VAR        "LOG_LEVEL"
#define UCM_ALLOC_ALIGN_VAR      "ALLOC_ALIGN"
#define UCM_EN_EVENTS_VAR        "EVENTS"
#define UCM_EN_MMAP_RELOC_VAR    "MMAP_RELOC"
#define UCM_EN_MALLOC_HOOKS_VAR  "MALLOC_HOOKS"
#define UCM_EN_MALLOC_RELOC_VAR  "MALLOC_RELOC"
#define UCM_EN_DYNAMIC_MMAP_VAR  "DYNAMIC_MMAP_THRESH"
#define UCM_EN_CUDA_HOOKS_VAR    "CUDA_HOOKS"


ucm_config_t ucm_global_config = {
    .log_level                  = UCS_LOG_LEVEL_WARN,
    .alloc_alignment            = 16,
    .enable_events              = 1,
    .enable_mmap_reloc          = 1,
    .enable_malloc_hooks        = 1,
    .enable_malloc_reloc        = 0,
    .enable_dynamic_mmap_thresh = 1,
#if HAVE_CUDA
    .enable_cuda_hooks          = 1
#endif
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

    ucm_config_print_doc(stream,
                         "Logging level", "<fatal|error|warn|info|debug|trace>",
                         print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_LOG_LEVEL_VAR,
            ucm_log_level_names[ucm_global_config.log_level]);

    ucm_config_print_doc(stream,
                         "Minimal alignment of allocated blocks",
                         "long integer", print_flags);
    fprintf(stream, "%s%s=%zu\n", UCM_ENV_PREFIX, UCM_ALLOC_ALIGN_VAR,
            ucm_global_config.alloc_alignment);

    ucm_config_print_bool_doc(stream,
                              "Enable memory events",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_EVENTS_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_events));

    ucm_config_print_bool_doc(stream,
                              "Enable installing mmap symbols in the relocation table",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_MMAP_RELOC_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_mmap_reloc));

    ucm_config_print_bool_doc(stream,
                              "Enable using glibc malloc hooks",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_MALLOC_HOOKS_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_malloc_hooks));

    ucm_config_print_bool_doc(stream,
                              "Enable installing malloc symbols in the relocation table.\n"
                              "This is unsafe and off by default, because sometimes glibc\n"
                              "calls malloc/free without going through the relocation table,\n"
                              "which would use the original implementation and not ours.",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_MALLOC_RELOC_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_malloc_reloc));


    ucm_config_print_bool_doc(stream,
                              "Enable dynamic mmap threshold: for every released block, the\n"
                              "mmap threshold is adjusted upward to the size of the size of\n"
                              "the block, and trim threshold is adjust to twice the size of\n"
                              "the dynamic mmap threshold.",
                              print_flags);
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_DYNAMIC_MMAP_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_dynamic_mmap_thresh));


#if HAVE_CUDA
    fprintf(stream, "%s%s=%s\n", UCM_ENV_PREFIX, UCM_EN_CUDA_HOOKS_VAR,
            ucm_config_bool_to_string(ucm_global_config.enable_cuda_hooks));
#endif
}

static void ucm_config_set_value_table(const char *str_value, const char **table,
                                       int *value)
{
    int i;

    for (i = 0; table[i] != NULL; ++i) {
        if (!strcasecmp(table[i], str_value)) {
            ucm_global_config.log_level = i;
            return;
        }
    }
}

static void ucm_config_set_value_bool(const char *str_value, int *value)
{
    if (!strcasecmp(str_value, "1") || !strcasecmp(str_value, "y") || !strcasecmp(str_value, "yes")) {
        *value = 1;
    } else if (!strcasecmp(str_value, "0") || !strcasecmp(str_value, "n") || !strcasecmp(str_value, "no")) {
        *value = 0;
    }
}

static void ucm_config_set_value_size(const char *str_value, size_t *value)
{
    char *endptr;
    size_t n;

    n = strtoul(str_value, &endptr, 10);
    if (*endptr == '\0') {
        *value = n;
    }
}

ucs_status_t ucm_config_modify(const char *name, const char *value)
{
    if (!strcmp(name, UCM_LOG_LEVEL_VAR)) {
        ucm_config_set_value_table(value, ucm_log_level_names,
                                   (int*)&ucm_global_config.log_level);
    } else if (!strcmp(name, UCM_ALLOC_ALIGN_VAR)) {
        ucm_config_set_value_size(value, &ucm_global_config.alloc_alignment);
    } else if (!strcmp(name, UCM_EN_EVENTS_VAR)) {
        ucm_config_set_value_bool(value, &ucm_global_config.enable_events);
    } else if (!strcmp(name, UCM_EN_MMAP_RELOC_VAR)) {
        ucm_config_set_value_bool(value, &ucm_global_config.enable_mmap_reloc);
    } else if (!strcmp(name, UCM_EN_MALLOC_HOOKS_VAR)) {
        ucm_config_set_value_bool(value, &ucm_global_config.enable_malloc_hooks);
    } else if (!strcmp(name, UCM_EN_MALLOC_RELOC_VAR)) {
        ucm_config_set_value_bool(value, &ucm_global_config.enable_malloc_reloc);
    } else if (!strcmp(name, UCM_EN_DYNAMIC_MMAP_VAR)) {
        ucm_config_set_value_bool(value, &ucm_global_config.enable_dynamic_mmap_thresh);
#if HAVE_CUDA
    } else if (!strcmp(name, UCM_EN_CUDA_HOOKS_VAR)) {
        ucm_config_set_value_bool(value, &ucm_global_config.enable_cuda_hooks);
#endif
    } else {
        return UCS_ERR_INVALID_PARAM;
    }

    return UCS_OK;
}

static void ucm_config_set(const char *name)
{
    char var_name[64];
    char *str_value;

    snprintf(var_name, sizeof(var_name), "%s%s", UCM_ENV_PREFIX, name);
    str_value = getenv(var_name);
    if (str_value != NULL) {
        ucm_config_modify(name, str_value);
    }
}

UCS_STATIC_INIT {
    if (RUNNING_ON_VALGRIND) {
        /* Valgrind limits the size of brk() segments to 8mb, so must use mmap
         * for large allocations.
         */
        ucm_global_config.enable_dynamic_mmap_thresh = 0;
    }
    ucm_config_set(UCM_LOG_LEVEL_VAR);
    ucm_config_set(UCM_ALLOC_ALIGN_VAR);
    ucm_config_set(UCM_EN_EVENTS_VAR);
    ucm_config_set(UCM_EN_MMAP_RELOC_VAR);
    ucm_config_set(UCM_EN_MALLOC_HOOKS_VAR);
    ucm_config_set(UCM_EN_MALLOC_RELOC_VAR);
    ucm_config_set(UCM_EN_DYNAMIC_MMAP_VAR);
}
