/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "parser.h"

#include <ucm/api/ucm.h>
#include <ucm/util/log.h>
#include <ucm/mmap/mmap.h>
#include <ucs/sys/compiler.h>


#define UCM_CONFIG_PREFIX   "MEM_"

static const char *ucm_mmap_hook_modes[] = {
    [UCM_MMAP_HOOK_NONE]   = "none",
    [UCM_MMAP_HOOK_RELOC]  = UCM_MMAP_HOOK_RELOC_STR,
#if UCM_BISTRO_HOOKS
    [UCM_MMAP_HOOK_BISTRO] = UCM_MMAP_HOOK_BISTRO_STR,
#endif
    [UCM_MMAP_HOOK_LAST]   = NULL
};

static const char *ucm_module_unload_prevent_modes[] = {
    [UCM_UNLOAD_PREVENT_MODE_LAZY] = "lazy",
    [UCM_UNLOAD_PREVENT_MODE_NOW]  = "now",
    [UCM_UNLOAD_PREVENT_MODE_NONE] = "none",
    [UCM_UNLOAD_PREVENT_MODE_LAST] = NULL
};

static ucs_config_field_t ucm_global_config_table[] = {
  {"LOG_LEVEL", "warn",
   "Logging level for memory events", ucs_offsetof(ucm_global_config_t, log_level),
   UCS_CONFIG_TYPE_ENUM(ucm_log_level_names)},

  {"ALLOC_ALIGN", "16",
   "Minimal alignment of allocated blocks",
   ucs_offsetof(ucm_global_config_t, alloc_alignment), UCS_CONFIG_TYPE_MEMUNITS},

  {"EVENTS", "yes",
   "Enable memory events",
   ucs_offsetof(ucm_global_config_t, enable_events), UCS_CONFIG_TYPE_BOOL},

  {"MMAP_HOOK_MODE", UCM_DEFAULT_HOOK_MODE_STR,
   "MMAP hook mode\n"
   " none   - don't set mmap hooks.\n"
   " reloc  - use ELF relocation table to set hooks.\n"
#if UCM_BISTRO_HOOKS
   " bistro - use binary instrumentation to set hooks.\n"
#endif
   ,ucs_offsetof(ucm_global_config_t, mmap_hook_mode), UCS_CONFIG_TYPE_ENUM(ucm_mmap_hook_modes)},

  {"MALLOC_HOOKS", "yes",
   "Enable using glibc malloc hooks",
   ucs_offsetof(ucm_global_config_t, enable_malloc_hooks),
   UCS_CONFIG_TYPE_BOOL},

  {"MALLOC_RELOC", "yes",
   "Enable installing malloc symbols in the relocation table.\n"
   "This is unsafe and off by default, because sometimes glibc\n"
   "calls malloc/free without going through the relocation table,\n"
   "which would use the original implementation and not ours.",
   ucs_offsetof(ucm_global_config_t, enable_malloc_reloc), UCS_CONFIG_TYPE_BOOL},

  {"CUDA_RELOC", "yes",
   "Enable installing CUDA symbols in the relocation table",
   ucs_offsetof(ucm_global_config_t, enable_cuda_reloc),
   UCS_CONFIG_TYPE_BOOL},

  {"DYNAMIC_MMAP_THRESH", "yes",
   "Enable dynamic mmap threshold: for every released block, the\n"
   "mmap threshold is adjusted upward to the size of the size of\n"
   "the block, and trim threshold is adjust to twice the size of\n"
   "the dynamic mmap threshold.\n"
   "Note: dynamic mmap threshold is disabled when running on valgrind.",
   ucs_offsetof(ucm_global_config_t, enable_dynamic_mmap_thresh),
   UCS_CONFIG_TYPE_BOOL},

  {"DLOPEN_PROCESS_RPATH", "yes",
   "Process RPATH section of caller module during dynamic libraries opening.",
   ucs_offsetof(ucm_global_config_t, dlopen_process_rpath),
   UCS_CONFIG_TYPE_BOOL},

  {"MODULE_UNLOAD_PREVENT_MODE", "lazy",
   "Module unload prevention mode\n"
   " lazy - use RTLD_LAZY flag to add reference to module.\n"
   " now  - use RTLD_NOW flag to add reference to module.\n"
   " none - don't prevent module unload, use it for debug purposes only."
   ,ucs_offsetof(ucm_global_config_t, module_unload_prevent_mode), UCS_CONFIG_TYPE_ENUM(ucm_module_unload_prevent_modes)},

  {NULL}
};

UCS_CONFIG_REGISTER_TABLE(ucm_global_config_table, "UCM", UCM_CONFIG_PREFIX,
                          ucm_global_config_t, &ucs_config_global_list)

UCS_STATIC_INIT {
    (void)ucs_config_parser_fill_opts(&ucm_global_opts, ucm_global_config_table,
                                      UCS_DEFAULT_ENV_PREFIX, UCM_CONFIG_PREFIX, 0);
}
