/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "parser.h"

#include <ucm/api/ucm.h>
#include <ucm/util/log.h>


#define UCM_CONFIG_PREFIX   "MEM_"

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

  {"MMAP_RELOC", "yes",
   "Enable installing mmap symbols in the relocation table",
   ucs_offsetof(ucm_global_config_t, enable_mmap_reloc), UCS_CONFIG_TYPE_BOOL},

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

  {"ENABLE_SYSCALL", "no",
   "Use syscalls when possible to implement the functionality of replaced libc routines",
   ucs_offsetof(ucm_global_config_t, enable_syscall),
   UCS_CONFIG_TYPE_BOOL},

  {NULL}
};

UCS_CONFIG_REGISTER_TABLE(ucm_global_config_table, "UCM", UCM_CONFIG_PREFIX,
                          ucm_global_config_t)

UCS_STATIC_INIT {
    (void)ucs_config_parser_fill_opts(&ucm_global_opts, ucm_global_config_table,
                                      NULL, UCM_CONFIG_PREFIX, 0);
}
