/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#if defined(__x86_64__)

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <ucs/arch/global_opts.h>
#include <ucs/config/parser.h>

ucs_config_field_t ucs_arch_global_opts_table[] = {
#if ENABLE_BUILTIN_MEMCPY
  {"BUILTIN_MEMCPY_MIN", "auto",
   "Minimal threshold of buffer length for using built-in memcpy.",
   ucs_offsetof(ucs_arch_global_opts_t, builtin_memcpy_min), UCS_CONFIG_TYPE_MEMUNITS},

  {"BUILTIN_MEMCPY_MAX", "auto",
   "Maximal threshold of buffer length for using built-in memcpy.",
   ucs_offsetof(ucs_arch_global_opts_t, builtin_memcpy_max), UCS_CONFIG_TYPE_MEMUNITS},
#endif
  {NULL}
};


void ucs_arch_print_memcpy_limits(ucs_arch_global_opts_t *config)
{
#if ENABLE_BUILTIN_MEMCPY
    char min_thresh_str[32];
    char max_thresh_str[32];

    ucs_config_sprintf_memunits(min_thresh_str, sizeof(min_thresh_str),
                                &config->builtin_memcpy_min, NULL);
    ucs_config_sprintf_memunits(max_thresh_str, sizeof(max_thresh_str),
                                &config->builtin_memcpy_max, NULL);
    printf("# Using built-in memcpy() for size %s..%s\n", min_thresh_str, max_thresh_str);
#endif
}

#endif
