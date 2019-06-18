/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_ARCH_GLOBAL_OPTS_H
#define UCS_ARCH_GLOBAL_OPTS_H

#include <ucs/config/parser.h>

#if defined(__x86_64__)
#  include "x86_64/global_opts.h"
#elif defined(__powerpc64__)
#  include "ppc64/global_opts.h"
#elif defined(__aarch64__)
#  include "aarch64/global_opts.h"
#else
#  error "Unsupported architecture"
#endif

extern ucs_config_field_t ucs_arch_global_opts_table[];

void ucs_arch_print_memcpy_limits(ucs_arch_global_opts_t *config);

#endif
