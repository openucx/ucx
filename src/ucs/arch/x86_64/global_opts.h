/**
* Copyright (C) Mellanox Technologies Ltd. 2019.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCS_X86_64_GLOBAL_OPTS_H_
#define UCS_X86_64_GLOBAL_OPTS_H_

#include <stddef.h>

#include <ucs/sys/compiler_def.h>

BEGIN_C_DECLS

#define UCS_ARCH_GLOBAL_OPTS_INITALIZER {   \
    .builtin_memcpy_min = UCS_MEMUNITS_AUTO, \
    .builtin_memcpy_max = UCS_MEMUNITS_AUTO  \
}

/* built-in memcpy config */
typedef struct ucs_arch_global_opts {
    size_t builtin_memcpy_min;
    size_t builtin_memcpy_max;
} ucs_arch_global_opts_t;

END_C_DECLS

#endif
